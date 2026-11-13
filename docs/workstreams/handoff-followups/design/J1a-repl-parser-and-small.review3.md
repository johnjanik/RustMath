# Review 3 of J1a-repl-parser-and-small (design with Revisions 1 and 2)

Reviewer scratch: `$X = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/J1a-repl-parser-and-small/review3`.

| Path | Content |
|---|---|
| `$X/proto/` | The revision-2 prototype rebuilt from `$S/r1/proto/src/repl.rs` plus `$S/r2/patch_r2.py` (md5 `8d3e32be...`, equal to `$S/r2/proto/src/repl.rs`); `build.sh` (test binary), `build.sh lib` (driver) |
| `$X/head/` | HEAD's `repl.rs` (md5 equal to the tree at c79a582) built the same way, so both drivers share every dependency |
| `$X/rel/` | build log of an optimized (`-O`) prototype driver, used for the nesting-depth rows |
| `$X/t1.txt` .. `$X/t9.txt`, `*_head.out`, `*_proto.out` | this review's REPL inputs and outputs |
| `$X/fz/` | differential fuzz of the parser: `gen.py` (3000 random exact expressions with signs, powers, `/`, groups, integer and rational variables), `gen_sym.py` (3000 with a symbolic variable), `ref.py` (Python `Fraction` reference, Python precedence) |
| `$X/f64/` | differential fuzz of `eval_to_f64`: HEAD and prototype with a public hook, `gen.py` (4000 bound texts with unary signs, `e`, `tau`, `PI`, decimals, Float/Integer/Rational variables, negative and rational exponents, groups), `cmpref.py` (classification against gp at 120 digits) |
| `$X/bigp.txt`, `$X/primes_gen.gp` | random primes of 3500 to 5000 bits (gp) for HEAD's `sqrt_mod` timing |
| `$X/c.gp`, `$X/s.gp` | gp checks of `count_bound`, `primepi`, `primes`, `nextprime` and the `sqrt_mod` rows |

Binaries are deleted (about 90 MB each); `proto/build.sh`, `head/build.sh` and the rustc lines in the design rebuild them.

Verdict: **approved**. No blockers. Eight minor issues, listed below. None of them changes an algorithm, and each
has a small fix that the implementer can apply.

## What I reproduced

- **Prototype.** 74 of 74 tests pass (63 existing, 11 new), 6.6 s in debug. HEAD has 63 tests in `repl.rs`.
- **Defects at HEAD.** On the HEAD driver: `2 * -3`, `2 ** -1`, `3 - -2`, `-x`, `(2+3)*4`, `"a-b"`, `s = "x,y,z"`,
  `'x,y'`, `print("x,y", 1)`, `Polynomial([gcd(4, 6), 1])` are SyntaxErrors; `-2**2` and `-2^2` give 4;
  `y + z - y` gives `(y + (z - y))`; `π*2` panics at `repl.rs:9597:32`; `h / r` with `r = 0/5` panics in
  `rational.rs`; `"(" + ")"` returns the whole text as one literal. `find_root(w^2 == 4, -3+2, 3)`,
  `(w^2 == 9, -pi-1, 0)`, `(w^2 == 16, 0, pi2)` with `pi2 = 5` give "No root found"; `(w^2 == 2, -2^2, 0)`
  gives +1.414; `(.., 0, 1/0)` and `(.., 0, 1e400)` give `inf`; `(.., 0, 2pi)` gives 1.414. `pochhammer(3, 4)`
  = 0, `pochhammer(5, 3)` = 60. `mean([r2])` = NaN, `variance([r2, 1])` = NaN, `fft([r2])` = `[NaN + 0i]`,
  `mean([2^1100])` = inf, `median([r2, 1, 2])` panics at `statistics.rs:60:44`, `1e400` = inf. The cited
  lines (`:2348`, `:2703`, `:2718`, `:3018`, `:3024`, `:3166-3204`, `:4579-4595`, `:5472-5485` and the five
  other option loops, `:5176-5181`, `:8874-8927`, `:10949-11015`) say what the design says.
- **Fixes on the prototype.** Every S9, S14 and N5 row I ran gives the designed value (`$X/t1b_proto.out`,
  `$X/t2_proto.out`, `$X/t5.txt`): `-6`, `""`, `1/2`, `-4`, `-5`, `5`, `20`, `((y + z) - y)`, `"éa"`, `(2 * π)`,
  `1/6`, `1/2`, `3/4`, `4`, `3/2`, `100000000000000000000/3`, `9/4`, `-1/4`, ZeroDivisionError for `0 ** -1`
  and `h / r`, `512`, `--5` = 5, `x1e-3` = 4, `-"a"` and `-[1, 2]` TypeErrors naming `String` and `List`, the
  find_root rows 2, -3, -1.414, 4, ZeroDivisionError, ValueError, SyntaxError, the review-2 regression row
  "No root found" and the absorbed-overflow row 2, the `|exponent|` messages; all `chart` rows of 5.4
  including `check_chart` in `exterior_derivative` and `wedge`; the N5 rows (9 arrows for `plot_points=3`,
  16 for `plot_points = 4`, 625 for `50/2`, the range, twice, non-integer and `contours`/`n_contours` errors).
- **Parser fuzz (`$X/fz/`).** 3000 random exact expressions: the prototype equals the Python `Fraction`
  reference on all 3000 (one row differs only by the driver's 300-character display cut). HEAD was right on
  247, wrong on 2 (`-1**2+20 ^ 2`, `-5^0`) and an error on the rest. No expression that HEAD evaluated
  changes value. 3000 expressions with a symbolic variable: no HEAD value becomes an error; the 17 symbolic
  values that both builds produce agree with the reference at w = 7/3 and w = -5/2. By hand I also went through
  every pair of `apply_binary_op` arms (`:9666-9968`) for which HEAD's grouping `A + (B - C)` or `A * (B / C)`
  differs from the new left fold: none gives a HEAD value and a new TypeError.
- **`eval_to_f64` fuzz (`$X/f64/`).** 4000 bound texts. Every text without a real operand that has a value
  in the double range (1344) gives the correctly rounded double of its exact value; the 7 outside the range are
  ValueErrors (HEAD gave `-inf` for 2 of them) and 2 divisions by zero are ZeroDivisionErrors. Texts with a real
  operand: 2018 correctly rounded, 595 within 4 ulps, 16 further off, all by cancellation after a real operand
  (for example `-66/21 + - PI+--tau`), which is the documented 53-bit semantics; 11 ValueErrors whose gp value
  is outside the double range, 2 ZeroDivisionErrors. No text that HEAD rounded correctly is changed. 5
  ValueErrors where gp has a double value: an exact operand above the double range (10^300 times 10^170)
  meets a real one, or a quotient underflows after coercion. These are honest errors, not values.
- **S11 library fix.** num-rational 0.4.2 `ratio_to_f64` (read in the registry source): fast path for parts
  within 2^53, early inf and 0 for exponent differences out of range, a 55-56-bit quotient with the
  remainder as sticky bit. It is correctly rounded and never NaN. rustmath-rationals already depends on it.
- **S15 graphs.** HEAD release peaks: `G = complete_graph(3585)` 392 MB (model 397 MB), `G = path_graph(1187500)`
  393 MB (model 399 MB), bare `Graph(2770833)` 263 MB (model 266 MB). The model holds within 2 %.
- **S15 primes and prime_pi.** gp reproduces every value: `count_bound` 6, 43, 2171, 144764, 1000000 at
  7943969 (2y/ln y = 1000000.907, no rounding edge), 1000001, 1006608; `primepi(10^6)` 78498,
  `primepi(6*10^9)` 279545368; both `primes` lists of 6.4; 69, 297, 75 for the next primes after 10^400,
  2^1000, 2^512; 258 and 7 primes; 29 primes above 2^1024 starting at +643, +1081, +2113. HEAD release:
  `prime_pi(10^5)` 2.9 s, `primes(0, 10^5)` 2.8 s, `primes(10^30, 10^30 + 20000)` 0.71 s,
  `primes(2^512, 2^512 + 2000)` 0.45 s, `primes(10^30, 10^30 + 10^5)` 4.2 s (design: 3.5 s). I recomputed the
  two-pass model for four rows of 6.3 (2.2 s, 2.9 s, 101 s, refusal at k = 3): same numbers.
- **sqrt_mod.** gp reproduces the 2^127+1, p20·q21, psi_12 (4 and 73) and 41 rows. Python reproduces the
  eight roots of 9 mod 1009·1013·(2^89 - 1) and T = 2000, 1010, 500 (with integer division, see m6).
  HEAD release on a prime n: 3500 bits 10.2 s, 4000 bits 15.8 s, 4096 bits 17.0 s, 4200 bits 18.6 s, 5000 bits
  35 s (`$X/bigp.txt`). So the 4096-bit bound of phase 2 refuses no prime n that HEAD answered in 10 s; above
  it HEAD needed more than 17 s. I re-derived the "no new refusal" argument for composites: HEAD runs up to
  three trial divisions in 10 s, while phase 3 gets one at the full budget on cofactors no larger than HEAD's,
  and every cofactor HEAD could leave is below T_old^2 < psi_12, so I2's BPSW decides it. The argument holds.
- **Interfaces of earlier units** (read in their designs): I1 `is_probable_prime` (exact below psi_12, trial
  division below 1000, then BPSW), `try_is_prime` with its verified cache, `for_each_prime(lo, hi)` half-open,
  `try_sqrt_mod_prime`; I2 `FactorizationLimits { trial_division_limit, pollard_rho_limit, mpqs_limit, proof }`
  with `Some(x < 25)` disabling the sieve, `LimitedFactorization { sign, primes, probable_primes, composites }`;
  MF1 4.3 and 4.4 item 5; C1 and J1b leave `pochhammer` to J1a and switch the arms after it. The names match.
- **Citations.** ch. 25 §25.1.2 (`Chapter_25_Real_and_Complex_Fields.md:56-58`) says what 2.2 item 10 uses:
  in a binary operation with one real and one integer or rational operand, the real's field is used.
- **Callers.** `Plot3DOptions`, `parse_from_args`, `eval_simple_arithmetic`, `sqrt_mod_report`, the graph,
  `prime_pi`, `primes` and `pochhammer` arms occur only in `rustmath-jupyter/src/repl.rs`; no crate depends on
  rustmath-jupyter; no other design changes `Rational::to_f64`; F1 does not touch the finitefields square-root
  functions; `eval_to_f64` is called only for ranges and bounds, never per point.

## Blockers

None.

## Minor

- **m1. N5: `parse_color` still ignores what it cannot parse.** `Plot3DOptions::parse_color` (`repl.rs:277-294`)
  maps an unknown color name to the default blue (`_ => "#6666ff"`) and accepts any text after `#` or `0x`.
  That is the N5 defect ("ignore values they cannot parse") inside the function the design rewrites, but 1.2
  does not list colors among its errors. Required: an unknown color name, and a `#`/`0x` value that is not 3 or
  6 hex digits, is a ValueError naming the key; add `parse_from_args(&["color='foo'"])` is Err to the 1.4 test.
  (The function has no caller, so this is minor.)
- **m2. S11: `N` of a huge exact number still shows `inf`.** `N`/`evalf` (`:1935-1941`) uses
  `try_eval_to_f64` (`:10287-10288`), whose `Expr::Integer(n) => n.to_f64()` gives `Some(inf)` above 2^1024.
  `N(2^1100)` returns the String `"inf"` on HEAD and on the prototype (`$X/t4_p.out`), and after the library fix
  `N` of a rational above the double range gives `"inf"` too. 4.2 says "an inf there is filtered by the plotting
  code", but `N` is not plotting code. `N(r2)` (HEAD `"NaN"`) does become `1.0000000000` through the library
  fix. Required: `N` refuses a non-finite result with the `exact_to_f64` wording, or section 9 lists it and 4.2
  drops the sentence.
- **m3. S11 tests miss the two sites the HANDOFF names first.** `repl.rs:3018` and `:3024` are the
  `integrate_num` bounds. 4.2 rewrites them, but 4.4 has no `integrate_num` row. HEAD:
  `integrate_num(x*1, x, 0, Rational(2*big + 1, big))` = NaN and `integrate_num(x*1, x, 0, 2^1100)` = inf
  (`$X/t8.txt`). Add both: the first is 2 (gp `intnum(x=0, 2, x)` = 2; the bound rounds to 2.0), the second a
  ValueError "integrate_num: upper bound: ... is outside the range of a double". Note also that the prototype
  rewrites only `list_to_floats` and `eval_to_f64`: on it `fft([r2])` is still `[NaN + 0i]` and the 4-argument
  `find_root(x - 1, x, 0, Rational(2*big + 1, big))` is still 0 (`$X/t7.txt`), so those two rows of 4.4 are
  unverified. The design specifies the fix for `list_to_complexes`, `value_to_f64` and `integrate_num`; the
  implementer must not stop at the prototype.
- **m4. A new abort path for deep alternating nesting.** HEAD answers `-(-(...(1)...))` and `(1+(1+...))` of any
  depth with a SyntaxError (no grouping). After J1a these recurse once per level. Debug prototype driver: 600
  levels pass, 1000 abort (stack overflow, which kills the kernel). Optimized prototype driver: 3000 pass, 5000
  abort. HEAD's nested calls `abs(abs(...))` already abort near 3000 in release, so the threshold is no worse
  than the existing one, and section 9 documents it. But this unit adds the inputs that reach it, and
  `LexState` already tracks the depth. Recommended: refuse a bracket depth above a fixed bound (CPython uses 200)
  with a SyntaxError in `top_level_chars`/`group_inner`, with a test; at least say in 2.3 that these inputs
  change from an error to an abort beyond a few thousand levels.
- **m5. sqrt_mod phase 3: the cost bound assumes the cofactor does not shrink.** 7.2 says phase 3 costs at most
  about π(T)·c(bits) = 2·10^10 ns / ln T. That holds when no factor is found. When phase 3 finds many primes,
  T(bits(c_cur)) grows as c_cur shrinks. The only invariant is (55 + bits/11) ns <= WORK / p at each tried prime
  p, so the total is bounded by the sum of WORK/p over the primes from 10^4 to T_end, about
  WORK · (ln ln 3.6·10^8 - ln ln 10^4) = 0.76 · 20 s, about 15 s. Reaching it needs an adversarial n of about
  2·10^7 bits. Recommended: count the modelled cost of the steps taken and stop phase 3 when it exceeds
  `trial_work`. That keeps "no new refusal": HEAD's step cost follows the same model and HEAD steps through every
  odd number, so HEAD needed more than the budget on any input the cap stops.
- **m6. State the integer arithmetic of T.** The expected values of `test_sqrt_mod_factor_phases` hold only
  with u64 integer division in `T = trial_work / (55 + bits / 11)` (109 / 11 = 9). With real division T is
  1972, 995 and 492: the first case still factors, but the second then finds no factor and gives the other
  message ("no prime factor up to 995, not split by Pollard rho ..."), and the third says 492. Write the formula
  as u64 arithmetic in 7.2 and in the constant's comment.
- **m7. Two numbers disagree with their sources.** 7.2 cites "I1 release: 3.9 s mean, 5.4 s max at 1024 bits",
  and the `SQRT_MOD_PROOF_BITS` comment says "<= 5.4 s"; I1 §0 and §2.3.7 give 3.3 s mean and 4.3 s max, which
  6.3's `PRIMES_PROOF_SECONDS` uses. 6.3's table gives HEAD 3.5 s for `primes(10^30, 10^30 + 10^5)`; I
  measured 4.2 s (same machine class; the row's conclusion is unchanged). Align the numbers.
- **m8. Wording of the pass-2 refusal.** The message says "at least {k} BPSW probable primes above 2^64", while
  the test that triggers it uses k_big, the probable primes at or above psi_12 (those between 2^64 and psi_12
  are already proven by the 12 bases). Print k_big and say "BPSW probable primes that need a proof (at or above
  psi_12)". Also, the release check of 8.2 accepts `prime_pi(6000000000)` anywhere between 5 and 20 s; at 20 s an
  admitted call is twice the 10 s the limit stands for. Refit when it falls outside about 7 to 14 s.

## Coverage, correctness, plan

- **Coverage.** All seven items (N5, S9, S10 REPL part, S11, S14, S15, sqrt_mod) are covered and none is declared
  out of scope; none needs to be. The S9 grammar (unary below power, power right-associative with a signed right
  operand, left-associative sums and products) is the Python, PARI and MAGMA rule, and the fuzz confirms the
  implementation of it. S10 matches the ch. 49 digest (line 292) and the workspace's own `pochhammer.rs`.
- **Decided and Unresolved.** sqrt_mod separates proven primes, BPSW probable primes (caveat text) and
  unfactored composites (ValueError naming the piece); Jacobi -1 on any piece, composite pieces included, is
  correctly used as a proof. `primes` lists only proven primes, an unresolved ECPP proof is a ValueError, and
  a refusal answers nothing. No bounded search decides. No facade or silent truncation is introduced: the
  clamps become range errors, the integer and rational conversions refuse values outside the double range,
  and `float_arith` refuses overflow, underflow and division by zero.
- **New refusals.** The only refusals of calls HEAD answered within 10 s are `primes` answers with probable
  primes at or above psi_12 whose ECPP proofs exceed the budget (for example `primes(10^30, 10^30 + 10^5)`,
  HEAD 4.2 s). The design states this and the reason: HEAD's lists there were unproven, and after I1 each entry
  needs a proof. Review 2 asked for exactly this behaviour. I agree.
- **Plan.** Implementable in the stated order: the rationals and finitefields changes first, then S11 (its
  helper is used by S9), S9 (its scanner is used by S14 and N5), S14 with the test rewrites in the same step
  (`test_int_arg_ranges_named`, `test_form_component_limits` including MF1's R^1 block, `test_form_without_components`;
  I read all three at HEAD), N5, S10, S15, sqrt_mod. The acceptance commands cover the crate, the two libraries,
  the workspace build and the full workspace test against the 14,173 baseline, which covers every caller of
  `Rational::to_f64`.
