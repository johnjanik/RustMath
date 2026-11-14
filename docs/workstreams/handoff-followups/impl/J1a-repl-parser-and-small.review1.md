# Review 1 of the J1a implementation (REPL parser, plot options, chart, limits, sqrt_mod)

Reviewer scratch: `$X = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/J1a-repl-parser-and-small.rev1/`
(`repl.diff`, the acceptance logs `check.log`, `rat.log`, `ff.log`, `jup.log`, `rel_test.log`, `clippy.log`,
REPL inputs `t1.txt`..`t4.txt`, `rel_rows.txt`, `rel_nest.txt`, `nest_*.txt`, `p30.txt`, `p100.txt` with
outputs, gp scripts `chk.gp`, `chk2.gp`, `chk3.gp`; the driver source `drive.rs`, binaries deleted,
`target/release` deleted).

Verdict: **approved**. No blockers. Tests pass. Six minor notes below; none changes a result.

## What I checked

- **Diff read in full** (`git diff`, 3659 lines; four files, all of this unit): `Rational::to_f64`, the
  finitefields count function and re-export, and every hunk of `repl.rs` (scanner, splitters, chains, signs,
  grouping, Q arithmetic, `eval_to_f64`/`float_arith`/`real_op`, `exact_to_f64`, the S11 sites, `N`, chart,
  plot options, `Plot3DOptions`, pochhammer, graph model, `prime_pi`, `checked_primes`, `count_bound`, time
  models, `sqrt_mod_factor`, `sqrt_mod_report_with`, the depth counter, docs, tests).
- **Design coverage.** Every item of the revised design is implemented: N5 (six commands and
  `Plot3DOptions`), S9 items 1-10, S10, S11 (library fix, `exact_to_f64`, `integrate_num`, `list_to_floats`,
  `list_to_complexes`, `value_to_f64`, `eval_to_f64`, float literals), S14 (names, `check_chart` in the three
  arms, test rewrites), S15 (graphs, `prime_pi`, `primes` with both passes), sqrt_mod (phases 1-4, the report,
  the count function). The prototype tests are all present, plus the design's new ones.
- **Review 3 minors.** m1 (colors), m2 (`N`), m3 (`integrate_num` rows and the `fft`/4-argument `find_root`
  rows; I reran them), m4 (a recursion counter; better than a bracket bound, since the 10^5-parenthesis row
  still evaluates), m6 (u64 formula), m7 (doc numbers), m8 (k_big wording, `prime_pi` band) are in. m5 is in
  as asked, but see minor 1.
- **Earlier units.** I1's `PROOF_BITS` is reused (`SQRT_MOD_PROOF_BITS = super::PROOF_BITS`); I1's
  `is_probable_prime` (exact below psi_12 by the twelve bases, `primality.rs:399-411`), `try_is_prime` (cache),
  `for_each_prime` (half-open, lazy segments) and I2's `factorization_with_limits` (`primes` are below psi_12
  or ECPP-proven when `proof` is false, `factorization.rs:85-97, 210-223`) are used within their contracts. So
  every `Prime` piece is either below psi_12 or cached by `try_is_prime`, and the `assert_prime` checks of
  the finitefields functions cost nothing and cannot panic. S1 and MF1 did not land (`git log c79a582..HEAD`),
  as the report says; the test rewrites target HEAD's text correctly.
- **Correctness of sqrt_mod.** The Jacobi shortcuts (odd part of n, then each odd piece, composites
  included) are proofs. Phase 2 `Prime(m)` (loop stopped at p^2 > m, or m < trial^2 after all primes below
  `trial`) and phase 3 `Prime(cur)` (p > isqrt(cur) with every smaller prime tried; cur < 3.7e8^2 < psi_12) are
  proofs. The bound named in the "no prime factor up to {T}" message is always a bound below which every
  prime was tried. A `Composite` piece is a ValueError naming it; a `Probable` piece keeps the caveat. No
  undecided result is returned as decided.
- **Correctness of primes.** `count_bound` is a valid bound (trivial bound for y <= 54, Montgomery-Vaughan
  above, applied on (lo - 1, lo - 1 + y] with lo - 1 >= 1); `n_odd` = floor(b/2) - floor(L/2) is right; only
  proven primes are listed; an unresolved ECPP is a ValueError.
- **Reference values, recomputed** (gp 2.15 `chk.gp`, `chk2.gp`, `chk3.gp`; Python `Fraction`): all four
  sqrt_mod lists of `test_sqrt_mod_factored` (2^127+1, p20·q21, psi_12 with 4 and 73) and their first roots
  (CRT of the least roots: 79487054939, 173837379523 give 75322749390115309241067), `kronecker(41, psi12)` =
  -1, `kronecker(3, P)` = `kronecker(3, Q)` = -1 (so the per-piece shortcut decides `sqrt_mod(3, P·Q)`), the
  eight roots of 9 mod 1009·1013·(2^89-1), 1013·(2^89-1) = 627016629898045109236406418443, the 79-digit
  p40·q40, `count_bound` 2, 6, 43, 2171, 144764, 1000000 (2·7943969/ln = 1000000.907), 1000001, 1006608,
  `primepi(10^6)` = 78498, `primepi(6*10^9)` = 279545368, both `primes` lists, 69 and 297, 29 primes above
  2^1024 at +643, +1081, +2113, the counts 258, 7, 1389; the seven `to_f64` edge cases; every graph byte count
  and edge count in `test_graph_limits` (Python mirror of the model). Extra REPL rows I ran (`t3.txt`):
  `sqrt_mod(9, 360)` 24 roots, r = 297; `sqrt_mod(4, 3·5·7·11·13)` 32 roots, r = 10012;
  `sqrt_mod(-1, 221)` = [21, 47, 174, 200]; `sqrt_mod(4, (2^61-1)(2^89-1))` and `sqrt_mod(4, psi12^2)` lists;
  `sqrt_mod(2, 2^127-1)` least root 2^64: all equal to gp.
- **Parser on the debug driver** (`t1.txt`, `t2.txt`): `2 ** -1 * 3` = 3/2, `2 ^ -1 - 3` = -5/2,
  `2*-3^2` = -18, `2**-3**2` = 1/512, `-2^-2` = -1/4, `3--2` = 5, `pie-3` (a name ending in e) = 1,
  `x1e - 3` = 4, `1.e-3`, `"it's" + "x"`, `abs(-(2+3))` = 5, `+-+-5` = 5, `1/2/3` = 1/6, `a/3*3` = 10,
  `1/2^-2` = 4, `0^-1` and `(0/3)^-2` ZeroDivisionErrors, `2 * * 3` a SyntaxError. Bounds: `-pi^2` = -9.87,
  `pi^0.5`, `.5e1`, `5.`, `2**2`, `e^1`, `(pi)`, `((pi))` read correctly; `(-8)^(1/3)` (NaN), `10^400 + pi`
  (out of range), `0*pi^2000` (intermediate overflow), `1e-400 + 2` (underflow) are ValueErrors; `3pi`,
  `pi pi` SyntaxErrors; `nan`, `inf`, `sqrt(9)`, `"3"` TypeErrors. No silent inf, NaN or 0.
- **Tests fail on the old code.** Each new test has at least one row that HEAD 2e000c9 answers differently
  (SyntaxErrors, NaN/inf, the hang of `sqrt_mod(4, 2^127+1)`, `prime_pi(6000000001)` answered, unlimited
  graphs, pochhammer 0, unchecked charts and options) or that does not compile there (new functions, the
  `Result` of `parse_from_args`). No assertion was removed or weakened: the only removed assertion line
  (`coordinate_form(Mbig, Cb, 0)`) is replaced by the same expected message with the R^3 chart, an input change
  forced by S14.
- **No unsafe**, no em-dash in added lines, no clippy warning on an added line (`clippy.log` matched against
  the 2342 added lines), the 7 rustc warnings of rustmath-jupyter are all on old lines.
- **Release checks** (built here, then deleted): see the table below. The nesting bound holds in release on
  the 8 MB main thread (where the kernel evaluates: `#[tokio::main]` runs `handle_execute` in its own future,
  `handlers.rs:81`): 199 nested `abs`, 66 levels of `abs(-(...))` (198 calls), 199 nested lists, 99 levels of
  `-(...)` and of `(1+(...))` all evaluate.

## Acceptance

| Command | Result |
|---|---|
| `cargo test -p rustmath-rationals --no-fail-fast` | 102 passed, 0 failed, 3 ignored |
| `cargo test -p rustmath-finitefields --no-fail-fast` | all passed (lib 202, doctests incl. `num_square_roots_mod_prime_power`) |
| `cargo test -p rustmath-jupyter --no-fail-fast` | 89 passed, 0 failed, 1 ignored; 445 s in debug (`test_table_limits`, C1's, see the report's handoff) |
| `cargo test --release -p rustmath-jupyter --lib -- --ignored test_sqrt_mod_refusal_release` | 1 passed, 1.05 s |
| `cargo check --workspace --tests` | ok, 0 errors |
| `grep -n unsafe` on the four files | no hits |
| `cargo test --workspace` | not run (orchestrator) |

Release timing (my driver over the real `ReplContext`):

| Call | Time | Design |
|---|---|---|
| `prime_pi(6000000000)` | 11.28 s, 279545368 | 7-14 s band |
| `primes(10^30, 10^30 + 20000)` | 1.87 s | model 2.2 s |
| `primes(2^512, 2^512 + 2000)` | 2.27 s | model 2.9 s |
| `primes(10^400, 10^400 + 10)`, `primes(2^1000, 2^1000 + 200)` | below 0.05 s, `[]` | < 0.1 s |
| `primes(10^30, 10^30 + 10^5)`, `primes(2^1024, 2^1024 + 20000)` | refused at k = 1191 (0.13 s), k = 3 (0.11 s) | |
| `sqrt_mod(4, 2^127+1)`, `sqrt_mod(4, p20·q21)` | below 0.05 s, 0.06 s | < 1 s |
| `complete_graph(3585)` | 1.28 s | < 2 s |

## Blockers

None.

## Minor

1. **The phase-3 cost cap of m5 never fires.** `sqrt_mod_factor` stops phase 3 when the modelled cost
   `used` exceeds `lim.trial_work` = 2·10^10 ns, that is 20 s modelled. Review 3 bounded the uncapped worst
   case by the sum of WORK/p over the primes from 10^4 to 3.6·10^8, about 0.76 · WORK = 15 s. So the cap is
   above the worst case and above the 10 s policy; the adversarial input of m5 still costs up to about 15 s.
   Also `used` restarts for each piece of `rest`. A cap of `trial_work / 2` (10 s modelled) over all pieces
   keeps "no new refusal": HEAD tries every odd number up to p, so it spends (p/2)/π(p) >= ln(p)/2 >= 4.6 times
   the modelled phase-3 cost to reach p. Reaching the case needs an n of about 10^7 bits, so this is minor.
   Recommended, not required.
2. **Quadratic stripping of a repeated factor.** The new `odd` loop in `sqrt_mod_report_with` divides by 2
   once per bit of v_2(n), and phases 1 and 3 strip p^e one division at a time. With
   `prime_power_sqrts(.., 2, k)` (finitefields, a bit-by-bit lift) the call is quadratic in k:
   `sqrt_mod(1, (2^10000)^30)` takes 59 s and `sqrt_mod(1, (2^10000)^100)` 579 s in a debug build
   (`$X/p30.out`, `$X/p100.out`). HEAD did the same work through `IntegerMod` (trial division three times), so
   this is not a regression. `trailing_zeros` with a shift for the odd part would remove the new loop's
   share; the finitefields lift belongs to the HANDOFF.
3. **Probable primes above 4096 bits are now refused.** `sqrt_mod(4, 2^4423 - 1)` (a Mersenne prime) is a
   ValueError "... 4423 bits with no prime factor up to 43763676; testing it exceeds the REPL limit (4096 bits)"
   after 1.03 s in release. I1's version at HEAD 2e000c9 answered it with the probable-prime caveat after one
   BPSW test (about 3 strong tests, 0.2 s at this size by `PRIMES_STRONG_TEST_SECONDS`). The design approved
   the 4096-bit gate against c79a582, where such an n took more than 17 s, so this follows the design. A cheap
   improvement: for an untested cofactor of up to about 8192 bits, run `is_probable_prime` after phase 3 (about
   1 s at 8192 bits) and record a `Probable` piece. For the HANDOFF.
4. **`Rational::to_f64` allocates.** It clones both parts into a `BigRational` on every call, also for small
   values where the old code did two conversions without allocation. A fast path for parts within 2^53 (one
   division, the same double) would avoid it. Performance only; the value is right.
5. **`N` of NaN.** `N`/`evalf` reports every non-finite result as "outside the range of a double (the double
   computation gives NaN)". For NaN the first half is inaccurate. Cosmetic.
6. **Handoff for MF1 and S1** (not in the report). Both units are still to land and both edit `repl.rs` after
   this one. MF1 §4.3 replaces the R^3 block of `test_form_component_limits` by an R^1 block because the
   library refuses a one-name chart on R^3; after J1a the charts have three names, so that rewrite is not
   needed, and MF1's library dimension check in `exterior_derivative` overlaps `check_chart`. S1's design
   fixes the symbolic parser precedence (`expr("z^2/2")` displays `(z ^ 1)`, so `diff(z^2/2, z)` = 1 on the
   REPL today; S1 `test_parse_rational_precedence`), which J1a does not touch.

## Skipped review-3 minors and other handoffs

The report folds in all eight review-3 minors. m5 is folded in literally; minor 1 above explains why its
threshold has no effect. I3's handoff ("J1a/J1b": the stale binomial timing sentences of `BINOMIAL_WORK`) is
not in J1a's design; J1b's design recalibrates `BINOMIAL_WORK`, so it belongs there. F1's handoff to "J1"
(switch `IntegerMod::sqrt`, `all_sqrts`, `num_sqrts` and three other finitefields sites to
`group_order::factor_proven`) is a finitefields change outside J1a's design; the REPL no longer calls those
functions, so it stays open for the HANDOFF.
