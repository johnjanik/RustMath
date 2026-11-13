# Implementation report: Unit I3-integers-arith

Design: `fu2/design/I3-integers-arith.md` (Revision 1 body), reviews 1 and 2 (approved).
Base: HEAD `c79a582` (I3 is the first unit of the round, so no earlier unit reports existed and
no handoffs were addressed to it). Nothing committed (the orchestrator commits). All ten items
are implemented; nothing is declared out of scope.

## Per item: what changed

### Item 1: i32 / i64 gcd, lcm, xgcd (rustmath-core)

- `rustmath-core/src/traits.rs:440` `gcd_u64` (Euclid on magnitudes), `:453` `xgcd_i128`
  (truncated extended Euclid, triple negated when the last remainder is negative).
- `:471` `impl EuclideanDomain for i32` and `:558` for i64: `norm` = `unsigned_abs` (total),
  `div_rem` with `checked_div`/`checked_rem` (`MIN / -1` is `Err(NumericalError)`, was a panic),
  overrides of `gcd`, `lcm` (u64 / u128 intermediate), `extended_gcd` (i128). Documented
  `# Panics` only when MAGMA's value does not fit (gcd 2^31 / 2^63, lcm above MAX).
- Tests: `traits.rs:1041` `test_i32_i64_gcd_lcm_xgcd_signs_match_pari` (generic over both
  types), `:1047` `test_i64_euclidean_extremes`, `:1066` `test_i32_euclidean_extremes`, five
  `#[should_panic(expected = "does not fit in i64/i32")]` tests (`:1083`-`:1111`). Every value has
  its gp command in a comment (gp 2.15.4, re-run here: `[gcd(4,-6),...]`, `gcdext(-2^63,3)`,
  `gcdext(2^31-1,-2^31)`, `lcm(-2^62,3)`, ...).
- `rustmath-integers/tests/ch18_signs_roots.rs:536` `gcd_grid_fixture_through_i32_and_i64`
  (all 3,721 gp rows of `gcd_grid.csv` through both impls), `:573`
  `machine_word_gcd_agrees_with_integer` (100,000 LCG pairs incl. MIN and 0, against Integer
  and, added here, against num-integer's BigInt extended Euclid).
- Callers: generic `R: EuclideanDomain` with i32/i64 (`UnivariatePolynomial<i64>` content via
  `normalization_unit`; `ch23_quotrem_gcd.rs` `gcd_lcm_over_i64` passes). No test pinned a
  negative gcd.

### Item 2: ArithInt / ArithLLong (rustmath-integers/src/fast_arith.rs)

- API changes: `ArithInt::gcd -> u32` (`:223`), `ArithInt::lcm -> u64` (`:230`),
  `ArithLLong::gcd -> u64` (`:364`), `ArithLLong::lcm -> u128` (`:370`), all total.
  `add_mod`/`mul_mod` (`:164`, `:179`, `:306`, `:320`) return `Result`: `Err(DivisionByZero)`
  for m = 0, else the least non-negative residue modulo |m| (pow_mod's conventions).
  `is_prime` (`:240`, `:380`) is `u64::try_from(n).is_ok_and(is_prime_u64)` (deterministic).
  Private `ArithLLong::miller_rabin` deleted; `small_primes`/`get_small_primes` kept.
  New private helper `gcd_magnitudes` (`:106`). Module header documents the conventions.
- Callers: only this crate (grep over the workspace and rustmath-py). Unit-test expectations in
  `fast_arith.rs` became `Ok(..)`; `tests/ch18_residues.rs:767` became `Ok(9_223_...805)`.
- Tests (`tests/ch18_residues.rs`): `:819` `fast_arith_gcd_lcm_extremes`, `:842`
  `fast_arith_add_mul_mod_any_modulus`, `:866` `fast_arith_add_mul_mod_agree_with_integer`
  (grid plus extremes, against `Integer::modulo`), `:914` `fast_arith_int_is_prime_near_i32_max`
  (gp `isprime`, `factor(2146654199)`, and agreement with `Integer::is_prime` on
  [2^31 - 20001, 2^31 - 1] and [2^63 - 2001, 2^63 - 1]). Renamed
  `fast_arith_miller_rabin_after_pow_mod_change` to `fast_arith_llong_is_prime_mersenne_61`
  with the new comment (asserts unchanged).

### Item 3: Integer::sqrt (Isqrt)

- `integer.rs:232` `sqrt` = `BigUint::sqrt` of the magnitude (exact floor; Newton from a float
  start). Doc cites §18.4.8 and Brent-Zimmermann Algorithm 1.13. Module header line added.
- `prime.rs`: private `isqrt` deleted; `is_prime_trial_division` uses `n.sqrt()`.
- Fixtures: `tests/data/gen_i3.gp` (copy of the scratch generator with a header in the style of
  `gen_signs_roots.gp`), run in an empty directory: its four CSVs are byte-identical to the
  design's `gp/` fixtures and are committed (`isqrt_big.csv`, `ispower_root.csv`,
  `xgcd_lehmer.csv`, `binom_big.csv`, 109 KB).
- `ch18_signs_roots.rs`: `within(secs, f)` copied from `ch23_quotrem_gcd.rs` (`:55`), `m127()`
  helper, `:646` `isqrt_big_fixture` (90 rows + floor property), `:661`
  `isqrt_of_a_100k_bit_number` (`within(10)`, gp `sqrtint(3^63093+12345) % (2^127-1)` and
  `#binary`; 0.09 s debug).

### Item 4: Lehmer gcd and xgcd (half-gcd: not now)

- New `rustmath-integers/src/lehmer.rs` (crate-private, `mod lehmer;` in lib.rs): header cites
  MAGMA §18.6, [Knu97] §4.5.2 Algorithm L, [Web95], [Sch71]. `shr_u64` (`:29`), `lehmer_matrix`
  (`:50`, i128 arithmetic, entries leave through `i64::try_from(..).expect(..)`), `combine`
  (`:72`), `gcd` (`:103`, u64 fast path, `debug_assert` only when both operands exceed 128 bits,
  review 2 m4), `gcd_cofactor` (`:145`), `xgcd_u64` (`:185`), `xgcd_magnitudes` (`:202`, t by
  exact division).
- `integer.rs:1576` `EuclideanDomain::gcd` and `:1606` `extended_gcd` use the module;
  `(g, sign(a)·s, sign(b)·t)`; docs of the inherent gcd/xgcd (`:117`, `:137`), the overrides
  and the module header name Lehmer. `lcm` unchanged over the new gcd.
- Tests: `ch18_signs_roots.rs:406` `xgcd_lehmer_fixture` (24 gp rows), `:450`
  `lehmer_gcd_agrees_with_stein_and_extended_euclid` (400 LCG pairs up to 3,000 bits against
  Stein and num-integer's extended Euclid), `:500` `xgcd_of_large_operands`, `:516`
  `gcd_and_xgcd_of_100k_bit_operands` (only the two Lehmer calls inside `within(5)`, Stein and
  the identities after it, review 2 m1; 1.5 s total in debug under load). Unit tests in
  `lehmer.rs`: `lehmer_matrix_follows_euclid`, `combine_sign_cases`,
  `gcd_and_cofactor_small_and_structured`.
- Half-gcd: no code (design §4.4, estimated crossover 10^5 to 10^6 bits).

### Item 5: IsPower(n, k) with the root

- `integer.rs:454` new `pub fn exact_root(&self, k: u32) -> Option<Self>` (contract as in the
  design: positive root for even k, negative for negative odd-k input, k = 0 only for 1).
  `:433` `is_power` is `exact_root(k).is_some()`.
- New `rustmath-integers/src/perfect_power.rs` (crate-private): header cites §18.4.5, [Ber98],
  [BS93], [vzGG99] ch. 9, Knuth §4.6.3. `may_be_qth_power` and the `POWER_SIEVE_*` constants
  moved here unchanged with their test (`:47`); `OddPart`/`odd_part` (`:77`), `root_2adic`
  (`:146`, 2-adic Newton, `debug_assert` r^k ≡ a), `exact_root_odd` (`:194`),
  `exact_root_biguint` (`:229`, odd root first, then e square roots each after the
  quadratic-residue sieve, as the design specifies).
- `rustmath-rationals/src/sage_wrapper.rs:472` `SageRational::nth_root` uses `exact_root` on
  |numerator| and |denominator| (design Plan step 6).
- Tests: `ch18_signs_roots.rs:908` `exact_root_inline` (gp `ispower(..,&r)` values; the four
  `3^100003+2` cases under `within(10)`), `:943` `exact_root_fixture` (2,486 gp rows). Unit
  tests in `perfect_power.rs`: `exact_root_odd_exhaustive_small` (n ≤ 20000, odd k ≤ 15),
  `root_2adic_is_a_root` (k in {3,5,7,63,65,1001}, prec in {1,2,63,64,65,128,1000}),
  `exact_root_biguint_composite_exponents`, `power_residue_sieve_never_refutes_a_power` (moved).

### Item 6: is_perfect_power essentially linear

- `integer.rs:644` `is_perfect_power`: HEAD's prime-exponent loop with the new root step
  (sieve only for q < 32, `BigUint::sqrt` for q = 2, `exact_root_odd` on the cached `OddPart`
  for odd q, recomputed only when the base changes). Doc block rewritten (algorithm, proof
  sketch, measured costs); module header updated.
- Tests: `ch18_signs_roots.rs:888` `is_perfect_power_huge_inputs` (`within(5)`, gp
  `ispower(2^1000000+1)` etc.; 0.46 s debug with the big test run in parallel). Existing
  fixtures pass; the comment of `is_perfect_power_sieve_inline` updated (asserts unchanged).

### Item 7: SageInteger::binomial

- New `rustmath-integers/src/binomial.rs` (crate-private): header cites §18.8, Goetgheluck
  1987, Legendre, [vzGG99] §10.1. `product_tree`, `product_u64`, `binomial_nonneg` (`:76`).
  Review 2 m2 (optional part) folded in: one signed count per prime instead of `exps` and `got`.
- `sage_wrapper.rs:444` `SageInteger::binomial`: sign/zero cases kept, K = 0 and 1 return
  early, then the 2^32-bit refusal (`panic!("binomial: C(n, k) has more than 2^32 bits; not
  computed")` when K > 2^32 or K·max(1, bits(N) - 1 - bits(K)) > 2^32), then
  `binomial_nonneg`. Doc: algorithm, cost, `# Panics` with the proof sketch and the real memory
  figure ("about 10 GB" near the cap: sieve plus three words per prime, review 2 m2), and the
  §10 Sage/gp/GAP sentence.
- Tests: `ch18_signs_roots.rs:179` `binomial_big_fixture` (80 gp rows, `within(60)`, 5.0 s
  debug), `:189` `binomial_central_rows_within_10s` (0.74 s debug), `:203`
  `binomial_beyond_2_pow_32_bits_is_refused` (three refusals by `catch_unwind` and payload
  text, boundary guard `binomial(2^33,3)`), `:259` `binomial_agrees_with_the_product_formula`
  (HEAD's loop as a test-local reference, n in -40..=300, k in -3..=|n|+2), `:272`
  `binomial_1000_500`. Unit tests in `binomial.rs`: `product_tree_edges`,
  `binomial_nonneg_across_the_u64_boundary` (n = 2^64 - 1, 2^64, 2^64 + 1 and all n ≤ 60).

### Item 8: Integer::sigma exponent overflow

- `integer.rs:372` `sigma`: every exponent (a + 1)·k is formed in u64 and converted with
  `u32::try_from` in a pre-pass before any power (`Err(NumericalError)` naming the 2^31-bit
  size); then one loop for k = 1 and k > 1. `num_divisors` (`:322`) forms
  `u64::from(multiplicity) + 1`. `# Errors` documented.
- Tests: `ch18_signs_roots.rs:969` `sigma_exponent_overflow_is_an_error` (`within(10)`, four
  calls incl. `3.sigma(2^31)`; 0.01 s), `integer.rs:1876` `test_sigma_values_pari` (gp
  `sigma(2^10*3^5,3)`, `sigma(7^40,5) % (2^127-1)`, `#binary`).

### Item 9: checked to_i64 and the migration

- `integer.rs:96` `pub fn to_i64(&self) -> Option<i64>`. `integer.rs:1307`
  `impl_try_from_integer!` gives `TryFrom<&Integer>` and `TryFrom<Integer>` for i8..i128, isize,
  u8..u128, usize with `MathError::NumericalError("an integer of {bits} bits[ (negative)] does
  not fit in {ty}")`. Tests `integer.rs:1725` `test_to_i64_is_checked`, `:1736`
  `test_try_from_integer` (boundaries of every width, message text).
- Migrated sites: all 137 Integer sites of Appendix A, by class: G with
  `T::try_from(..).expect("<invariant>")`, T with `.unwrap()`, R-fix exact replacements, R-keep
  explicit `.expect("<function>: ... beyond i64 is not supported")` with a comment naming the
  reason, D deleted (`solve.rs` `expr_to_rational`, `expr_to_integer`). Where the target was not
  i64 (`as usize`, `as u8`, `as u32`, `as u64`) the conversion goes straight to that type
  (`usize::try_from`, `u8::try_from`, `u32::try_from`, `u64::try_from`), which removes the
  silent `as` truncation as well. CRLF line endings of `rustmath-algebraic/src/descriptor.rs`
  and `radicals.rs` were preserved.
- R-fix tests (each panicked or was wrong on HEAD per the design's `rev1/` reproductions):
  `rational.rs:597` `numeric_conversion_to_i64_is_checked`; `substitute.rs`
  `eval_float_of_huge_integers`; `integrate.rs` `polynomial_degree_beyond_i64_is_none`
  (`checked_add` in the Mul branch, doc of `polynomial_degree` updated); `solve.rs`
  `constant_inequality_with_huge_values` (also calls `solve_constant_inequality` directly,
  review 2 m5); `series.rs` `taylor_at_a_huge_point` (taylor, series and series_with_big_o at
  10^30, checked by evaluating at 10^30 + 1); `orthogonal_polys.rs`
  `huge_degree_stays_unevaluated`; `prime_pi.rs` `prime_pi_of_huge_arguments`;
  `factorization.rs:1372` `test_hensel_lift_prime_beyond_i64_is_refused`;
  `unramified.rs:662` `unramified_modulus_above_2_pow_63` (2^89 - 1: modulus [1, 0, 1],
  `with_modulus` degree 2, not Conway); `numberfields lib.rs:832`
  `class_number_quadratic_is_honest`.
- rustmath-py (`rustmath-py/src/rationals.rs:228-229`, R-keep) is outside the workspace and was
  edited without a compile check (review 2 m6; the commit message should say so).

### Item 10: Binomial(n, r) for r < 0

- Doc of `SageInteger::binomial` (`sage_wrapper.rs:244`ff): the Sage documentation sentence and
  examples, the gp and GAP list (re-run here: both `[0, 0, 0, -6, 0, 0, 0]`), "MAGMA
  unconfirmed" kept. `binomial_negative_upper_index_inline` gains binomial(3,-1) = 0,
  binomial(-5,-2) = 0, binomial(-2,5) = -6.

## API and behaviour changes (callers checked)

- Breaking: `Integer::to_i64 -> Option<i64>`; all 137 Integer call sites migrated (24 crates);
  the compiler confirms none is left in the workspace (`cargo check --workspace --tests`).
  The remaining `.to_i64()` calls are on BigInt, rug, MPFR, BigFloat, ComplexMPFR or generic
  `NumericConversion` receivers (the design's class N).
- Breaking (crate-internal callers only): `ArithInt`/`ArithLLong` `gcd`, `lcm` return unsigned
  wider types; `add_mod`, `mul_mod` return `Result`.
- New: `Integer::exact_root`; `TryFrom<&Integer>` / `TryFrom<Integer>` for every primitive
  integer type.
- Values unchanged, only faster: `Integer::sqrt`, gcd, lcm, extended_gcd (Lehmer gives
  Euclid's quotients), `is_power`, `is_perfect_power`, `SageInteger::binomial`,
  `SageRational::nth_root`.
- Errors instead of panics or wrong values: i32/i64 `div_rem(MIN, -1)`, `sigma` with an
  exponent beyond u32, `NumericConversion::to_i64` for Rational, `eval_float`, the R-fix sites
  of Appendix A, `hensel_lift` with p beyond i64 (`Err(InvalidArgument)`), bsdratio's Sha
  fields (`Err` instead of a silent `as u32`), quadratic class numbers (exact or
  `Err(NotImplemented)`, never the placeholder 1).
- New documented panic: `SageInteger::binomial` beyond 2^32 result bits (proven lower bound).
- Existing assertions: none weakened. The `fast_arith.rs` unit tests and
  `ch18_residues.rs:767` keep their values wrapped in `Ok(..)` (the return type changed).
  Which new tests fail on HEAD follows the design's HEAD reproductions (`rev1/`, review 2);
  they were not re-run against HEAD here.

## Deviations from the design (with evidence)

1. Class number (review 2 m3, optional part taken). Instead of `Err(NotImplemented)` for every
   quadratic field outside the nine class-one discriminants, `class_number_quadratic`
   (`rustmath-numberfields/src/lib.rs:419`) now computes imaginary quadratic class numbers
   exactly for |d| ≤ 10^7: the fundamental discriminant of the squarefree part
   (`fundamental_discriminant`, `lib.rs:595`) and the existing reduced-form count
   `hcf::class_number_imag_quadratic` (Cohen Algorithm 5.3.5). Real quadratic fields, larger
   |d| and non-integral d give `Err(NotImplemented)`. Consequences against the design's test
   list: x^2 + 5 gives `Ok(2)` (not `Err`), and `test_class_number_quadratic` (x^2 + 3) keeps
   passing honestly (h(-3) = 1), so it was not moved to x^2 + x + 1. Test values (gp):
   `quadclassunit(-20).no` = 2, `(-15)` = 2, `(-23)` = 3, `(-3)` = `(-4)` = `(-8)` = 1,
   `poldisc(x^2+9995)` = -39980, `coredisc` -9995, `quadclassunit(-9995).no` = 40 (while the
   order of discriminant -39980 has 120, so the reduction to the field discriminant is tested);
   x^2 - 2, x^2 - 10 (real) and x^2 + 10^30 + 3 give `Err(NotImplemented)`.
2. `rustmath-numberfields/src/ideals.rs:535` is listed in Appendix A as T ("test value"), but
   it is in non-test code (`rational_prime_factors`, used by ideal factorisation at `:546`;
   the test module `ideal_arith_tests` ends at `:425`). Treated as R-keep: explicit
   `expect("rational_prime_factors: a prime factor beyond i64 is not supported")`, same
   behaviour as HEAD (dropping the prime would make the factorisation incomplete).
3. `lehmer_matrix` unit test: the design says "(2^63 - 1, F_92) gives Fibonacci cofactors".
   The quotients of (2^63 - 1, F_92) are 1, 4, ... (not all 1), so the entries are not
   Fibonacci numbers there. The test checks instead, for (F_92, F_91), (2^63 - 1, F_92) and
   2,000 random pairs, that the matrix maps (uh, vh) to two consecutive Euclid remainders with
   Euclid's cofactors and determinant ±1; for (F_92, F_91) the entries are Fibonacci numbers and
   at least 40 steps are accepted.
4. `perfect_power.rs`: a debug-only check `pow_trunc(r, k, prec) == a mod 2^prec` in
   `root_2adic` as the design says; `exact_root_biguint` takes the odd root first and then the
   square roots (the design's order; the prototype did squares first).
5. Additional Integer fast paths in `lehmer.rs` not in the design text: `gcd` and
   `xgcd_magnitudes` use u64 Euclid when both magnitudes fit in 64 bits (every Rational
   normalisation of word-size values). Same quotients, so the same values; covered by the
   grid fixtures and the num-integer comparison in `machine_word_gcd_agrees_with_integer`.
6. Formatting: `rustfmt` was applied to the new files and to the touched files that were
   rustfmt-clean at HEAD (bsdratio.rs, division.rs, rizzo.rs, torsion.rs, lseries.rs,
   berlekamp.rs, ch23_berlekamp.rs, square_roots.rs, ch18_residues.rs, ch18_signs_roots.rs);
   other touched files were not reformatted.
7. Two imports became unused through the migration and were removed:
   `NumericConversion` in `rustmath-numberfields/src/lib.rs:24` (its only use was the deleted
   `from_i64(100)` placeholder) and `rustmath_rationals::Rational` in
   `rustmath-symbolic/src/solve.rs` (its only use was the deleted dead code; the test module
   imports it).

## Review notes: folded in / skipped

Review 2: m1 (Stein outside the bound) folded in; m2 (memory figure; optional one signed count
per prime) both folded in; m3 (hcf.rs:88; optional delegation) folded in, see deviation 1; m4
(debug_assert cost) folded in with the 128-bit threshold; m5 (direct
`solve_constant_inequality` test) folded in; m6 (rustmath-py unchecked) accepted, noted above.
Nothing skipped.

## Test counts (debug, this container)

All runs `--no-fail-fast`, debug profile, final tree:

| command | passed | failed | ignored |
|---|---|---|---|
| `cargo test -p rustmath-core` | 49 | 0 | 1 |
| `cargo test -p rustmath-integers` (lib 185, factorization_drivers 6, residues 30, signs_roots 38, ring_predicates 1, doc 38) | 298 | 0 | 3 |
| `cargo test -p rustmath-rationals` | 101 | 0 | 3 |
| `cargo test -p rustmath-polynomials -p rustmath-matrix` | 672 | 0 | 5 |
| `cargo test -p rustmath-symbolic -p rustmath-numberfields -p rustmath-ellipticcurves -p rustmath-modular -p rustmath-numbertheory -p rustmath-powerseries -p rustmath-crypto -p rustmath-padics -p rustmath-curves -p rustmath-algebraic -p rustmath-liealgebras -p rustmath-galois -p rustmath-groups -p rustmath-combinatorics -p rustmath-quadraticforms -p rustmath-rings -p rustmath-functionfields -p rustmath-geometry -p rustmath-finitefields -p rustmath-manifolds -p rustmath-jupyter` | 7973 | 0 | 153 |

`cargo check --workspace --tests`: passes (no error). `cargo clippy -p rustmath-core -p
rustmath-integers -p rustmath-rationals`: no warning in the new or rewritten code (the two
`is_multiple_of` hints in binomial.rs / perfect_power.rs were fixed); the remaining warnings
are pre-existing (ecm.rs, quadratic_sieve.rs, integer.rs:578/:813, prime.rs:671,
rationals sage_wrapper.rs:736, special_numbers.rs:40). Two import removals in numberfields
lib.rs and symbolic solve.rs happened while the big run was compiling; after them
`cargo check --workspace --tests` passes and the affected tests (`class_number*`,
`solve*`, `constant_inequality*`) were rerun green. The cost tests run well inside their
bounds in debug: isqrt 0.09 s (bound 10), is_perfect_power 0.46 s (5), Lehmer gcd + xgcd
about 0.4 s of a 1.5 s test (5), binomial central rows 0.74 s (10), binomial fixture 5.0 s
(60), sigma 0.01 s (10). No release build was made; the leftover `target/release` (539 MB,
from the design phase) was deleted at the start. No `rustmath-misc/output.txt` or
`.tmp_dir_*` changes.

## Left undone

- Full `cargo test --workspace` (the orchestrator runs it).
- rustmath-py is not compiled (outside the workspace, no pyo3 in the cache).
- No change to docs/*, PROJECT_TRACKER.md or docs/magma/* (orchestrator).

## Handoffs

- J1a/J1b (REPL): `repl_limits::BINOMIAL_WORK` (repl.rs:10480-10488) and the
  `checked_binomial` doc (repl.rs:10605-10626) model and measure the old K-step loop
  ("C(524288, 262143) takes 10.5 s"); after I3 that call takes about 16 ms in release (0.2 s
  in debug). The limits stay safe (over-estimates) but the timing sentences are stale. The REPL
  caps results at 2^22 bits before calling `SageInteger::binomial`, so it never reaches the new
  2^32-bit panic. The REPL has no `.to_i64()` call left; the comment at repl.rs:11999 about the
  inherent panicking `to_i64` is now historical.
- S1 (symbolic): the S7 sub-item "expansion point in series.rs:34" is fixed here (series.rs
  :34/:206/:233 use `is_zero()`); the parser sub-item (parser.rs:148-154) is untouched.
  Symbolic files edited here (rebase over them): expand.rs, integrate.rs, integrate_exact.rs,
  limits.rs, polynomial.rs, radical.rs, risch.rs, series.rs, simplify.rs, solve.rs,
  substitute.rs, specialfunctions/{orthogonal_polys,other,prime_pi,transcendental}.rs.
- C1 (combinatorics): the u32 `binomial` (HANDOFF S2/S4) can call `SageInteger::binomial`,
  now quasi-linear. `symmetric_group_representations.rs:598` was migrated here.
- Trackers (orchestrator), per design Plan step 12: PROJECT_TRACKER rows 355 (core i32/i64
  gcd/lcm/xgcd now MAGMA signs, div_rem Err for MIN/-1), 400 (sigma: exponent pre-pass, Err
  beyond u32), 417 (gcd: Lehmer measured; half-gcd absent, estimated crossover 10^5 to 10^6
  bits), 418 (IsPower: 2-adic roots, `exact_root`), 421 (fast_arith: unsigned gcd/lcm,
  Result add_mod/mul_mod, deterministic is_prime; remove "Open:"), 424 (Isqrt via
  BigUint::sqrt), 427/428 (binomial: Goetgheluck splitting, 2^32-bit refusal, Sage/gp/GAP
  cross-check for r < 0), 1625 (chevalley_basis comment fixed: no 0 fallback, explicit panic),
  1799 (class_number_quadratic: placeholder removed; imaginary quadratic exact for |d| ≤ 10^7
  by reduced forms; real quadratic and larger |d| `Err(NotImplemented)`; regulator needed for
  real fields, Cohen §5.7), 2144 (`eval_float` of a huge Integer fixed). New public API:
  `Integer::exact_root`, `TryFrom<&Integer>`/`TryFrom<Integer>` for all primitive integers,
  `Integer::to_i64 -> Option<i64>` (breaking). `docs/magma/status_overrides.csv`:
  ch018.IsPower.2 (partial -> verified with exact_root), ch018.IsPower, ch018.Isqrt,
  ch018.GreatestCommonDivisor, ch018.ExtendedGreatestCommonDivisor, ch018.Binomial,
  ch018.DivisorSigma. HANDOFF: remove the fixed follow-ups (Binomial r < 0 now cross-checked
  with PARI, GAP and Sage, MAGMA unconfirmed; S1; S5; S7 first part; the I3 items) and add the
  notes below.
- Tracker notes found while implementing (design §11, still open):
  - `Integer::nth_root(n)` (MAGMA Iroot) is slow for large n on large inputs (num-bigint's
    Newton start); a start from the leading bits would fix it.
  - R-keep sites that guard pre-existing facades (now explicit panics with a comment):
    `rustmath-powerseries/src/algebraic.rs` `i128_of` callers turn `None` into an empty root
    list; `rustmath-algebraic/src/radicals.rs:39` returns alpha itself as its n-th root;
    `rustmath-numbertheory/src/quadratic_forms.rs` counts 0 solutions for m <= 0;
    `rustmath-quadraticforms/src/local_densities.rs:231` counts only the first 1000 residues.
    Other R-keep sites: `rustmath-symbolic/src/polynomial.rs` (degree_recursive, coefficient),
    `rustmath-geometry/src/plane_curve_ops.rs:90`, `rustmath-algebraic/src/descriptor.rs:555`,
    `rustmath-liealgebras/src/chevalley_basis.rs:276`, `rustmath-quadraticforms/src/
    theta_series.rs:136`, `rustmath-numberfields/src/ideals.rs:535`, `rustmath-py`.
  - `rustmath-symbolic/src/integrate.rs` `polynomial_degree` returns `Some(-1)` for x^(-1)
    (pre-existing, heuristic tier).
  - `SageInteger::binomial` near the 2^32-bit cap needs about 10 GB (a segmented sieve would
    remove the K-byte sieve).
