# Review 1 of the implementation of Unit I2 (rustmath-integers factorization)

Scratch dir: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/I2-integers-factorization.rev1/`
(`$R` below). Base: HEAD `7c4d057` (I3 and I1 committed on `c79a582`); the uncommitted diff
is this unit. I read the design (all 1260 lines, both revisions), the three design reviews, the
I1 and I3 implementation reports (handoffs), the implementation report, the whole diff and every
new file in full (`factorization.rs`, `quadratic_sieve.rs`, `quadratic_sieve/siqs.rs`,
`quadratic_sieve/gf2.rs`, both new integration test files).

Verdict: **approved**. No blocker. Tests pass. Seven minor items below, all cheap (docs, one
memory copy, two cost cliffs on numbers with many prime factors).

## Acceptance commands (run here)

| command | result |
|---|---|
| `cargo test -p rustmath-integers --no-fail-fast` | 400 passed, 0 failed, 6 ignored (lib 233, drivers 9, limited 14, primality 18 + 2 ign, QS 12 + 1 ign, residues 30, signs_roots 38, ring_predicates 1, doc 45 + 3 ign) |
| `cargo test --release -p rustmath-integers --test ch18_quadratic_sieve -- --ignored` | 1 passed, 3.66 s (acceptance < 5 s); `target/release` deleted afterwards |
| `cargo test -p rustmath-numbertheory --no-fail-fast` | 46 passed, 0 failed |
| `cargo check --workspace --tests` | passes |
| `cargo clippy -p rustmath-integers --tests` | no warning in new or changed lines (remaining: `prime.rs:913`, `prime.rs:1509`, `ecm.rs:155/201/619`, `integer.rs`, `sage_wrapper.rs`, all pre-existing) |
| `rustfmt --edition 2021 --check` on the 6 new files | clean; crate-wide `cargo fmt --check` fails only on pre-existing diffs (prime.rs 8 = 8 at HEAD, ecm.rs 10 = 10, factorint.rs 4 < 6; integer.rs, sage_wrapper.rs untouched), review 3 m2 |
| `grep -rn unsafe` on the new files | no hit |
| debug CPU time | `ch18_quadratic_sieve` 4.9 s user (< 8 s), `ch18_limited_factorization` 5.2 s user (< 6 s) |

`git status`: only the unit's files; no `rustmath-misc/output.txt` or `.tmp_dir_*` change.

## Reference values

`$R/gp/refs.gp`, `refs2.gp` (PARI/GP 2.15.4): all nine semiprime rows of the QS table
(`factor(p*q) == [p,1;q,1]`), N150 (150 digits, 497 bits), F7, the p q r, p^2 q and 10000019 inputs,
psi_12, P30, p20, 2207, 2209, 666958488061768516158211, 101 * 103 * P30, p15 q15, q12^2, 42525,
101 p20 q20 and `#Str(p20 q20)` = 39, 79839848800024750353128, p25 q25, precprime(2^40),
nextprime(2^40), the next prime 1099511627803, the product with P30, the forprime count 2363,
primepi(2^16) = 6542, `#Str(2^80)` = 25, 12780049, 15347, 165731, 3127, 10403, 2 P30, 6 p15 q15,
`#binary(N40)` = 131, the k = 3 factor base list and t_5..t_29 = [2,2,1,1,3,7,7,13], the Tonelli
values of p = 23616331489 (least non-residue 107), 101 * P30, 270050147102784401. Every value in
the tests matches.

## Independent checks

- **Fuzz against gp** (`$R/fuzz.rs`, built against the current debug rlib with `$R/build.sh`; inputs
  `$R/gp/inputs.txt` from `gen.gp` / `gen2.gp`, 940 lines with gp's `factor`): 300 semiprimes with
  both primes in (650, 6000) (just above p_max of the default base), 200 balanced and unbalanced
  semiprimes, 80 p^2 q, 80 p q r, 30 p^3 q^2, 40 Carmichael numbers (6k+1)(12k+1)(18k+1), 60
  primes of 3 to 40 digits, 30 prime powers p^2..p^5 with p > 47, 40 inputs 2^a 3^b 47^c times a
  prime, 10 primes just above psi_12 with p, 3p, p q, 30 twin-like semiprimes p * nextprime(p),
  20 balanced 25- to 30-digit semiprimes. Per input: `quadratic_sieve_factor` (proper divisor, or
  `None` exactly for primes), `quadratic_sieve_factor_complete` of n and -n equal to gp,
  `factorization_with_limits` with the defaults equal to gp and complete, with `proof: false`
  (no composites), and with five capped settings ((100, 0, Some(0)), (0, 50, Some(0)),
  (10, 0, Some(25)), (1000, 300, Some(30)), (2, 1, Some(24))); `factor_pollard_rho_limited` with
  (c, s, k) = (1, 1, 100), (0, 2, 500), (-2, 3, 500), (-1, -5, 1000), (7, 0, 1);
  `trial_division_with_primes` with an unsorted list; and for n <= 90 bits three random
  `QsParams` each (F 30..330, M 64..20064, multiplier 1..1000, slack 0..39, extra 1..64, budget
  1..3000, random seed). Every LimitedFactorization was checked: value() = n, pairwise coprime,
  sorted, every `primes` entry in gp's factorization with gp's exponent, every composite not in it
  and failing BPSW. **0 failures, no panic** (`$R/fuzz_all.txt`, 332 s, worst input 4.7 s in
  debug).
- **Edge cases** (`$R/edge.rs`, `$R/edge.txt`): extreme `QsParams` on the 30-digit row (F = 30 with
  M = 64, slack 255, multiplier u64::MAX; M = 2^24 with 1024 extra; F = 2000 with M = 64) end in
  `OutOfPolynomials` in 0.03 to 0.7 s; N150 with `for_bits` and budget 1 in 0.5 s; 53^7 and
  53 * P30 give 53; rho with c = -2, s = 2, k = u64::MAX splits (10^9+7)(10^9+9) in 32 ms; c = 0,
  s = 0 splits 91; 2^400 + 1 with `mpqs_limit: Some(30)` returns in 80 ms with value() = n.
  `engine_requeues_composite_divisors` really reaches the composite path: with seed 1 the sieve
  returns 3000000040000000133 = 1000000007 * 3000000019 for the p q r input (seed 2 gives the
  composite p^2 = 10000001400000049 for the p^2 q input).
- **Release timings** (`$R/rtime*.rs`, against the release rlib before deleting it): QS 40 digits
  51 ms, 50 digits 0.34 s, 60 digits 2.3 s (design 77 ms, 0.35 s, 2.4 s); p12 * P30 0.081 s by the
  sieve, 0.095 s by `factorization_with_limits` with the defaults (design 0.08 s); trial division
  of p15 q15 to 10^8 in 0.33 s and to 10^9 in 3.2 s (doc: "10^9 about 4 s"). The cost claims hold.
- **Algorithm re-read**: factor base (Euler criterion, p | N caught for every p <= p_max, p = N
  gives `IsPrime`), a-supply guard and Fisher-Yates draw, B_l and b^2 = kN mod a, Gray code (sign
  rule and root update r -= e Bainv2), u32 root arithmetic without overflow (r + p < 2^32),
  candidate check (completeness of the root test, a-factor exponent +1, exact assertion), |y|
  de-duplication before the cap test, cap of 16 F first partials, stop at R rows, budget before
  every polynomial and every draw, GF(2) elimination (non-pivot rows only receive pivot rows;
  histories independent), square root over the multiset of relations, gcd checked against 1 and
  N. Brent rho with the exact budget (the replay always ends inside the last block, because q was
  coprime to m before it), `rho_rounds` (at least one step per round), `classify`,
  `has_at_most_digits`, `coprime_refine` (product preserved, sum of log2 bases drops), the
  two-call trial division across 2^40, the engine queue and the refinement step. No path reports
  an undecided value as decided: `primes` entries come from the sieve, exactness below psi_12, or
  `try_is_prime` = Ok(true); `composites` from a BPSW failure or Ok(false); a bounded search that
  fails returns `None` / `OutOfPolynomials` / a composite entry.
- **Tests and assertions**: every design test of 1.6, 2.4 and 3.5 exists (plus five extra). The two
  HEAD assertions that pinned the broken sieve (`quadratic_sieve_factor(165731) == Some(3127)`,
  `quadratic_sieve_factor(3127) == None`) are corrected with gp's `factor` (the report says so);
  the complete-driver assertions are kept. `test_quadratic_sieve_small` / `_larger` now pin the
  exact divisor. The deleted `test_legendre_symbol`, `test_trial_factor`,
  `test_factor_base_generation` went with their functions (fate table). Nothing is weakened.
  The new tests fail on HEAD (new API; HEAD's sieve returns `None` on every row).
- **Callers**: no use of the changed or new names outside `rustmath-integers` (grep of `*.rs`
  outside `target/`); `rustmath-numbertheory` (glob re-export of `prime::*`) builds and passes.
- **Handoffs**: I1's handoff (BPSW for splitting, `try_is_prime` to certify, `for_each_prime`
  contract, `pow_mod_u64`) is used as stated; I3 has none for I2. Review 3's four minors and its
  note are folded; nothing skipped.

## Minor

- **n1. The matrix is stored twice during elimination, so the documented memory is low.**
  `run` keeps `rows` alive (`siqs.rs:959-961`) and `left_nullspace` copies it
  (`gf2.rs:31`, `rows.to_vec()`). The peak of the matrix step is then rows + copy + history,
  about 3/2 of the stated R (F + 1 + R) / 8 bytes: about 1.6 GB at F = 65536 instead of the
  1.07 GB of the `QsParams` doc (`quadratic_sieve.rs:73-77`), and about 2 GB at the caps instead of
  "about 1.5 GB". Required change: pass the rows by value (`left_nullspace(rows: Vec<Vec<u64>>,
  ..)`, moved from `run`), which makes the doc true, or correct the numbers.
- **n2. Trial division above 2^16 runs one BPSW test of the whole cofactor per prime found.** On a
  number with many prime factors in (2^16, limit] this dominates. Release, n = product of the
  primes in [65536, hi) with `trial_division_limit: hi` (`$R/rtime3.txt`): 232 primes 1.15 s,
  393 primes 9.4 s, 586 primes 46.6 s; `trial_division_with_primes` with the same list 0.4 to 2 ms,
  and `factorization_with_limits` with the default limit 10000 factors the 393-prime product in
  1.05 s (`$R/rtime4.txt`). The `FactorizationLimits` cost paragraph only states the per-prime
  remainder cost. Required change: document the cost (one BPSW test of the remaining cofactor
  for each prime found above 2^16), or test less often, for example only when the number of primes
  visited since the last test has reached the number visited before it (at most a factor 2 more
  primes after the cofactor becomes prime, O(log) tests; the early stop of
  `limits_trial_division_huge_limit` and `trial_division_crosses_2_40` still fires at the first
  prime above 2^16 because the first test happens there).
- **n3. `coprime_refine` is quadratic in the number of trial-division primes.** Step 4 of
  `limited_engine` refines every recorded piece, the trial-division primes included, although they
  are distinct primes that were divided out completely, so they are coprime to everything else.
  Release: primorial(40000) with (40000, 0, Some(0), false) 2.3 s (4203 primes; the trial division
  itself is a few ms), 10000! with the defaults 0.74 s, 40000! with limit 40000 18 s
  (`$R/rtime.txt`, `rtime4.txt`); debug is several times slower. Required change: refine only the
  pieces recorded by the queue and append the trial-division primes afterwards (they keep
  `Class::Prime`), or document the cost.
- **n4. `trial_division_with_primes` doc overstates which entries are checked.** The doc says
  "Every list entry that divides n is checked to be prime" and panics "if ... an entry that
  divides n is proven composite" (`factorint.rs:416-424`). The code tests an entry only when it
  divides the cofactor left at that point: (12, [2, 4]) returns primes [(2, 2), (3, 1)] and does
  not panic, although 4 | 12 (`$R/edge.txt`). The output is correct. Required change: say "an
  entry that divides the cofactor left when it is reached".
- **n5. Comments that go stale at commit.** New comments name "HEAD" as the old code
  (`quadratic_sieve.rs:423, 434, 474, 489`, `gf2.rs:104`, `siqs.rs:979, 1001`,
  `prime.rs:1767`, `factorint.rs` tests "HEAD reported ...", "HEAD stopped ..."), "the design
  table" (`siqs.rs:972, 1239`), and the scratch path `scratchpad/i2impl/brent.py`
  (`prime.rs:1769`), which is not in the repository. Required change: say "the sieve before
  this change" (or name the commit), describe the semiprime rows by the gp command that the
  integration test header already gives, and state the Brent step count as a regression value of
  the budget rule without the scratch path. The repository has precedents for scratch-path
  comments, so this is cosmetic.
- **n6. Process reference in a public doc.** `factorization.rs:28-30` says "one design review
  measured 0.45 s per curve and 3 of 160 curves ...". Required change: state the measurement
  itself ("measured: 0.45 s per curve in release, 3 of 160 curves found the factor").
- **n7. Small doc slips.** `quadratic_sieve_factor_with_params` says a prime n > 47 "runs the sieve
  and gives NoFactor"; it gives `NoFactor { dependencies: 0 }` without sieving when n <= p_max (the
  scan meets n), and can give `OutOfPolynomials` when the budget runs out. `choose_a` gives up
  after 1001 repeats (`repeats > MAX_REPEATED_A`), the doc says 1000. `sieved` (`siqs.rs:438`)
  says "j >= 1" but does not test it (all callers start at 1). Required change: align the three
  docs with the code.

## Items covered

| item | status |
|---|---|
| 1 QS sieve stage | SIQS ported as designed (multiplier, factor base on the segmented sieve, guard, Gray code, log sieve, single large prime, de-duplication, cap, stop at R rows, per-polynomial budget, dense GF(2), square root); verified on gp inputs from 6 to 60 digits and on random parameters; pinned tests replaced with gp values. n1, n7 |
| 2 `trial_division_with_primes` | `LimitedFactorization`, no early break, entries classified, cofactor classified without a proof. n4 |
| 3 capped drivers, third return value | `factor_pollard_rho_limited`, `factorization_with_limits`, `LimitedFactorization` with the MAGMA semantics of the design (ECMLimit := 0, MPQSLimit < 25 disables, composites only with a set limit, Proof); exact Brent budget; complete QS driver on the engine with the I1 panic policy. n2, n3, n6 |
| 4 QS `legendre_symbol` | deleted; word Euler criterion and word Tonelli-Shanks; no error value swallowed |
| 5 ECM module doc | the design's text; the curve comment matches `ecm.rs:331-386` |
