# Review 3 of design I2 (rustmath-integers factorization)

Reviewer scratch dir: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/I2-integers-factorization/review3/`
(`$R` below; `$X` is the designer's dir). I read the whole design (1260 lines, Revisions 1 and 2 included),
reviews 1 and 2, and the I1 design sections the design depends on (I1 §2.1, §2.5, §2.8).

Verdict: **approved**. No blocker. Four minor items, all documentation or acceptance wording. Review 2's
blocker (the polynomial budget) is fixed in the prototype and the design, and I re-ran it.

## What I checked, and what holds

### Defects at HEAD (`$R/repro3.rs`, linked against the HEAD debug rlib)

| claim | reproduced |
|---|---|
| `quadratic_sieve_factor`: 3127 None, 165731 Some(3127), 10403 None, 15347 None, (10^9+7)(10^9+9) None, the 20- and 30-digit rows None | yes |
| `quadratic_sieve_factor_with_params(270050147102784401, 300, 10^6)` None | yes |
| `trial_division_with_primes`: (1225, [2,3]) [(1225,1)]; -1225 [(-1,1),(1225,1)]; (1001, [2,3]) [(1001,1)]; (8, [5,2]) [(8,1)]; (12, [4,3]) [(12,1)]; also (50, [2,3]) [(2,1),(25,1)] | yes |
| `gaussian_elimination_gf2` returns column supports, `try_dependency` takes a floor root of a reduced product (`quadratic_sieve.rs:282-337`, `:371-376`) | yes (read) |
| `legendre_symbol` `.unwrap_or(Integer::one())` at `:57-58`, and (a/2) = 1 at `:46-48` (gp `kronecker(4,2)` = 0) | yes (read, gp) |
| `ecm.rs:14-15` describes p - 1 / p + 1; `:322-323` calls the curve "Montgomery parameterization"; stage 2 is `.take(100)` single multiplications | yes (read `ecm.rs:297-399`) |
| `factor_pollard_rho` has no limit; `brent_rho` starts at 2 and can overshoot its budget (`prime.rs:917-970`) | yes (read) |
| HANDOFF lines (`docs/HANDOFF.md:273-282`), tracker lines `PROJECT_TRACKER.md:254, 395, 399, 402, 406, 407, 415, 3100`, the four `status_overrides.csv` rows | correct |

### Reference values (`$R/refs3.gp`)

All nine semiprime rows of 1.6 (`factor(p*q) == [p,1;q,1]` is 1 for each), N150 (150 digits, 497 bits), P30,
p20, q20, p15 q15, 6 p15 q15, 2 P30, q12^2, F7, 101 p20 q20 and `#Str(p20 q20)` = 39,
`factor(79839848800024750353128)`, `precprime(10000)` = 9973, `precprime(2^40)`, `nextprime(2^40)` and the
next prime 1099511627803, `forprime` count 2363, psi_12, 101 * 103 * P30, H18E8 product, `#Str(2^80)` = 25,
666958488061768516158211, the p^2 q, p q r and 10000019 (10^22+9) inputs, p25 q25, 42525, 2^64+1, 12780049,
`#Str(1000000016000000063)` = 19, the k = 3 factor base list (52 odd primes) and t_5..t_29 = [2,2,1,1,3,7,7,13],
101 * P30, p12 * P30: every value matches the design.

### Algorithm (re-derived)

- b^2 = kN mod a for every b of the Gray family (B_l = +-t mod q_l and 0 mod the other q_j); the Gray rule (new
  bit = 1 XOR bit l+1 of i, so e = +1 iff (i >> l) = 3 mod 4) visits all 2^(s-1) sign patterns; the root
  update r' = r - e Bainv2[l].
- Completeness of the candidate check: every prime <= p_max dividing g is in the base (p | a: an a-factor;
  p | kN: p | k, since p | N was caught by the scan; otherwise kN is a non-zero square mod p); so v < B_L is
  prime. Pairing only needs equal L anyway.
- GF(2): a non-pivot row only receives pivot rows, so its cleared columns stay clear and the XOR from word c/64
  loses nothing; each returned history holds its own bit and only pivot bits, so rows - rank independent
  vectors, at least `extra`.
- X^2 = prod y_i^2 = prod Q_i = Y^2 (mod N) over the multiset of relations (a first partial used by several pair
  rows enters with its multiplicity). For N = p^2 q the gcd is a union of whole prime-power components, so the
  requeue test `engine_requeues_composite_divisors` reaches p^2 and the perfect-power split.
- |y| de-duplication, the first-partial cap (order: duplicate, cap, insert, store), the stop at R rows (the first
  R rows are the same as with a trim), the a-supply guard and the Fisher-Yates draw.
- Termination: the budget is now checked before every polynomial (`$X/rev2/siqs_r3.rs:495-500`), and before every
  draw; accepted draws are at most `max_polynomials`; 1000 rejected draws in a row end the run.
- `has_at_most_digits` (bits <= 3d true, bits > 4d false, else exact), `coprime_refine` (H18E8 values checked with
  an independent Python refinement, `$R/refine3.py`), the trial-division stage split at 2^40, the engine's queue
  (multiset order), the classification lists of `LimitedFactorization`, `split_composite`'s order.
- MAGMA digest semantics (§18.10, §18.10.1, §18.10.3): cascade, defaults 10000 / 8191 / infinity / Proof true,
  "MPQSLimit < 25 disables MPQS", "third value only if both ECMLimit and MPQSLimit are set", "if MPQS is
  disallowed, ECM continues indefinitely" (here: `ECMLimit := 0` is set, so `composites` may be non-empty only
  with `mpqs_limit` set). `PollardRho(n, c, s, k)` defaults 1, 1, 8191. Read correctly.
- Citations (Pomerance 1985, Silverman 1987, Alford-Pomerance 1995, Contini 1997, Crandall-Pomerance §6.1.1-6.1.6,
  Brent 1980, Bach-Driscoll-Shallit 1993, Bays-Hudson 1977, Lenstra 1987, Cohen 1.5.1 and §10.4): as I know them.

### Prototype runs (my copy `$R/siqs_r3.rs`, built with `$R/build.sh` against the workspace's num-* rlibs)

| check | design | measured here |
|---|---|---|
| N150, F = 40 / 60 / 100, `max_polynomials = 10` | `OutOfPolynomials { relations: 0, needed: 73 / 93 / 133 }`, s = 31 / 28 / 26, 3-5 ms release, 14-24 ms debug | same values, s and window; 2.5-4.5 ms release, 13-16 ms debug |
| N150, F = 40, default budget 40000 | 1.4 s release | 1.7 s |
| N150, F = 30 / 31 (guard) | `needed: 63 / 64`, polys 0 | same |
| 40-digit row, budgets 1, 3, 16, 17 | polys = b, a = 1, 1, 1, 2, needed 835, s = 5 | same (relations 1, 9, 60, 64) |
| release grid 20..60 digits | 6 ms .. 2.4 s | 3 ms, 6 ms, 10 ms, 29 ms, 58 ms, 0.15 s, 0.40 s, 1.08 s, 2.49 s |
| debug grid 20..45 digits | 2.2 s in all | 2.2 s in all |
| 2207, p20 (primes) | `NoFactor`, 36 and >= 32 dependencies | 36 and 39 |
| 666958488061768516158211, seeds 1..6, debug | a factor each time | yes, 1 to 5 dependencies |
| doc and test inputs (270050147102784401 with `for_bits(58)`, F7, the p q r, p^2 q, p q inputs, psi_12, p20 q20) | proper divisors | yes (debug 8 ms to 0.41 s) |
| robustness, `$X/robust.txt` with new seeds 7 and 13, `$X/rev1/small.txt` with seed 7 | 0 failures | 0 failures (at most 11 / 10 dependencies) |
| **independent inputs** (`$R/gen3.gp`, 2699 gp inputs: 1500 semiprimes with both primes in (650, 6000), just above p_max of the default base; 300 p^2 q; 300 p q r; 200 balanced semiprimes of 20 to 40 digits; 200 unbalanced p * q with p < 10^4; 200 primes of 3 to 40 digits), seeds 1, 2, 3 | | 0 failures, worst input 0.41 s, at most 16 dependencies tried (`$R/inputs3_rev3.txt`) |

Other costs: `$X/rev2/cross40_*` gives 2365 primes visited (2363 below 2^40), 11.5 ms release and 53 ms debug
(design 12 / 60 ms); `$X/rev2/tdrange` gives 62, 83, 104, 162, 279 ns per prime near 1, 10^9, 10^10, 10^11, 10^12
and 5.1 to 6.5 µs above 2^40 (design 65, 86, 113, 171, 308 ns and about 5.7 µs). Budgeted Brent (`$R/brent3.py`,
an independent copy of the budget rule): psi_12 (None, 8191) with 8191 and 399165290221 after 450558 steps with
2^22; (10^9+7)(10^9+9) with c = 2, x0 = 2: 1000000009 after 31870 steps, (None, 1) with budget 1; p15 q15 (None,
8191); 101 * 103 * P30 gives 10403 after 30; 2^64+1 gives 274177 after 1918; 12780049 gives 53 after 14;
q12^2 stays unsplit by rho (the perfect-power test splits it, as `split_composite` orders). All as stated.

### Callers (my grep of `*.rs` outside `target/`)

`trial_division_with_primes`: only `factorint.rs` tests `:783-828`. `quadratic_sieve_factor_with_params`:
`quadratic_sieve.rs:422` (rewritten) and tests `:723`, `:733`. `brent_rho`, `split_odd_composite`,
`split_brent_and_trial_division`: `prime.rs` only. `quadratic_sieve::` paths: this crate only.
`rustmath-numbertheory` glob-re-exports `prime::*`; none of the new names (`factor_pollard_rho_limited`,
`POLLARD_RHO_DEFAULT_ITERATIONS`) exists anywhere, and no crate defines `LimitedFactorization`,
`FactorizationLimits`, `QsParams`, `QsOutcome`. The existing `tests/ch18_factorization_drivers.rs` asserts outcomes
only, so it keeps passing. One behaviour change the design states only in 1.4: `quadratic_sieve_factor(-15)` is
`None` at HEAD and becomes `Some(3)`; no caller depends on it.

### I1 interface

Everything I2 uses is in I1's design with the same names and shapes: `is_probable_prime` (exact below psi_12),
`try_is_prime -> Result<bool>` (Err only when ECPP is unresolved), `is_prime`, `is_prime_u64` in `primality.rs`,
`sieve::for_each_prime(lo, hi, FnMut(u64) -> ControlFlow<()>)` over [lo, hi), `sieve::primes_up_to`,
`fast_arith::pow_mod_u64` `pub(crate)`. The 2^40 split holds under both readings of I1's survivor rule
("isqrt(hi - 1) > 2^20" or "hi > 2^40"): the first call has hi <= 2^40. I1's review 2 (n3) asks I1 to name
`try_is_prime` for I2's certification, which is what I2 already uses.

### Plan order

Step 1 leaves `rho_rounds` unused until step 3 (a dead-code warning; the crate has no `deny(warnings)` and CI
runs plain `cargo build` / `cargo test`). Step 2 keeps HEAD's complete-driver body, which only needs
`quadratic_sieve_factor` and `factor_pollard_rho`; the fate-table tests pass with it (165731 and 3127 give 53
from the factor-base scan). Steps 3 and 4 use only earlier items. Each step builds.

## Minor

- **m1. The `ECMLimit` cost argument is measured on one size only.** The design leaves ECM out because on the
  41-digit p12 * P30 the sieve takes 0.08 s and ECM curves 0.15 s each. That holds where the sieve is fast. On a
  100-digit composite with a 12-digit factor (gp `100000000003 * nextprime(10^88)`), the crate's ECM takes 0.45
  to 0.48 s per curve in release and found the factor with 3 of 160 curves (0/40 and 3/120, `$R/ecm100.rs`), so
  about 25 s expected. The default engine sends that number to the sieve at 100 digits, which by the design's own
  growth rate (39 s at 70 digits, about 4.5 times per 5 digits) needs about 90 hours. Also, my rerun of
  `$X/rev2/ecm_rate` on the 41-digit input found the factor with 7 of 80 curves, not 2 of 80 (sigma comes from
  `thread_rng`). Required change (docs only): say in the `factorization.rs` module doc and in 3.2 and 8 that,
  without ECM, a large composite with a medium-size factor costs the sieve's time for the whole composite
  (hours to days from about 90 digits), that a larger `pollard_rho_limit` is the in-crate way to reach factors up
  to about 12 to 14 digits (about sqrt(p) rho steps), and quote the per-curve success as "a few percent" rather
  than "2 of 80".
- **m2. `cargo fmt -p rustmath-integers -- --check` fails at HEAD.** It exits 1 with 46 diffs in 8 files,
  among them `integer.rs`, `sage_wrapper.rs` and `tests/ring_predicates.rs`, which this unit does not touch.
  Required change: check only the files this unit writes (`rustfmt --edition 2021 --check <files>`), or say that a
  crate-wide format is a separate commit.
- **m3. The memory bound of `QsParams` leaves out the set of used a-values.** `ASelector::used` keeps one entry
  (s indices) per accepted a, at most min(ceil(max_polynomials / 2^(s-1)), number of distinct a-values). Under
  `for_bits` this is small (below about 60000 entries from 20 to 100 digits). With a caller's large
  `max_polynomials` and small s it grows past the bound the doc states. Required change: add the term to the
  memory paragraph of `QsParams` (or cap the set and stop with `OutOfPolynomials` when it is full).
- **m4. "MAGMA-shaped form" of `TrialDivision` is not exact.** Section 2.3 records
  `factorization_with_limits` with `pollard_rho_limit = 0, mpqs_limit = Some(0)` as the MAGMA-shaped form of
  `TrialDivision(n, B)`. The engine still splits perfect powers and, with `proof` true, proves a prime cofactor
  by ECPP: n = 7 * 1009^2 with B = 100 gives primes [(7, 1), (1009, 2)], where MAGMA's `TrialDivision` leaves
  1018081 as an unfactored part. Required change: word the tracker note as "close to `TrialDivision(n, B)`:
  perfect powers are split too, and the cofactor is classified (use `proof: false` to skip ECPP)".

Note, no change needed: the F values in the 1.3 list are computed from the digit count (ln n = d ln 10), while
`for_bits` uses bits ln 2. So the 40-digit row (131 bits) has F = 802, as the budget test's `needed: 835` says;
the `QsParams` doc's "F = 841 (40 digits)" is the digit-count value. A short clause would avoid confusion.

## Items covered

| item | covered | notes |
|---|---|---|
| 1 QS sieve stage | yes: SIQS (Contini 1997, Crandall-Pomerance §6.1), factor base from L(n), log sieve, single large prime, de-duplication, dense GF(2) left nullspace, square root, gcd; gp semiprimes 20 to 60 digits with measured times (reproduced); the two pinned tests rewritten | m3 |
| 2 `trial_division_with_primes` | yes: `LimitedFactorization`, cofactor classified, list entries that divide n checked prime, no early break | |
| 3 capped driver and return type | yes: `factor_pollard_rho_limited`, `factorization_with_limits`, `LimitedFactorization` with MAGMA's sequence, sign and unfactored composites, plus unproven probable primes; memory bounded for every argument | m1, m4 |
| 4 QS `legendre_symbol` | yes: deleted; word Euler criterion and word Tonelli-Shanks, no error path | |
| 5 ECM module doc | yes: doc only; the text matches `ecm.rs:297-399` | |
| I1 interface | yes: section 6 states what I2 needs and the fallbacks | |

Nothing is declared out of scope. `ECMLimit` is fixed at 0 as a stated choice with a hand-off (m1 asks for the
cost of that choice to be stated for large inputs too).
