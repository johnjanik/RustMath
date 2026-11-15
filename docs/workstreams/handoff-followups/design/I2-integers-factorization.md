# Design I2: rustmath-integers factorization (QS, capped drivers, trial division)

Unit I2 of the "open follow-ups" plan (docs/HANDOFF.md, "Open follow-ups", integer factorization bullet,
lines at `9c24389`; HEAD `c79a582` has the same code). Crate: `rustmath-integers`. Implemented after I3
(arithmetic) and I1 (primality). Spec: MAGMA Handbook ch. 18, digest
`rustmath-lava/docs/Chapter_18_Ring_of_Integers.md` §18.10 (Factorization), §18.10.1 (`Factorization` with
`TrialDivisionLimit`, `PollardRhoLimit`, `ECMLimit`, `MPQSLimit`, `Proof`; the third return value),
§18.10.3 (`TrialDivision`, `PollardRho`, `ECM`, `MPQS`).

Experiments: `$X` = `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/I2-integers-factorization/`.
The reference prototype is `$X/rev2/siqs_r3.rs`: a complete SIQS (num-bigint only, zero `unsafe`, about 790
lines) with every Revision 1 change (de-duplication, a-supply guard, caps, `for_bits` clamp) and every
Revision 2 change (the polynomial budget checked before every polynomial, the cap on first partial
relations, the stop at R rows inside the candidate loop). The implementation ports it (section 1.2 lists
what to change on the way). `$X/rev1/siqs_r2.rs` is the Revision 1 prototype and `$X/siqs.rs` the first
version, both kept for the record; `$X/rev1/design_before_rev1.md` and `$X/rev2/design_before_rev2.md` are
this document before Revision 1 and before Revision 2.

## Revision 2

Review 2 (`I2-integers-factorization.review2.md`) found 1 blocker and 6 minor items. All are fixed below; no
item is rejected (m1 is fixed by the second option the review offers). New experiments are in `$X/rev2/`
(built with `$X/rev2/build.sh`, as in Revision 1).

| Issue | What changed |
|---|---|
| B1: the polynomial budget is checked once per a-value | Confirmed: the Revision 1 prototype tests `polys >= max_polys` only before an a is drawn (`$X/rev1/siqs_r2.rs:396-399`), and one a then sieves all 2^(s-1) of its polynomials. On N150 with F = 40 and `max_polynomials = 10` (s = 31) the reviewer's copy (`review2/siqs_probe2.rs`) ran past 306000 polynomials in 10 s here. Fix: the budget is checked before every polynomial, the first one of each a included, and before every a draw (1.2 step 4, "Termination" and "Changes to port" corrected; the sentence that the prototype "already has `max_polynomials`" is gone). The new prototype `$X/rev2/siqs_r3.rs` has the check (`:495-500`). The same call returns `OutOfPolynomials { relations: 0, needed: 73 }` after exactly 10 polynomials, in about 3 ms release and 14 to 24 ms debug. F = 60 and F = 100 give `needed: 93` and `needed: 133` in the same time. The default budget (1000 F = 40000 polynomials at F = 40) ends in 1.4 s release and 41 s debug. On the 40-digit row (s = 5, 16 polynomials per a) the budgets 1, 3, 16 and 17 stop after exactly 1, 3, 16 and 17 polynomials, with 1, 1, 1 and 2 a-values. Runs that end before the budget are unchanged: the 581 robustness inputs with 6 seeds and the 1445 small inputs with 3 seeds give 0 failures with the same counts of dropped \|y\| and dependencies as Revision 1, and every grid row from 18 to 70 digits is factored (`$X/rev2/robust_r3.txt`, `small_r3.txt`, `grid_rel_r3*.txt`). New tests: `qs_polynomial_budget_is_per_polynomial` (integration, the review's input) and `run_stops_at_the_polynomial_budget` (unit, through the new internal `RunStats`). |
| m1: `ECMLimit` left out for a weak reason | Still left out, but now as a choice backed by measurements, not as an infeasibility. The Revision 1 reason (sigma from `thread_rng`) is dropped: a seeded sigma would be a few lines. The real reason is cost. The crate's ECM (affine Weierstrass curves, a 100-prime stage 2) takes 164 ms per curve in release at B1 = 2000 on the 41-digit p12 · P30, and only 2 of 80 curves found the 12-digit factor (`$X/rev2/ecm_rate.rs`; 0 of 10 at B1 = 500 and 2000 in `ecm_time.rs`). The sieve splits the whole 41-digit number in about 0.08 s. So an `ecm_limit` on this ECM would mostly cost time. A useful one needs a new ECM (hand-off, section 8). Sections 0, 3.2 and 8 say this. |
| m2: I1's hand-off of `prime::factor` and `prime_divisors` is dropped | Section 8 now hands them on, with what they do at HEAD (unbounded trial division, `prime.rs:415` and `:469`), their callers and the fix (rebuild on `factorization_with_limits`). Plan step 7 carries it into HANDOFF. |
| m3: `has_at_most_digits(0, d)` returns true | Now false for every c <= 0, so the doc "true iff 0 < c < 10^d" is the whole contract. `has_at_most_digits_edges` adds (0, 5) and (-7, 5), both false. |
| m4: the plan steps do not build in order | Reordered (section 7): `prime.rs` internals first; then the sieve, with `quadratic_sieve_factor_complete` keeping HEAD's body for that step (it calls the new `quadratic_sieve_factor` and `factor_pollard_rho`, which both exist); then `factorization.rs` together with `factor_pollard_rho_limited` and the switch of the complete driver to `limited_engine`; then `factorint.rs`. Each step builds and tests green on its own. |
| m5: "at most about 1.2 GB" was not a bound | The relation store is now capped, so the stated bound holds. (1) At most 16 F large primes get a stored first partial relation (the entries of the large-prime map); once there are 16 F, a partial relation with a new large prime is dropped (`Added::Dropped`), while partials that pair with a stored one are still kept. (2) The run stops as soon as it has R rows, inside the candidate loop, instead of finishing the polynomial and trimming (`$X/rev2/siqs_r3.rs:632-641`, `:662-666`). The first R rows are the same either way, so the outcome is the same (36 grid rows from 20 to 55 digits compared: same polynomials, dependencies and factor; the 581 robustness inputs with 6 seeds and the 1445 small inputs: 0 failures and the same dependency counts, `$X/rev2/robust_r3_final.txt`, `small_r3_final.txt`). Fewer repeated \|y\| are seen (226 to 336 per seed instead of 314 to 468), because candidates after the R-th row are no longer checked. So at most R + 16 F relations are stored. Measured: 3.7 to 4.9 F stored relations (20 to 55 digits) and at most 4.8 F first partials (every grid run from 18 to 70 digits), 0 dropped, so the cap does not bind there. With a cap of F instead, the 40-, 45- and 50-digit rows are still factored, with about 25 % more polynomials. The memory docs of `QsParams` and `FactorizationLimits` now give bounds that hold (about 1.5 GB at the caps). New unit test `relation_store_caps_first_partials`. |
| m6: trial-division cost above 2^40 depends on I1 | The engine splits the trial-division range at 2^40: one `for_each_prime` call with hi <= 2^40 (no survivor test under I1 §2.5) and one from 2^40 up. So I2's cost does not depend on whether I1 confirms all survivors or only those above 2^40 (section 6 states what I2 needs). The cost doc now gives the measured rate by range (`$X/rev2/tdrange.rs`, an odd-only segmented sieve as I1 describes it, one word remainder per prime on a 30-digit cofactor, release): 65 ns per prime near 10^7, 86 ns at 10^9, 113 ns at 10^10, 171 ns at 10^11, 308 ns at 10^12, and about 5.7 µs above 2^40, where each survivor also gets HEAD's 12-base `is_prime_u64` (`$X/rev2/td40.rs`). So a limit of 10^12 on a cofactor without small factors costs about 2.5 hours, not the 45 minutes Revision 1 stated. The stage is now the crate-internal `trial_division_stage(m, lo, limit)` (3.2), with the new unit test `trial_division_crosses_2_40`: precprime(2^40) and nextprime(2^40) are both found across the split, with 2365 primes visited (gp counts 2363 in [2^40 - 2^16, 2^40), then two above), in 12 ms release and 60 ms debug (`$X/rev2/cross40.rs`). |

## Revision 1

Review 1 (`I2-integers-factorization.review1.md`) found 2 blockers and 11 minor items. All are fixed below; no
item is rejected. Three cells are superseded by Revision 2 and say so: the trial-division cost (B2), the
reason for leaving out `ECMLimit` (m3), and the memory statement (B2). New experiments are in `$X/rev1/` (built with `$X/rev1/build.sh`, which links against the
workspace's own num-* rlibs).

| Issue | What changed |
|---|---|
| B1: a-value draw hangs when the window is too small | Confirmed with the revised prototype: the 150-digit n = nextprime(10^74) nextprime(3 10^75) with F = 30 (M from `for_bits`) needs s = 34 a-factors, but only 22 primes are eligible. Now `run` checks, before any draw, that s <= 64, that at least s primes are eligible (p > 30, p not dividing k) and that the window holds at least max(s - 1, 1) of them. Otherwise it returns `OutOfPolynomials { relations: 0, needed }` at once. The s - 1 window indices are drawn by a partial Fisher-Yates shuffle, so a draw takes exactly s - 1 steps. The termination argument (end of 1.2) is rewritten and counts every loop. New test `qs_a_supply_too_small_returns_promptly` (F = 30 and 31 on that n: `OutOfPolynomials` in about 2 ms, release or debug, in the prototype). |
| B2: huge allocations or overflow from accepted arguments | (1) Trial division in the limits engine now runs over the primes up to bound = min(limit, isqrt(m)), from I1's segmented `sieve::for_each_prime` (32 KiB segments). From the first prime above 2^16 on it stops early once the cofactor passes `is_probable_prime` (tested only when the cofactor has changed), so small limits pay no BPSW test. No `(limit + 1)^2` and no `limit + 1` remain (the end is `bound.saturating_add(1)`). Memory is O(1) in the limit. The cost is about 70 ns per prime in release and 0.5 µs in debug on a 30-digit cofactor (`$X/rev1/tdrate.rs`), and it is documented (Revision 2, m6: this rate holds near 10^7; the doc now gives it by range). (2) The MPQS digit limit is compared through bit-length bounds, with one exact 10^d only inside a narrow band (`has_at_most_digits`, 3.2). (3) Caps: F <= 65536 (dense GF(2) matrix about F^2/4 bytes, 1.08 GB at the cap), M <= 2^24 (sieve 2M bytes = 32 MiB), extra relations in [1, 1024], max_polynomials >= 1, large prime bound by `saturating_mul`. `QsParams` and `FactorizationLimits` document memory and time (Revision 2, m5: the relation store is now capped too). (4) `for_bits` clamps F to [60, 65536] and M, lp_mult to the ends of the table, so its output always validates. New tests: `limits_trial_division_huge_limit` (u64::MAX and 10^12 on 100 and on 2 P30), `qs_params_validation` (F = 65537, M = 2^24 + 1 and others rejected; `for_bits` in range for bits from 0 to u64::MAX). |
| m1: de-duplication is load-bearing; the evidence was gathered without it | Step 7 now says why \|y\| de-duplication is a correctness step. The prototype now has it, and the robustness runs were redone with the L(n) parameters of `for_bits`. With de-duplication: 581 inputs (`$X/robust.txt`) x 6 seeds and 1445 new gp inputs of 6 to 18 digits (`$X/rev1/small.txt`) x 6 seeds give 0 failures; at most 11 dependencies were tried on any composite; 314 to 468 repeated \|y\| were dropped per seed. Without it the same 581 inputs needed up to 36 dependencies (the matrix guarantees only 32), and the reviewer's run failed on 666958488061768516158211 with seed 4. New unit test `relation_store_drops_repeated_y`, and the integration test `qs_seed_sweep_24_digits` (that n, seeds 1 to 6). |
| m2: `with_params` for \|n\| <= 47 | New step 0 of the algorithm (1.2) is the quick-path list. It runs before the multiplier, so the SIQS core always has N > 47 with no prime factor <= 47, and gcd(k, N) = 1 holds. `with_params` gives `Ok(NoFactor { dependencies: 0 })` for n = +-1 and for +-p with p a prime <= 47; it gives `Err` for 0. Test `qs_with_params_tiny_inputs`. The dangling "step 0 below" now points to step 0. |
| m3: MAGMA `ECMLimit` / `MPQSLimit` semantics | `FactorizationLimits` and section 3.2 now say: the engine is MAGMA's `Factorization` with `ECMLimit := 0`. The crate's ECM draws sigma from `rand::thread_rng` (`ecm.rs:317`), so results would not be reproducible, and it is the weak ⏳ row. A seeded ECM with a curve budget is needed first. (Revision 2, m1: superseded; the reason is now the measured cost of this ECM, section 3.2.) `mpqs_limit` now follows MAGMA: `Some(x)` with x < 25 disables the sieve. With x >= 25 the sieve takes composites of at most x digits, and here also those below 25 digits, because MAGMA's SQUFOF and ECM for them are absent. With `ECMLimit` fixed at 0, MAGMA's rule "the third value only if both limits are set" becomes "`composites` is empty when `mpqs_limit` is None". New test `limits_mpqs_limit_below_25_disables_sieve`. |
| m4: prime powers split once per exponent | `split_composite` now returns all pieces with exponents: c = b^j gives [(b, j)], an even c gives [(2, v), (c / 2^v, 1)]. The engine pushes (b, j e). Measured: `is_perfect_power(3^2000)` = (3, 2000) in 0.32 ms debug (`$X/rev1/pp2.rs`), against 0.53 s for the 1999 splits the reviewer measured. The complete QS driver keeps trial division by the primes <= 47 (engine with limit 47). New tests `rho_limited_prime_powers` and `qs_complete_prime_powers`. |
| m5: ECM doc overstates stage 2 | Section 5's text now says what the code does: stage 1 succeeds when the order of the point is B1-powersmooth (`ecm.rs:304`), and stage 2 tries only the first 100 primes in (B1, B2] (`ecm.rs:389`). |
| m6: multiplier list | The design uses the 30 multipliers of the measured prototype (no 46, `siqs.rs:182-185`). |
| m7: I1 and I3 coupling | I1's revision (its §2.4 and §2.8) leaves the policy to I2. Decision: the three `Vec` drivers keep I1's panic on an unresolved proof. `quadratic_sieve_factor_complete` panics with a message that names `factorization_with_limits`, which never panics and reports such a factor in `probable_primes` (3.4, 6). Plan step 6 re-enables the QS case of I1's `psi12_is_split_by_every_driver` if I1 moved it to `#[ignore]`. I3's edits to `quadratic_sieve.rs` (`:50`, `:60`, `:93`, `:96`, `:112`, `:145`, `:228`, `:376`) are all in functions this rewrite deletes. The rewrite supersedes them (6). |
| m8: HEAD tests and doc tests | Section 1.6 has a table with the fate of every HEAD test in `quadratic_sieve.rs` and of the three doc examples. The complete-driver doc comment "the sieve finds no factor" is replaced. |
| m9: `limits_trial_division_proves_small_cofactor` | Dropped, with the "m < (limit + 1)^2 is prime" shortcut it was meant to test. A cofactor the shortcut could prove is below 2^32 when the limit is at most 2^16 (then `classify` decides it exactly), and for larger limits the new early stop catches a prime cofactor first; only a limit above 5.6 * 10^11 (hours of trial division) would let the shortcut save a proof. Replaced by `limits_trial_division_respects_limit` and `limits_trial_division_huge_limit`. |
| m10: `factor_trial_division(n, None)` stops at 10^6 above 2^128 | Confirmed at `factorint.rs:190-192`. Outside this unit's items. The `ch018.TrialDivision` tracker note records it, and it is handed to a follow-up (section 2.3, section 8). |
| m11: `Proof` flag of `PollardRho` | `factor_pollard_rho_limited` takes `proof: bool` (MAGMA §18.10.3: every function there takes `Proof`, default true). New test `rho_limited_proof_flag`. |

## 0. Summary

| # | Item (HANDOFF) | At HEAD | Fix |
|---|---|---|---|
| 1 | QS sieve stage broken (`quadratic_sieve.rs:282`, `:340`), driver is rho in practice, two tests pin the sieve's output (`:778`, `:792`) | Confirmed: the GF(2) step returns non-nullspace sets; the square root is a floor root of a non-square; `quadratic_sieve_factor` returns `None` on every semiprime tried (10403, 20 to 30 digits) | New self-initialising quadratic sieve (SIQS): Knuth-Schroeppel multiplier, factor base sized from L(n), Contini's polynomials with Gray-code switching, log sieve with small-prime and single-large-prime variations, exact relation check, \|y\| de-duplication, dense GF(2) left nullspace by Gaussian elimination, exact square-root step, gcd. Validated parameters with memory caps, a polynomial budget checked before every polynomial, and a cap on stored partial relations. Prototype factors every gp semiprime from 18 to 72 digits (60 digits: 2.4 s release, 16 s debug; PARI 2.4 s), and 2026 mixed inputs from 6 to 45 digits with 6 seeds each, without a failure. The pinned tests are replaced. |
| 2 | `trial_division_with_primes` reports a leftover cofactor as a prime (`factorint.rs:401`, `:434-436`) | Confirmed: `([2, 3], 1225)` gives `[(1225, 1)]`, `([2, 3], 1001)` gives `[(1001, 1)]` | Returns the new `LimitedFactorization`: proven primes, unproven probable primes and proven composites in separate lists, plus the sign. Each list entry that divides n is checked prime. The early `break`, which is wrong for unsorted lists, goes. |
| 3 | `factor_pollard_rho` has no capped variant returning the unfactored composites (MAGMA third return value) | Confirmed: no such API; `factor_pollard_rho` has no limit (for two 20-digit primes it does not finish) | `prime::factor_pollard_rho_limited(n, c, s, k, proof)` (MAGMA `PollardRho(n, c, s, k : Proof)`) and `factorization::factorization_with_limits(n, &FactorizationLimits)` (MAGMA `Factorization(n : TrialDivisionLimit, PollardRhoLimit, MPQSLimit, Proof)` with `ECMLimit := 0`), both returning `LimitedFactorization`. It holds MAGMA's sequence, sign and unfactored composites, plus a fourth list for probable primes whose proof is unresolved, as the I1 interface asks. Memory is bounded for every argument. |
| 4 | QS `legendre_symbol` maps an error to "residue" (`quadratic_sieve.rs:45-68`) | Confirmed at `:57-58` (`.unwrap_or(Integer::one())`); unreachable for p >= 3; it also returns 1 for (a/2) with a even | Deleted. The factor base uses Euler's criterion on machine words (exact for an odd prime p < 2^32, no error path); the square roots come from a word Tonelli-Shanks that returns `None` exactly for non-residues. |
| 5 | `ecm.rs:14-15` module doc describes p - 1 / p + 1, not ECM | Confirmed; also the comment at `ecm.rs:322-323` calls the curve construction "Montgomery parameterization", which it is not | Doc fix: what ECM needs (the order of E(F_p) smooth; cost L_p[1/2, sqrt 2]) and what this module's two stages actually test (B1-powersmooth; the first 100 primes in (B1, B2]); the comment says what the code does. |

Nothing is declared out of scope. One choice inside item 3 (Revision 2, m1): `factorization_with_limits` has
no `ECMLimit`; it is MAGMA's `Factorization` with `ECMLimit := 0`. This is not an infeasibility (a seeded
sigma and a curve budget would be a few lines). It is a choice on measured cost: the crate's ECM takes 164 ms
per curve in release at B1 = 2000 on a 41-digit n and found its 12-digit factor with 2 of 80 curves, while
the sieve splits that n in about 0.08 s (3.2). A useful `ECMLimit` needs a new ECM (section 8).

Not covered, because no item of this unit names them (listed so the reviewers know):
- SQUFOF and MAGMA's `SQUFOFLimit`, and `Bases` (MAGMA's Miller-Rabin count for `Proof := false`; here BPSW,
  from I1).
- Block Lanczos (dense elimination is fast enough up to about 75 digits, 1.3), the double large prime
  variation, MAGMA `MPQS(n, D)`'s working directory and slave processes.
- `prime::factor` and `prime_divisors` (unbounded trial division; I1's notes list them under I2, but no
  HANDOFF item names them). Handed on in section 8 (Revision 2, m2).
- `factorint::factor_trial_division(n, None)` stops at 10^6 when |n| > 2^128 (review 1, m10; section 2.3).

## 1. Item 1: the quadratic sieve

### 1.1 Defect at HEAD

Code: `rustmath-integers/src/quadratic_sieve.rs`.

1. `gaussian_elimination_gf2` (`:282-337`): when a column has no pivot it returns, as a "dependency", the set
   of all rows with a 1 in that column (`:307-319`). That is not a left-nullspace vector, and real
   dependencies are never produced. Reproduction (`$X/repro_gauss.rs`, a verbatim copy run on the 3 x 2
   all-ones matrix): it returns one vector, `[true, false, false]`, whose row sum is `[true, true]`; the left
   nullspace has dimension 2 ({r0, r1}, {r0, r2}).
2. `try_dependency` (`:340-397`): y is `q_product.sqrt().unwrap_or(Integer::one())` (`:376`) where
   `q_product` is the product of the q_i already reduced mod n (`:371`). That is not a square, so the floor
   root gives no congruence of squares; gcd(x - y, n) splits n only by accident.
3. One polynomial x^2 - n on 1000 points with 50 primes (`:421-423`); `sieve_for_smooth_relations`
   (`:223-260`) trial-divides every value (no sieve).

Reproduction (`$X/repro_head.rs`, linked against the release rlib of HEAD; the reviewer reproduced it on
the debug rlib):

| call | HEAD | reference |
|---|---|---|
| `quadratic_sieve_factor(3127)` | `None` | gp `factor(3127)` = [53, 1; 59, 1] |
| `quadratic_sieve_factor(165731)` | `Some(3127)` (composite) | gp `factor(165731)` = [53, 2; 59, 1] |
| `quadratic_sieve_factor(10403)` | `None` | gp `factor(10403)` = [101, 1; 103, 1] |
| `quadratic_sieve_factor(1000000016000000063)` | `None` | gp: [1000000007, 1; 1000000009, 1] |
| `quadratic_sieve_factor(270050147102784401)` (18 digits) | `None` | gp: [314159311, 1; 859596191, 1] |
| `quadratic_sieve_factor(27005010885249061781)` (20 digits) | `None` | gp: [3141592661, 1; 8595961921, 1] |
| `quadratic_sieve_factor(270050107561429816771648674403)` (30 digits) | `None` | gp: [314159265359057, 1; 859596190017779, 1] |
| `quadratic_sieve_factor_with_params(270050147102784401, 300, 1000000)` | `None` (0.2 s) | as above |

`quadratic_sieve_factor_complete` (`:549-610`) is correct only because it hands every cofactor the sieve
cannot split to `factor_pollard_rho`. The tests at `:775-797` pin the broken outputs (165731 gives 3127, 3127
gives `None`).

### 1.2 Fix: the self-initialising quadratic sieve (SIQS)

References (cite in the new file headers):
- [Pom85] C. Pomerance, *The quadratic sieve factoring algorithm*, EUROCRYPT '84, LNCS 209 (1985) 169-182.
- [Sil87] R. D. Silverman, *The multiple polynomial quadratic sieve*, Math. Comp. 48 (1987) 329-339
  (multiple polynomials, threshold, the multiplier function, large primes).
- [AP95] W. R. Alford, C. Pomerance, *Implementing the self-initializing quadratic sieve on a distributed
  network*, in: Number Theoretic and Algebraic Methods in Computer Science (Moscow 1993), World Scientific
  1995, 163-174.
- [Con97] S. P. Contini, *Factoring integers with the self-initializing quadratic sieve*, MSc thesis,
  University of Georgia, 1997 (a = q_1 ... q_s, B_l, Gray code, Bainv2 root update).
- [CP05] R. Crandall, C. Pomerance, *Prime Numbers: A Computational Perspective*, 2nd ed., Springer 2005,
  §6.1.1 (basic QS, L(n)), §6.1.3 (matrix step), §6.1.4 (large prime variation), §6.1.5 (multiple
  polynomials), §6.1.6 (self-initialization).
- [Knu97] D. E. Knuth, TAOCP vol. 2, 3rd ed., §4.5.4 (the Knuth-Schroeppel multiplier).
- [Coh93] H. Cohen, GTM 138, Algorithm 1.5.1 (Tonelli-Shanks), §10.4 (QS and MPQS).
- [Knu97b] D. E. Knuth, TAOCP vol. 2, 3rd ed., §3.4.2, Algorithm P (the shuffle used for the a-factor draw).
- MAGMA ch. 18 §18.10.3 `MPQS` (Lenstra's MPQS; SIQS is the same family) and §18.10.1 (`MPQSLimit`).

#### Step 0: quick paths (in `quadratic_sieve_factor_with_params` and `quadratic_sieve_factor`)

Input n. `with_params` returns `Err(InvalidArgument)` for n = 0 and for parameters out of range (1.4). Let
N = |n|, then in this order:
1. N = 1: `NoFactor { dependencies: 0 }`.
2. N even: N = 2 gives `NoFactor { dependencies: 0 }`, else `Factor(2)`.
3. For the odd primes p = 3, 5, ..., 47: if p | N, then p = N gives `NoFactor { dependencies: 0 }` (N is a
   prime <= 47), else `Factor(p)`.
4. `N.is_perfect_power()` = Some((b, j)): `Factor(b)` (b is the minimal base, so 1 < b < N).
5. Otherwise N is odd, N >= 53, N has no prime factor <= 47 and is not a perfect power: run the SIQS core
   (steps 1 to 9).

`quadratic_sieve_factor(n)` maps `Factor(d)` to `Some(d)` and `NoFactor` to `None`; n = 0 gives `None`.
After item 4 it also returns `None` when `is_probable_prime(N)` (I1; exact below psi_12, BPSW above), and
only then runs the core with `QsParams::for_bits(N.bits())` and seeds 1, 2, 3 until `Factor`. Every
composite N <= 47 has a prime factor p <= 47 with p < N, so items 1 to 3 decide every N <= 47 (a test runs
them all, 1.6).

#### The SIQS core (run on N odd, N > 47, no prime factor <= 47, not a perfect power)

1. **Multiplier** [Knu97, Sil87]. For each squarefree k in {1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19,
   21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 47} (the 30 values of the measured
   prototype, `$X/rev2/siqs_r3.rs` `knuth_schroeppel`, unchanged since Revision 1; 46 is not used) score
   f(k) = -ln(k)/2 + c_2(kN) + sum over the first 300 odd primes p of: ln(p)/p if p | k, 2 ln(p)/(p - 1) if
   kN is a non-zero square mod p; c_2 = 2 ln 2 if kN = 1 mod 8, ln 2 if 5 mod 8, ln(2)/2 otherwise (also
   for kN even). Take the k with the largest f. Every prime factor of k is <= 47, and N has none (step 0),
   so gcd(k, N) = 1. kN is never a square: k > 1 would need k | N; k = 1 would make N a square.
2. **Factor base** (F primes). primes[0] = 2 always. Then the odd primes p in increasing order, from I1's
   segmented `sieve::for_each_prime(3, 2^31, ..)` (Break once the base has F primes; memory O(1)), with
   r = kN mod p:
   - r = 0 and p | k: include, root unused (`divides_k`), never sieved.
   - r = 0 and p does not divide k: p | N. If p < N return `Factor(p)`; if p = N then N is prime: return
     `NoFactor { dependencies: 0 }`.
   - r^((p-1)/2) = 1 mod p (Euler's criterion on u64 words, exact for a prime p): include with
     t_p = sqrt(r) mod p (word Tonelli-Shanks, `sqrt_mod_u64`, section 4).
   logp[j] = round(log2 p_j) as u8. Every prime up to p_max = primes[F-1] that divides N is found here,
   which also handles all small inputs: a composite N < p_max^2 always returns at this step. About half of
   the primes qualify, so F = 65536 is reached near p = 1.8 * 10^6. If the scan reaches 2^31 with fewer
   than F primes (no realistic N does; it would need kN to be a non-residue modulo almost every prime
   below 2^31), return `OutOfPolynomials { relations: 0, needed }`. So p_max < 2^31 and primes fit a u32.
3. **Parameters** (1.3, validated in 1.4): F, half width M (x in [-M, M), M a multiple of 64), large prime
   bound B_L = min(lp_mult.saturating_mul(p_max), p_max^2 - 1) (p_max < 2^31, so p_max^2 fits a u64),
   threshold T, extra relations (default 32).
4. **Polynomials** [Con97 §4]. Q(x) = (a x + b)^2 - kN = a g(x) with g(x) = a x^2 + 2 b x + c,
   c = (b^2 - kN)/a.
   - Size: log2 target = (log2(2kN))/2 - log2(M). qbits = min(11, log2(p_max) - 1), at least 3;
     s = max(1, round(log2 target / qbits)).
   - Eligible set E: the factor base indices j with p_j > 30 and p_j not dividing k. Window W: the indices
     of E with |log2 p_j - log2 target / s| <= 1, widened by one index on each side until it has 4s + 8
     entries or covers all of E. So W is a subset of E.
   - **a-supply guard (Revision 1, B1).** If s > 64, or |E| < s, or |W| < max(s - 1, 1), return
     `OutOfPolynomials { relations: 0, needed: F + 1 + extra }` before any draw. (s > 64 also keeps
     2^(s-1) in a u64 and the Bainv2 table at s x F words; it happens only for n above about 430 digits
     with `for_bits`, or with a factor base far too small for n.)
   - Draw: the s - 1 indices are the first s - 1 entries of a partial Fisher-Yates shuffle of a copy of W
     [Knu97b], driven by splitmix64 seeded with `params.seed`: for t = 0 .. s - 2 swap entry t with a
     uniform entry of t .. |W| - 1. So the draw takes exactly s - 1 steps and the indices are distinct.
     The last index is the index of E, not drawn, whose log2 p is closest to log2 target minus the sum of
     the drawn logs (it exists because |E| >= s). For s = 1 the one index is a uniform entry of W.
   - The sorted index set must be new (a `HashSet`); after 1000 consecutive repeats the supply is
     exhausted: return `OutOfPolynomials { relations, needed }`.
   - **Polynomial budget (Revision 2, B1).** A counter of sieved polynomials is compared with
     `max_polynomials` before every polynomial, the first polynomial of each a included, and before every
     a draw. Once it has reached `max_polynomials`, the run returns `OutOfPolynomials { relations, needed }`
     at once. A check once per a is not enough: one a has 2^(s-1) polynomials, and s can be 31 with valid
     arguments (N150 with F = 40, 1.6), so a check per a lets one a sieve 2^30 polynomials, about 11 hours
     in release.
   - B_l = (a/q_l) * gamma_l with gamma_l = t_{q_l} * ((a/q_l)^-1 mod q_l) mod q_l, replaced by
     q_l - gamma_l if larger than q_l/2. b = B_1 + ... + B_s. Then b^2 = kN mod a (each b = +-t mod q_l).
   - For every factor base prime p not dividing a k (index j >= 1): ainv = a^-1 mod p,
     Bainv2[l][j] = 2 B_l ainv mod p, and the sieve roots shifted to array indices
     r1 = (ainv (t_p - b) + M) mod p, r2 = (ainv (-t_p - b) + M) mod p.
   - Gray code [Con97 §4.3]: for i = 1 .. 2^(s-1) - 1, l = trailing_zeros(i), u = i >> l; e = +1 if
     u = 3 mod 4, else -1; b <- b + 2 e B_l; r <- r - e Bainv2[l][j] mod p for both roots. This visits all
     2^(s-1) sign patterns of B_1 .. B_{s-1} with B_s fixed. c is recomputed per b (one BigInt division).
5. **Sieve** [Pom85, CP05 §6.1.1]. Byte array of length 2M filled with init = 128 - T (`fill`), then for each
   index j with p_j > 30 (small prime variation; p_j does not divide a k) add logp[j] at r1, r1 + p, ...
   and at r2, ... (wrapping_add). A location is a candidate iff its byte is >= 128; scan 8 bytes at a time
   (`chunks_exact(8)`, mask 0x8080808080808080). T = round(log2 M + (log2 kN - 1)/2 - log2 B_L - slack),
   clamped to [1, 128]. With init + (sum of logs) <= 128 + log2 B_L + slack + about 10 < 256 the byte cannot
   wrap below 128 for n up to about 85 digits; above, T is clamped and only more candidates appear.
6. **Candidates** (`check_candidate`). x = i - M; g = (a x + 2b) x + c (BigInt); skip g = 0. Divide |g| by 2
   (trailing zeros); for each odd index j: if p_j divides a or k test |g| mod p_j directly, else test
   i mod p_j in {r1, r2}; on a hit divide out all powers of p_j. The exponent of each a-factor gets +1
   (Q = a g). Cofactor v: v = 1 gives a full relation; 1 < v < B_L gives a partial relation with large prime
   v; else discard. Then check exactly (always, not only in debug)
   y^2 - kN = (-1)^negative * prod p_j^e_j * large_prime with y = a x + b; a failure is an internal error
   (`assert!` with a message).
   - Completeness: for p not dividing a k, p | g(x) iff (a x + b)^2 = kN mod p iff
     x = a^-1(+-t_p - b) mod p, so the root test finds every such prime. A prime p <= p_max outside the
     factor base cannot divide g: p | g gives kN a square mod p, so p is in the base or divides kN, and
     p | N was excluded in step 2. Hence v has only prime factors > p_max, and v < B_L < p_max^2 makes v
     prime.
7. **Relations** (`RelationStore`). `Relation { y: BigInt (= a x + b), negative: bool, factors: Vec<(u32
   index, u32 exp)>, large_prime: u64 (1 if full) }`. `RelationStore::add(rel) -> Added` with
   `enum Added { Full, PartialFirst, PartialPaired, Duplicate, Dropped }`:
   - **De-duplication (a correctness step, Revision 1, m1).** If |y| is in `seen_y: HashSet<BigUint>`, the
     relation is dropped (`Duplicate`): no row, no pairing. Otherwise |y| is inserted. Why it is needed:
     different polynomials can produce the same y. With small s, a-values that share a prime q_l have
     overlapping progressions y = b mod a, and y and -y give the same Q. A repeated relation (or a pair row
     {r, r'} with y' = +-y and the same large prime) is a dependency by itself, and its congruence is
     X = +-Y, which never splits N. Such trivial vectors take part of the nullspace that the matrix
     guarantees (at least `extra` = 32 vectors). Measured in the prototype on the 581 robustness inputs over
     6 seeds: with de-duplication at most 11 dependencies were tried on any composite (314 to 468 repeats
     dropped per seed; 226 to 336 with the Revision 2 stop at R rows, which checks fewer candidates);
     without it up to 36 (`$X/rev1/robust_dedup_r2.txt`, `$X/rev1/robust_nodedup.txt`).
     On 666958488061768516158211 (`= 765944099509 * 870766533079`), 8 seeds need 1 to 5 dependencies with
     de-duplication and 3 to 19 without. The reviewer's run of the first prototype, without it, ended with
     `None` after 42 dependencies on that n with seed 4, and 50 of its 1010 relations repeated a |y|.
   - Full relation: a matrix row by itself (`Full`). Partial [CP05 §6.1.4]: a `HashMap<u64, usize>` maps a
     large prime to the first partial with it (`PartialFirst`); every later partial with the same prime
     gives a matrix row {first, this} (`PartialPaired`).
   - **Cap on first partials (Revision 2, m5).** At most 16 F large primes get a stored first partial
     relation, that is, the `HashMap` has at most 16 F entries (`FIRST_PARTIALS_PER_PRIME = 16`, a private
     constant of `siqs.rs`). Once it has that many, a partial relation whose large prime is not in the map
     is dropped (`Dropped`): no row, and its |y| is not inserted. A partial relation whose large prime is in
     the map is still stored and paired. The order in `add` is: duplicate test, cap test, insert |y|,
     store. A stored first partial is never removed, so the set of large primes with a stored partial
     never shrinks, and a dropped relation is simply lost (if the same y comes again, it is dropped
     again). Measured: at most
     4.8 F map entries in every grid run from 18 to 70 digits, so the cap does not bind in normal
     runs; it bounds memory in pathological runs (a tiny factor base, a large prime bound far above the
     factor base, a huge budget). Dropping costs speed only, never soundness: with a cap of F instead of
     16 F the 40-, 45- and 50-digit rows still factor, with about 25 % more polynomials.
   - **Stop at R rows (Revision 2, m5).** R = F + 1 + extra. The run stops as soon as R rows exist, inside
     the candidate loop; the remaining candidates of that polynomial are not checked. Revision 1 finished the
     polynomial and kept the first R rows. Both give the same R rows, hence the same matrix and outcome
     (36 grid rows from 20 to 55 digits compared, `$X/rev2/grid_rel_r3b.txt`), but the stop stores no
     surplus. More rows than R would only make the elimination and, for a prime N, the square-root loop
     longer (the first prototype without the trim tried 290 dependencies on a 20-digit prime, 0.68 s in
     debug; with it 0.17 s).
   - Memory: at most R + 16 F stored relations (fulls and paired partials make at most R rows; first
     partials are at most 16 F). Measured 3.7 to 4.9 F from 20 to 55 digits, each about 300 bytes there
     (estimate from the measured sizes: |y| of 8 to 16 bytes, stored twice with the `seen_y` key, 9 to 17
     (index, exponent) pairs of 8 bytes, the row, vector headers and map entries).
8. **Linear algebra** (`gf2.rs`, [CP05 §6.1.3]). Columns: 0 = sign, 1 + j = primes[j]. Row bits = parity
   of the summed exponents of its relations. Left nullspace by Gaussian elimination on bit-packed rows (u64
   words) with an appended identity ("history"): for each column c, take the first non-pivot row with bit
   c as pivot and XOR it (matrix part from word c/64 on, plus history) into every other non-pivot row with
   bit c. Return the histories of the rows that never became pivots.
   - Correctness: a non-pivot row only receives XORs of pivot rows, so after column c every non-pivot row
     has bit c clear and keeps it clear (later pivots have bit c clear too); at the end its matrix part is
     0 and its history is the set of rows that sums to it. A pivot row's history holds only its own and
     earlier pivots' bits, so each returned history contains its own row bit and no other returned row's
     bit: the vectors are independent. There are rows - rank of them, so they are a basis of the left
     nullspace. With R = F + 1 + extra rows and rank <= F + 1 there are at least `extra` of them.
   - Memory: R x (F + 1 + R) bits, about F^2/4 bytes; time O(R^2 (F + R) / 64) word operations.
9. **Square root** [CP05 §6.1.1]. For a nullspace vector, expand to the multiset of relations; X = prod y_i
   mod N; E_j = sum of exponents, s = number of negative values, E_L per large prime; assert all even; Y =
   prod p_j^(E_j/2) * prod L^(E_L/2) mod N. Then X^2 = prod y_i^2 = prod Q_i (mod N) (kN = 0 mod N)
   = (-1)^s prod p^E prod L^E_L = Y^2 (mod N); assert `X^2 = Y^2 mod N` (internal invariant). g =
   gcd(X - Y, N); return `Factor(g)` for the first 1 < g < N. If none: `NoFactor { dependencies }`.

Soundness (independent of all parameters): `Factor(d)` always has 1 < d < N and d | N (a quick-path prime,
a perfect-power base, a factor base prime that divides N, or a gcd checked against 1 and N). A run that
finds nothing says `NoFactor` or `OutOfPolynomials`; neither is a primality verdict. Each congruence
X^2 = Y^2 with X != +-Y splits N; for N with two distinct prime factors each non-trivial dependency succeeds
with probability about 1/2, so with 32 or more dependencies a failure on a composite is about 2^-32 per run
(the public wrappers retry with new seeds). De-duplication (step 7) removes the trivial dependencies that
would otherwise eat into those 32.

Termination (Revision 1, B1; Revision 2, B1). Every loop of the core is bounded:
- Steps 1 and 2: 30 multipliers x 300 primes; the factor base scan visits at most the primes below 2^31,
  and in practice stops near 1.8 * 10^6 at F = 65536 (half of the primes are residues, Chebotarev).
- Step 4: the guard runs before any draw; a draw is s - 1 shuffle steps plus one scan of E (at most F).
- The polynomial loop (Revision 2, B1): the budget is checked before every polynomial, the first one of
  each a included, and before every draw. So at most `max_polynomials` polynomials are sieved in all,
  whatever s is. Between two sieved polynomials there is either one Gray-code step (O(F) word operations
  and one BigInt addition) or the start of a new a: at most 1000 rejected draws in a row (then
  `OutOfPolynomials`), one accepted draw and its initialisation (O(s) BigInt operations and O(s F) word
  operations). A new a is drawn only while the counter is below the budget, and its first polynomial is
  then sieved, so there are at most `max_polynomials` accepted draws and at most
  1001 * (max_polynomials + 1) draws in all.
- Per polynomial: the sieve is O(M + sum 2M/p), the scan O(M), each candidate O(F) word remainders. A large
  `threshold_slack_bits` makes most of the 2M locations candidates (T is clamped at 1), so a polynomial can
  then cost O(M F) word operations; this only costs time.
- Steps 8 and 9: R columns and rows; at most R dependencies.

#### Changes to port from the prototype `$X/rev2/siqs_r3.rs`

The Revision 2 prototype has the |y| de-duplication, the cap on first partials and the stop at R rows of
step 7, the a-supply guard and the Fisher-Yates draw of step 4, the polynomial budget checked before every
polynomial (step 4; the Revision 1 prototype `$X/rev1/siqs_r2.rs` checked it only once per a, which is the
defect of review 2 B1, so do not take that file as the model for this loop), the `p = N` case of step 2,
the `Outcome` enum (here `QsOutcome`), the caps and `validate` (`QsParams::validate`), and `for_bits` with
the L(n) factor base and the clamp. On the way:
- Use I1's `sieve::for_each_prime` for the factor base scan and `sieve::primes_up_to` for the 300
  multiplier primes (the prototype's eager `primes_up_to` is not ported), and the crate's word modexp
  (`fast_arith::pow_mod_u64`, `pub(crate)` after I1). Keep the word helpers `mul_mod_u64`, `inv_mod_u64`,
  `sqrt_mod_u64` private to `siqs.rs`.
- Implement step 0 in `quadratic_sieve.rs` (the prototype assumes its precondition).
- Split the prototype's single function as listed in 1.4, with `RelationStore::add` returning `Added`.
- Keep the budget check inside the Gray-code loop, before the root update of each polynomial, and not only
  at the top of the a-loop (`$X/rev2/siqs_r3.rs:495-500`). `qs_polynomial_budget_is_per_polynomial` hangs
  without it (caught by `within`), and `run_stops_at_the_polynomial_budget` counts the polynomials.
- `run` returns `(QsOutcome, RunStats)` with `RunStats { a_factors, a_values, polynomials,
  dropped_partials }` for the unit tests; the public functions drop the stats. The prototype's timing
  fields and its env switches (`NODEDUP`, `CAPMUL`) are not ported.
- Assertions stay `assert!` (relation check, even exponents, X^2 = Y^2): they are O(1) per relation or
  dependency.
- No `f64` decisions except parameter choice and the threshold (both only change speed, never results).
  The window and the last-index choice use f64 logs; they only steer which valid a is drawn.

### 1.3 Parameters (factor base from L(n)) and measured cost

L(n) = exp(sqrt(ln n ln ln n)) [CP05 §6.1.1]. The theoretical optimum B = L(n)^(1/2 + o(1)) has large o(1)
terms at these sizes; a fit to the measured optimum of the prototype is

    F = clamp(round(3.4 * L(n)^0.27), 60, 65536),   ln n = max(bits, 40) * ln 2

(F = 123, 210, 344, 546, 841, 1267, 1873, 2723, 3901, 5520, 7721 at 20, 25, ..., 70 digits; 14663 at 80,
48000 at 100; the cap 65536 is reached at about 106 digits). M and the large prime multiplier come from a
table in bits, linearly interpolated, M rounded to a multiple of 64; below 40 and above 280 bits the end
values are used:

| bits <= | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 | 220 | 240 | 260 | 280 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M | 4096 | 8192 | 16384 | 16384 | 32768 | 32768 | 32768 | 32768 | 65536 | 65536 | 65536 | 98304 | 131072 |
| lp_mult | 10 | 20 | 30 | 40 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 100 | 100 |

Other defaults: threshold slack 12 bits, extra relations 32, `max_polynomials = 1000 * F`, seed 1. So
`for_bits(bits)` validates for every u64 (F in [60, 65536], M <= 131072, lp_mult <= 100; for bits = u64::MAX
the f64 L(n) is infinite and the clamp gives 65536). Sweeps (`$X/siqs.rs` with `LPM`, `MM`, slack 0..15,
F x 1..3) showed a flat optimum: slack 9 to 15 and F x 2 to 2.5 of the msieve-like table are within 15 % of
each other; halving or doubling M or lp_mult changes 50 to 60 digits by less than 20 %.

Measured on this machine (gp semiprimes of section 1.6; release = `rustc -O`, debug = opt-level 0 against
the workspace's debug rlibs, as `cargo test` runs). The release and debug rows from 20 to 60 digits are the
Revision 1 prototype (`$X/rev1/grid_rel_r1.txt`, `$X/rev1/grid_dbg_r1.txt`); 65 to 72 digits and PARI are
from the first prototype (`$X/rel_lfb.txt`, `$X/pari_times.gp`). The Revision 2 changes do not alter these
runs: the same polynomials, dependencies and factors from 18 to 60 digits (`$X/rev2/grid_rel_r3.txt`,
`grid_rel_r3b.txt`), and every row from 61 to 70 digits factored with 0 dropped partials
(`$X/rev2/grid_rel_r3cap_61_70.txt`; its times are up to 1.4 times the table's, measured while other
builds ran on the machine):

| digits | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 | 65 | 70 | 72 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| release | 6 ms | 18 ms | 14 ms | 43 ms | 77 ms | 0.15 s | 0.35 s | 1.1 s | 2.4 s | 8.7 s | 39 s | 65 s |
| debug | 33 ms | 64 ms | 97 ms | 0.23 s | 0.46 s | 1.3 s | 3.0 s | 7.8 s | 16 s | | | |
| PARI 2.15.4 `factor` | 1 ms | | 4 ms | | 32 ms | | 0.33 s | 0.91 s | 2.4 s | 8.1 s | 31 s | |

Peak memory (release): 14 MB at 60 digits, 34 MB at 70 digits. Linear algebra share: 0.15 s of 2.4 s at 60
digits (debug 2.8 s of 16 s); it is O(F^2 (F + R)/64) word operations, so at 80 digits (F about 15000)
about 20 s release, still below the sieve time. At the cap F = 65536 the matrix takes 1.08 GB and about
12 minutes in release (scaled from 0.15 s at F = 3901 by (65536/3901)^3).

Robustness (Revision 1, with the `for_bits` parameters and de-duplication): `$X/robust.txt` (581 gp inputs,
8 to 45 digits: balanced and unbalanced semiprimes, p^2 q, p q r, and 62 primes), seeds 1 to 6: a proper
divisor for every composite and `NoFactor` for every prime, 0 failures, 13 to 14 s per seed (release),
worst single input 0.31 s (`$X/rev1/robust_dedup_r2.txt`). `$X/rev1/small.txt` (1445 gp inputs, 6 to 18
digits, generator `$X/rev1/small.gp`: two-prime composites, p^3 q, p^2 q^2 r, 53 q, three-prime
composites, primes; every composite without a prime factor <= 47 and not a perfect power), seeds 1 to 6: 0
failures, about 4.5 s per seed. The reviewer's independent run (seeds 1 to 6, the 581 inputs plus 1849 of
6 to 18 digits) also gave 0 failures.

### 1.4 API and file layout

`rustmath-integers/src/quadratic_sieve.rs` (rewritten; public API, step 0, drivers, docs) with submodules
`src/quadratic_sieve/siqs.rs` (multiplier, factor base, word arithmetic, polynomials, sieve, relations,
square root) and `src/quadratic_sieve/gf2.rs` (left nullspace). Each file header cites MAGMA ch. 18
§18.10.3 (`MPQS`) and the papers above.

```rust
/// Parameters of one SIQS run (MAGMA ch. 18 §18.10.3 `MPQS`; Contini 1997).
///
/// Memory of a run, with R = F + 1 + extra: the sieve array, 2 M bytes (at most 32 MiB); the factor base,
/// roots and Bainv2 table, about (20 + 4 s) F bytes with s <= 64 (at most 18 MB); the relations, at most
/// R + 16 F of them (at most 16 F large primes keep a first partial relation, and the run stops at R rows),
/// measured 3.7 to 4.9 F from 20 to 55 digits at about 300 bytes each; the dense GF(2) matrix,
/// R (F + 1 + R) / 8 bytes, about F^2 / 4 bytes (0.2 MB at F = 841 (40 digits), 3.9 MB at F = 3901
/// (60 digits), 1.08 GB at the cap F = 65536). Measured peak: 14 MB at 60 digits, 34 MB at 70 digits. At
/// the caps (F = 65536, M = 2^24, for n near 110 digits) about 1.5 GB (estimate: 1.1 million relations of
/// about 350 bytes; a relation grows slowly with n, as |y| has about bits(n)/2 bits). Time of the matrix
/// step: O(R^2 (F + R) / 64) word operations (0.15 s in release at F = 3901, about 12 minutes at the cap).
/// Time of the sieve: at most `max_polynomials` polynomials (checked before every one), each
/// O(M + sum M/p) plus O(F) word operations per candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct QsParams {
    /// Number of primes in the factor base, 2 included. MIN_FACTOR_BASE_SIZE ..= MAX_FACTOR_BASE_SIZE.
    pub factor_base_size: usize,
    /// Half width M of the sieve interval [-M, M). MIN_SIEVE_HALF_WIDTH ..= MAX_SIEVE_HALF_WIDTH; rounded up
    /// to a multiple of 64.
    pub sieve_half_width: usize,
    /// Partial relations with one prime cofactor below min(this * p_max, p_max^2) are kept (the product
    /// saturates). >= 1.
    pub large_prime_multiplier: u64,
    /// Bits subtracted from the expected log2 |g(x)| in the sieve threshold. Any value.
    pub threshold_slack_bits: u8,
    /// Matrix rows beyond F + 1; the nullspace has at least this dimension. 1 ..= MAX_EXTRA_RELATIONS.
    pub extra_relations: usize,
    /// Polynomials sieved before the run gives up with `OutOfPolynomials`; the count is checked before
    /// every polynomial, so no more than this many are sieved. >= 1.
    pub max_polynomials: u64,
    /// Seed of the splitmix64 generator that draws the a-factors. Any value.
    pub seed: u64,
}
impl QsParams {
    pub const MIN_FACTOR_BASE_SIZE: usize = 30;
    /// The dense matrix takes about F^2 / 4 bytes: 1.08 GB here.
    pub const MAX_FACTOR_BASE_SIZE: usize = 65536;
    pub const MIN_SIEVE_HALF_WIDTH: usize = 64;
    /// The sieve array takes 2 M bytes: 32 MiB here. The default table never exceeds 2^17.
    pub const MAX_SIEVE_HALF_WIDTH: usize = 1 << 24;
    pub const MAX_EXTRA_RELATIONS: usize = 1024;
    /// The defaults of section 1.3 for an n of `bits` bits. Defined for every u64 and always valid:
    /// F is clamped to [60, MAX_FACTOR_BASE_SIZE], M and the multiplier to the ends of the table.
    pub fn for_bits(bits: u64) -> QsParams;
    /// Ok(()) iff every field is in its documented range; Err(InvalidArgument) names the field and range.
    pub fn validate(&self) -> Result<()>;
}

/// Result of one sieve run. None of the variants is a statement about the primality of n.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QsOutcome {
    /// A proper divisor d of |n| (1 < d < |n|, d | n), not necessarily prime.
    Factor(Integer),
    /// Every dependency gave gcd 1 or |n|. Always so for a prime (and `dependencies: 0` when step 0 or the
    /// factor base scan meets |n| itself); probability about 2^-dependencies for a composite that is not a
    /// prime power.
    NoFactor { dependencies: usize },
    /// The polynomial budget or the supply of a-values ran out before `needed` matrix rows were found;
    /// `relations` is the number of rows found. `relations: 0` can also mean that the a-supply guard
    /// fired before any sieving: the factor base has too few primes above 30 to build an a of the size n
    /// needs (or s > 64).
    OutOfPolynomials { relations: usize, needed: usize },
}

/// One factor of n, or None (step 0 of section 1.2, then `is_probable_prime(|n|)` gives None, then SIQS
/// with `QsParams::for_bits` and seeds 1, 2, 3 until `Factor`). None never claims that n is prime.
pub fn quadratic_sieve_factor(n: &Integer) -> Option<Integer>;

/// One run with the given parameters after step 0, without the probable-prime check (a prime n > 47 runs
/// the sieve and gives `NoFactor`). Ok(NoFactor { dependencies: 0 }) for n = +-1 and n = +-p with p a prime
/// <= 47. Errors (`InvalidArgument`): n = 0; `params.validate()` fails.
pub fn quadratic_sieve_factor_with_params(n: &Integer, params: &QsParams) -> Result<QsOutcome>;

/// Complete factorization of |n| (section 3.4), primes ascending, empty for 0 and +-1. Panics when a
/// factor's primality proof is unresolved (I1's policy); `factorization::factorization_with_limits` never
/// panics and reports such a factor in `probable_primes`.
pub fn quadratic_sieve_factor_complete(n: &Integer) -> Vec<(Integer, u32)>;
```

Crate-internal (`siqs.rs`), the prototype's single function split up:

```rust
fn multiplier(n: &BigUint) -> u32;
struct FactorBase { k: u32, kn: BigUint, primes: Vec<u32>, sqrt_kn: Vec<u32>, logp: Vec<u8>,
                    divides_k: Vec<bool>, sieve_start: usize }
enum BaseScan { Base(FactorBase), Divisor(u32), IsPrime }   // step 2: p | N with p < N, or p = N
fn factor_base(n: &BigUint, k: u32, size: usize) -> BaseScan;
fn sqrt_mod_u64(a: u64, p: u64) -> Option<u64>;
/// None: the guard of step 4 failed (s > 64, |E| < s or |W| < max(s - 1, 1)).
fn a_selector(fb: &FactorBase, m: usize, seed: u64) -> Option<ASelector>;
struct ASelector { window: Vec<usize>, eligible: Vec<usize>, s: usize, log2_target: f64, rng: u64,
                   used: HashSet<Vec<usize>> }
fn choose_a(fb: &FactorBase, sel: &mut ASelector) -> Option<Vec<usize>>;   // None: 1000 repeats in a row
struct Polynomial { a: BigUint, a_factors: Vec<usize>, bl: Vec<BigUint>, b: BigInt, c: BigInt, index: u64 }
struct Roots { r1: Vec<u32>, r2: Vec<u32>, bainv2: Vec<Vec<u32>>, is_a_factor: Vec<bool> }
fn init_polynomial(fb: &FactorBase, m: usize, a_factors: &[usize]) -> (Polynomial, Roots);
fn next_polynomial(fb: &FactorBase, poly: &mut Polynomial, roots: &mut Roots); // one Gray-code step
fn sieve(fb: &FactorBase, roots: &Roots, init: u8, buf: &mut [u8]) -> Vec<usize>; // candidate indices
fn check_candidate(fb: &FactorBase, poly: &Polynomial, roots: &Roots, m: usize, i: usize,
                   lp_bound: u64) -> Option<Relation>;                         // includes the exact check
struct RelationStore { rels: Vec<Relation>, rows: Vec<Vec<usize>>, partial_of: HashMap<u64, usize>,
                       seen_y: HashSet<BigUint>, first_partial_cap: usize }
/// Dropped: a partial relation with a new large prime while `partial_of` has `first_partial_cap` entries.
enum Added { Full, PartialFirst, PartialPaired, Duplicate, Dropped }
const FIRST_PARTIALS_PER_PRIME: usize = 16;   // first_partial_cap = 16 F in `run`
impl RelationStore { fn new(first_partial_cap: usize) -> RelationStore; fn add(&mut self, rel: Relation) -> Added; }
fn square_roots(n: &BigUint, fb: &FactorBase, store: &RelationStore, deps: &[Vec<usize>]) -> QsOutcome;
/// Counters of one run (unit tests): s, the accepted a-values, the polynomials sieved, the dropped partials.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RunStats { a_factors: usize, a_values: u64, polynomials: u64, dropped_partials: u64 }
fn run(n: &BigUint, params: &QsParams) -> (QsOutcome, RunStats);   // steps 1 to 9; precondition of step 0
```

MAGMA's `MPQS(n)` requires n > 5 * 10^24 (digest §18.10.3); the functions here accept every n: step 0 and
the factor-base scan settle small n, and the sieve itself works from about 8 digits on (the robustness
inputs start at 6).

The unit tests sit in `siqs.rs`'s own `mod tests` and call these private items directly. `gf2.rs`:
`pub(super) fn left_nullspace(rows: &[Vec<u64>], ncols: usize) -> Vec<Vec<usize>>` (each result lists row
indices). Internally the sieve works on `num_bigint::BigUint`/`BigInt` through `Integer::as_bigint` and
`Integer::new`; no wrapper types. Negative n: the public functions work on |n| and return positive divisors.

`lib.rs`: add `QsParams, QsOutcome` to the existing `pub use quadratic_sieve::{...}`.

### 1.5 Callers and behaviour changes (grep of the whole workspace)

- `quadratic_sieve_factor`, `quadratic_sieve_factor_complete`: re-exported in `lib.rs:24`; used only in
  `rustmath-integers` (module docs, `tests/ch18_factorization_drivers.rs:26`, `:150-155`, `:188`). No other
  crate uses `rustmath_integers::quadratic_sieve` (grep `quadratic_sieve` over `*.rs`: only this crate; the
  reviewer's grep of `*.rs`, `*.md`, `*.csv` agrees, and no REPL or lava binding calls these functions).
  Behaviour: `quadratic_sieve_factor` now finds factors (165731 gives 53; 3127 gives 53; 10403 gives 101).
- `quadratic_sieve_factor_with_params(n, usize, usize) -> Option<Integer>` becomes
  `(n, &QsParams) -> Result<QsOutcome>`. Callers: only `quadratic_sieve.rs` tests `:723`, `:733`.
- `quadratic_sieve_factor_complete`: same signature and contract; now uses the sieve (section 3.4).
- Removed private items: `legendre_symbol`, `tonelli_shanks` (ported as `sqrt_mod_u64`),
  `generate_factor_base`, `trial_factor`, `sieve_for_smooth_relations`, `build_exponent_matrix`,
  `gaussian_elimination_gf2`, `try_dependency`, `SmoothRelation`.
- Docs: `PROJECT_TRACKER.md` rows `:395` (🔴), `:406`, `:407` (⏳), card `:254` (T-223), blocker `:3100`
  (B-04 note); `docs/magma/status_overrides.csv` `ch018.MPQS` (broken); `docs/HANDOFF.md:275-277`.

### 1.6 Tests (item 1)

New integration file `rustmath-integers/tests/ch18_quadratic_sieve.rs` (header: MAGMA ch. 18 §18.10.3
`MPQS`, [Con97], [CP05 §6.1]). The semiprimes are reproduced by

```
gp: default(realprecision, 120); d = 40; h = d\2; p = nextprime(floor(Pi*10^(h-1)));
    q = nextprime(floor(sqrt(10)*exp(1)*10^(d-h-1))); [p, q, p*q, factor(p*q) == [p,1; q,1]]
```

| d | n | p | q |
|---|---|---|---|
| 20 | 27005010885249061781 | 3141592661 | 8595961921 |
| 25 | 2700501075624869745621467 | 314159265359 | 8595961900213 |
| 30 | 270050107561429816771648674403 | 314159265359057 | 859596190017779 |
| 35 | 27005010756136059108789661003408639 | 31415926535897999 | 859596190017769361 |
| 40 | 2700501075613600140300554153516920373243 | 31415926535897932429 | 85959619001776934567 |
| 45 | 270050107561360013478271527386296955458256719 | 3141592653589793238499 | 85959619001776934511781 |
| 50 | 27005010756136001347490384749379527844355732904871 | 3141592653589793238462773 | 8595961900177693451170027 |
| 55 | 2700501075613600134748901272440129406658374730908723747 | 314159265358979323846264367 | 8595961900177693451169944141 |
| 60 | 270050107561360013474890101164188225207220555802656708135689 | 314159265358979323846264338521 | 859596190017769345116994409009 |

Other values (gp `$X/refs.gp`, `$X/rev1/refs_r1.gp`): P30 = nextprime(10^29) =
100000000000000000000000000319; p20 = nextprime(10^19) = 10000000000000000051; N150 =
nextprime(10^74) * nextprime(3*10^75) =
300000000000000000000000000000000000000000000000000000000000000000000000627700000000000000000000000000000000000000000000000000000000000000000000013869
(150 digits).

| test | input | expected | reference | HEAD |
|---|---|---|---|---|
| `qs_factors_gp_semiprimes_20_to_45_digits` | rows 20..45 | `quadratic_sieve_factor(n)` is p or q; rows 20 and 30 also as `-n` | gp above | F (`None`) |
| `qs_factors_named_inputs` | F7 = 2^128 + 1; 10000019 * 10000000000000000000009; 1000000007 * 2000000011 * 3000000019; 100000007^2 * 10000000000000061 | a proper divisor of n each | gp `factor(2^128+1)` = [59649589127497217, 1; 5704689200685129054721, 1]; `factor(100000190000000000000090000171)`; `factor(6000000113000000706000001463)`; `factor(100000014000001100000085400002989)` = [100000007, 2; 10000000000000061, 1] | F |
| `qs_complete_matches_pari` | the 40-digit row, F7, the p q r and p^2 q inputs above, psi_12 = 318665857834031151167461 (psi_12 needs I1's BPSW) | `quadratic_sieve_factor_complete(n)` equals gp `factor(n)` (-n for the p q r input) | gp `factor(318665857834031151167461)` = [399165290221, 1; 798330580441, 1] | F (does not finish for the 40-digit row; psi_12 unsplit) |
| `qs_complete_prime_powers` (Revision 1, m4) | 3^2000; 1000003^50; 2^1000 * 3^500 | [(3, 2000)]; [(1000003, 50)]; [(2, 1000), (3, 500)]; each inside `within(5, ..)` | gp `factor(3^2000)` = Mat([3, 2000]), `factor(1000003^50)` = Mat([1000003, 50]), `factor(2^1000*3^500)` | P (HEAD debug: 1.7 ms and 27 ms, `$X/rev1/head_pp.rs`); a regression guard for the engine-based driver |
| `qs_prime_gives_no_factor` | P30; p20 | `quadratic_sieve_factor(P30)` = None (early BPSW return); `quadratic_sieve_factor_with_params(p20, &QsParams::for_bits(64))` = `Ok(NoFactor { dependencies })` with dependencies >= 32 | gp `isprime(nextprime(10^29))` = 1, `isprime(nextprime(10^19))` = 1 | F (signature) |
| `qs_with_params_tiny_inputs` (Revision 1, m2) | n in [1, -1, 2, -2, 3, 5, 43, 47, -47]; n in [4, 9, 15, 25, 35, 45, 46, 49, 2209]; 2207; 0 | `Ok(NoFactor { dependencies: 0 })`; `Ok(Factor(d))` with d = 2, 3, 3, 5, 5, 3, 2, 7, 47 (step 0 order); `Ok(NoFactor { dependencies })` with dependencies >= 32 (a prime above 47 runs the sieve; prototype: 36 with k = 2); `Err(InvalidArgument)` | gp `primes(15)`, `factor(2209)` = Mat([47, 2]), `isprime(2207)` = 1 | F (signature) |
| `qs_quick_paths` | `quadratic_sieve_factor` of 100; 3 * P30; 1000003^3; 10403; 0; 1; -1; 2; 3; 47 | `Some(2)`, `Some(3)`, `Some(1000003)`, 101 (factor-base scan), `None` for 0, +-1, 2, 3, 47 | gp `factor(1000003^3)` = Mat([1000003, 3]); `factor(10403)` | partly F |
| `qs_out_of_polynomials` | 40-digit row, `max_polynomials = 1` | `OutOfPolynomials { relations, needed }` with relations < needed | | N |
| `qs_a_supply_too_small_returns_promptly` (Revision 1, B1) | N150 with `QsParams { factor_base_size: 30, ..QsParams::for_bits(bits) }` and with 31 | `Ok(OutOfPolynomials { relations: 0, needed: 63 })` and `needed: 64`, inside `within(5, ..)` (prototype: s = 34 needed, 22 eligible; 2.5 ms debug) | gp `n150` above | N (HEAD has no such parameters; the first prototype hung) |
| `qs_params_validation` (Revision 1, B2) | `validate` of `for_bits(64)` with one field changed: F = 29, 65537; M = 63, 2^24 + 1; extra 0, 1025; lp_mult 0; max_polynomials 0; and `with_params(10403, ..)` with F = 65537 | each `Err(InvalidArgument)` naming the field; `for_bits(b).validate()` is Ok and F is non-decreasing for b in [0, 1, 2, 40, 64, 200, 332, 400, 2000, 10^6, u64::MAX]; `for_bits(u64::MAX).factor_base_size == 65536`; `for_bits(0).factor_base_size == 60` | | N |
| `qs_polynomial_budget_is_per_polynomial` (Revision 2, B1) | N150 with `QsParams { factor_base_size: F, max_polynomials: 10, ..QsParams::for_bits(497) }` for F = 40, 60, 100 (497 = `N150.bits()`, gp `#binary(n150)`; the review's `for_bits(498)` gives the same M and multiplier, both past the table's end at 280 bits; the guard passes with s = 31, 28, 26) | `Ok(OutOfPolynomials { relations: 0, needed: 73 })`, `needed: 93`, `needed: 133`, all three inside `within(5, ..)` (prototype `$X/rev2/siqs_r3.rs`: 3 to 5 ms release, 14 to 24 ms debug) | gp `n150` above; needed = F + 1 + 32 | N (no such parameters); a port of the Revision 1 loop hangs here (the first a has 2^30 polynomials) |
| `qs_seed_sweep_24_digits` (Revision 1, m1) | 666958488061768516158211 with `for_bits(80)` and seeds 1..=6 | `Ok(Factor(d))`, d in {765944099509, 870766533079}, every seed (prototype debug: 0.6 s for all six) | gp `factor(666958488061768516158211)` | N |
| `#[ignore] qs_factors_gp_semiprimes_50_to_60_digits` | rows 50, 55, 60 | p or q | gp above | F |

`#[ignore]` reason string: "about 30 s in a debug build; run with `cargo test --release -p rustmath-integers
--test ch18_quadratic_sieve -- --ignored` (about 4 s)". `within(secs, f)` is a copy of the helper at
`rustmath-polynomials/tests/ch23_quotrem_gcd.rs:66-84` (I3 copies it into `tests/ch18_signs_roots.rs`; each
integration file of this unit that uses it gets its own copy). Debug cost of the default QS tests (prototype
timings): rows 20..45 2.2 s, named inputs 0.7 s, complete driver 1.3 s, primes and tiny inputs 0.3 s, seed
sweep 0.6 s, budget tests 0.1 s: about 5.6 s of CPU, less wall time since the test harness runs tests in
parallel.

Unit tests in `quadratic_sieve/siqs.rs` and `gf2.rs`:

| test | check | reference | HEAD |
|---|---|---|---|
| `sqrt_mod_u64_least_non_residue_above_100` | port of `quadratic_sieve.rs:640-666`: p = 23616331489, roots of 2, 1000, and p - 2 (for -2), `None` for 107 | gp `lift(sqrt(Mod(2, p)))` etc. (values in the HEAD test) | P (ported) |
| `sqrt_mod_u64_small_primes_exhaustive` | port of `:628-637` (sqrt(2) mod 7 in {3, 4}) and `:668-688` (brute force over odd p < 200, with `is_prime_u64`) | brute force | P |
| `sqrt_mod_u64_rejects_composite` | `#[should_panic(expected = "sqrt_mod_u64: p = 15 is not an odd prime below 2^63")]` | gp `isprime(15)` = 0 | P |
| `factor_base_matches_kronecker` | N = 1000000016000000063, k = 3, size 53: odd primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 53, 61, 67, 71, 79, 83, 103, 109, 127, 139, 149, 151, 181, 191, 197, 199, 211, 223, 227, 233, 251, 269, 281, 293, 313, 337, 347, 349, 353, 359, 383, 389, 397, 401, 409, 419, 431, 433, 457, 509, 523]; 3 has `divides_k`; t_p^2 = 3N mod p for the others (t_5..t_29 = +-[2, 2, 1, 1, 3, 7, 7, 13]) | gp `select(p -> kronecker(3*1000000016000000063, p) >= 0, primes(100))`; `lift(sqrt(Mod(3*N, p)))` | N |
| `factor_base_reports_a_small_prime_factor` | N = 101 * P30 = 10100000000000000000000000032219, k = 1: `Divisor(101)`; N = 101: `IsPrime`, and `run` turns it into `NoFactor { dependencies: 0 }` | gp | N |
| `a_selector_guard` (Revision 1, B1) | the factor base of N150 with size 30 (k from `multiplier`): `a_selector` is `None`; the 30-digit row with `for_bits`: `Some`, and 200 calls of `choose_a` return sorted, distinct index sets of length s, each a subset of E | prototype: s = 34, \|E\| = 22 | N |
| `polynomial_roots_survive_gray_code` | 30-digit row: for every one of the 2^(s-1) polynomials of the first a and every j not dividing a k: (a (r - M) + b)^2 = kN mod p_j for r = r1, r2 | exact arithmetic | N |
| `relations_are_exact` | 30-digit row: every collected relation satisfies y^2 - kN = +-prod p^e * L with L = 1 or p_max < L < B_L prime | exact arithmetic, `is_prime_u64(L)` | N |
| `relation_store_drops_repeated_y` (Revision 1, m1) | `RelationStore::new(1000)`, synthetic relations: full y = 5 then y = -5: `Full`, `Duplicate`, rows unchanged; partial L = 101 with y = 7: `PartialFirst`; L = 101 with y = -7: `Duplicate` and no pair row; L = 101 with y = 9: `PartialPaired`, last row = {first, this} | hand | N |
| `relation_store_caps_first_partials` (Revision 2, m5) | `RelationStore::new(2)`: partial L = 101, y = 7 and L = 103, y = 9: `PartialFirst` twice; L = 107, y = 11: `Dropped`, 2 relations stored, rows unchanged; L = 101, y = 13: `PartialPaired` (pairing still works at the cap); full y = 15: `Full`; L = 107, y = 11 again: `Dropped` (not `Duplicate`: a dropped \|y\| is not recorded) | hand | N |
| `run_stops_at_the_polynomial_budget` (Revision 2, B1) | 40-digit row, `QsParams { max_polynomials: b, ..QsParams::for_bits(131) }`, b in {1, 3, 16, 17}, through `run` | outcome `OutOfPolynomials { relations, needed: 835 }` with relations < 835; `RunStats` with `a_factors == 5`, `polynomials == b`, `a_values == ceil(b / 16)` (1, 1, 1, 2: each accepted a sieves its 16 polynomials before the next draw) | prototype `siqs_r3 np 0 0 1 b n40`: polys = 1, 3, 16, 17 and a = 1, 1, 1, 2 | N |
| `gf2_left_nullspace_hand` | port of `:799-810`: rows [1,0,1], [0,1,1], [1,1,0]: exactly one vector {r0, r1, r2}; 3 x 2 all ones: 2 vectors, each row sum 0 | hand | F (HEAD's function returns `[r0]` for the all-ones matrix, row sum [1, 1]) |
| `gf2_left_nullspace_random` | seeded random matrices 64 x 64 dense, 120 x 100 at 5 % density, 200 x 150: each vector sums to 0; count = rows - rank, rank computed in the test by a separate column-echelon elimination of the transpose | independent elimination | N |

Fate of every HEAD test and doc example in `quadratic_sieve.rs` (Revision 1, m8):

| HEAD item | fate |
|---|---|
| `test_legendre_symbol` (`:616`) | deleted with the function; `factor_base_matches_kronecker` covers the residue test |
| `test_tonelli_shanks` (`:628`) | ported into `sqrt_mod_u64_small_primes_exhaustive` |
| `test_tonelli_shanks_least_non_residue_above_100` (`:640`) | ported as `sqrt_mod_u64_least_non_residue_above_100` |
| `test_tonelli_shanks_small_primes` (`:668`) | ported as `sqrt_mod_u64_small_primes_exhaustive` |
| `test_tonelli_shanks_composite_modulus_panics` (`:691`) | ported as `sqrt_mod_u64_rejects_composite` (new message) |
| `test_factor_base_generation` (`:698`) | replaced by `factor_base_matches_kronecker` |
| `test_trial_factor` (`:706`) | deleted with `trial_factor`; `relations_are_exact` covers the candidate check |
| `test_quadratic_sieve_small` (`:719`) | rewritten: `quadratic_sieve_factor_with_params(143, &QsParams::for_bits(8))` = `Ok(Factor(11))` (step 0) |
| `test_quadratic_sieve_larger` (`:729`) | rewritten: `quadratic_sieve_factor_with_params(1001, &QsParams::for_bits(10))` = `Ok(Factor(7))` |
| `test_quadratic_sieve_prime` (`:741`) | kept (97 gives `None`) |
| `test_quadratic_sieve_even` (`:748`) | kept (100 gives `Some(2)`) |
| `test_complete_factorization` (`:755`) | kept (60) |
| `test_complete_factorization_composite_sieve_divisor` (`:774`) | renamed `test_complete_factorization_repeated_prime`; first assertion `quadratic_sieve_factor(165731) == Some(53)` (the factor-base scan meets 53 first; a property of the scan order, not of random choices); second assertion kept; doc comment rewritten. The composite-divisor path of the driver is tested by `engine_requeues_composite_divisors` (3.5) |
| `test_complete_factorization_sieve_failure` (`:788`) | renamed `test_complete_factorization_3127`; `quadratic_sieve_factor(3127) == Some(53)`; complete assertion kept |
| `test_gaussian_elimination` (`:799`) | ported into `gf2_left_nullspace_hand` |
| module doc example (`:18-28`, 15347) | becomes the 20-digit row, with an assertion that the factor is p or q (debug about 40 ms) |
| `quadratic_sieve_factor` doc example (`:409-420`, 15347) | kept with `assert_eq!(quadratic_sieve_factor(&n), Some(Integer::from(103)))` and the sentence "a prime factor below the factor base bound is found by the factor-base scan" |
| `quadratic_sieve_factor_complete` doc example (`:533-548`) | 1001 kept; the 12780049 comment "the sieve finds no factor, so the cofactor is finished by Pollard rho" becomes "12780049 = 53 · 59 · 61 · 67 (PARI/GP: factor(12780049)); Brent rho with 8191 iterations splits it before the sieve is needed"; a third example factors the 30-digit row into its two primes |
| new `quadratic_sieve_factor_with_params` doc example | 270050147102784401 with `QsParams::for_bits(58)`: `Ok(Factor(d))`, d in {314159311, 859596191} |

## 2. Item 2: `trial_division_with_primes`

### 2.1 Defect at HEAD

`factorint.rs:401-440`: the loop breaks at the first list prime with p^2 > remaining (`:415-417`, which is
also wrong for an unsorted list), then pushes the leftover as `(remaining, 1)` (`:434-436`). The doc
(`:397-400`) promises a factorization. `$X/repro_head.rs` (the reviewer adds (12, [4, 3]) -> [(12, 1)]):

| call | HEAD | gp |
|---|---|---|
| `trial_division_with_primes(1225, [2, 3])` | `[(1225, 1)]` | `factor(1225)` = [5, 2; 7, 2] |
| `trial_division_with_primes(-1225, [2, 3])` | `[(-1, 1), (1225, 1)]` | |
| `trial_division_with_primes(1001, [2, 3])` | `[(1001, 1)]` | `factor(1001)` = [7, 1; 11, 1; 13, 1] |
| `trial_division_with_primes(8, [5, 2])` | `[(8, 1)]` (breaks at 5 since 25 > 8) | `factor(8)` = Mat([2, 3]) |

### 2.2 Fix

New signature (MAGMA `TrialDivision(n, B)` returns the primes found and the unfactored part, §18.10.3):

```rust
/// Trial division of n by the given list (any order, duplicates allowed). The result is a
/// `LimitedFactorization` (section 3.2): the listed primes that divide n with their exact exponents in
/// `primes`; the cofactor m = |n| / prod p^e, when m > 1, classified without a proof attempt:
/// `primes` if m < psi_12 and `is_probable_prime(m)` (exact there), `composites` if
/// `!is_probable_prime(m)` (a proof), else `probable_primes`. The sign of n is `sign`; 0 gives sign 0
/// and empty lists.
///
/// # Panics
/// If an entry is < 2, or an entry that divides n is proven composite (it would otherwise be reported
/// as a prime). Entries that do not divide n are not tested.
pub fn trial_division_with_primes(n: &Integer, primes: &[Integer]) -> LimitedFactorization;
```

Algorithm: m = |n|; for each entry p, in list order and without an early break: if p | m, strip all powers
of p (exponent e) and classify p with `classify(p, true)` (section 3.2): `Prime` goes to `primes`,
`ProbablePrime` (only for an entry >= psi_12 whose ECPP proof is unresolved) to `probable_primes`,
`Composite` panics with "trial_division_with_primes: the list entry {p} is not prime". After the list,
classify the cofactor m > 1 with `classify(m, false)`: no proof attempt, because a large cofactor is the
normal case of trial division and its proof can cost seconds; m < psi_12 is still decided (`Prime` or
`Composite`), above psi_12 a BPSW pass gives `ProbablePrime`. Sort the lists. Cost: one remainder per entry,
one classification per dividing entry, one BPSW test for the cofactor.

Invariants: bases pairwise coprime (distinct primes and a cofactor without listed prime factors); product
times sign = n.

### 2.3 Callers

Only `factorint.rs` tests `:783-828`. The module doc of `factorint.rs` gains a line on MAGMA `TrialDivision`.
Tracker: `PROJECT_TRACKER.md:399` (🛑), card T-223 text; `docs/magma/status_overrides.csv` `ch018.TrialDivision`
keeps pointing at `factor_trial_division` (Sage port, ⏳, composite cofactor) and adds
`factorization_with_limits` with `pollard_rho_limit = 0, mpqs_limit = Some(0)` as the MAGMA-shaped form.

The same `ch018.TrialDivision` note records a defect of `factor_trial_division` that is outside this unit
(review 1, m10): with `limit = None` it computes the bound as
`remaining.sqrt().unwrap_or(Integer::from(1000000)).to_u64().unwrap_or(1000000)` (`factorint.rs:190-192`).
For |n| > 2^128 the square root does not fit a u64, so trial division silently stops at 10^6 and appends a
composite cofactor, although the doc says that `None` means sqrt(n). Hand-off (section 8): bound the loop by
`min(isqrt(n), u64::MAX)` with the segmented sieve and an early stop on a probable-prime cofactor, as the
limits engine does (3.2), or document the 10^6 cap.

### 2.4 Tests (replace `factorint.rs:782-828`)

| test | input | expected | gp | HEAD |
|---|---|---|---|---|
| `trial_division_with_primes_full` | 210, [2, 3, 5, 7] | primes [(2,1),(3,1),(5,1),(7,1)], other lists empty, sign 1 | `factor(210)` | P (shape changes) |
| `trial_division_with_primes_prime_cofactor` | 30, [2, 3] | primes [(2,1),(3,1),(5,1)] (5 < psi_12 is decided) | `factor(30)` | P |
| `trial_division_with_primes_composite_cofactor` | 1225, [2, 3]; -1225; 1001, [2, 3] | composites [(1225,1)], sign -1 for -1225; composites [(1001,1)] | `factor(1225)`, `factor(1001)` | F |
| `trial_division_with_primes_unsorted_list` | 8, [5, 2] | primes [(2,3)] | `factor(8)` | F |
| `trial_division_with_primes_probable_prime_cofactor` | 2 * P30 = 200000000000000000000000000638, [2] | primes [(2,1)], probable_primes [(P30,1)] | `isprime(nextprime(10^29))` = 1 | F |
| `trial_division_with_primes_large_composite_cofactor` | 6 * p15 * q15 = 120000000000053400000000005022, [2, 3] | primes [(2,1),(3,1)], composites [(20000000000008900000000000837,1)] | `factor(6*nextprime(10^14)*nextprime(2*10^14))` | F |
| `trial_division_with_primes_zero_and_units` | 0; 1; -1 | sign 0 / 1 / -1, empty lists | | F (type) |
| `trial_division_with_primes_rejects_composite_entry` | 12, [4, 3] | `#[should_panic(expected = "trial_division_with_primes: the list entry 4 is not prime")]` | | F |

## 3. Item 3: capped drivers and MAGMA's third return value

### 3.1 Defect at HEAD

`prime.rs:808-860` `factor_pollard_rho` has no limit and returns only `Vec<(Integer, u32)>`; there is no
function that stops and reports what it could not split (MAGMA `Factorization(n : ECMLimit, MPQSLimit)`,
`PollardRho(n, c, s, k)`, `TrialDivision(n, B)` all return the unfactored composites, §18.10.1, §18.10.3).
For 101 * p20 * q20 (gp row in 3.5) `factor_pollard_rho` needs about sqrt(10^19), some 3 * 10^9 rho steps
(its own doc: 10^6 steps take about 20 s in debug), so it does not finish in practice; the tracker rows
(`docs/magma/status_overrides.csv` `ch018.PollardRho`: "no c,s,k params, no Brent, no factorization
output"; `ch018.Factorization`: "No sign output and no Proof, Bases, ECMLimit or MPQSLimit parameters")
confirm the gap.

### 3.2 Fix: `LimitedFactorization` and the limits engine

New file `rustmath-integers/src/factorization.rs`. Header: MAGMA ch. 18 §18.10.1 (`Factorization` and its
parameters, the third return value), §18.10.3 (`TrialDivision`, `PollardRho`, `MPQS`), §18.10.4
(`CoprimeBasis`); E. Bach, J. Driscoll, J. Shallit, *Factor refinement*, J. Algorithms 15 (1993) 199-222
(gcd-only refinement); [Bre80] R. P. Brent, *An improved Monte Carlo factorization algorithm*, BIT 20 (1980)
176-184, for rho; C. Bays, R. H. Hudson, BIT 17 (1977) 121-127 (segmented sieve, via I1); I1's
`try_is_prime` (BPSW, then ECPP) for proofs. Name chosen to avoid MAGMA's `PartialFactorization` (a
different intrinsic, §18.10.4).

MAGMA semantics (digest §18.10 and §18.10.1) and how this engine relates to them (Revision 1, m3; Revision
2, m1). The module doc of `factorization.rs` states these four points:
- MAGMA's cascade is Cunningham lookup, trial division, perfect power, Pollard rho (Brent), SQUFOF, ECM,
  MPQS. This engine is trial division, perfect power, Brent rho, SIQS, with `ECMLimit := 0` fixed and no
  Cunningham table or SQUFOF. Leaving ECM out is a choice on measured cost, not an infeasibility: a seeded
  sigma (`try_one_curve` already takes a seed, `ecm.rs:316`) and a curve budget would be a few lines. But
  the crate's ECM (affine Weierstrass arithmetic with one modular inverse per group operation, a stage 2
  of 100 primes) takes 164 ms per curve in release at B1 = 2000 on the 41-digit
  p12 * P30 = 10000000000300000000000000031900000000957 (gp: [100000000003, 1; P30, 1]), and 2 of 80 curves
  found the 12-digit factor (`$X/rev2/ecm_rate.rs`; also 0 of 10 at B1 = 500 and at B1 = 2000, and 1 of 10
  at B1 = 11000 in debug, 5.3 s per curve, `$X/rev2/ecm_time.rs`). The sieve splits that whole number in
  about 0.08 s in release. So curves of this ECM would mostly cost time; an `ECMLimit` is worth offering
  only with a new ECM (section 8).
- `MPQSLimit`: MAGMA's default is infinity, and a value below 25 disables MPQS. Here `mpqs_limit: None` is
  infinity, and `Some(x)` with x < 25 disables the sieve, as in MAGMA. With x >= 25 the sieve takes the
  composites of at most x decimal digits. MAGMA sends composites below 25 digits to SQUFOF and ECM instead;
  this engine has neither, so the sieve takes those too (it is fast there: 6 ms at 20 digits in release).
- The third return value: MAGMA returns it only when both `ECMLimit` and `MPQSLimit` are set. `ECMLimit` is
  fixed at 0 here, so the rule becomes: `composites` can be non-empty only when `mpqs_limit` is `Some`. With
  `None` the engine does not stop until the factorization is complete, as MAGMA's default does.
- `Proof` (default true) as in MAGMA. `Bases` is not offered: the probable-prime test is BPSW (I1), not 20
  Miller-Rabin bases.

```rust
/// A factorization of n computed under limits: MAGMA's factorisation sequence, sign and sequence of
/// unfactored composites (ch. 18 §18.10.1), plus the probable primes whose proof was not obtained
/// (I1 interface, section 6).
///
/// Invariants: every base is > 1; the bases of all three lists together are pairwise coprime; each list
/// is sorted by base; for n != 0, sign * prod b^e over the three lists equals n; n = 0 gives sign 0 and
/// empty lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LimitedFactorization {
    /// -1, 0 or 1.
    pub sign: i8,
    /// (p, e): p proven prime (trial division, decided below psi_12, or a verified ECPP certificate).
    pub primes: Vec<(Integer, u32)>,
    /// (q, e): q passed BPSW (`is_probable_prime`), q >= psi_12, and q was not proven prime (`proof` was
    /// false, or the proof was unresolved).
    pub probable_primes: Vec<(Integer, u32)>,
    /// (m, e): m proven composite (`is_probable_prime(m)` is false, or the proof found m not prime) and not
    /// split by the enabled methods within their limits.
    pub composites: Vec<(Integer, u32)>,
}
impl LimitedFactorization {
    /// `probable_primes` and `composites` are both empty.
    pub fn is_complete(&self) -> bool;
    /// sign * prod b^e over the three lists (n itself; 0 for sign 0).
    pub fn value(&self) -> Integer;
}

/// MAGMA `Factorization(n : TrialDivisionLimit, PollardRhoLimit, MPQSLimit, Proof)` with `ECMLimit := 0`
/// (the module doc lists the four differences to MAGMA).
///
/// Cost and memory. Trial division: one word remainder per prime up to min(limit, isqrt(cofactor)), from
/// I1's segmented sieve, called once below 2^40 and once from 2^40 up (memory O(1) in the limit: 32 KiB
/// segments and base primes up to 2^20). Beyond 2^16 it stops early once the cofactor passes
/// `is_probable_prime`. Measured cost per prime in release on a 30-digit cofactor: 65 ns near 10^7,
/// 86 ns near 10^9, 113 ns near 10^10, 171 ns near 10^11 and 308 ns near 10^12 (the sieve's share grows
/// with the number of base primes), and about 6 µs above 2^40, where the sieve confirms each survivor
/// with `is_prime_u64`. A debug build is about 7 times slower (0.5 µs per prime near 10^7). So a limit
/// of 10^9 on a cofactor without small factors costs about 4 s in release, 10^12 about 2.5 hours, 2^40
/// about 2.7 hours, and each further 10^9 above 2^40 about 3.5 minutes. Rho: `pollard_rho_limit`
/// iterations per composite, memory O(size of n). The sieve: `QsParams::for_bits` of the composite:
/// memory 14 MB at 60 digits and 34 MB at 70 digits (measured), at most about 1.5 GB (the caps of
/// `QsParams`, reached from about 106 digits; estimate); time about 2.4 s at 60 digits and 39 s at
/// 70 digits in release, growing about 4.5 times per 5 digits. With `mpqs_limit: None` there is no time
/// limit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorizationLimits {
    /// Trial division by the primes up to this bound (MAGMA default 10000; 0 or 1: none). Any u64.
    pub trial_division_limit: u64,
    /// Brent-rho iterations per composite (MAGMA default 8191; 0: no rho).
    pub pollard_rho_limit: u64,
    /// None: the sieve runs on every composite left (MAGMA default infinity) and the result is complete.
    /// Some(x) with x < 25: no sieve (MAGMA: "< 25 disables MPQS"). Some(x) with x >= 25: the sieve runs on
    /// composites of at most x decimal digits.
    pub mpqs_limit: Option<u32>,
    /// Prove every probable prime >= psi_12 (MAGMA `Proof`, default true).
    pub proof: bool,
}
impl Default for FactorizationLimits { /* 10000, 8191, None, true */ }

pub fn factorization_with_limits(n: &Integer, limits: &FactorizationLimits) -> LimitedFactorization;
```

In `prime.rs`:

```rust
/// MAGMA `PollardRho(n, c, s, k : Proof)` (ch. 18 §18.10.3; [Bre80]): every composite piece of |n| gets at
/// most k iterations of Brent's rho with x^2 + c from x_0 = s; when a run ends with gcd = piece (all cycles
/// closed together), the next run uses c + 1, c + 2, ... from s with the remaining iterations. Perfect
/// powers and the factor 2 are split off first. MAGMA's defaults are c = 1, s = 1, k = 8191
/// (`POLLARD_RHO_DEFAULT_ITERATIONS`), proof = true. With `proof` false, factors >= psi_12 that pass BPSW go
/// to `probable_primes`. No trial division, no sieve.
pub fn factor_pollard_rho_limited(n: &Integer, c: &Integer, s: &Integer, k: u64, proof: bool)
    -> LimitedFactorization;
pub const POLLARD_RHO_DEFAULT_ITERATIONS: u64 = 8191;
```

Crate-internal (in `factorization.rs`):

```rust
pub(crate) enum Class { Prime, ProbablePrime, Composite }
/// c > 1. is_probable_prime(c) false -> Composite; true and c < psi_12 -> Prime (exact there); else if
/// proof: try_is_prime(c) (I1; BPSW, the cache of proven primes, then ECPP): Ok(true) -> Prime,
/// Ok(false) -> Composite, Err (unresolved) -> ProbablePrime; else ProbablePrime.
pub(crate) fn classify(c: &Integer, proof: bool) -> Class;

/// Which composites the sieve may take.
pub(crate) enum QsPolicy { Never, UpToDigits(u32), Always }
pub(crate) struct SplitPlan { pub rho_c: Integer, pub rho_x0: Integer, pub rho_steps: u64,
                              pub qs: QsPolicy, pub complete: bool }
/// Pieces of the composite c > 1, or None (only when !plan.complete). Some(pieces): c = prod b^k over the
/// pieces, every 1 < b < c. In this order: c even -> [(2, v), (c / 2^v, 1)] (the second dropped if 1);
/// c.is_perfect_power() = Some((b, j)) -> [(b, j)]; prime::rho_rounds(c, rho_c, rho_x0, rho_steps) = Some(d)
/// -> [(d, 1), (c / d, 1)]; the sieve if `qs` allows c: quadratic_sieve_factor_with_params(c,
/// for_bits(c.bits()) with seeds 1, 2, 3) until Factor(d) -> [(d, 1), (c / d, 1)]; if complete:
/// [(prime::split_odd_composite(c), 1), (c / that, 1)] (always succeeds); else None.
pub(crate) fn split_composite(c: &Integer, plan: &SplitPlan) -> Option<Vec<(Integer, u32)>>;

/// True iff 0 < c < 10^d (Revision 2, m3: c <= 0 gives false), without forming 10^d unless needed:
/// c <= 0 -> false; bits(c) <= 3d -> true (0 < c < 8^d <= 10^d); bits(c) > 4d -> false
/// (c >= 2^(4d) = 16^d > 10^d; for d = 0 every c > 0 has bits > 0); otherwise compare with 10^d exactly
/// (then 10^d has about as many bits as c). 3d and 4d in u64.
pub(crate) fn has_at_most_digits(c: &Integer, d: u32) -> bool;

/// Factor refinement [BDS93]: while two bases share g = gcd > 1, replace (a^e, b^f) by
/// ((a/g)^e, (b/g)^f, g^(e+f)) (merge when a = b), dropping 1s. The product is preserved; the sum of
/// log2 of the bases drops by at least 1 per step, so it ends after at most (total bits) steps.
pub(crate) fn coprime_refine(pieces: Vec<(Integer, u32)>) -> Vec<(Integer, u32)>;

/// Step 2 of `limited_engine` (Revision 2, m6): trial division of m by the primes p with lo <= p <= limit
/// and p <= isqrt(m), ascending, through two `sieve::for_each_prime` calls split at 2^40. Divides each
/// prime found out of m with its exact exponent, and stops early as step 2 says. Returns the primes
/// found and the number of primes passed to the closure (for the unit test).
pub(crate) fn trial_division_stage(m: &mut Integer, lo: u64, limit: u64) -> (Vec<(Integer, u32)>, u64);

/// The engine behind the public drivers.
pub(crate) fn limited_engine(n: &Integer, trial_division_limit: u64, plan: &SplitPlan, proof: bool)
    -> LimitedFactorization;
```

`limited_engine` (Revision 1, B2 and m4):
1. n = 0: sign 0, empty. Else sign = signum(n), m = |n|.
2. Trial division (`trial_division_stage(&mut m, 2, trial_division_limit)`), only if
   `trial_division_limit >= 2` and m > 1:
   - bound = min(limit, isqrt_sat(m)), where isqrt_sat(m) = `m.sqrt()` converted to u64, saturating at
     u64::MAX (so m >= 2^128 gives u64::MAX). No `limit + 1` and no square of the limit is formed.
   - Two calls of `sieve::for_each_prime` with the same closure (I1 §2.5; segmented, 32 KiB segments,
     memory O(1) in the bound). First over [lo, min(bound + 1, 2^40)) (lo = 2 in the engine; skipped if
     lo >= 2^40); then, only if bound >= 2^40 and the first call ended without Break (`for_each_prime`
     returns `()`, so the closure sets a flag when it breaks), over [max(lo, 2^40), bound.saturating_add(1))
     (at bound = u64::MAX the end excludes only 2^64 - 1, which is composite). Why two calls (Revision 2,
     m6): a call with hi <= 2^40 sieves with base primes up to isqrt(hi - 1) < 2^20, so every survivor is
     prime and I1 does no survivor test; only the second call has survivors that I1 confirms with
     `is_prime_u64` (each above 2^40). So the cost below 2^40 is the same whether I1 confirms all
     survivors of a call with hi > 2^40 or only those above 2^40.
     For each p: if p > bound, Break. If p > 2^16 and m has changed since the last probable-prime test (or
     was never tested): Break if `is_probable_prime(m)`. If m mod p = 0 (a word remainder,
     `m.as_bigint().magnitude() % p`): divide out p^e, record (p, e, Prime), set
     bound = min(bound, isqrt_sat(m)), and Break if m = 1.
   - Then, if m > 1, push (m, 1) to the queue.
   Correctness: each recorded p is prime (it comes from the sieve) with its exact exponent. The stage stops
   only when every prime up to the bound was tried (so m has no prime factor <= min(limit, sqrt(m))), or
   m = 1, or m passed `is_probable_prime`. In the last case m is prime (exact below psi_12; BPSW above, no
   counterexample known); if it is a BPSW pseudoprime after all, `classify` in step 3 finds that with a proof
   (`Composite`, then split), or reports m in `probable_primes` without one. No prime is ever claimed
   without a proof. Cost: at most pi(bound) word remainders (plus one `is_prime_u64` per sieve survivor
   above 2^40), and at most 1 + (number of distinct primes above 2^16 found) BPSW tests; none when
   bound <= 2^16. The default limit 10000 costs 1229 remainders and
   no BPSW test, as at HEAD; the limit 47 of the complete QS driver costs 15 remainders. A huge limit on
   2 * P30 costs the 6542 primes below 2^16 (about 3 ms in debug) and one BPSW test.
3. Queue of (c, e). Pop: `classify(c, proof)`; Prime / ProbablePrime: record (c, e, class). Composite:
   `split_composite(c, plan)`; Some(pieces): push (b, k e) for each (b, k); None: record (c, e, Composite).
   The product of the recorded b^e times the queue's c^e is always m, and every split replaces c by bases
   below c, so the loop ends (the multiset of queued bases decreases in the multiset order). A prime power
   costs one perfect-power test (3^2000: 0.32 ms in debug, `$X/rev1/pp2.rs`), not one split per exponent.
4. `coprime_refine` over all recorded pieces. Pieces that equal a recorded base keep their class; new
   pieces (quotients and gcds of recorded composites) are classified with `classify(_, proof)` and are not
   split again (a composite piece of a refinement is a divisor of an unsplit composite; it is reported as
   a composite). Perfect powers among composite pieces: (b^j, e) becomes (b, j e) and b is classified.
5. Sort the three lists by base.

The public drivers:
- `factorization_with_limits(n, l)` = `limited_engine(n, l.trial_division_limit, &SplitPlan { rho_c: 1,
  rho_x0: 1, rho_steps: l.pollard_rho_limit, qs, complete: l.mpqs_limit.is_none() }, l.proof)` with
  qs = `Always` for None, `Never` for Some(x < 25), `UpToDigits(x)` for Some(x >= 25). With
  `mpqs_limit = None` the result has no `composites` (the third value appears only when the limit is set);
  if the sieve fails three times on a composite the complete rho fallback finishes it.
- `factor_pollard_rho_limited(n, c, s, k, proof)` = `limited_engine(n, 0, &SplitPlan { rho_c: c, rho_x0: s,
  rho_steps: k, qs: Never, complete: false }, proof)`.

Changes to `prime.rs` internals:
- `brent_rho(m, c, x0, max_steps) -> (Option<Integer>, u64)`: start at x0 (HEAD: 2); exact budget: the
  doubling round advances min(r, budget - steps) and each block min(128, r - k, budget - steps), so at most
  `max_steps` new sequence terms are computed (the replay after a block gcd of m recomputes at most 128
  terms of that block); returns the steps used. HEAD's version can overshoot by 2x (`prime.rs:761-762`).
- `pub(crate) fn rho_rounds(m, c, x0, k) -> Option<Integer>`: runs `brent_rho` with c, c + 1, ... sharing k.
- `split_brent_and_trial_division` calls `brent_rho(m, c, 2, budget).0`; the `# Cost` text of
  `factor_pollard_rho` (`:753-763`) loses the "overshoot by 2x" clause.
- `split_odd_composite` becomes `pub(crate)`.

Measured (`$X/rho_check.rs`, `$X/rho_check2.rs`, release): `rho_rounds` from c = 1, x0 = 1 with k = 8191
splits every odd composite below 2 * 10^5 that is not a perfect power (81761 inputs, at most 156 steps);
a single run with c = 1 fails on 15 and 25 (gcd = m), which is why rounds are used. Single runs:
(10^9+7)(10^9+9) needs 50430 steps (c = 1, x0 = 1), 2^64 + 1 gives 274177 after 1918, 101 * 103 * P30 gives
10403 after 30 (a composite divisor, split again in its own queue entry), psi_12 needs 450558.

### 3.3 Correctness argument

- Product: invariant of steps 2 and 3; refinement preserves the product; so `value() == n`.
- Lists: `primes` entries are proven (sieve primes of trial division, exactness of `is_probable_prime`
  below psi_12, or `try_is_prime` = Ok(true), whose ECPP certificate I1 verifies); `composites` entries are
  proven composite (a BPSW failure or `try_is_prime` = Ok(false) rests on a witness); `probable_primes` are
  exactly the rest.
- Coprimality and sortedness: by `coprime_refine` and the final sort.
- Termination: step 2 visits at most pi(bound) primes; step 3 ends as stated; each method of
  `split_composite` is bounded (rho by k, the sieve by the termination argument of 1.2), and the complete
  fallback ends because the windows reach the least prime factor (`prime.rs:885-887`).
- Memory: trial division O(1) in the limit; the sieve within the caps of `QsParams` (1.4).

### 3.4 The complete QS driver

`quadratic_sieve_factor_complete(n)` keeps its signature and contract (primes ascending, |n|, empty for 0
and +-1). It becomes `limited_engine(n, 47, &SplitPlan { rho_c: 1, rho_x0: 1, rho_steps: 8191, qs: Always,
complete: true }, true)`: trial division by the primes <= 47 as at HEAD (review 1, m4), then rho, the sieve
and the complete fallback. `composites` is empty (complete). If `probable_primes` is not empty, a factor's
ECPP proof was unresolved; the driver panics, as I1's `is_prime` does for the other two complete drivers,
with "quadratic_sieve_factor_complete: the factor {q} passed BPSW but ECPP found no proof; use
factorization::factorization_with_limits, which reports it in probable_primes". Otherwise it returns
`primes`. A BPSW pseudoprime (none known) is found by `try_is_prime` and split. Docs: "The sieve's cost
depends on the size of the composite, not of its factors. Rho with 8191 iterations runs first and usually
removes prime factors below about 10^7 at once", plus the table of 1.3 and the panic sentence.

Behaviour change: the driver now finishes for 40-digit semiprimes (HEAD: rho, no practical end) and splits
psi_12 through the sieve in milliseconds (Revision 1 prototype, debug: 69 ms; I1's planned test
`psi12_is_split_by_every_driver` then runs fast for QS). Check of the engine: `$X/rev1/driver_r1.rs` is the
Revision 1 engine (segmented trial division with the early stop, pieces with exponents, `QsPolicy`,
`has_at_most_digits`, the Revision 1 sieve; HEAD's `is_prime` stands in for `is_probable_prime`, so psi_12
comes out as a probable prime there). In debug (`$X/rev1/driver_r1_dbg.txt`): the 22 PARI rows of
`tests/ch18_factorization_drivers.rs` with trial division and rho off (the sieve, the perfect-power test and
the factor 2 do every split): 0 mismatches. Every expected value of `limits_*` and of the prime-power tests in
3.5 comes out as stated: 2 * P30 with limit u64::MAX or 10^12 tries 6542 primes and one probable-prime test
(81 ms in debug, most of it the 240 Miller-Rabin rounds of the stand-in); 100 tries 3 primes; the `mpqs_limit`
rows split 101 * p20 * q20 only for Some(39) and None (0.42 s); Some(24) leaves (10^9+7)(10^9+9) unsplit and
Some(25) splits it; 3^2000 takes 34 ms with limit 47 and 0.30 s with no trial division;
`has_at_most_digits(2^4000, u32::MAX)` takes 18 µs.

### 3.5 Tests (item 3)

New integration file `rustmath-integers/tests/ch18_limited_factorization.rs` (header: MAGMA ch. 18 §18.10.1
and §18.10.3; [Bre80]; [BDS93]). Every test also checks the invariants of `LimitedFactorization` with a
helper (`value() == n`, pairwise coprime, sorted, entries of `primes` pass `is_prime`, entries of
`composites` fail `is_probable_prime`).

Values (gp `$X/refs.gp`, `$X/rev1/refs_r1.gp`): P30 = nextprime(10^29) = 100000000000000000000000000319;
p15 = nextprime(10^14) = 100000000000031; q15 = nextprime(2*10^14) = 200000000000027; p20 = nextprime(10^19)
= 10000000000000000051; q20 = nextprime(3*10^19) = 30000000000000000041; q12 = nextprime(10^12) =
1000000000039.

| test | call | expected | gp | HEAD |
|---|---|---|---|---|
| `rho_limited_leaves_large_composite` | `factor_pollard_rho_limited(p15*q15 = 20000000000008900000000000837, 1, 1, 8191, true)` | composites [(n, 1)], others empty | `factor(20000000000008900000000000837)` | F (no API) |
| `rho_limited_splits_small_factors` | 101*103*P30 = 1040300000000000000000000003318557, (1, 1, 8191, true) | primes [(101,1), (103,1), (P30,1)] (P30 proven by I1's ECPP) | `factor(1040300000000000000000000003318557)` | F |
| `rho_limited_proof_flag` (Revision 1, m11) | the same n, (1, 1, 8191, false) | primes [(101,1), (103,1)], probable_primes [(P30,1)] | `isprime(P30)` = 1 | F |
| `rho_limited_budget` | (10^9+7)(10^9+9) = 1000000016000000063 with k = 8191 and with k = 2^20 | composites [(n,1)]; primes [(1000000007,1), (1000000009,1)] | `factor(1000000016000000063)` | F |
| `rho_limited_powers_signs_and_units` | -42525; q12^2 = 1000000000078000000001521; 2^64+1; 0; 1; -1 | sign -1, primes [(3,5),(5,2),(7,1)]; primes [(q12, 2)]; primes [(274177,1),(67280421310721,1)]; sign 0 / 1 / -1 and empty | `factor(42525)`, `factor(q12^2)`, `factor(2^64+1)` | F |
| `rho_limited_prime_powers` (Revision 1, m4) | 3^2000; 2^1000 * 3^500; 1000003^50, all with (1, 1, 8191, true), inside `within(5, ..)` | primes [(3, 2000)]; [(2, 1000), (3, 500)]; [(1000003, 50)] | gp `factor(3^2000)`, `factor(2^1000*3^500)`, `factor(1000003^50)` | F |
| `rho_limited_psi12` (needs I1's BPSW) | psi_12, k = 8191 and k = 2^22 | composites [(psi_12,1)]; primes [(399165290221,1),(798330580441,1)] | `factor(318665857834031151167461)` | F (HEAD's `is_prime` passes psi_12) |
| `limits_default_is_complete` | 101*p20*q20 = 30300000000000000195940000000000000211191, `Default` | primes [(101,1),(p20,1),(q20,1)], complete | `factor(30300000000000000195940000000000000211191)` | F |
| `limits_mpqs_limit` | same n with mpqs_limit Some(0), Some(24), Some(25), Some(38), Some(39) | the first four: primes [(101,1)], composites [(300000000000000001940000000000000002091,1)]; Some(39): complete | `#Str(p20*q20)` = 39 | F |
| `limits_mpqs_limit_below_25_disables_sieve` (Revision 1, m3) | (10^9+7)(10^9+9) (19 digits) with (10000, 0, Some(24), true) and with Some(25) | composites [(n,1)]; primes [(1000000007,1), (1000000009,1)] | `#Str(1000000016000000063)` = 19 | F |
| `limits_trial_division_respects_limit` (Revision 1, m9) | 8 * 9973 * 10007 * p15 = 79839848800024750353128 with (10000, 0, Some(0), true) | primes [(2,3),(9973,1)], composites [(1000700000000310217,1)] (10007 > 10000 stays in the cofactor) | `factor(79839848800024750353128)` = [2, 3; 9973, 1; 10007, 1; 100000000000031, 1]; `precprime(10000)` = 9973 | F |
| `limits_trial_division_huge_limit` (Revision 1, B2) | trial_division_limit u64::MAX and 10^12 (rho 8191, mpqs None, proof true) on 100, on 2 * P30 = 200000000000000000000000000638, on -1 and 0; inside `within(5, ..)` | primes [(2,2),(5,2)]; primes [(2,1),(P30,1)]; sign -1 / 0, empty | `factor(100)`, `isprime(P30)` | N (no API; the first design allocated 10^10 bytes) |
| `limits_proof_flag` | 2 * P30 with proof false and true (Default otherwise) | probable_primes [(P30,1)] and primes [(2,1)] / primes [(2,1),(P30,1)] | `isprime(P30)` = 1 | F |
| `limits_two_25_digit_primes` | p25*q25 = 5000000000000000000000076000000000000000000000287, `Default` | primes [(1000000000000000000000007,1),(5000000000000000000000041,1)] | `factor(...)` | F |
| `coprime_refine_values` (unit, `factorization.rs`) | [6, 10]; [1380, 675, 3408, 654]; [(12,1),(12,2)] | [(2,2),(3,1),(5,1)]; [(2,7),(3,6),(5,3),(23,1),(71,1),(109,1)]; [(12,3)] | `factor(60)`; `factor(1380*675*3408*654)` | N |
| `coprime_refine_random` (unit) | 200 seeded random lists of 2..6 products of small primes | pairwise coprime, product preserved | | N |
| `trial_division_crosses_2_40` (unit, Revision 2, m6) | `trial_division_stage(&mut m, 2^40 - 2^16, 2^40 + 2^16)` with m = precprime(2^40) * nextprime(2^40) * P30 = 120892581953546433750499900385647336431813123664094681 | found [(1099511627689, 1), (1099511627791, 1)], m = P30 afterwards (early stop on the probable prime), 2365 primes visited: the 2363 primes of [2^40 - 2^16, 2^40) (first call, no survivor test), then 2^40 + 15 and the next prime, where the early stop fires (prototype `$X/rev2/cross40.rs`: 12 ms release, 60 ms debug) | gp `precprime(2^40)`, `nextprime(2^40)`, `factor(n)`; `c=0; forprime(p=2^40-2^16, 2^40, c++); c` = 2363 | N |
| `has_at_most_digits_edges` (unit, Revision 1, B2; Revision 2, m3) | `has_at_most_digits(c, d)` for the pairs (c, d) on the right | (0, 5) false; (-7, 5) false; (1, 0) false; (9, 1) true; (10, 1) false; (10^24 - 1, 24) true; (10^24, 24) false; (2^80, 24) false and (2^80, 25) true (2^80 has 25 digits); (huge 2^4000, u32::MAX) true without forming 10^u32::MAX (returns in microseconds) | gp `#Str(2^80)` = 25 | N |
| `split_composite_plans` (unit) | plan {rho 0, qs Never, complete true} on 3127, 165731, 1000000016000000063 (the complete fallback: Floyd rho, then Brent rounds with trial-division windows); plan {rho 8191, qs Always, complete true} on the 40-digit row (the sieve stage); plan {rho 0, qs Never, complete false} on p15*q15; any plan on 3^40 and on 2^10 * 3127 | pieces whose product is c, each base in (1, c); None for p15*q15; [(3, 40)]; [(2, 10), (3127, 1)] | gp `factor` | N (replaces the pinned `quadratic_sieve.rs:775-797` tests; psi_12 is left out because Floyd needs 188311 steps, 4.6 s in debug per I1) |
| `engine_requeues_composite_divisors` (unit) | `limited_engine` with plan {rho 0, qs Always, complete true}, no trial division, on 1000000007 * 2000000011 * 3000000019 and on 100000007^2 * 10000000000000061 | primes equal gp `factor` (whatever the sieve's divisors are, composite ones go back to the queue) | gp `factor(6000000113000000706000001463)`, `factor(100000014000001100000085400002989)` | N |
| `brent_rho_budget` (unit, `prime.rs`) | p15*q15, c = 1, x0 = 1, 8191; (10^9+7)(10^9+9), c = 2, x0 = 2, 2^20 and 1 | (None, 8191); Some(1000000009 or 1000000007), and (None, 1) | measured 31870 steps | changes `test_brent_rho` (`prime.rs:1393-1402`) |

Debug cost of this file: the sieve on 39 and 49 digits (about 0.4 s and 3 s) plus I1 proofs of 25- and
30-digit primes; about 5 s in all.

## 4. Item 4: the QS `legendre_symbol`

Defect: `quadratic_sieve.rs:45-68`; `:57-58` maps a `mod_pow` error to 1 ("residue"); `mod_pow` errs only
for a zero modulus, so this is unreachable for p >= 3, but it is an error swallowed as a decision. Also
`legendre_symbol(4, 2)` returns 1 (`:46-48`) although the Kronecker symbol (4/2) is 0 (gp
`kronecker(4, 2)` = 0); the sieve never asked.

Fix: delete `legendre_symbol` and the `Integer`-based `tonelli_shanks`. In `siqs.rs`:

```rust
/// Square root of a modulo the odd prime p < 2^63: Some(r) with r^2 = a mod p, 0 <= r < p, or None
/// exactly when a is a non-residue (Euler's criterion, exact for a prime). Tonelli-Shanks [Coh93, 1.5.1]
/// on u64 words with u128 products. Panics with "sqrt_mod_u64: p = {p} is not an odd prime below 2^63"
/// if p is not one (`is_prime_u64`).
fn sqrt_mod_u64(a: u64, p: u64) -> Option<u64>;
```

The factor base calls Euler's criterion r^((p-1)/2) mod p on words; there is no error value anywhere. Tests:
the three ported `sqrt_mod_u64_*` tests and `factor_base_matches_kronecker` (section 1.6). The tracker row
`PROJECT_TRACKER.md:407` loses its `legendre_symbol` sentence.

## 5. Item 5: the ECM module doc

`ecm.rs:14-15` reads "The algorithm is particularly effective when n has a factor p where p-1 or p+1 is
smooth". That describes Pollard's p - 1 and Williams' p + 1 methods. What the module actually tests
(Revision 1, m5): stage 1 multiplies the point by k = the product of the largest prime powers <= B1
(`compute_stage1_multiplier`, `ecm.rs:299-314`, the `power <= b1 / p` loop at `:304`); stage 2 multiplies
the stage-1 point by each of the first 100 primes in (B1, B2], one at a time (`ecm.rs:383-399`, `.take(100)`
at `:389`). Replace lines 14-15 with:

```
//! ECM (H. W. Lenstra, *Factoring integers with elliptic curves*, Ann. of Math. 126 (1987) 649-673)
//! works in the group E(F_p) of a random curve, for an unknown prime p | n. The order of that group lies
//! in [p + 1 - 2 sqrt(p), p + 1 + 2 sqrt(p)] (Hasse) and changes with the curve, so every new curve is a
//! new chance for every p. The expected cost depends on the size of p, about L_p[1/2, sqrt 2]. Pollard's
//! p - 1 and Williams' p + 1 methods are the analogues with one fixed group each, so they need p - 1 or
//! p + 1 itself to be smooth.
//!
//! What this module tests on one curve with point P: stage 1 computes k P with k the product of the
//! largest prime powers <= B1, so it finds p when the order of P in E(F_p) divides k (the order is
//! B1-powersmooth). Stage 2 then computes q (k P) for each of the first 100 primes q in (B1, B2], one at
//! a time, so it also finds p when that order is q times a divisor of k for one of those q. Primes of
//! (B1, B2] beyond the first 100 are not tried.
//!
//! MAGMA: `ECM(n, B1)` (Handbook ch. 18 §18.10.3, digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`) uses GMP-ECM with Suyama's parametrisation and a
//! full stage 2; this module uses affine Weierstrass arithmetic, random curves and the short stage 2
//! above.
```

and the comment at `ecm.rs:322-323` ("Montgomery parameterization: use σ to generate curve / This is a standard
technique to ensure we have a valid point") with: "Suyama's u = sigma^2 - 5 and v = 4 sigma only seed a
random point (u/v, 1) and the coefficient a below; b is then fitted so that the point lies on y^2 = x^3 + a x
+ b. This is not Suyama's curve (no torsion of order 12 is forced); any curve with a known point is a valid
ECM curve." No code change, no test change (doc only). Tracker row `PROJECT_TRACKER.md:402` notes the doc fix.

## 6. Interface with I1 (and I3)

I1's design (`../I1-integers-primality.md` §2.1, §2.4, §2.5 and §2.8, Revision 1) provides, and this design
uses:

| I2 needs | I1 item | used in |
|---|---|---|
| `prime::is_probable_prime(&Integer) -> bool`: exact below psi_12, BPSW above; `false` always a proof | §2.1, §2.2 | every splitting decision (`classify`, the trial-division early stop, `quadratic_sieve_factor`'s early `None`, `trial_division_with_primes`) |
| `prime::try_is_prime(&Integer) -> Result<bool>`: a proof with a cache of proven primes; `Err(NotSupported)` when ECPP is unresolved | §2.1, §2.4 | `classify(_, proof = true)` for c >= psi_12 |
| `prime::is_prime`: a proof, panics on Unresolved | §2.1 | the invariant helper of the tests |
| `sieve::for_each_prime(lo, hi, f)` (segmented, overflow-free near 2^64) and `sieve::primes_up_to(b)` | §2.5 | trial division in `limited_engine`; the factor base scan (1.2 step 2); the 300 multiplier primes (1.2 step 1) |
| (Revision 2, m6) a `for_each_prime` call with hi <= 2^40 does no per-survivor primality test (base primes up to isqrt(hi - 1) < 2^20 leave only primes), and a call with hi > 2^40 delivers exactly the primes of [lo, hi) (survivors confirmed by `is_prime_u64`, as I1 §2.5 says) | §2.5 ("if isqrt(hi - 1) > 2^20, the survivors are confirmed") | `trial_division_stage` splits its range at 2^40, so I2 needs nothing more: below 2^40 the cost per prime is the sieve's (65 to 310 ns in release, 3.2); above 2^40 every survivor is above 2^40 and costs one `is_prime_u64` (about 6 µs in release with HEAD's 12 bases, `$X/rev2/td40.rs`). If I1 confirms only survivors >= 2^40 the split changes nothing |
| `is_prime_u64` (public in `primality.rs`) | §2.1 | `sqrt_mod_u64`'s precondition check, tests |
| I1's edits to `quadratic_sieve.rs` (`:450` early return, `:519-523` docs, `:572` guard) | §2.8 | superseded by the rewrite; keep their wording for the "proven prime" sentence |

I1's rule "do not call `is_prime` on a value that is only used for a splitting decision" holds: the
splitting decisions use `is_probable_prime`; `try_is_prime` runs only on values that will be returned as
primes.

Panic policy (Revision 1, m7). I1 §2.4 says the panic of `is_prime` is left "to the complete factorization
drivers (I2 gives them a `Result`)", and I1 §2.8 leaves the choice to I2. Decision: `factor_pollard_rho`,
`ecm_factor_complete` and `quadratic_sieve_factor_complete` keep their `Vec` return type and panic on an
unresolved proof (I1 measured no unresolved prime up to 2048 bits). The non-panicking complete
factorization is `factorization_with_limits(n, &FactorizationLimits::default())`: it never panics and
reports an unproven factor in `probable_primes`. The docs of all three `Vec` drivers name it (one sentence
each; this replaces I1's "I2 gives them a `Result`" in their docs). No `try_` variants are added: the
`LimitedFactorization` result already carries the undecided case.

If I1 lands under other names, map them; if it lands without `try_is_prime`/`prove_primality`,
`classify(_, true)` treats every probable prime >= psi_12 as `ProbablePrime` and the docs say so. If it
lands without `is_probable_prime`, use `is_prime_miller_rabin(n, 12)` (12 fixed bases; `false` is still a
proof). If it lands without `sieve::for_each_prime`, trial division iterates windows
`fast_arith::prime_range(lo, lo + 2^16)` (HEAD's version allocates only the window, and costs
O(2^16 + sqrt(lo + 2^16)) per window).

I1's test `psi12_is_split_by_every_driver` (`tests/ch18_factorization_drivers.rs`): I1 may have moved its QS
case to an `#[ignore]` test because Floyd rho takes about 5 s there. After this unit QS splits psi_12 through
the sieve in about 0.1 s in debug, so plan step 6 removes that `#[ignore]` (or puts the QS case back into
the main test).

I3 (`../I3-integers-arith.md`): `Integer::sqrt` and `is_perfect_power` get faster with the same values; no
interface change for I2. The rewrite does not use `prime.rs`'s private `isqrt` that I3 deletes. I3 edits
`quadratic_sieve.rs:50`, `:60`, `:93`, `:96`, `:112`, `:145` (the `to_i64` migrations of its Appendix A) and
`:228`, `:376` (`Integer::sqrt` callers). All of them are inside `legendre_symbol`, `tonelli_shanks`,
`sieve_for_smooth_relations` and `try_dependency`, which this rewrite deletes. The rewrite supersedes those
edits; nothing of them needs to survive, and the new code has no `to_i64` on an `Integer` and no floor
square root.

## 7. Plan

Implementation order (one PR, crate `rustmath-integers`; I3 and I1 already merged). Revision 2 (m4): the
order is chosen so that each step only uses items of earlier steps; after each step
`cargo test -p rustmath-integers` builds and passes, so each step can be its own commit.

1. `src/prime.rs` internals: `brent_rho` with x0, exact budget, steps returned; `rho_rounds`;
   `split_odd_composite` `pub(crate)`; `POLLARD_RHO_DEFAULT_ITERATIONS`; the `# Cost` text of
   `factor_pollard_rho` (overshoot clause); `test_brent_rho` -> `brent_rho_budget`. The only caller of
   `brent_rho`, `split_brent_and_trial_division` (`prime.rs:892-897`), is updated in the same step.
2. `src/quadratic_sieve.rs` rewritten with `src/quadratic_sieve/siqs.rs` and `src/quadratic_sieve/gf2.rs`
   (port of `$X/rev2/siqs_r3.rs`, section 1.2, with step 0 added); `QsParams` with its constants,
   `validate` and `for_bits`; `QsOutcome`; `quadratic_sieve_factor` and
   `quadratic_sieve_factor_with_params`; the unit tests of 1.6 (with `RunStats` and the cap test) and the
   HEAD tests as in the fate table; doc examples. `lib.rs` re-exports `QsParams, QsOutcome`. In this step
   `quadratic_sieve_factor_complete` keeps HEAD's body (`quadratic_sieve.rs:549-610`): it calls
   `quadratic_sieve_factor(&Integer) -> Option<Integer>` (same signature, now the new sieve) and
   `factor_pollard_rho`, so it builds and stays correct (composite divisors go through its recursion);
   the fate-table tests of the complete driver pass with either body.
3. `src/factorization.rs` (new): `LimitedFactorization`, `FactorizationLimits`, `factorization_with_limits`,
   `Class`, `classify`, `QsPolicy`, `SplitPlan`, `split_composite` (its sieve stage calls
   `quadratic_sieve_factor_with_params` from step 2), `has_at_most_digits`, `coprime_refine`,
   `trial_division_stage`, `limited_engine`, unit tests. `lib.rs`: `pub mod factorization;` and
   `pub use factorization::{ factorization_with_limits, FactorizationLimits, LimitedFactorization };`.
   In the same step: `prime::factor_pollard_rho_limited(n, c, s, k, proof)` (it is `limited_engine` with a
   plan) and the panic sentence naming `factorization_with_limits` in the doc of `factor_pollard_rho`;
   `quadratic_sieve_factor_complete` switches to `limited_engine` (3.4) with its new docs.
4. `src/factorint.rs`: `trial_division_with_primes` returns `LimitedFactorization` (uses `classify` from
   step 3); tests replaced; module doc line for MAGMA `TrialDivision`.
5. `src/ecm.rs`: doc lines 14-15, the comment at 322-323, and the panic sentence in the doc of
   `ecm_factor_complete`.
6. `tests/ch18_quadratic_sieve.rs`, `tests/ch18_limited_factorization.rs` (new);
   `tests/ch18_factorization_drivers.rs` unchanged except that it keeps passing, and the QS case of I1's
   `psi12_is_split_by_every_driver` is re-enabled if I1 had moved it to `#[ignore]`.
7. Docs and trackers: `PROJECT_TRACKER.md` rows `:395` (🔴 -> 🟢), `:399` (🛑 -> 🟢), `:402` (doc note),
   `:406` and `:407` (⏳ -> 🟢), `:415` (rho driver row: mention the capped variant) and new rows for
   `factor_pollard_rho_limited` and `factorization_with_limits`; card T-223 (`:254`) and blocker B-04
   (`:3100`) text; `docs/magma/status_overrides.csv` rows `ch018.MPQS`, `ch018.PollardRho`,
   `ch018.TrialDivision` (including the m10 note on `factor_trial_division`), `ch018.Factorization` (the
   `ECMLimit := 0` and `MPQSLimit` notes of 3.2), then `python3 tools/magma_tracker/build.py`;
   `docs/HANDOFF.md:275-282` (remove the five items, keep the I1 sentence if I1 left it, add the three
   hand-offs of section 8).

Acceptance commands:

```
cargo fmt -p rustmath-integers -- --check
cargo clippy -p rustmath-integers --tests
cargo test -p rustmath-integers --no-fail-fast
cargo test --release -p rustmath-integers --test ch18_quadratic_sieve -- --ignored
cargo test -p rustmath-numbertheory --no-fail-fast      # re-exports prime::*
cargo check --workspace --tests
grep -rn "unsafe" rustmath-integers/src                  # no new hits
```

Timing acceptance (debug, `cargo test -p rustmath-integers`): the QS tests below 8 s of CPU, the limited
factorization file below 6 s; release ignored test below 5 s. The ported SIQS should stay within 1.5x of the
prototype table in 1.3; if it is slower, profile the candidate loop first (in the prototype it is about
40 % of the time outside the sieve).

Risks:
- Parameter fit: tuned on one machine and one family of semiprimes. The robustness runs (2026 inputs, 6
  seeds) and the soundness argument mean a bad parameter only costs time; keep the table and the L(n)
  formula as written so the measured times stay comparable.
- De-duplication must stay in `RelationStore::add` before the partial pairing; without it the failure rate
  is no longer about 2^-32 (1.2 step 7). `relation_store_drops_repeated_y` guards it.
- Debug-time growth: the 60-digit case is `#[ignore]`; keep default tests at <= 45 digits.
- I1 timing: proofs of 25- to 30-digit primes in the limited-factorization tests depend on I1's ECPP
  speed (I1 measured 2^89 - 1 at 12 ms in debug).
- `classify` below psi_12 relies on `is_probable_prime` being exact there (I1 §2.1 states it).
- The byte-threshold trick limits exact thresholds to about 85 digits (clamped above); correctness is not
  affected.
- Large explicit limits are honoured: a `trial_division_limit` of 10^12 on a composite cofactor without
  small factors costs about 2.5 hours in release, and F = 65536 takes 1.08 GB and about 12 minutes of
  elimination. Both are documented on the parameters; the defaults stay far below.
- The polynomial budget must be checked inside the Gray-code loop (1.2 step 4, Revision 2 B1).
  `qs_polynomial_budget_is_per_polynomial` hangs (and `within` fails it) if the check sits only at the
  top of the a-loop, and `run_stops_at_the_polynomial_budget` counts the polynomials exactly.
- The first-partial cap (1.2 step 7) never binds in the measured runs; if a port drops partials in normal
  runs (`RunStats::dropped_partials > 0` on the grid rows), the cap test is in the wrong place.
- The `brent_rho` budget change touches the complete rho fallback; its tests (`prime.rs:1365-1425`) and the
  22 PARI rows of `tests/ch18_factorization_drivers.rs` cover it.

## 8. Hand-offs

- `factorint::factor_trial_division(n, None)` (`factorint.rs:190-192`): above 2^128 it stops at 10^6 and
  appends a composite cofactor, against its doc (review 1, m10). Recorded in the `ch018.TrialDivision`
  tracker note and in HANDOFF. Fix: `trial_division_stage` (3.2) can serve it.
- `prime::factor` (`prime.rs:415-452`) and `prime::prime_divisors` (`prime.rs:469-479`, built on `factor`)
  (Revision 2, m2; I1 §7 hands them to I2). `factor` is trial division by 2 and every odd number up to
  sqrt(cofactor) on `Integer`s. Its result is correct (a cofactor left after f^2 > cofactor is prime), but
  it costs about max(p2, sqrt(p1)) / 2 BigInt steps for the largest prime factor p1 and the second-largest
  p2: 3.4 s in release on 100000007 * 1000000007 (`$X/rev2/pfactor.rs`, 69 ns per step), so about
  6 minutes for a 10-digit p2 and 10 hours for a 12-digit one. `prime_divisors` has no doc on that cost
  and no caller outside its doc example. Neither is named in this unit's HANDOFF items, and `factor` is
  imported or called on 41 lines in 24 files of 8 crates (`rustmath-ellipticcurves` 8 files,
  `rustmath-finitefields` 7, this crate 3 (`integer.rs`, `factorint.rs`, `sage_wrapper.rs`),
  `rustmath-constants` 2, `rustmath-coding`, `rustmath-jupyter` (REPL), `rustmath-matrix`, `rustmath-py`),
  so changing it is its own unit. Fix:
  rebuild both on `factorization_with_limits(n, &FactorizationLimits::default())` (complete, never
  panics; a probable prime whose proof is unresolved has to be passed on or reported by the caller), or
  document the cost. Plan step 7 adds this line to HANDOFF.
- `ECMLimit` (Revision 2, m1): a new ECM first (projective Montgomery or Edwards curves with Suyama's
  parametrisation and a seeded sigma, as MAGMA's GMP-ECM; a full stage 2), then an `ecm_limit` curve
  budget in `FactorizationLimits` (default 0, MAGMA's "0 disables ECM") between rho and the sieve. The
  crate's present ECM is too slow for this to pay (164 ms per curve and about 2.5 % success at B1 = 2000 for
  a 12-digit factor, 3.2).

## Appendix: files in `$X`

| file | what |
|---|---|
| `rev2/siqs_r3.rs` | Revision 2 SIQS prototype (the one to port): `rev1/siqs_r2.rs` plus the polynomial budget checked before every polynomial (`:495-500`), the cap of 16 F first partials (`:632-641`, env `CAPMUL` changes the multiplier) and the stop at R rows inside the candidate loop (`:662-666`), with counters for dropped partials, first partials and relation sizes. Extra mode `np F M seed MAXP N...` (as `n`, with `max_polynomials = MAXP`) |
| `rev2/build.sh` | as `rev1/build.sh`, output in `rev2/bin/` |
| `rev2/robust_r3*.txt`, `rev2/small_r3*.txt` | robustness runs of the Revision 2 prototype (581 inputs x 6 seeds, 1445 inputs x 3 seeds): 0 failures |
| `rev2/grid_rel_r3.txt`, `grid_rel_r3cap.txt`, `grid_rel_r3b.txt`, `grid_rel_r3cap_61_70.txt`, `grid_dbg_r3.txt` | grids 18 to 70 digits after each Revision 2 change (budget; cap; stop at R rows), release, and 20 to 45 digits debug |
| `rev2/ecm_time.rs`, `rev2/ecm_rate.rs` | the crate's ECM (HEAD release and debug rlibs): time and success rate per curve on p12 * P30 |
| `rev2/td40.rs`, `rev2/tdrange.rs`, `rev2/cross40.rs` | trial-division cost per prime by range and above 2^40 (with HEAD's `is_prime_u64`); the two-call stage across 2^40 |
| `rev2/pfactor.rs` | HEAD's `prime::factor` on 100000007 * 1000000007 (section 8) |
| `rev2/design_before_rev2.md` | this design before Revision 2 |
| `rev1/siqs_r2.rs` | Revision 1 SIQS prototype (superseded by `rev2/siqs_r3.rs`; its polynomial budget is checked once per a, review 2 B1): de-duplication, a-supply guard and Fisher-Yates draw, caps, `validate`, `for_bits` with the L(n) base and clamp, `Outcome`. Modes: `n F M seed N...` (F = 0 / M = 0: defaults), `robust FILE seed...`, `grid FILE lo hi seed`; env `NODEDUP=1` turns de-duplication off. `rev1/siqs_r1.rs` is the same without the dependency counters |
| `rev1/build.sh` | `build.sh prog.rs rel` or `dbg`, links against the workspace's num-* rlibs |
| `rev1/robust_r1.txt`, `rev1/robust_dedup_r2.txt`, `rev1/robust_nodedup.txt`, `rev1/robust_small_nodedup.txt` | robustness runs with and without de-duplication (6 seeds each) |
| `rev1/small.gp`, `rev1/small.txt` | 1445 gp inputs of 6 to 18 digits |
| `rev1/grid_rel_r1.txt`, `rev1/grid_dbg_r1.txt` | timings 20 to 60 digits (release) and 20 to 45 (debug) |
| `rev1/tdrate.rs` | trial-division rate (segmented sieve, `&BigUint % u64`) |
| `rev1/pp2.rs`, `rev1/head_pp.rs` | `is_perfect_power` on prime powers; HEAD's QS drivers on 3^2000 and 1000003^50 (HEAD debug rlib) |
| `rev1/refs_r1.gp`, `rev1/n151.txt` | Revision 1 reference values; the 150-digit N150 |
| `rev1/design_before_rev1.md` | this design before Revision 1 |
| `siqs.rs` | first SIQS prototype (no de-duplication; kept for the record); modes: `FILE lo hi slack fbmul` (grid), `n slack fbmul N...`, `robust FILE`; env `LFB=1` (L(n) factor base), `LPM`, `MM` |
| `siqs_lib.rs` | the first prototype without `main`, used by `driver.rs` |
| `rev1/driver_r1.rs`, `rev1/siqs_r2_lib.rs`, `rev1/driver_r1_dbg.txt` | Revision 1 limits engine prototype and its debug run (3.4) |
| `driver.rs` | first prototype of `LimitedFactorization`, `coprime_refine`, budgeted Brent rho, the limits engine |
| `rho_check.rs`, `rho_check2.rs` | exact-budget Brent rho on the test inputs; rounds on all odd composites below 2 * 10^5 |
| `repro_head.rs`, `repro_gauss.rs` | HEAD reproductions of items 1 and 2 |
| `semiprimes.gp` / `.txt` | gp semiprimes of 18 to 72 digits, each checked with `factor` (4.4 min) |
| `robust.gp` / `.txt` | 581 gp inputs (semiprimes, p^2 q, p q r, primes) |
| `refs.gp`, `fb.gp`, `pari_times.gp` | reference values and PARI timings |
| `rel_grid.txt`, `rel_lfb.txt`, `dbg_grid.txt`, `rel_check.txt` | measured grids of the first prototype |
