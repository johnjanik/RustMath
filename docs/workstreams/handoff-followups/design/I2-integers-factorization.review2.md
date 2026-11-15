# Review 2 of design I2 (rustmath-integers factorization)

Reviewer scratch dir: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/I2-integers-factorization/review2/`
(`$R` below; `$X` is the designer's dir). I read the whole design (1107 lines, Revision 1 included) and review 1.

Verdict: **not approved**. One blocker (B1), six minor items. Everything else I checked holds, and most of it I
re-ran myself. The blocker is small to fix: the polynomial budget of the sieve is checked once per a-value in
the reference prototype, so one a can run 2^(s-1) polynomials, and the design tells the implementer to port
that prototype as it is.

## What I checked, and what holds

Defects at HEAD (`$R/repro.rs`, `$R/repro2.rs`, linked against the current debug rlib of HEAD):

| claim | reproduced |
|---|---|
| `quadratic_sieve_factor`: 3127 None, 165731 Some(3127), 10403 None, (10^9+7)(10^9+9) None, 270050147102784401 None, 666958488061768516158211 None | yes |
| `quadratic_sieve_factor_with_params(270050147102784401, 300, 10^6)` None | yes (1.5 s debug) |
| `trial_division_with_primes`: (1225, [2,3]) -> [(1225,1)]; -1225 -> [(-1,1),(1225,1)]; 1001 -> [(1001,1)]; (8, [5,2]) -> [(8,1)] | yes |
| `legendre_symbol` `.unwrap_or(one)` at `quadratic_sieve.rs:57-58`; `ecm.rs:14-15` describes p - 1 / p + 1 | yes (read) |
| m10: `factor_trial_division(n, None)` stops at 10^6 above 2^128 | yes: (2^128+1) * 1000003 comes back as one "factor" |

HANDOFF line numbers (`quadratic_sieve.rs:282`, `:340`, `:778`, `:792`, `:45-68`; `factorint.rs:401`, `:434-436`;
`prime.rs:808`; `ecm.rs:14-15`) and the tracker lines the plan edits (`PROJECT_TRACKER.md:254, 395, 399, 402,
406, 407, 415, 3100`; `status_overrides.csv` rows `ch018.MPQS`, `ch018.PollardRho`, `ch018.TrialDivision`,
`ch018.Factorization`) are correct.

Algorithm, re-derived: b^2 = kN mod a (B_l = +-t mod q_l, 0 mod the other q_j; a squarefree, q_l not dividing kN);
the Gray-code rule (flip bit l = tz(i); add 2 B_l iff (i >> l) = 3 mod 4, i.e. the new bit is 0) visits all
2^(s-1) sign patterns with B_s fixed; the root update r <- r - e Bainv2; the cofactor argument (every prime
<= p_max dividing g is in the base, so v < p_max^2 is prime); the GF(2) elimination (non-pivot rows keep cleared
bits, so XOR from word c/64 on loses nothing; a non-pivot history holds its own bit and only pivot bits, so the
vectors are independent and number rows - rank >= extra); X^2 = Y^2 mod N from kN = 0 mod N. |y| de-duplication
is sound (y fixes Q = y^2 - kN, so a repeated |y| is the same relation). The multiplier makes gcd(k, N) = 1 and
kN a non-square under the step-0 precondition. Step 0 decides every N <= 47. `coprime_refine` terminates (the
log-sum drops by >= 1 per step). `has_at_most_digits` is right for c > 0 (m3 for c = 0).

MAGMA digest (§18.10, §18.10.1, §18.10.3): read correctly (defaults 10000, 8191, infinity, Proof true; MPQSLimit
< 25 disables MPQS; third value only with both ECMLimit and MPQSLimit set; PollardRho(n, c, s, k) defaults
1, 1, 8191). The only §18.10 handbook example (H18E8, [1380, 675, 3408, 654]) is ported as `coprime_refine_values`.

Reference values (`$R/refs.gp`): all nine semiprime rows, P30, p15 q15, 6 p15 q15, 101 p20 q20 and
`#Str(p20*q20)` = 39, q12^2, N150 (150 digits), `factor(79839848800024750353128)`, F7, 101*103*P30,
666958488061768516158211, the p^2 q and p q r inputs, 10000019 * 10^22+9, the k = 3 factor base list and the
roots t_5..t_29, p25 q25, `#Str(2^80)` = 25, psi_12, `factor(1380*675*3408*654)`, 2 P30,
`factor(270050147102784401)`: every one matches the design.

Budgeted Brent (`$R/brent.py`, an independent Python copy of the specified budget rule): (10^9+7)(10^9+9) from
c = 1, x0 = 1 is (None, 8191) with k = 8191 and splits after 50430 steps with 2^20; (c = 2, x0 = 2, budget 1)
gives (None, 1); 2^64+1 gives 274177 after 1918; 101*103*P30 gives 10403 after 30; p15 q15 (None, 8191);
12780049 gives 53 after 14 (the new doc example); psi_12 and the p^2 q input stay unsplit with 8191. All as the
design says.

Timings (designer's binaries, this machine): debug rows 20..45 in 2.0 s (design 2.2 s); debug 50 and 55 digits
2.8 s and 8.1 s (design 3.0 s and 7.8 s); release 50, 55, 60 digits 0.32 s, 1.03 s, 2.29 s (design 0.35, 1.1,
2.4 s); trial-division rate 71 ns per prime release, 490 ns debug (design 70 ns, 0.5 µs); `is_perfect_power(3^2000)`
0.28 ms debug (design 0.32 ms); the a-supply guard on N150 with F = 30 and 31 returns
`OutOfPolynomials { relations: 0, needed: 63 / 64 }` in 1.6 ms release.

Robustness, independent inputs (`$R/gen.gp`, 722 gp inputs of 8 to 40 digits: balanced and unbalanced
semiprimes with a small factor just above 47, p^2 q, close factors p, nextprime(p), all base-2 Fermat
pseudoprimes in [10^6, 5 10^6] without small factors, psi_12, 3215031751, 99 primes), prototype `siqs_r2` with
`for_bits` and de-duplication, seeds 1, 2, 3, 7, 11: 0 failures, worst input 0.39 s, at most 12 dependencies
tried. Seed sweep on 666958488061768516158211, seeds 1..6: a factor every time. p20, 2207, 1009, P30 give
`NoFactor` with 39, 36, 34, 39 dependencies; 53 and 59 give `NoFactor { dependencies: 0 }` from the scan.

Small parameters (`$R/siqs_probe2.rs`, F in {30, 33, 40, 60}, M in {64, 512, default} on the 20-, 30- and
40-digit rows): every run ends (Factor or OutOfPolynomials) in milliseconds. These runs also show the overshoot
of B1 (polys = 512 with a budget of 300 when s = 9).

Callers (my grep of `*.rs` and `*.md` outside `target/` and the archive): `quadratic_sieve::*`,
`trial_division_with_primes`, `brent_rho`, `split_odd_composite` are used only inside `rustmath-integers`;
`rustmath-numbertheory` glob-re-exports `prime::*` and none of the new names collides; no REPL or lava binding
calls these functions (no `MPQS`, `PollardRho`, `TrialDivision` strings in any `.rs` outside the crate). The plan's
downstream commands (`rustmath-numbertheory`, `cargo check --workspace --tests`) are the right ones.

I1 interface: everything I2 uses exists in I1's revised design with the same names and shapes
(`is_probable_prime`, `try_is_prime -> Result<bool>`, `is_prime`, `is_prime_u64` public in `primality.rs`,
`sieve::for_each_prime(lo, hi, FnMut(u64) -> ControlFlow<()>)`, `sieve::primes_up_to`, `pow_mod_u64`
`pub(crate)`), and I1 §2.8 leaves the panic policy to I2. I3's edits to `quadratic_sieve.rs` are all inside
deleted functions, as the design says.

## Blocker

### B1. The polynomial budget is checked once per a-value, so one a can sieve 2^(s-1) polynomials

The design's termination argument (section 1.2, "Termination", line 254) says each iteration of the polynomial
loop sieves one counted polynomial, at most `max_polynomials` in all. The reference prototype does not do this:
it tests `stats.polys >= max_polys` only at the top of the a-loop (`$X/rev1/siqs_r2.rs:396-399`), and the
Gray-code loop (`:486-487`) then runs all 2^(s-1) polynomials of that a. The design's list of changes to port
says the prototype "already has ... `max_polynomials`" (line 264), so a faithful port keeps the defect, and no
test can see it: `qs_out_of_polynomials` uses the 40-digit row (s = 5, 16 polynomials per a), which ends either
way, and the guard test uses F = 30 and 31, where the guard fires before any sieving.

Reproduction (`$R/siqs_probe2.rs`, the prototype with an env override of `max_polys` and a progress print):
N150 with `factor_base_size = 40` (other fields from `for_bits(498)`, so M = 131072) passes the a-supply guard
with s = 31, |W| = 32, |E| = 32. With `max_polynomials = 10` the run went past 8000 polynomials with 0 relations
and was killed; the first a alone has 2^30 polynomials (about 11 hours in release at the measured 38 µs per
polynomial). F = 60 (s = 28) and F = 100 (s = 26) behave the same. This is a public call,
`quadratic_sieve_factor_with_params(N150, &QsParams { factor_base_size: 40, max_polynomials: 10, .. })`, with
validated arguments, which in practice does not return: the same class of defect as review 1's B1.

The fix works: `$R/siqs_probe3.rs` adds the same check before each polynomial inside the Gray-code loop. The
calls above then return `OutOfPolynomials { relations: 0, needed: 73 / 93 / 133 }` in 2 to 4 ms, and with the
default budget 1000 F = 40000 the F = 40 run ends in 1.5 s release.

Required change: state in step 4 and in "Termination" that the budget is checked before every polynomial (the
first one of each a included), not once per a. Add this to the list of changes to port, and correct the sentence
that the prototype already has `max_polynomials`. Add a test, for example `qs_polynomial_budget_is_per_polynomial`:
N150 with `QsParams { factor_base_size: 40, max_polynomials: 10, ..QsParams::for_bits(498) }` gives
`Ok(OutOfPolynomials { relations: 0, needed: 73 })` inside `within(5, ..)`. A port with the per-a check hangs on
it. Optionally also assert, on the 40-digit row with `max_polynomials = 3` (s = 5), that the run stops after
3 polynomials; this needs a counter in the outcome or a unit test on the run loop.

## Minor

- **m1. ECMLimit: the stated reason is weak.** Section 3.2 leaves out `ECMLimit` because the crate's ECM draws
  sigma from `rand::thread_rng` (`ecm.rs:317`). But `try_one_curve(n, b1, b2, seed)` already takes a seed
  (`ecm.rs:316`; it is only added to the random draw), and `ecm_factor` already takes a curve count. A seeded
  sigma (for example splitmix64 of the seed and the curve index) and an `ecm_limit: u64` curve budget in
  `FactorizationLimits` (default 0, which is MAGMA's "0 disables ECM", so no default behaviour changes) are a
  few lines in this crate. The HANDOFF item names `ECMLimit`. Either add it, or say plainly that leaving it at 0
  is a choice (ECM is the weak ⏳ row and would mostly cost time), not an infeasibility. The interim behaviour
  is already documented honestly.
- **m2. An I1 hand-off is dropped.** I1's design (§7, line 1294) says "`prime_divisors` (`prime.rs:469`) and
  `factor` (trial division) are I2's". I2 §0 says they are not covered, which is right for the HANDOFF items, but
  §8 does not hand them on, so the plan's HANDOFF edit would lose them. Add a §8 line (they can be rebuilt on
  `factorization_with_limits`), next to the `factor_trial_division` hand-off.
- **m3. `has_at_most_digits` and c = 0.** The doc says "True iff 0 < c < 10^d", but bits(0) = 0 <= 3d returns
  true for c = 0. Callers only pass c > 1, so state the precondition c > 0 (or return false for 0).
- **m4. Plan order.** Plan step 1 adds `factor_pollard_rho_limited`, which is `limited_engine` with a plan, and
  `limited_engine` only exists after step 2. Move that function to step 2. Step 2's `split_composite` calls the
  new `quadratic_sieve_factor_with_params`, which exists only after step 4. Say how the crate stays
  build-green in between (for example, step 2 lands with `QsPolicy::Always` mapped to no sieve and a `todo`
  comment, or steps 2 and 4 are one commit).
- **m5. The memory bound "at most about 1.2 GB".** The `FactorizationLimits` doc gives this as a bound. The
  relation store is not capped: partial relations were 4 to 5 F in every measured run up to 70 digits (my
  check of `$X/rel_lfb.txt`: 4.4 to 5.3 F at 65 to 70 digits), but nothing bounds them beyond the polynomial
  budget. Write "typically, from runs up to 70 digits", or cap the stored partials.
- **m6. Trial-division cost above 2^40 (interface with I1).** I1 §2.5 confirms sieve survivors with
  `is_prime_u64` when hi > 2^40. The engine calls `for_each_prime(2, bound + 1)` with bound up to u64::MAX
  (limit u64::MAX and m >= 2^80). If I1 confirms every survivor, and not only survivors >= 2^40, the stated cost
  of about 70 ns per prime does not hold for such bounds. Every survivor below (2^20)^2 is prime without a check.
  State what I2 needs from I1 (confirm only survivors >= 2^40), or add the higher cost to the doc. The tests are
  not affected: 2 P30 stops at the first prime above 2^16.

## Items covered

| item | covered | notes |
|---|---|---|
| 1 QS sieve stage | yes: SIQS from Contini 1997 / Crandall-Pomerance §6.1, L(n) factor base, log sieve, dense GF(2) elimination, square root, gcd; gp semiprimes of 20 to 60 digits with measured times (reproduced) | B1 |
| 2 `trial_division_with_primes` | yes: `LimitedFactorization`, cofactor classified as prime, probable prime or composite, no early break | |
| 3 capped driver and return type | yes: `factor_pollard_rho_limited`, `factorization_with_limits`, `LimitedFactorization` (MAGMA's sequence, sign, unfactored composites, plus unproven probable primes) | m1, m4, m5, m6 |
| 4 QS `legendre_symbol` | yes: deleted; word Euler criterion and word Tonelli-Shanks | |
| 5 ECM module doc | yes: doc only; the proposed text matches what `ecm.rs:297-399` does | |

Nothing is declared out of scope. `ECMLimit` is fixed at 0, and the design states why (see m1).
