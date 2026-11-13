# Review 1 of design I2 (rustmath-integers factorization)

Reviewer scratch dir: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/I2-integers-factorization/review1/`
(`$R` below). Design read in full (779 lines, no Revision sections).

Verdict: **not approved**. Two blockers (B1, B2), eleven minor items. The algorithms are right and the
prototype is strong: I re-ran it and reproduced every timing and robustness number I sampled. The blockers
are about termination and resource use of the new public APIs on in-range arguments. Both have small fixes.

## What I checked, and what holds

Defects at HEAD (built `$R/repro.rs` against the HEAD debug rlib):

| claim | reproduced |
|---|---|
| QS: 3127 -> None, 165731 -> Some(3127), 10403 -> None, 15347 -> None, (10^9+7)(10^9+9) -> None, 20-digit row -> None | yes, all |
| `gaussian_elimination_gf2` on the 3 x 2 all-ones matrix returns [r0], row sum [1, 1] | yes (traced by hand from `quadratic_sieve.rs:282-337`) |
| `try_dependency` floor root of a non-square (`:376`) | yes |
| `trial_division_with_primes`: 1225 -> [(1225,1)], -1225 -> [(-1,1),(1225,1)], 1001 -> [(1001,1)], (8,[5,2]) -> [(8,1)] | yes; also (12,[4,3]) -> [(12,1)] |
| `legendre_symbol` `.unwrap_or(one)` at `:57-58`; `mod_pow` errs only for modulus 0 (`integer.rs:153-156`); (4/2) -> 1 | yes |
| `ecm.rs:14-15` describes p - 1 / p + 1; `:322-323` "Montgomery parameterization" is not what the code does | yes (x = u/v, y = 1, a = (v-u)^3(3u+v)/v^3 - 2, b fitted) |
| `factor_pollard_rho` has no cap; psi_12 comes back unsplit from the QS driver | yes (`qs_complete(psi12) = [(psi12, 1)]`) |

Algorithm (re-derived): Q = (ax+b)^2 - kN = a g(x); b^2 = kN mod a by CRT from B_l = +-t mod q_l and
B_l = 0 mod q_j (j != l); the Gray-code sign rule (+ iff (i >> v) = 3 mod 4) is Contini's
(-1)^ceil(i/2^(v+1)); the root update r <- r - e Bainv2 follows from r = a^-1(+-t - b) + M; the completeness
argument for the cofactor (prime factors > p_max, v < p_max^2 so v prime) is right; the GF(2) elimination
argument (non-pivot rows keep cleared bits, histories contain their own row bit and no other non-pivot
bit, count = rows - rank, rank <= F + 1 so >= extra dependencies) is right; X^2 = Y^2 mod N follows from
kN = 0 mod N. Byte overflow only loses candidates. `coprime_refine` preserves the product and its
log-sum argument terminates; the limits engine's trial-division primality shortcut (m < (limit+1)^2) is
sound. Citations checked (Silverman 1987, Pomerance 1985, Contini 1997, Bach-Driscoll-Shallit 1993, Brent
1980, Lenstra 1987, Cohen GTM 138); the MAGMA digest (§18.10, §18.10.1, §18.10.3) is read correctly apart
from m3.

Reference values: I ran every gp value of sections 1.6, 2.4 and 3.5 (`$R/refs1.gp`, `$R/fb.gp`): the nine
semiprime rows, F7, the three named inputs, psi_12, P30, p15, q15, p20, q20, q12, 101*103*P30, 101*p20*q20,
`#Str(p20*q20)` = 39, p25*q25, 2^64+1, 42525, 1380*675*3408*654, the factor-base list and roots for
k N = 3 * 1000000016000000063. All match the design. The `coprime_refine` expectations hold under every
input order (`$R/refine.py`).

Timings (release, `$R/siqs_rel`, LFB mode): 20 to 60 digits match section 1.3 (60 digits 2.19 s, 55 digits
0.92 s, 50 digits 0.37 s). Debug (`$R/siqs_dbg`, against the debug rlibs): rows 20..45 take 2.2 s in all
(design: 2.5 s). Brent on psi_12 with k = 2^22: 450558 steps, 1.7 s debug. Rho rounds: all 81761 odd
composites below 2*10^5 split, max 156 steps (designer's binary re-run). Budgeted Brent values of
`brent_rho_budget` (31870 steps; (None, 1); (None, 8191)) reproduced.

Robustness with the parameters the design specifies (L(n) factor base; the designer's robust mode used the
table times 2 instead): `$R/siqs_rev2.rs` with |y| de-duplication, seeds 1 to 6, the designer's 581 inputs
plus 1849 new gp inputs of 6 to 18 digits (`$R/small.gp`: random composites without factors <= 47, p^3 q,
p^2 q^2 r, 53 q, primes): 0 failures, worst 0.36 s. See m1 for the run without de-duplication.

Callers (my grep of all `*.rs`, `*.md`, `*.csv`): `quadratic_sieve::*` and `trial_division_with_primes` are
used only inside `rustmath-integers`; `brent_rho`, `split_odd_composite` are private; `rustmath-numbertheory`
re-exports `prime::*` and no new name collides (`LimitedFactorization`, `FactorizationLimits`,
`factorization_with_limits`, `factor_pollard_rho_limited`, `POLLARD_RHO_DEFAULT_ITERATIONS`, `QsParams`,
`QsOutcome` do not exist anywhere). No REPL or lava binding calls these functions. Tracker line numbers
(`PROJECT_TRACKER.md:254, 395, 399, 402, 406, 407, 415, 3100`; `status_overrides.csv` rows) are correct.

## Blockers

### B1. The a-value selection does not terminate for in-range parameters; the termination claim is false

Section 1.2 step 4 draws "s - 1 distinct window indices" and section "Termination" says the polynomial
loop ends after `max_polynomials` or 1000 repeated a-choices and "the other steps are finite".
`QsParams` accepts `factor_base_size >= 30`. With a small factor base and a large n, s exceeds the number
of eligible primes, and the draw loop (prototype `siqs.rs:342-349`, ported as `choose_a`) never ends:

```
$R/siqs_probe n 12 0.0034 <nextprime(10^74) * nextprime(3*10^75)>     (F = 31, 151 digits)
s=33 window=23 nfb=31 pmax=257
a-selection spinning: need 32 distinct of 23        (killed by timeout)
```

`quadratic_sieve_factor_with_params(n, &QsParams { factor_base_size: 31, .. })` is a public call with
in-range arguments, so this is a hang in the public API, and the `max_polynomials` budget does not bound it.

Required change: before drawing, if the eligible set (p > 30, p not dividing k) has fewer than s entries,
or the window has fewer than s - 1, return `OutOfPolynomials` at once (or reject the parameters with
`InvalidArgument` in validation, stating the rule). Put the guard in the termination argument and add a
test (F = 30 on a 150-digit n returns promptly with `OutOfPolynomials` or `Err`).

### B2. Accepted arguments lead to huge allocations or overflow (limits engine and `QsParams`)

1. `limited_engine` step 2: "For p in `fast_arith::prime_range(2, limit + 1)`, stopping at the first p
   with p^2 > m". `prime_range` sieves eagerly with `vec![true; end - start]` (`fast_arith.rs:487-500`),
   so the cost is set by the limit, not by m. `factorization_with_limits(&100, &{trial_division_limit:
   10^10, ..})` allocates 10 GB before the first division and aborts. `limit + 1` overflows at `u64::MAX`
   (panic in debug), and `(limit + 1)^2` overflows u64 for limit >= 2^32. MAGMA's `TrialDivisionLimit`
   takes any value.
2. `split_composite`: "qs_max_digits is None or c < 10^limit". Read literally, `mpqs_limit = Some(u32::MAX)`
   forms 10^(4.3 * 10^9) (about 1.8 GB). Compare decimal digit counts instead (bits bound, then exact).
3. `QsParams::factor_base_size <= 2_000_000`: the dense GF(2) step needs rows x (cols + rows) bits, about
   (F + 33) * 2F / 8 bytes, which is 10^12 bytes at F = 2 * 10^6. This is reachable: with a 30-digit n and
   F = 2 * 10^6 nearly every sieve value is smooth, so the relations arrive in seconds and then the
   matrix allocation aborts. `sieve_half_width <= 2^30` is a 2 GB sieve array. `large_prime_multiplier *
   p_max` can overflow u64 (use `saturating_mul`).
4. `QsParams::for_bits` follows the L(n) formula without a cap, so it leaves its own documented range for
   large n (F is about 140000 at 120 digits, 4.9 GB of matrix; above 2 * 10^6 near 200 digits). The
   automatic path must clamp to the validated range.

Required change: iterate trial-division primes only up to min(limit, isqrt(m)), in bounded windows
(segmented sieve; I1 §2.5 specifies one), and do the "m < (limit + 1)^2" test in `Integer`; specify the
digit-count comparison; lower the `factor_base_size` cap to what the dense elimination can hold (state the
memory, about F^2 / 4 bytes, and pick a bound such as 65536), and the sieve width cap (for example 2^26);
make `for_bits` clamp; document memory in `QsParams` and `FactorizationLimits`. Tests:
`factorization_with_limits` with `trial_division_limit = u64::MAX` and `10^12` on 100 and on 2 * P30
(fast, correct), `with_params` rejecting F above the cap.

## Minor

- **m1. De-duplication is load-bearing, but the evidence was gathered without it.** The prototype
  `$X/siqs.rs` has no `seen_y` set (grep: none), and its robust mode uses the table times 2, not the L(n)
  base the design specifies. Without de-duplication I get a failure:
  `666958488061768516158211` (= 765944099509 * 870766533079), L(n) base, seed 4: `None` after 42
  dependencies, with 50 repeated |y| among 1010 relations (s = 2, a-values sharing a prime give the same
  y). With de-duplication it splits, and 0 failures over 6 seeds x 2430 inputs. Required: say in step 7
  that de-duplication is a correctness step (why: shared a-factors), add a unit test that the relation
  store drops a repeated |y| (and does not create a pair row from it), and restate the robustness claim
  with the L(n) parameters and de-duplication (numbers above).
- **m2. `with_params` below 48 is unspecified.** The SIQS precondition is |N| > 47 with no prime factor
  <= 47, and the quick path "p | n, p < |n|" does not catch a prime n <= 47. The multiplier then picks
  k = n (prototype: k = 47 for N = 47, k = 43 for N = 43), so "gcd(k, N) = 1" fails. Required: specify the
  outcome of `quadratic_sieve_factor_with_params` for n = +-1, 2, 3 and primes <= 47 (for example
  `Ok(NoFactor { dependencies: 0 })` without running the sieve) and test it. Also, "step 0 below" in step
  1 refers to nothing.
- **m3. MAGMA parameter semantics.** The HANDOFF item names "MAGMA's third return value with `ECMLimit` /
  `MPQSLimit`", but section 0 lists `ECMLimit` as "not in this unit's list". State the honest interim
  behaviour in the API doc: the engine is MAGMA with `ECMLimit := 0` (no ECM stage), and say why (the
  crate's ECM draws sigma from `rand::thread_rng`, `ecm.rs:317`, so results would not be reproducible;
  it is the ⏳ row) and what would be needed (a seeded ECM with a curve budget). Also document two
  deviations: MAGMA disables MPQS for `MPQSLimit` < 25, while `Some(x)` with x < 25 here still sieves
  composites of up to x digits; and MAGMA returns the third value only when both limits are set.
- **m4. Prime powers cost one split per exponent.** `split_composite` returns only the base b of
  c = b^j, so the engine pushes (b, e) and (b^(j-1), e) and pays j perfect-power tests. The complete QS
  driver (section 3.4) also drops HEAD's trial division by the primes <= 47. Measured in debug: HEAD
  `quadratic_sieve_factor_complete(3^2000)` 2.8 ms; the design's splitting of 3^2000 alone 0.53 s
  (1999 perfect-power splits), and it grows about quadratically in j. Required: on a perfect power push
  (b, j * e) (as `$X/driver.rs` does), and keep a small trial division in the complete QS driver.
- **m5. ECM doc precision.** The stage-1 multiplier is the product of the largest prime powers <= B1
  (`ecm.rs:304`), so the condition is "B1-powersmooth", and stage 2 tries only the first 100 primes above
  B1 (`ecm.rs:389`, `take(100)`), not every prime up to B2. The proposed text "apart from at most one prime
  below B2" overstates this module. Say what the code does.
- **m6. Multiplier list.** Section 1.2 step 1 lists 31 values including 46; the measured prototype uses 30
  (no 46, `siqs.rs:182-185`). Pick one and keep the measurements consistent.
- **m7. I1 coupling.** The I1 review (blockers B1, B2) questions panicking `is_prime` in existing callers;
  `quadratic_sieve_factor_complete` keeps `is_prime` (panics on Unresolved, which I1's reviewer measured
  at 6 of 16 random 2048-bit primes). Follow whatever policy the I1 revision adopts, and note that
  `factorization_with_limits` is the non-panicking alternative. If I1 moved the QS part of
  `psi12_is_split_by_every_driver` to `#[ignore]`, plan step 6 should re-enable it. I3 also edits
  `quadratic_sieve.rs` (`:50`, `:60`, `:96`, `:228`, `:376`); say that the rewrite supersedes those edits.
- **m8. Disposition of HEAD tests and doc tests.** List what happens to `test_quadratic_sieve_prime`,
  `_even`, `test_complete_factorization`, `test_factor_base_generation`, `test_trial_factor`, the module
  doc example (15347) and the complete-driver doc example whose comment says "the sieve finds no factor,
  so the cofactor is finished by Pollard rho" (`quadratic_sieve.rs:543-547`).
- **m9. `limits_trial_division_proves_small_cofactor` does not test what its name says.** 10007 is below
  psi_12, so `classify` decides it anyway, and the trial-division loop stops early at p = 101. The
  "m < (limit + 1)^2" path is only an optimisation; rename the test or drop the claim.
- **m10. Out-of-unit observation for the tracker rows this unit touches.** `factor_trial_division(n,
  None)` uses `sqrt(n).to_u64().unwrap_or(1000000)` (`factorint.rs:192`): for |n| > 2^128 it divides
  only up to 10^6 and appends a composite cofactor, while the doc says `None` means sqrt(n). Section 2.3
  calls its cofactor "documented". The `ch018.TrialDivision` note should state this, or hand it to a
  follow-up.
- **m11. `factor_pollard_rho_limited` fixes `proof = true`.** The digest says every §18.10.3 function takes
  `Proof` (default true). Optional: take a `proof` flag, or say why not.

## Items covered

| item | covered | notes |
|---|---|---|
| 1 QS sieve | yes (SIQS, L(n) base, log sieve, dense GF(2), square root, gcd; 20-60 digits vs gp, timed) | B1, B2.3-4, m1, m2, m6 |
| 2 `trial_division_with_primes` | yes | m10 (sibling function) |
| 3 capped driver and return type | yes (`LimitedFactorization`, `factor_pollard_rho_limited`, `factorization_with_limits`) | B2.1-2, m3, m4, m11 |
| 4 QS `legendre_symbol` | yes (deleted; word Euler criterion) | |
| 5 ECM module doc | yes (doc only) | m5 |

Nothing is declared out of scope. `ECMLimit` is left out without the interim-behaviour statement (m3).
