# Review 1 of design I1 (integers, primality)

Design: `I1-integers-primality.md` (921 lines, no Revision sections). Reviewer scratch:
`I1-integers-primality/review1/` (called `$R` below). Tree at HEAD `c79a582`, clean. All HEAD claims were
reproduced with my own standalone program (`$R/r1_head.rs`, linked against a fresh copy of the HEAD debug
rlib), all reference values with my own gp scripts (`$R/g1.gp` .. `$R/g7.gp`).

Verdict: **not approved**. Three blockers (B1-B3), fifteen minor items. The core algorithms (BPSW, the
Jacobian order check, the certificate verifier, the sieve) are correct and well prototyped; the problems
are the coverage of the prover at sizes the design did not measure, the callers that acquire a new panic,
and one silent truncation.

## What I checked and found correct

HEAD defects (all reproduced, `$R/r1_head.rs`, debug):
- `is_prime(psi_12)` = `is_prime(psi_13)` = true; `factor_pollard_rho(psi_12)` = `[(psi_12, 1)]` in 22 ms.
- `next_prime(0)` = `next_prime(1)` = `next_prime(-5)` = 3.
- `is_prime_miller_rabin(3215031751, 4)` = false, `(3825123056546413051, 11)` = false (all 12 bases run);
  `(2^127-1, 20)` 19.5 ms against 0.96 ms for one round (design: 30 ms / 1.6 ms; same ratio).
- `nth_prime(2000)` 1.8 s, `prime_pi(20000)` 2.3 s: consistent with the design's 11 s for 10^4 / 10^5.
- `ArithInt::new().is_prime(2147483647)` panics "attempt to multiply with overflow" (fast_arith.rs:246);
  `ArithLLong::new().is_prime(i64::MAX)` = false.
- `square_root_mod_prime`: `assert_prime` (integer_mod.rs:652, :877-883) and again `is_prime` inside
  `Integer::sqrt_mod_prime` (integer.rs:883). Confirmed. (There is a third test on the prime-power path,
  see m7.)
- No type `FastArith64` exists; the reading "duplicates the 64-bit fast-arithmetic type" is reasonable.

Reference values (gp 2.15.4, `$R/g1.gp`..`$R/g4.gp`): every value in the test table that I sampled
matches: the factorizations of psi_12, psi_13, 561, 2047, 1000036000099, 3215031751 (strong pseudoprime
to 2, 3, 5, 7, not 11), 3825123056546413051 (to 2..31, not 37); `nextprime(2^k) - 2^k` = 13, 13, 277, 51, 7,
235; `precprime(10^30)` = 10^30 - 11; the next/previous prime values at 2^64, 2^127, 10^40; `prime(10^5)`,
`prime(10^6)`, `primepi(10^6)`, `primepi(10^7)`; the three `primes([..])` windows; the repunit list
[2, 19, 23, 317]; `isprime(2^31-1)`, `factor(2147395601)`, `isprime(9223372036854775783)`; the 46 base-2
strong pseudoprimes below 10^6 (first 2047, last 983401); `qfbsolve(Qfb(1,1,6), 1000000000933)`; the roots
of `polclass(-23,1)`, `polclass(-23)`, `polclass(-56,5)` and their cubes mod the stated primes; the table's
counts per class number (my count of `$X/table_h20.txt` gives the design's 20 numbers).

BPSW: I implemented Selfridge's method A and the strong Lucas test independently in gp (`$R/g2.gp`).
psi_12 and psi_13 have Selfridge D = -7 and both fail the strong Lucas test (both pass the base-2 strong
test), so `NotPrime(StrongLucas(-7))` is right. The composites below 20000 passing my strong Lucas test are
exactly [5459, 5777, 10877, 16109, 18971] (OEIS A217255). The prototype's BPSW agrees with PARI
`ispseudoprime` on 202 numbers (`$R/g7.gp`): 60 primes of 65-464 bits, 60 semiprimes, 40 base-2 strong
pseudoprimes above 2^64 of the form (k+1)(2k+1), psi_12, psi_13, 40 random odd numbers. 0 mismatches.

ECPP soundness (re-derived): the theorem of 2.3.1 is the Goldwasser-Kilian theorem as PARI states it, with
gcd(N, 6) = 1 and the discriminant unit making E_p an elliptic curve for each p | N. The Jacobian argument of
2.3.2 holds for the exact formulas: Z3 = 2 Y1 Z1 (dbl-1998-cmo-2) and Z3 = Z1 H (the mixed addition is
madd-2004-hmv, not "add-1998-cmo-2" as the prototype comment says, but the formulas are the right ones);
Z = 0 is absorbing, so a final Z that is a unit mod N makes every step a true group operation mod every
p | N; U = -T then gives [q]T_p = O. Completeness for prime N also holds: in left-to-right double-and-add an
addition always follows a doubling, so the accumulator is jT with 2 <= j <= q - 2 and never +-T.
`iroot4_ceil` and the bound q > (r + 1)^2 imply q > (N^(1/4) + 1)^2. The verifier uses neither the table nor
randomness. PARI's own validity conditions (`??primecert`) match the design's checks (the design is
slightly stricter: gcd(N, 6) = 1 and the discriminant unit; a valid PARI certificate of a prime satisfies
both).

ECPP prototype, run by me (`$X/bin/ecpp_rs`, debug, no genus filter): random 256-bit prime 0.61 s
(12 levels), random 384-bit 0.82 s (14), nextprime(10^40) 94 ms, 2^127 - 1 88 ms, nextprime(2^200) 0.21 s.
gp `primecertisvalid` = 1 for all five certificates (`$R/cert_*.txt`). PARI rejects the tampered
nextprime(2^200) certificates (s doubled, x + 1, t + 2, N + 2) and accepts it with its first step removed
(see m3). Timings agree with section 2.3.7.

Sieve and S6: the segmented sieve design (base primes to min(isqrt(hi - 1), 2^20), `is_prime_u64`
confirmation above 2^40, Rosser-Schoenfeld bound for `nth_prime`) is correct. The citations I checked
(Baillie-Wagstaff 1980, Pomerance-Selfridge-Wagstaff 1980, Baillie-Fiori-Wagstaff 2021, Atkin-Morain 1993,
Goldwasser-Kilian 1999, Sorenson-Webster 2017, Bays-Hudson 1977, Rosser-Schoenfeld 1962, Yui-Zagier 1997,
Watkins 2004, Enge-Sutherland 2010, Enge 2009) have the right journals, volumes and pages. The MAGMA digest
semantics (§18.9.1: `IsPrime` with `Proof := true`, `IsProbablePrime`, `PrimalityCertificate`,
`IsPrimePower` with p proven; §18.9.2: `NextPrime` least prime > n, `NthPrime` to 10^10) are read
correctly.

The I2 interface (section 2.8) is consistent with the I2 design (`I2-integers-factorization.md` lines
393-399, 693-705), and F1 consumes `try_is_prime`, `is_probable_prime` and `try_sqrt_mod_prime` as
specified here.

## Blockers

### B1. The prover's coverage above 1024 bits is unmeasured and poor; with `is_prime` panicking, existing callers regress at realistic sizes

The design measures the "stuck at the top level" rate up to 1024 bits (3 %) and calls Unresolved "rare
above 512 bits". It has no data for larger inputs. I ran the designer's own counter
(`$X/bin/ecpp_rs_rel table_h20.txt count 2048 16`, release, h <= 20, trial bound 2^16):

```
bits 2048 count 16 candidates histogram {0: 6, 1: 2, 2: 2, 3: 5, 4: 1}
```

6 of 16 random 2048-bit primes (37 %) have no top-level candidate, so `prove_primality` is `Unresolved`
and `is_prime` panics for them. This fits the heuristic: the expected number of top-level candidates is
about sum over rows of 1/h times 2 traces times e^gamma ln(2^16) / ln N, which halves when the size
doubles (about 3 at 1024 bits, 1.4 at 2048, e^-1.4 = 25 % stuck). Two further full proofs at 2048 bits
(`stats 2048 2`) were both Unresolved, so 8 of 18 in all. At 1536 bits, 3 of 3 were proven, but at a mean
of 29.5 s in release, with 11 failed inner subtrees ("level fail bits 1383 d_scanned 1420" and so on).
Each failure is a full scan of the 1420 rows. The table is in the addendum.

Why it is a blocker:
- The task asks to keep `is_prime` fast for the existing callers and to state which numbers are proven
  within what time. At 2048 bits the design's `is_prime` panics on more than a third of primes, and it
  costs a full ECPP proof per new prime, against 240 Miller-Rabin rounds at HEAD. Callers that generate
  or check primes of this size exist: `rustmath-crypto/src/prng_number.rs` (`rsa_modulus(b)` with
  b = 4096 needs two 2048-bit primes), the finite-field and Galois-ring constructors, the REPL.
- Bulk functions multiply the risk. `prime_range` above 2^64 and `random_prime` (which calls
  `prime_range`) prove every prime in the range. A range with 100 primes of 1024 bits panics with
  probability 1 - 0.97^100 = 95 %.
- The design calls run-time class polynomials "feasible but a project of its own" (section 7). By the
  task's rule that is in scope unless shown infeasible. The design does not show that.

Required change. Measure and state the Unresolved rate and the mean and maximum time (release and debug)
at 1536 and 2048 bits, and say what happens at 4096 bits. Then do one of the following:
- (a) Close the gap with a feasible algorithm: class polynomials computed at run time for larger h
  ([AM93] §7, Enge 2009; fixed-point complex arithmetic on BigInt, so no new numeric wrapper), or a
  complete fallback for an unresolved top level. APR-CL (Cohen-Lenstra 1984; Cohen GTM 138 §9.1) decides
  every n, but gives no certificate, so `Primality` needs a variant for it.
- (b) Keep the static table, but justify why (a) is infeasible in this unit, and change the policy so
  that no existing caller gets a new panic at the sizes where Unresolved is common. Examples: prime
  generators skip unresolved candidates (B2); bulk enumerators prove only what they must return and
  document the residual; and `is_prime`'s doc and the HANDOFF state the measured rates per size.

Whichever option is chosen, correct the summary and section 3 ("can panic for an unresolved prime (none
below 512 bits)", "rare above 512 bits") to the measured numbers.

### B2. Result-returning callers acquire a panic path; the design's own rule (section 2.4) is not applied to them

Section 2.4 says: "Callers that return Result use `try_is_prime` (section 3 lists which are switched now
and which are handed to their units)". Section 3 marks the following callers "unchanged". All of them
return `Result`, and no unit design switches them (I grepped F1's design: it covers
`check_prime_characteristic`, `Gfpn`, the generators and `PrimeField::sqrt`, but not these):
- `rustmath-finitefields/src/finite_field.rs:147` (`FiniteField::new`), `:185` (`with_modulus`);
  `galois_ring.rs:78` (`GaloisRing::new`), `:106` (`with_modulus`). All four return
  `Err(InvalidArgument)` for a composite p today. After I1 they panic for an unresolved p instead of
  returning `Err(NotSupported)`.
- `rustmath-crypto/src/prng_number.rs:78`, `:97` (`random_prime_bits`, `random_prime_3mod4_bits`), used by
  `rsa_modulus`, `rsa_modulus_with_exponent`, `random_sequence_rsa`, `bbs_modulus` and `blum_blum_shub`,
  all `Result`. Section 3 sizes this caller as "20 bits in tests", but the public `b` is unbounded and
  2048- or 4096-bit moduli are the normal use. No unit owns rustmath-crypto in this run, so I1 must fix
  it. The natural honest fix: a random-prime search skips a candidate whose proof is unresolved and goes
  on scanning. It needs some proven prime, not a particular one. (MAGMA `RSAModulus` uses `RandomPrime`,
  `Proof := true`, ch. 158 digest line 54, so keeping a proof is right. The skip only shifts the
  distribution slightly, so say so in the doc.)
- `rustmath-jupyter/src/repl.rs:10964` (`sqrt_mod_report`, which returns `Result`) keeps `is_prime(n)`.
  The REPL has no `catch_unwind` (grep), so a panic ends the session. The REPL `next_prime` (`:2680`) and
  `primes` (`:2718`) paths reach `is_prime` too. Map them to `EvalError` with `try_is_prime`, as the design
  already does for the `is_prime` command.
- `rustmath-integers/src/prime.rs` `random_prime(a, b)` returns `Option`. The same skip rule applies (see
  m6).

Required change: list every `Result`- or `Option`-returning caller with its decision: switched now, or
handed to a named unit with the exact site. Switch the crypto generators and the REPL paths in this unit.
Hand the four finite-field constructors to F1 explicitly, or switch them here; they are four one-line
`match try_is_prime(&p)` edits.

### B3. `pollard_p_minus_1` would silently truncate its bound

Section 2.5: "`pollard_p_minus_1` (`:650-653`): iterate `small_primes()` instead of testing every
p <= bound". `small_primes()` holds the primes below 2^16, while `bound` is a `u32`. For bound > 65536,
stage 1 would silently stop at 65521, and the function would return `None` where HEAD finds a factor.
That is a silent truncation, which the rules forbid. Required change: iterate the primes up to `bound`
(`sieve::for_each_prime(2, bound as u64 + 1, ..)` or `primes_up_to(bound)`). Add a test with a bound
above 2^16 whose factor needs a prime above 2^16. Example: p with p - 1 = 2^a · 65537 · (small), and the
gp check `factor(p-1)`.

## Minor items

- m1. Section 2.1 says no `PrimalityCertificate` exists in the workspace. In fact `rustmath-constants`
  exports `pub trait PrimalityCertificate` (`certificates.rs:16`, `lib.rs:17`). Nothing collides today,
  since no file glob-imports both crates (grep). Still, state the clash. Either name the new type
  differently (for example `EcppCertificate`, with `PrimalityCertificate` as MAGMA's name in the doc), or
  note that importers must alias it.
- m2. Test `sieve_matches_is_prime_u64` uses `primes_in_range(lo, lo + 10^4)` with lo = 2^64 - 10^4. The
  value lo + 10^4 = 2^64 does not fit in a `u64`. Use `primes_in_interval(lo, u64::MAX)` for that case.
  More generally, specify overflow-free segment arithmetic near 2^64: the prototype's `start + 2*SEG` and
  `p * p` overflow there.
- m3. Test `ecpp_rejects_tampered_certificates`, "a step removed": if the removed step is the first one,
  the rest is a valid certificate for q_1. PARI says 1, and `from_pari` would set n = q_1, so `verify` is
  true. Remove a middle step, or keep n = N with the steps from 2 on (then `st.n != cur` gives false).
- m4. Section 2.8 changes the `debug_assert` of `split_odd_composite` (`prime.rs:871`) to
  `!is_probable_prime(m)`. The design's own path for a BPSW pseudoprime that ECPP refutes reaches
  `split_odd_composite` with exactly such an m: the loop guard `is_prime` is false, but BPSW passes. The
  assert would then fire in debug builds. Use `!is_prime(m)` (cheap for an ordinary composite), or assert
  only that m is odd and > 1. In the same case, the early `None` of `pollard_rho` via `is_probable_prime`
  sends the cofactor straight to the Brent and trial-division fallback. That still terminates; say so in
  the doc.
- m5. `try_sqrt_mod_prime` (section 2.7) promises `Err` when "the computation proved p composite" or "a
  bounded search ran out". For a composite perfect square p ≡ 1 (mod 4), the least-non-residue loop in
  `sqrt_mod_prime_checked` (`integer.rs:1180-1186`) has bound p, because every Jacobi symbol is 0 or 1.
  So for a large square p it never returns. All callers the design and F1 name test p first, so they are
  safe. Either add an `is_perfect_square` check that returns `Err`, or state the precondition (p prime)
  and the non-termination in the doc.
- m6. `random_prime(a, b)` keeps enumerating with `prime_range`, which now proves every prime in the
  range. Above psi_12 this is slower than HEAD and multiplies the Unresolved risk (B1). Enumerate with
  `is_probable_prime`, pick one, and prove only the pick (skip an unresolved pick).
- m7. The item names `square_root_mod_prime`, and the fix covers it. But `square_root_mod_prime_power`
  and `all_square_roots_mod_prime_power` still test twice: once in `checked_prime_power_args` (`:938`),
  then again in `square_root_mod_prime` via `prime_power_sqrts` (`:815`). The cache makes the second test
  cheap above psi_12. Below psi_12 it is still 12 strong tests. An unchecked internal variant
  (`try_sqrt_mod_prime`) removes it. Optional, but mention it.
- m8. `bpsw` (section 2.2) does not check gcd(n, Q). `NotPrimeWitness::StrongLucas` requires
  gcd(n, 2DQ) = 1. That is unreachable for n without prime factors below 1000 unless |D| > about 4000, but
  the witness contract needs the check: return `Divisor(gcd)`.
- m9. Section 6's risks say "do not prove 512-bit numbers in tests", but the test
  `is_prime_mersenne_and_neighbours` asserts `is_prime(2^521 - 1)`, a 25-level proof of about 4 s in debug.
  Resolve the contradiction: keep it and count it in the test-time estimate, or test 2^521 - 1 only with
  `is_probable_prime`.
- m10. Embedding the 1.3 MB table with `include_str!` puts it into every binary that links
  `rustmath-integers` and reaches `is_prime`. That is most of the 228 test executables in
  `target/debug/deps`, about 300 MB more on a disk with 8.6 GB free. State the cost. Options: the h <= 16
  table (574 KB) with an Unresolved-rate measurement, or a compact encoding (the gamma_2 and Weber
  coefficients are small).
- m11. `prove_level` recursion depth is the number of levels: 56 at 1024 bits, more than 100 at 2048.
  Each debug frame holds several BigInts and the row loop state. Test threads have 2 MB stacks. Specify an
  iterative driver (an explicit stack of (N, row index, candidate index)), or bound the depth and
  measure it.
- m12. `try_is_prime` trusts the prover's steps. Calling `cert.verify()` before caching costs about a
  fifth of the proof (section 2.3.7) and guards against prover bugs. At least do it in debug builds and in
  the tests (the roundtrip test does it for four numbers only).
- m13. The designer's gp scripts fail as written on a default PARI stack: `$X/t15_certs.gp` stops with
  "the PARI stack overflows" in `primecert(nextprime(2^200))`, and so does `polclass(-23,1)`. The
  committed `data/ecpp_class_polynomials.gp` and every gp command quoted in test comments that needs more
  stack must start with `default(parisizemax, 2^30)`.
- m14. Section 2.1's API table: `prove_primality` for 2^64 <= n < psi_12 can return `Unresolved` for a
  number the strong tests have proven prime (the design says so in 2.3.4). Since `Primality::Prime` needs
  an ECPP certificate, that is consistent. But document on `prove_primality` that `try_is_prime` and
  `is_prime` decide such n without ECPP.
- m15. Where the PARI test vectors (`$X/c127.txt`, `c40.txt`, `c65.txt`, `c200.txt`) go is not stated. Put
  them in the test file as string constants (c200 is 1.1 KB) with the gp command that made each, and let
  `from_pari` accept line breaks and spaces (gp's pretty printer wraps long vectors).

## Items covered (summary)

| Item | Covered | Note |
|---|---|---|
| A (proving is_prime) | yes, with a static-table ECPP | B1: coverage and policy above 1024 bits |
| S6 | yes | B3 (p - 1 bound), m2, m6 |
| is_prime_u64 dedup | yes | overlaps I3 (`ArithInt::is_prime`), handled |
| square_root_mod_prime twice | yes | m5, m7 |
| drivers and I2 interface | yes, consistent with I2 | m4 |

## Addendum: measured ECPP cost at 1536 and 2048 bits (release prototype, h <= 20)

`$X/bin/ecpp_rs_rel $X/table_h20.txt stats B n` (release, no genus filter; logs `$R/stats1536.*`,
`$R/stats2048.*`, `$R/count2048.log`):

| Bits | Samples | Proven | Release time | Levels | Failed inner subtrees |
|---|---|---|---|---|---|
| 1536 | 3 | 3 | mean 29.5 s, max 34.3 s | 90 | 11 in total |
| 2048 | 2 | 0 (both "level fail bits 2048 d_scanned 1420") | 6.8 s to give up | 0 | |
| 2048 (count mode) | 16 | 10 with at least one top-level candidate | | | |

So 8 of 18 random 2048-bit primes cannot be proven with the static table. A 1536-bit proof takes about
30 s in release, so several minutes in debug, against well under a second for HEAD's 240 Miller-Rabin
rounds. The 11 failed subtrees in 3 proofs at 1536 bits also show that `MAX_FAILED_SUBTREES = 64` is not
"never reached" at these sizes.
