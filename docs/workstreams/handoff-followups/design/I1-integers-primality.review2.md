# Review 2 of design I1 (integers, primality)

Design: `I1-integers-primality.md` (1301 lines, with the "Revision 1" section that answers review 1). Tree at
HEAD `c79a582`, clean. Reviewer scratch: `I1-integers-primality/review2/` (called `$V` below). I reproduced the
HEAD claims with my own programs (`$V/h1.rs`, `$V/h2.rs`, `$V/h3.rs`). They link against the HEAD debug rlibs
(`librustmath_integers-a1ad7d0876b91f69`, `librustmath_finitefields-b7ae0c869da07454`). I took the reference
values from my own gp scripts (`$V/g1.gp` .. `$V/g6.gp`, PARI/GP 2.15.4). I re-ran the designer's prototype
(`rev1/bin/ecpp_rt_{dbg,rel}`) for a sample of the cost claims.

Verdict: **approved**. I found no blocker. Review 1's three blockers are resolved (details below). Twelve minor
items follow. Each is a clarification, a test that should be stronger, or a plan detail. None changes the
algorithms or the API.

## What I checked and found correct

### HEAD defects (reproduced)

- `is_prime(psi_12)` and `is_prime(psi_13)` both return true. `factor_pollard_rho(psi_12)` returns
  `[(psi_12, 1)]` in 20 ms.
- `next_prime(0)`, `next_prime(1)` and `next_prime(-5)` all return 3.
- `is_prime_miller_rabin(3215031751, 4)` and `is_prime_miller_rabin(3825123056546413051, 11)` are false, because
  all 12 bases run. `is_prime(2^127-1)` takes 19-20 ms; one round takes 1.0 ms (the design says 30 ms / 1.6 ms).
- `nth_prime(3000)` takes 2.6 s (consistent with the design's 11 s for 10^4).
- `ArithInt::is_prime(2147483647)` panics with "attempt to multiply with overflow" (`fast_arith.rs:246`).
  `ArithLLong::is_prime(i64::MAX)` is false.
- `FiniteField::new(psi_12, 1)` and `GaloisRing::new(psi_12, 2, 1)` are both `Ok` at HEAD, so the design's new
  finite-field test fails at HEAD as claimed.
- `square_root_mod_prime`: `assert_prime` (`integer_mod.rs:652`, `:877-883`), then `is_prime` again inside
  `Integer::sqrt_mod_prime` (`integer.rs:883`), plus `unit_sqrts_odd` (`:815`) on the prime-power path.
  Confirmed.
- No type `FastArith64` exists. A grep finds only `HANDOFF.md:330`.
- Every line reference I sampled holds at HEAD: `prime.rs:556`, `:641`, `:651`, `:829`, `:833`, `:871`;
  `quadratic_sieve.rs:87`, `:182`, `:450`, `:572`, `:679`; `ecm.rs:441`, `:523`, `:534`; `integer.rs:701`, `:832`,
  `:880`, `:906`, `:976`, `:999`, `:1154`; `repl.rs:914`, `:2680`, `:2696`, `:2710`, `:2725`, `:10964`.

### Reference values (gp, `$V/g1.gp`, `$V/g3.gp`, `$V/g4.gp`, `$V/g6.gp`)

Every value of section 4 that I sampled matches gp:
- psi_12 and psi_13 factor as stated. Both have Selfridge D = -7. Both pass the base-2 strong test and fail my
  independent gp strong Lucas test, so the witness `StrongLucas(-7)` is right.
- The strong Lucas pseudoprimes below 20000 are [5459, 5777, 10877, 16109, 18971]. There are 46 base-2
  strong pseudoprimes below 10^6 (first 2047, last 983401).
- 1000036000099 has 2 as a strong witness. 1032247 = 1013 · 1019, gcd with (1 - 4053)/4 is 1013, and
  (4053/1032247) = -1.
- `nextprime(2^k) - 2^k` = 13, 13, 277, 51, 7, 235. For every k, `nextprime(2^k)+2` and `2^k+1` are composite.
  `precprime(10^30) = 10^30 - 11`.
- The base patterns of 3215031751 and 3825123056546413051 are as stated (the latter passes 2..31 and fails 37).
  psi_12 passes all 12 bases.
- H_{-15} and H_{-23} match. deg H_{-71} = 7. For D = -455, -58507, -1271, -7991: h, largest coefficient bits
  (471, 1434, 1039, 3336) and H_D(12345) mod 2^61-1 all match.
- `qfbsolve` for D = -23 gives u = -1967590, v = 74772. The roots of H_{-23} mod 1000000000933 match.
- The safe primes, the repunits [2, 19, 23, 317], the next/previous prime values, the 10^30 window
  {57, 99, 211, 231, 271}, the three `primes([..])` windows, `prime(10^5)`, `prime(10^6)`, `primepi(10^6)` and
  `primepi(10^7)` all match.
- The p - 1 test is right: the stage-1 gcd is 917519 for B = 70000 and 1 for B = 65535. The p - 1 factorizations
  are as stated.
- The `ispower` values, the `fast_arith` values (`factor(2147395601) = 13 · 165184277`, `isprime(2^63-25)`),
  the `iroot4` edges at 10^20 and `kronecker(41, psi_12) = -1` all match.
- The discriminant list `rev1/dlist100.txt` has 273,422 bytes, the stated sha256, 42,272 entries, counts
  [9, 18, 16, 54, 25, 51, 31, 131, 34, 87, ...] and 1736 for h = 100. The h <= 20 part has 1817 rows (9,302
  bytes, max |D| 58507). Each line is ascending and the max |D| is 2,383,747. These are Watkins' numbers.
- PARI certificates: C127, C40, C65 and C200 are valid. The tampered C200 variants are invalid (s doubled, x + 1,
  t + 2, N + 2, step 3 removed). `c[2..6]` is valid with N = q_1, as the revised test expects.
- All four PARI certificates also pass the design's stricter checks: gcd(N, 6) = 1, a unit discriminant, and
  q > (ceil(N^(1/4)) + 1)^2. So the verifier accepts them.
- I compared one more Hilbert polynomial, D = -1856563 (h = 100, 9215-bit coefficients), between the prototype
  and `polclass`. H(12345) mod 2^61-1 is 1901883014239404625 in both.

### Algorithms (re-derived)

- **BPSW.** The Lucas doubling and increment formulas (P = 1) and the strong condition are right. My gp
  implementation of the same steps reproduces OEIS A217255. Selfridge's D values are all 1 mod 4. The gcd(n, Q)
  check (review 1, m8) is in, with the "cannot trigger for |D| < 4000" argument, which is correct.
- **Goldwasser-Kilian theorem (2.3.1).** Right. PARI's documented theorem (`??primecert`) is the same up to
  wording. The design's per-prime form is the precise one.
- **Jacobian order check (2.3.2).** The formulas are dbl-1998-cmo-2 (Z3 = 2 Y1 Z1) and madd-2004-hmv
  (Z3 = Z1 H, H = x2 Z1^2 - X1). The three facts hold over F_p, p >= 5. gcd(N, 6) = 1 makes every p | N at least
  5. The exceptional cases (Y1 = 0; H = 0, which covers T = +-P2) are exactly the cases that give Z3 = 0, and
  Z = 0 is absorbing. So a final Z that is a unit mod N certifies every step mod every p | N. U = -T then gives
  [q]T_p = O.
- **Completeness for prime N.** q is odd, so the last bit of q - 1 is 0. Every addition follows a doubling of
  a prefix, so the accumulator is jT with 2 <= j < q - 1. q does not divide s, because s only has prime factors
  below 2^16 and q is the cofactor.
- **Verifier (2.3.3).** It is sound. The checks t^2 < 4N and s | m are not needed for soundness, only for PARI
  compatibility, which is fine. The verifier is independent of the list, H_D and the randomness.
- **Prover details.** Cornacchia (Cohen 1.5.3) is right. The traces for D = -3 and D = -4 are right. The genus
  filter keeps every row that can succeed for a prime N. The curve (3jc, 2jc^2) has j-invariant j, and with its
  twist it covers both twists. For D = -4, a non-square g generates the quartic classes; for D = -3, a non-square
  non-cube g generates the sextic classes.
- **Hilbert class polynomials.** Weber's formula j = (256 g + 1)^3 / g with g = q R^24 is correct
  (f_2^24 = 2^12 Delta(2 tau)/Delta(tau)). The bound log2 prod(1 + |j_i|) <= sum(pi sqrt|D|/(a ln 2) + 3.5) is
  correct. Every q_i keeps at least `guard` significant bits, since |q_i| >= 2^-est. So the 744 term
  (-24 + 768) is resolved even when |w| is huge.
- **Iterative driver (2.3.4).** The invariant steps.len() = stack.len() - 1 holds. Composite and Exhausted pop
  the child and its step, and the parent continues. `Prime` is only ever returned after `verify`.
- **Decided / Unresolved.** `Primality::{Prime(cert), NotPrime(witness), Unresolved}` and
  `try_is_prime -> Result<bool>` separate the three outcomes. Every bounded search (the row scan, 64 failed
  subtrees, 64 root-split attempts, g up to 2^20, 64 witness bases) ends in Unresolved or "next candidate",
  never in a decision.
- **MAGMA semantics.** The digest says: `IsPrime` has `Proof := true` and uses ECPP [AM93] for larger n;
  `IsProbablePrime`; `PrimalityCertificate` / `IsPrimeCertificate`; `IsPrimePower` with p proven; `NextPrime` is
  the least prime > n; `NthPrime` goes to 10^10. The design reads these lines (digest 290-313) correctly.

### Callers (my own grep)

I grepped `prime::is_prime`, `.is_prime()`, `next_prime`, `previous_prime`, `random_prime`, `prime_range`,
`nth_prime`, `prime_pi`, `is_prime_power`, `is_prime_miller_rabin`, `fast_arith::prime_range`, the residue
functions, the `numbertheory` re-export and the glob imports. The result is the same set as sections 3.1
and 3.2.
- No new name collides: no crate glob-imports `rustmath_numbertheory::*`, and only numbertheory glob-imports
  `prime::*`.
- `is_prime_miller_rabin` has no caller outside `prime.rs`.
- No test pins `next_prime(0)` or `next_prime(1)`.
- No test in the workspace runs `is_prime` on a prime above about 180 bits, except the new ones. So the debug
  suites do not get a slow proof.
- Every `Result`- or `Option`-returning caller has a decision, as review 1, B2 asked.
- `data/ecpp_discriminants.txt` is not matched by `.gitignore`.

### Cost (sampled with the prototype)

| Case | Design says | Measured here |
|---|---|---|
| 256-bit primes, debug | 0.36 s mean, 0.52 s max | 0.42 s mean, 0.48 s max (4 primes) |
| 1024-bit primes, release | 3.3 s mean | 2.76 s mean (3 primes) |
| 2^521 - 1, debug | about 4 s | 4.4 s to prove, plus 0.7 s to verify |
| H_{-7991}, debug | 2.1 s | 2.27 s |
| Sieve prototype, debug | `prime_count(10^8)` 0.75 s, `nth_prime(10^7)` 1.5 s | 0.69 s and 1.31 s |
| `factor_pollard_rho`, QS driver, zero-curve ECM driver on a semiprime of psi_12's size (399165290221 · 798330580517, not a pseudoprime), HEAD debug | about 5 s each | 4.65 s, 4.67 s, 4.70 s |

PARI accepts my debug run's 25-level certificate of 2^521 - 1 (`$V/g2.gp`) and rejects it with x + 1 in the last
step.

### Review 1's blockers

- **B1 (coverage).** Resolved by run-time H_D up to h = 100. The coverage and timing tables are measured and
  consistent with the logs: no unresolved prime up to 2048 bits, 65 failed subtrees in 12 proofs at 2048 bits,
  at most 10 in one proof against the limit of 64. The design states the residual above 2048 bits honestly. It
  treats faster ECPP for very large inputs as a limit of the chosen algorithm, not as an item of the list. I
  accept that: the item asks to state what is proven within what time, and 2.3.7 does.
- **B2 (callers).** Resolved (section 3.1).
- **B3 (p - 1 truncation).** Resolved: the sieve runs to `bound`, and there is a test above 2^16.

## Minor items

**n1. `bpsw`'s precondition and its test do not match.** Section 2.2 defines `bpsw(n)` for "odd n > 1 with no
prime factor below 1000". The test `bpsw_rejects_strong_base2_pseudoprimes` calls it on the 46 base-2 strong
pseudoprimes below 10^6. Every composite below 10^6 has a prime factor below 1000, so all 46 inputs are outside
the contract, and the test does not say which witness to expect.

Required change:
- State that `bpsw` is correct for every odd n > 1. The 1000 bound only matters for the "gcd(n, Q) cannot
  trigger" remark.
- Add base-2 strong pseudoprimes with no factor below 1000 and require `StrongLucas(_)` for them. Examples
  (`$V/g5.gp`): (k+1)(2k+1) with both factors above 1000, namely 2284453, 5489641, 8725753, 9863461; Chernick
  numbers 27278026129 and 65700513721 (least factors 1657 and 2221).
- In `strong_lucas_pseudoprimes_below_20000`, state the enumeration: odd non-square composites, Selfridge D,
  skipping n with (D/n) = 0 or gcd(n, Q) > 1. That is A217255's definition, and with it my gp gives exactly the
  listed five numbers.

**n2. Composite verdicts inside the prover are underspecified (2.3.6, 2.3.4).**
- "Cornacchia inconsistency" is ambiguous. For a prime N, a row whose Cornacchia "require" fails (|D| does not
  divide 4N - b^2, or c is not a square) is a normal miss: N is not represented by the principal form. If an
  implementer took that for a contradiction, a prime N would run a 64-base witness search and could end
  `Unresolved` at the first such row.
- `advance` has three outcomes (Found, Composite, Exhausted), but 2.3.6 can also produce
  "contradiction, no witness found", and the driver does not say what happens then.

Required change: say that a failed Cornacchia "require" skips the row. Only an `Err` from a square root (or a
proper gcd) is a contradiction. Give `advance` a fourth outcome, or map that case: at depth 0 return
`Unresolved(..)`; at depth > 0 pop the frame like `Composite`.

**n3. `prove_primality` is used where `try_is_prime` fits better.** Two places:
- `integer_constant_is_irreducible` (2.9).
- The I2 interface (2.8, "Certifying each final factor p: `prime::prove_primality(&p)`").

`prove_primality` has no cache. As the design itself says (2.1, m14), it can also be `Unresolved` for
2^64 <= p < psi_12 although the 12 strong tests prove p prime. So a proven prime could come back as `Err`, or as
an "unproven probable prime" factor. I2's design in fact uses `try_is_prime` (I2 lines 739-740, 970).

Required change: use `try_is_prime` in `integer_constant_is_irreducible`: Ok(b) gives Ok(b), Err gives
NotSupported. In 2.8, say that I2 certifies with `try_is_prime`, and uses `prove_primality` only when it wants
to keep the certificate, and then not for p < psi_12.

**n4. The acceptance commands miss REPL tests that I1's edits can break.**
`cargo test -p rustmath-jupyter sqrt_mod` and `cargo test -p rustmath-jupyter prime` do not match:
- `test_int_arg_ranges_named` (`repl.rs:12135`), which pins the `nth_prime` argument messages that I1 changes
  (the new ValueError above pi(2^64));
- `test_combinatorics_reference_values` (`:12052`, `nth_prime(100)`).

Required change: run `cargo test -p rustmath-jupyter --no-fail-fast` in full, or name these two tests.

**n5. The REPL `is_prime` and `next_prime` arms have no size bound.** Section 2.3.7 says a caller that needs a
fast answer "bounds the size itself, as the REPL does (J1a)". J1a's design bounds `primes` (`proof_bits`),
`prime_pi` and `sqrt_mod`, but not `is_prime` or `next_prime`. After I1 these two arms run an unbounded ECPP. A
4096-bit prime takes about 1.5 h in release and a day in debug, and the REPL has no `catch_unwind` or time
limit.

Required change: correct the sentence. Then either add a proof-size limit to those two arms in I1 (a ValueError
that names `is_probable_prime` / a probable-prime command), or hand the limit to J1a explicitly, with the sites
`repl.rs:914` and `:2680`.

**n6. `ecpp_driver_backtracks` cannot fail on a driver that never backtracks.** Its expected result is "Prime
... or Unresolved". A driver that gives up at the first failed child would pass. The prover is deterministic,
because it is seeded from N.

Required change: expose a failed-subtree counter (a test-only field, or a statistics struct returned by
`prove_chain`). Pin the measured outcome: `Prime`, a verified certificate, and at least one failed subtree with
H_INNER = 1. If nextprime(2^200) has no failing inner level under these parameters, pick an input that does.

**n7. No test reaches the `Err` of `try_is_prime` or the panic of `is_prime`.** Both are one-line mappings.
They are still the only paths that turn `Unresolved` into an answer, and the REPL, crypto, rings and
finite-field code rely on them.

Required change: add a crate-internal `try_is_prime_with(n, &ProverParams)` (or a `#[cfg(test)]` override of
the defaults). Test that H_TOP = 0 on 2^127 - 1 gives `Err(NotSupported)` with the documented message, and that
the cache is not filled.

**n8. `NotPrimeWitness::confirms` must validate the witness data.** For `StrongLucas(D)`, Q = (1 - D)/4 must be
an integer. The documented conditions ((D/n) = -1, gcd(n, 2DQ) = 1) do not say D = 1 (mod 4). So a hand-built
`StrongLucas(-8)` must give `false`, not a test with a truncated Q. Required change: add "D = 1 (mod 4)" to the
variant's contract and to `confirms`, plus a test that `StrongLucas(-8).confirms(n)` is false.

**n9. The existing arbitrary-precision complex arithmetic is not discussed.** CLAUDE.md asks to consume
existing numeric crates. The workspace already ships `rug` with MPC (`rustmath-complex/Cargo.toml:17`;
`rustmath-rationals` and `rustmath-reals` use rug too). The design writes private fixed-point complex
routines on BigInt: Machin's pi, a Taylor exp, products and quotients. Keeping the L0 crate free of GMP/MPFR is
a defensible reason, and soundness does not depend on H_D. Required change: say so in 2.3.5 and in the
`hilbert.rs` header, so the next reader does not take it for an overlooked dependency.

**n10. The I3 overlap is larger than section 2.6 says.** I3 is implemented first, and its design already makes
`ArithInt::is_prime` and `ArithLLong::is_prime` call `is_prime_u64` and deletes `ArithLLong::miller_rabin`
(I3 design §2, lines 221-262). It also renames `ch18_signs_roots.rs`'s `fast_arith_miller_rabin_after_pow_mod_change`
to `fast_arith_llong_is_prime_mersenne_61`, with a comment that names `fast_arith.rs:70`, and adds
`fast_arith_int_is_prime_near_i32_max`, which overlaps I1's `arith_int_is_prime_no_overflow`.

Required change: in 2.6 and in Plan step 3, say that after I3 only these remain:
- move `is_prime_u64` to `primality.rs`;
- update its imports (`fast_arith`, and `integer.rs` `may_be_qth_power`);
- update the renamed test's comment to `primality.rs`;
- drop or merge the duplicate overflow test.

**n11. The docs of the residue functions overstate self-verification.** Section 2.7 says the answers of
`legendre_symbol`, `is_quadratic_residue_prime`, `sqrt_mod_prime` and `quadratic_residues` "already verify
themselves". That holds for a returned root and for a Jacobi symbol of -1. It does not hold for
`legendre_symbol` returning 1, because for a composite modulus the value is only the Jacobi symbol.

Required change: keep HEAD's sentence ("for a pseudoprime the value is the Jacobi symbol"), reworded for BPSW,
in the `legendre_symbol` doc.

**n12. The 2^521 - 1 test costs more than stated.** With the mandatory `verify` (m12), the proof costs about
5.1 s in debug, not 4 s (measured above). Adjust section 5's estimate. No other change is needed.

## Items covered (summary)

| Item | Covered | Note |
|---|---|---|
| A: proving `is_prime` | yes: BPSW plus ECPP with a verified PARI-format certificate, three-valued API, `is_prime` a proof (MAGMA `Proof := true`) | n2, n3, n6, n7, n8, n9; the residual above 2048 bits is measured and honest |
| S6: bases repeated; `nth_prime` / `prime_pi` per candidate | yes: k distinct bases, segmented sieve, sieve-backed p - 1 | n1 (tests only) |
| `is_prime_u64` dedup | yes | n10 (I3 already does part of it) |
| `square_root_mod_prime` tests twice | yes (`try_sqrt_mod_prime`, one test; also the prime-power path) | n11 |
| Drivers and the I2 interface | yes: docs rewritten, psi_12 split, early returns use BPSW | n3 |

Nothing is declared out of scope except faster ECPP above 2048 bits. The design measures that as the limit of
the chosen algorithm, and the interim behaviour (`Unresolved`, `Err`, or a documented panic) is honest.
