# Review 1 of the implementation of I1 (integers, primality)

Tree: HEAD `80130e7` (I3) plus the uncommitted I1 changes (32 modified files, 5 new paths). Design:
`fu2/design/I1-integers-primality.md` (Revision 1 body) with review 2's minor notes. Report:
`fu2/impl/I1-integers-primality.md`. Scratch: `fu2/impl/I1-integers-primality.rev1/` (`$S` below).

Verdict: **approved**. No blocker. Four minor notes, all cosmetic or docs. Tests pass.

## What I ran

| Command | Result |
|---|---|
| `cargo test -p rustmath-integers --no-fail-fast` | 349 passed, 0 failed, 5 ignored (`$S/t_int.log`) |
| `cargo test -p rustmath-finitefields --no-fail-fast` | 262 / 0 / 0 |
| `cargo test -p rustmath-polynomials --no-fail-fast` | 377 / 0 / 4 |
| `cargo test -p rustmath-crypto -p rustmath-constants --no-fail-fast` | 198 / 0 / 0 |
| `cargo test -p rustmath-rings --no-fail-fast` | 1914 / 0 / 82 |
| `cargo test -p rustmath-jupyter --no-fail-fast` (full run, review 2 n4) | 64 / 0 / 0 |
| `cargo test -p rustmath-numbertheory -p rustmath-ellipticcurves -p rustmath-coding -p rustmath-quadraticforms -p rustmath-curves --no-fail-fast` | 660 / 0 / 13 |
| `cargo check --workspace --tests` | exit 0. The touched crates have no warning in new code. |
| `cargo clippy -p rustmath-integers --tests` | exit 0. All 16 warnings are in old code (git blame on `prime.rs:1413` and others). |
| `rustfmt --check` on the new files | clean |
| `cargo test --release ... -- --ignored ecpp_1024_release` | 2^1023 + 3^640 + 14: 54 levels in 3.29 s (design: 3.3 s mean). I deleted `target/release` (80 MB) afterwards. |

All counts match the report. `git status` shows no `rustmath-misc/output.txt` and no `.tmp_dir_*`.

## Independent checks

**PARI accepts the prover's certificates.** I ran `print_certificates_for_pari`, then
`primecertisvalid` in gp 2.15.4 after `default(parisizemax, 2^30)` (`$S/gp/v3.gp`). The certificates
of 2^127 - 1 (3 levels), nextprime(10^40) (4), nextprime(2^200) (6) and the 256-bit prime (12) are
all valid. With x + 1 in step 1, PARI rejects all four. With x + 1 in the last step, PARI rejects
three. For nextprime(2^200) it stays valid, as the report says: the last curve has a = 0, so the
moved point lies on another sextic twist of the same order, which is still a valid step. PARI's
format for a small prime is the bare integer (`primecert(7)` = 7), so `to_pari` matches.

**BPSW and the prover against PARI on 1090 numbers.** `$S/gp/gen.gp` (seeded) writes 1090 numbers
with `ispseudoprime`:
- 400 random primes from 2^79 to 2^679;
- 400 semiprimes with factors of 40 to 340 bits;
- 100 squares of primes;
- 100 random odd numbers;
- 90 base-2 strong pseudoprimes above psi_12, of the forms p(2p - 1) with p about 2^40 and
  p(4p - 3) with p about 2^60.

`$S/bpswcheck.rs` (linked against the debug rlib) compares `is_probable_prime` with PARI: 0
mismatches. `prove_primality` gave `NotPrime` for all 690 composites, every witness `confirms`, and
`try_is_prime` is `Ok(false)` for all of them. It proved all 144 primes below 260 bits with a
certificate that `verify` accepts, for the right `n()`. Nothing came back `Unresolved`.

**Reference values.** I recomputed with gp (`$S/gp/v1.gp`, `v2.gp`, `v4.gp`, `v5.gp`) the values the
implementation added beyond the design. All match:
- psi_12 passes the bases 2..37 and fails 41. Selfridge's D is -7 for psi_12 and psi_13.
- The six base-2 strong pseudoprimes without a factor below 1000 have `ispseudoprime` = 0.
- nextprime(10^45) = 10^45 + 9, and `primecertisvalid` of its certificate is 1.
- nextprime(2^200 + 236) = 2^200 + 697. nextprime(2^127 + 10^6) = 2^127 + 1000121, and
  2^127 + 1000123 is composite.
- 2147395601 = 13 · 165184277. 2^63 - 25 is prime. (4053/1032247) = -1.
- `#primes([10^6, 13·10^5])` = 21523. 2^1279 - 1 is a probable prime, and 17 divides 2^1100 + 1.
- `primes([2^64-100, 2^64+39])`. ellcard = 100 and ellorder([3,6]) = 5 on [2,3]/F_97.
  `Vecrev(polclass(-71))`. `qfbsolve(Qfb(1,1,1), 2^127-1)`.
- The base patterns of 3215031751 and 3825123056546413051. 1000036000099 has 2 as a witness.
- The 10^30 window.

**Boundaries** (`$S/edges.rs`), all correct (gp):
- next_prime(L - 1) = L, next_prime(L) = next_prime(2^64 - 1) = 2^64 + 13, with
  L = 18446744073709551557;
- previous_prime(L) = ..533. previous_prime at L + 1, 2^64 + 1 and 2^64 + 13 is L.
  previous_prime(2^64 + 14) = 2^64 + 13;
- prime_range(L, 2^64 + 14) = [L, 2^64 + 13], and prime_range(-10, 10) = [2, 3, 5, 7];
- random_prime(L, 2^64 + 14) gives both primes, and the empty window gives Ok(None);
- prime_pi(2^32) = 203280221, nth_prime_u64(203280221) = 4294967291, and the primes in
  [2^64 - 201, 2^64 - 1] are right;
- prime_power((2^89-1)^2) = (2^89-1, 2), prime_power(4) = (2, 2), and prime_power(-8) = None.

**Cost** (`$S/timing2.rs`, debug, idle machine). Means over random-looking primes of the stated size
(`$S/timing2.rs`):

| Bits | Primes | Mean | Max | Design claim (mean, max) |
|---|---|---|---|---|
| 128 | 6 | 0.059 s | 0.105 s | 0.065 s, 0.094 s |
| 256 | 6 | 0.44 s | 0.56 s | 0.36 s, 0.52 s |
| 512 | 3 | 2.5 s | 3.2 s | 3.2 s, 4.5 s |

Other measurements:
- `is_prime_u64` takes 0.65 µs per call near 2^64 (claim: about 10 µs).
- A cached `is_prime` takes 0.8 to 6.6 ms. That is trial division plus BPSW, run before the cache
  lookup.
- 200 odd candidates near 2^511 take 12.6 ms each on average, including one 512-bit proof.
- The release claim was checked above: 3.29 s at 1024 bits.

## Design items: all implemented

Everything in sections 2.1 to 2.9 is present and does what the design says:
- **Probable primes:** the word test, BPSW with the gcd(n, Q) check, and `NotPrimeWitness`.
  `confirms` validates the witness data (review 2 n8).
- **Certificates:** Jacobian `jdbl`/`jmadd` with exactly the formulas of 2.3.2, and `order_check`.
  The independent iterative verifier is never randomized, uses no list and no H_D, and never
  panics. The PARI parser ignores all whitespace.
- **Prover:**
  - the frame scan (Kronecker, genus filter, Cornacchia);
  - the D = -3 and D = -4 traces, and stripping by the primes below 2^16 with k >= 2;
  - H_D at run time with the rounding check, and polymod root finding;
  - the twists and points;
  - the iterative driver with the invariant steps.len() = stack.len() - 1, the 64-subtree limit,
    and `verify` before every `Prime`.
- **`try_is_prime`:** the cache, the `try_` family, the sieve and its rewired callers, the
  distinct-base Miller-Rabin and `prime_power`.
- **Square roots:** `try_sqrt_mod_prime` with the square check. `square_root_mod_prime` and the
  prime-power path test p once.
- **Callers:** the driver early returns, plus every downstream site of section 3.1.

I checked these soundness points by hand:
- **Order check.** A unit Z_T and a unit Z_U, with U = -T, give [q]T_p = O for every p | N.
  gcd(N, 6) = 1 is checked first.
- **q bound.** `iroot4_ceil` and `q_bound` are exact (tested at 10^20 ± 1).
- **Chain.** A step with s = 1 and q > N stays sound, because the theorem does not need q < N.
- **Witnesses.** Every `Divisor` comes from a gcd that was checked to be neither 1 nor N. In
  `polymod::inverse`, a 0 mod N cannot reach it, because every polynomial is trimmed. Every
  `MillerRabin(a)` has 2 <= a < N.
- **Bounded searches.** Each one ends in "next candidate", `Unresolved` or `Err`:
  - the row scan, 64 split attempts, g up to 2^20, 4 × 64 points, 64 witness bases, the subtree
    limit, and 3 precision attempts in H_D;
  - `selfridge_d` runs only after the square check;
  - the sieve's segment arithmetic cannot overflow up to `u64::MAX` (u128 multiples, saturating
    segment ends, the stop test `seg_hi >= e - 1`).
- **Memory.** No new unbounded allocation: the H_D cache holds 256 entries, the proven-prime
  cache 64, and the list is fixed. `random_prime` keeps the whole range, as before and as
  documented.

**Assertions.** None was weakened without a reference value. These changed, each because it pinned
the old wrong answer:
- `prime_residues.rs`: `is_prime(psi_12)` = true. gp says `isprime` = 0.
- `ch23_is_irreducible.rs`: `Err` for 2^89 - 1 is now `Ok(true)`. gp says `isprime` = 1.
- The REPL psi_12 rows: removed as the design says. The 2^127 - 1 row is now an exact equality.
- `ch18_residues.rs`: only the dead branch was removed. The new refusal assertions are stronger.
- `split_odd_composite`'s `debug_assert`: it lost `!is_prime(m)`, as review 1 m4 asked.
- The `is_prime_u64` tests moved to `primality.rs` and kept their values.

**Tests fail on the old code:** the psi_12 and psi_13 rejection, the three psi_12 driver splits,
`next_prime` for n < 2, the base counts, the finite-field constructors, `ch23` 2^89 - 1 and the
REPL commands. They can also fail on a broken new implementation: the prover round trip, PARI's
certificates, the tampered certificates, `ecpp_driver_backtracks` with a pinned failure count
(review 2 n6), and the unresolved `Err` (review 2 n7).

**Deviations.** The report's deviations 1 to 13 are justified:
- 3: the F_97 claim of the design is false for points of small order, and the replacement checks
  soundness and exactness for k < ord/2;
- 4: pinned backtracking;
- 9: `pollard_p_minus_1(2)` returned 2, which is not a proper divisor;
- 13: a cache-fresh prime.

**Handoffs.** No earlier unit hands anything to I1: I3's report has none, and I3 had already
deduplicated `ArithInt` and `ArithLLong` (review 2 n10). The design's assumptions about I3 hold at
HEAD. The I1 handoffs to I2, F1, P1, R1 and J1a are stated with sites.

**Project rules.**
- No `unsafe` and no em-dash in the added lines.
- The CRLF files keep CRLF.
- Every new file cites MAGMA ch. 18 §18.9 (with the digest path) and its papers.
- `hilbert.rs` records the list's sha256 (verified: `33c5463d...2ee3b`) and why it uses no `rug`.
- The gp generator is committed with `default(parisizemax, 2^30)`.
- No tracker or docs file was edited.

## Minor notes (none blocks)

1. **`unit_sqrts_odd` doc is too narrow** (`rustmath-finitefields/src/integer_mod.rs:812-813`). It
   says "The callers reach this only after `checked_prime_power_args` proved p prime".
   `IntegerMod::sqrt`, `all_sqrts` and `num_sqrts` (`:351`, `:398`, `:434`) also reach it through
   `prime_power_sqrts`, with primes from `prime::factor`. Those come from trial division, so they
   are prime too, and the code is right. Suggested wording: "p is prime in every caller (proven by
   `checked_prime_power_args`, or a factor from trial division in `IntegerMod`)".
2. **`test_try_is_prime_unresolved_is_err` doc names the wrong prime** (`prime.rs:1255-1270`). It
   says the proof "of 2^127 - 1" is unresolved, but the test uses 2^127 + 1000121. The `m127`
   variable only feeds `assert!(m127 < p)`. Cosmetic.
3. **`ecpp_cornacchia_traces` ends with an unrelated comment** (`ecpp/mod.rs:1181-1182`). The
   comment ("not represented with D = -23 iff (-23/N) != 1") does not describe the assertion that
   follows, which is (-3/N) = 1. Cosmetic.
4. **`test_is_prime_panics_when_unresolved` restates the mapping** (`prime.rs:1285-1294`). It panics
   through `try_is_prime_with(..).unwrap_or_else(panic!)` rather than through `is_prime`, so it would
   not catch a change to `is_prime`'s own mapping. Without a way to inject parameters into `is_prime`
   this is the best available, and `is_prime` is one line. Accept as is.

For J1a, not a defect of I1: the REPL `primes` arm has no proof-size limit yet. `is_prime`,
`next_prime` and `sqrt_mod` stop at `PROOF_BITS` = 1024. A range of 2000-bit numbers would run one
ECPP proof per prime, about 76 s each in release. An unresolved proof is a ValueError, not a panic.
J1a's design bounds this arm, and the I1 report hands it over.

## Review 2 notes in the implementation

All twelve (n1 to n12) are folded in as the report says. I checked n1 (contract and the six
pseudoprimes), n2 (`Ok(None)` from Cornacchia skips the row, `Composite(None)` handling), n3
(`try_is_prime` in `integer_constant_is_irreducible`), n5 (`PROOF_BITS`), n6, n7, n8 (`StrongLucas(-8)`),
n9 (the `hilbert.rs` header), n10, n11 (the Jacobi caveat in `legendre_symbol`) and n12 in the code.
The report skips none.
