# Design I1: integers, primality (rustmath-integers)

Unit I1 of the HANDOFF follow-ups (`docs/HANDOFF.md`, "Open follow-ups", lines at `9c24389`, same code as
HEAD `c79a582`). Implemented after I3 (integers arithmetic) and before I2 (factorization drivers).

Experiments, prototypes and logs: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/I1-integers-primality/`
(called `$X` below); the Revision 1 experiments are in `$X/rev1/` (called `$R1`). The most useful files for the
implementer:

| File | What it is |
|---|---|
| `$R1/ecpp_rt.rs` | Working Rust prototype (num-bigint) of this design: BPSW, the ECPP prover with Hilbert class polynomials computed at run time, the Jacobian order check, the verifier, the discriminant list loader. Modes: `hpoly D...`, `polytime h...`, `rootbench bits D`, `one N`, `stats bits count`, `count bits count`. Env: `HIN` (inner h limit, default 20), `HTOP` (top-level h limit, default 100), `SEED`, `BUDGET`. Not production code: recursive (the design is iterative, 2.3.4), no cache bound. |
| `$R1/build.sh` | `build.sh prog.rs rel|dbg`: links against the workspace's own num-bigint rlibs, release (`target/release/deps`) or debug (`target/debug/deps`). "Release" and "debug" timings below mean these two builds. |
| `$R1/dlist.gp`, `$R1/dlist100.txt` | Generator (PARI/GP 2.15.4) and the list of all 42,272 fundamental discriminants D < 0 with h(D) <= 100 (273,422 bytes, sha256 `33c5463d...2ee3b`). |
| `$R1/dcheck.rs` | Exact, PARI-independent check of the list: counts primitive reduced forms for every \|D\| <= 2.4·10^6 (13 s release). Result: all 100 lines equal. |
| `$R1/hpcheck.gp`, `$R1/hp_rust.txt` | 54 run-time Hilbert class polynomials (h = 1 to 100, \|D\| up to 1,856,563) compared with PARI `polclass(D)`: 0 differences. |
| `$R1/count*.log`, `$R1/stats*.log` | Coverage and timing statistics quoted in sections 0 and 2.3.7. |
| `$R1/pm1.gp`, `$R1/hpref.gp`, `$R1/gptime.gp` | Reference values for the new tests; PARI `primecert` timings. |
| `$X/ecpp_rs.rs`, `$X/ecpp_proto.py` | The first prototypes (static table); kept for the record. |
| `$X/c127.txt`, `c40.txt`, `c65.txt`, `c200.txt` | PARI `primecert` certificates used as test vectors (section 4 embeds them). |

## Revision 1

Review 1 (`I1-integers-primality.review1.md`) found 3 blockers and 15 minor items. All are fixed below; no
item is rejected. Summary of the changes:

| Issue | What changed |
|---|---|
| B1: prover coverage above 1024 bits | Closed by option (a). The static 1.3 MB table of class polynomials is gone. Hilbert class polynomials H_D are computed at run time by evaluating j at the reduced forms in fixed-point complex arithmetic on BigInt (Cohen GTM 138 §7.6; [AM93] §7; Enge 2009), for every fundamental D with h(D) <= 100 (42,272 discriminants, a 273 KB list, complete by Watkins 2004). The top level scans to h = 100, inner levels to h = 20 up to 2048 bits and to h = 100 above (2.3.4, 2.3.5). Measured: top-level coverage from 1024 to 4096 bits (212 primes), full proofs in release up to 3072 bits and in debug up to 2048 bits (sections 0 and 2.3.7): every prime up to 2048 bits was proven (61 proofs, 180 coverage runs); 2 of 32 primes at 3072 and 4096 bits have no top-level candidate; a 2048-bit proof takes 76 s in release and 16 min in debug; the "rare above 512 bits" and "none below 512 bits" sentences are replaced by those numbers. The policy part of option (b) is applied as well: prime generators skip an unresolved candidate, and bulk enumerators prove only what they return (B2, m6). |
| B2: Result/Option callers acquire a panic path | Section 3.1 lists every caller that returns `Result` or `Option` and takes an `Integer` of any size, with its decision. Switched in I1: crypto `random_prime_bits` and `random_prime_3mod4_bits` (skip an unresolved candidate); the REPL arms `is_prime`, `next_prime`, `primes`, `prime_pi`, `nth_prime` and `sqrt_mod_report` (EvalError); `prime::random_prime`; `FiniteField::new`, `FiniteField::with_modulus`, `GaloisRing::new`, `GaloisRing::with_modulus`; `berlekamp::check_prime`; the seven p-adic constructors and `compute_s_unit_group` in rustmath-rings; `PrattCertificate::generate`. Handed over with the exact site: F1 (`cantor_zassenhaus.rs:143`, already in F1's design). New fallible functions `try_next_prime`, `try_previous_prime`, `try_prime_range`, `try_random_prime` (2.1). |
| B3: `pollard_p_minus_1` truncation | Stage 1 iterates `sieve::for_each_prime(2, bound + 1, ..)`, all primes up to `bound`. New test `pollard_p_minus_1_bound_above_2_16`: n = 917519 · 2000000000123, bound 70000 finds 917519 (917519 - 1 = 2·7·65537); gp `pm1.gp` (2.5, 4). |
| m1: name clash | The certificate type is `EcppCertificate`; its doc says it is MAGMA's `PrimalityCertificate` and that `rustmath_constants::PrimalityCertificate` is an unrelated trait (2.1). |
| m2: sieve overflow near 2^64 | The test uses `primes_in_interval(lo, u64::MAX)`. Segment and multiple arithmetic is specified overflow-free (u128 or checked adds, `p <= hi / p`) (2.5). |
| m3: "a step removed" | Replaced by "the middle step removed" and "the first step removed, n kept" (both invalid); the test also states that the first-step-removed certificate is valid for its own n() = q_1 (4). |
| m4: `split_odd_composite` assert | `debug_assert!(m.is_odd() && *m > 1)` only; the doc says that `pollard_rho`'s early `None` then sends a BPSW pseudoprime to the complete Brent and trial-division fallback (2.8). |
| m5: `try_sqrt_mod_prime` on squares | It returns `Err` when p is a perfect square (checked first); its doc states the precondition and why every other input terminates (2.7). |
| m6: `random_prime` proves the range | It enumerates probable primes, picks one, proves only the pick, and drops an unresolved or refuted pick (2.5). |
| m7: prime-power square roots test twice | `unit_sqrts_odd` calls `try_sqrt_mod_prime` (no test); p was proven by `checked_prime_power_args` (2.7). |
| m8: gcd(n, Q) in BPSW | Step 3 of `bpsw` checks gcd(n, Q) and returns `Divisor` for a proper one (2.2). |
| m9: 2^521 - 1 in tests | Kept and counted: about 4 s in debug (5). |
| m10: binary size | The class polynomial table is gone. The discriminant list is 273 KB, so about 62 MB over 228 test binaries instead of 300 MB. The smaller lists (h <= 64: 108 KB) were measured and rejected (2.3.5). |
| m11: recursion depth | The prover is an iterative driver with an explicit stack of frames (2.3.4). Measured chain length: 120 at 2048 bits, 182 at 3072 bits (186 at most). |
| m12: verify before caching | `prove_primality` verifies every certificate it builds before it returns `Prime` (always, 2 to 7 % of the proof time); `try_is_prime` caches only verified results (2.3.4, 2.4). |
| m13: gp stack | Every committed gp script and every quoted gp command that needs it starts with `default(parisizemax, 2^30)` (4, 6). |
| m14: `prove_primality` below psi_12 | Documented on `prove_primality` (2.1, 2.3.4). |
| m15: test vectors | The four PARI certificates are string constants in `tests/ch18_primality.rs` with their gp commands; `from_pari` ignores all ASCII whitespace (2.1, 4). |

Other changes made while revising:
- The design no longer embeds polynomials whose coefficients are trusted from a file. A wrong H_D still could not
  make a certificate valid (the verifier never uses it), and run-time H_D are checked against `polclass` in tests.
- Discriminants with 3 | D are now used for every h (the table excluded them above h = 6 because of their j
  coefficient size), so the list for h <= 20 has 1817 rows instead of 1420.
- Root finding uses lazy reduction (one reduction per coefficient), measured in 2.3.5.
- Frames above 2048 bits scan to h = 100 like the top level (`H_INNER_BITS`), because an h <= 20 scan fails
  for half or more of such primes (2.3.4, 2.3.7).
- `next_prime(0)` and `next_prime(1)` are 2 (adjacent fix, unchanged from the first version).

## 0. Summary

| Item | Verdict at HEAD | Fix |
|---|---|---|
| A. No proving `is_prime` above psi_12 | Confirmed: `is_prime(psi_12)` and `is_prime(psi_13)` are `true`; all three drivers return psi_12 unsplit. | BPSW (strong Lucas) as the probable-prime test; ECPP (Atkin-Morain) with run-time Hilbert class polynomials, a PARI-compatible certificate and an independent verifier; `is_prime` becomes a proof (MAGMA `IsPrime`, `Proof := true`); three-valued `prove_primality` and `try_is_prime`. |
| S6. 12 bases repeated 20 times; `nth_prime` / `prime_pi` test every candidate | Confirmed: 240 Miller-Rabin rounds per prime; `nth_prime(10^4)` 11 s, `prime_pi(10^5)` 10.6 s (debug). | `is_prime_miller_rabin(n, k)` runs k distinct bases once; a segmented sieve serves `nth_prime`, `prime_pi`, `prime_range`, `primes_first_n`, `fast_arith::prime_range` and Pollard p - 1; `next_prime` / `previous_prime` pre-sieve. Adjacent: `next_prime(0) = next_prime(1) = 3` fixed. |
| `fast_arith::is_prime_u64` duplicates "FastArith64" | Confirmed (no type `FastArith64` exists; the 64-bit fast-arithmetic struct is `ArithLLong`, with its own Miller-Rabin; `ArithInt` has a third test that overflows and panics for 2^31 - 1). | One word-size test, `prime::is_prime_u64`, used by `is_prime`, `ArithLLong::is_prime`, `ArithInt::is_prime`, the perfect-power sieve and ECPP leaves. |
| `square_root_mod_prime` runs `is_prime` twice | Confirmed (`integer_mod.rs:652` and again inside `Integer::sqrt_mod_prime`, `integer.rs:883`; a third time on the prime-power path). Included here, not handed to F1. | New `Integer::try_sqrt_mod_prime` (no primality test, three outcomes); `square_root_mod_prime` and the prime-power functions test once. |
| Drivers and their docs | Docs promise "a proof below psi_12, a strong probable prime above"; psi_12 unsplit. | Drivers keep `is_prime` (now a proof) as loop guard, use `is_probable_prime` for their "n is prime" shortcuts; docs rewritten; interface for I2 specified (section 2.8). |

Nothing is declared out of scope. What `is_prime` proves, and how fast (prototype `$R1/ecpp_rt.rs`: h <= 100 at
the top level and in frames above 2048 bits, h <= 20 elsewhere; random primes of the given size; every
certificate verified; full table in 2.3.7):

| Bits | Proven / tried | Release mean (max) | Debug mean (max) | HEAD `is_prime` (240 rounds), release / debug |
|---|---|---|---|---|
| 128 | 10 / 10 | 0.024 s (0.038) | 0.065 s (0.094) | 20 ms debug |
| 256 | 10 / 10 | 0.070 s (0.13) | 0.36 s (0.52) | 90 ms debug |
| 512 | 8 / 8 (60 / 60 in the first version) | 0.41 s (0.57) | 3.2 s (4.5) | 0.7 s debug |
| 768 | 6 / 6 | 1.7 s (2.4) | about 15 s (estimate, ratio 9) | |
| 1024 | 10 / 10 (+ 2 / 2 debug) | 3.3 s (4.3) | 38 s (41) | 0.22 s / 3.1 s |
| 1536 | 5 / 5 (+ 1 / 1 debug) | 20.6 s (27.8) | 6.4 min (one proof; the same prime takes 25 s in release) | |
| 2048 | 12 / 12 (+ 1 / 1 debug) | 76 s (115) | 16 min (one proof; the same prime takes 61 s in release) | 1.0 s / 25.8 s |
| 3072 | 2 / 2 | 16 min (17) | | |
| 4096 | coverage only (12 primes) | about 1.5 h (estimate) | | |

Top-level coverage (the top level cannot backtrack, so a prime with no top-level candidate is `Unresolved`;
`count` and `minh` modes, random primes; the full table with the mean number of candidates is in 2.3.7):

| Bits | Primes | No top-level candidate with h <= 20 | h <= 32 | h <= 48 | h <= 64 | h <= 100 |
|---|---|---|---|---|---|---|
| 1024 | 100 | 3 (3 %) | 0 | 0 | 0 | 0 |
| 1536 | 40 | 2 (5 %) | 1 (2.5 %) | 1 (2.5 %) | 0 | 0 |
| 2048 | 40 | 6 (15 %) | 1 (2.5 %) | 0 | 0 | 0 |
| 3072 | 20 | 10 (50 %) | 6 (30 %) | 2 (10 %) | 2 (10 %) | 1 (5 %) |
| 4096 | 12 | 8 (67 %) | 5 (42 %) | 2 (17 %) | 1 (8 %) | 1 (8 %) |

`prove_primality` returns `Unresolved` only when the top level has no candidate, or every candidate's subtree
failed, or 64 subtrees failed in all (2.3.4); `try_is_prime` then returns `Err(NotSupported)` and `is_prime` panics with a message (it returns
`bool`). Up to 2048 bits no prime was unresolved in any experiment: 61 full proofs from 128 to 2048 bits, and
every one of the 180 primes of the coverage runs at 1024, 1536 and 2048 bits had a top-level candidate. Above
2048 bits the prover still works, but slowly, and 2 of 32 primes (1 of 20 at 3072 bits, 1 of 12 at 4096 bits)
had no top-level candidate with h <= 100. These rates are stated in the docs of `is_prime`, `try_is_prime` and
in HANDOFF, and no existing caller that returns `Result` or `Option` keeps a panic path (section 3.1). For
comparison, PARI/GP 2.15.4 `primecert` (C with GMP) takes 1.4 s, 5.3 s and 16 s for one random prime of 1024,
1536 and 2048 bits (`$R1/gptime.gp`).

## 1. Defects confirmed at HEAD

### 1.1 Item A: no primality proof above psi_12

Code: `rustmath-integers/src/prime.rs:90-101` (`is_prime`): trial division below 1000, otherwise
`is_prime_miller_rabin(n, 20)` (`prime.rs:31-88`) with the fixed bases 2, ..., 37. Sorenson and Webster
(Math. Comp. 86 (2017) 985-1003) show this is a proof only for n < psi_12 = 318665857834031151167461.

Reproduction (`$X/t1_bench.rs`, `$X/t12_ref.gp`; reproduced independently by review 1):

| Input | HEAD | Reference |
|---|---|---|
| `is_prime(318665857834031151167461)` | `true` | gp `isprime(318665857834031151167461)` = 0, `factor` = 399165290221 * 798330580441 |
| `is_prime(3317044064679887385961981)` (psi_13) | `true` (passes all 12 bases) | gp `factor` = 1287836182261 * 2575672364521 |
| `factor_pollard_rho(psi_12)` | `[(psi_12, 1)]` in 20 ms | gp `factor(psi_12)` = [399165290221, 1; 798330580441, 1] |
| `quadratic_sieve_factor_complete(psi_12)`, `ecm_factor_complete(psi_12, 1000, 10000, 10)` | `[(psi_12, 1)]` both (`$X/t18_drivers.rs`; same loop guard) | same |

MAGMA (digest `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md` §18.9.1, lines 290-294): `IsPrime(n)` has
`Proof := true` by default and "invokes ECPP for larger n [AM93]"; `IsProbablePrime(n: Bases := 20)`;
`PrimalityCertificate(n)` / `IsPrimeCertificate(cert)`; `IsPrimePower(n)` with "primality of p is proven";
`RandomPrime(n: Proof := true)` (lines 215, 309).

Tests that pin the HEAD behaviour and must change:
- `rustmath-finitefields/tests/prime_residues.rs:303` asserts `is_prime(&psi)` is true.
- `rustmath-jupyter/src/repl.rs:11746-11771` (`test_sqrt_mod_probable_prime_caveat_and_refusal`) expects
  psi_12 to take the prime path; after the fix it takes the composite path, whose trial-division factoring
  of psi_12 (smallest factor 399165290221) would not finish.
- `rustmath-integers/tests/ch18_residues.rs:649-659` branches on `n.is_prime()`; it stays green, the dead
  branch goes.
- `rustmath-polynomials/tests/ch23_is_irreducible.rs:596-598` expects `Err` for the constant 2^89 - 1; section
  2.9 changes the polynomial code, so the test flips to `Ok(true)`.

### 1.2 Item S6: repeated bases, candidate-by-candidate prime functions

- `prime.rs:55-58`: `for _ in 0..rounds { let witnesses = [2, ..., 37]; for &a_val in &witnesses ... }`
  runs the same 12 bases `rounds` times; `is_prime` passes `rounds = 20`, so a prime costs 240 strong
  tests. Measured: `is_prime_miller_rabin(2^127 - 1, 20)` 30 ms, `(.., 1)` 1.6 ms; `is_prime(2^521 - 1)`
  0.7 s; random 1024-bit prime 0.22 s release, 3.1 s debug; 2048-bit 1.0 s release, 25.8 s debug
  (`$R1/headbench.rs`).
- `nth_prime` (`prime.rs:240`), `prime_pi` (`:356`), `prime_range` (`:277`), `primes_first_n` (`:318`) call
  `is_prime` on every odd candidate. Measured at HEAD (`$X/t10_head.rs`, debug): `nth_prime(10^4)` 11.1 s,
  `prime_pi(10^5)` 10.6 s, `prime_range(10^6, 10^6 + 10^4)` 1.1 s, `primes_first_n(10^4)` 11.4 s. PROJECT_TRACKER
  row `prime.rs:137` says the same ("prime_pi(10^8) is not reachable").
- `pollard_p_minus_1` (`:650-653`) tests every integer p <= bound with `is_prime` to find the stage-1 primes.
- Adjacent defect in the same functions (tracker row `ch018.NextPrime`): `next_prime(0) = 3`,
  `next_prime(1) = 3`, `next_prime(-5) = 3` at HEAD; MAGMA `NextPrime(n)` is the least prime > n, so 2
  (gp `nextprime(1) = 2`, `nextprime(2) = 2`; PARI's `nextprime(n)` is the least prime >= n, so the MAGMA
  value is gp `nextprime(n + 1)`).

### 1.3 Item: duplicate word-size tests

There is no `FastArith64` anywhere in the workspace (`grep -rn FastArith64` finds only `HANDOFF.md:330`).
Reading the item as "duplicates the 64-bit fast-arithmetic type", the duplicates in
`rustmath-integers/src/fast_arith.rs` are:
- `is_prime_u64` (`:70-99`, crate-internal): trial division by the 12 primes below 41, then the 12 bases.
- `ArithLLong::is_prime` (`:391-414`) + `ArithLLong::miller_rabin` (`:417-462`): trial division by the
  cached primes below 10^4, then the same 12 bases. Same answers, second implementation.
- `ArithInt::is_prime` (`:223-253`): trial division with `i * i <= n` in i32, which overflows. Measured
  (`$X/t14_arith.rs`, debug): `ArithInt::is_prime(2147483647)` and `(2147483629)` panic with "attempt to
  multiply with overflow" (gp `isprime(2^31 - 1) = 1`, `isprime(2147483629) = 1`).
- `prime::is_prime` itself has its own u64 path (trial division below 1000, then 240 rounds).
- Both `ArithInt` and `ArithLLong` carry a private Eratosthenes (`compute_small_primes`, `:119`, `:285`)
  that duplicates `fast_arith::prime_range`.

(Other crates keep private copies: `rustmath-modular/src/modsym/{lseries.rs:137, hecke.rs:51,
decomposition.rs:191}`, `rustmath-ellipticcurves/src/lfunction.rs:87`, `rustmath-galois/src/sieve.rs:38`.
They are listed for the owners of those crates; section 7.)

### 1.4 Item: `square_root_mod_prime` runs `is_prime` twice

`rustmath-finitefields/src/integer_mod.rs:651-667`: `assert_prime("square_root_mod_prime", p)` (`:652`,
`:877-883`, calls `is_prime`), then `b.kronecker_symbol(p)` (`:656`), then `b.sqrt_mod_prime(p)` (`:660`),
which calls `is_prime(p)` again (`integer.rs:883`) and computes the Kronecker symbol again inside
`sqrt_mod_prime_checked` (`integer.rs:1167`). At HEAD the two tests cost 2 x 20 ms for p = 2^127 - 1; with a
proving `is_prime` they would be two proofs. The prime-power functions `square_root_mod_prime_power` and
`all_square_roots_mod_prime_power` test p in `checked_prime_power_args` (`:886`, `:938`) and again through
`unit_sqrts_odd` (`:814-815`), which calls `square_root_mod_prime`.

### 1.5 Item: the factorization drivers and their docs

- `factor_pollard_rho` (`prime.rs:808-860`), `quadratic_sieve_factor_complete` (`quadratic_sieve.rs:549-610`)
  and `ecm_factor_complete` (`ecm.rs:514-565`) use `!is_prime(&remaining)` as the loop guard and
  `is_prime(&divisor)` before recursing. Their docs (`prime.rs:727-730`, `quadratic_sieve.rs:519-523`,
  `ecm.rs:486-489`) say a factor above psi_12 "is only a strong probable prime and may be composite:
  psi_12 ... itself is returned unsplit".
- `pollard_rho` (`prime.rs:556`), `pollard_p_minus_1` (`:641`), `ecm_factor` (`ecm.rs:441`) and
  `quadratic_sieve_factor` (`quadratic_sieve.rs:450`) call `is_prime(n)` only to return `None` early.

## 2. The fix

### 2.1 API (MAGMA ch. 18 §18.9.1 and §18.9.2)

All public paths stay; new items are added. `rustmath_numbertheory` re-exports `prime::*`. Names checked with
grep: no `is_probable_prime`, `prove_primality`, `try_is_prime`, `Primality`, `EcppCertificate`, `EcppStep`,
`try_next_prime`, `try_previous_prime`, `try_prime_range`, `try_random_prime` exist in the workspace; the only
`is_prime_u64` definitions are private to other crates, which do not glob-import `prime::*`. There is a
`pub trait PrimalityCertificate` in `rustmath-constants` (`certificates.rs:16`, re-exported at `lib.rs:17`, the
Pratt certificates); that is why the new type is called `EcppCertificate` and not MAGMA's name.

New module `rustmath-integers/src/primality.rs` (word-size test, BPSW, witnesses):

```rust
/// Deterministic primality of a 64-bit word (moved from fast_arith; now public).
pub fn is_prime_u64(n: u64) -> bool;

/// MAGMA `IsProbablePrime(n)` / `IsPrime(n: Proof := false)`. Exact for n < psi_12 (n < 2^64 by
/// `is_prime_u64`, then strong tests to the 12 prime bases 2..37). Above psi_12 the Baillie-PSW test:
/// a strong probable prime to base 2 and a strong Lucas probable prime with Selfridge's parameters.
/// `false` is always a proof; `true` above psi_12 is not (no counterexample is known).
pub fn is_probable_prime(n: &Integer) -> bool;

/// Why a number is not prime; every variant can be re-checked cheaply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotPrimeWitness {
    /// n < 2.
    BelowTwo,
    /// A divisor d with 1 < d < n.
    Divisor(Integer),
    /// n is odd and not a strong probable prime to this base (a Miller-Rabin witness), 2 <= base < n.
    MillerRabin(u32),
    /// n is odd, (D/n) = -1, gcd(n, 2 D Q) = 1 for Q = (1 - D)/4, and n is not a strong Lucas probable
    /// prime for (P, Q) = (1, Q). Every prime with these conditions passes (Baillie-Wagstaff 1980), so the
    /// witness does not depend on D being Selfridge's first choice; the prover records Selfridge's D.
    StrongLucas(i64),
    /// n > 1 is a perfect square.
    Square,
}
impl NotPrimeWitness {
    /// Re-runs the check behind the witness; true iff it shows that n is not prime.
    pub fn confirms(&self, n: &Integer) -> bool;
}
```

New module `rustmath-integers/src/ecpp/` (`mod.rs`: certificate, verifier, prover driver; `hilbert.rs`:
reduced forms, fixed-point complex arithmetic, Hilbert class polynomials and their cache; `polymod.rs`: dense
polynomials mod N and root finding; data file `rustmath-integers/data/ecpp_discriminants.txt`):

```rust
/// One Atkin-Morain step in PARI's layout [N, t, s, a, [x, y]]: the curve y^2 = x^3 + a x + b over Z/NZ
/// with b = y^2 - x^3 - a x, m = N + 1 - t, q = m / s; q is the N of the next step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EcppStep { pub n: Integer, pub t: Integer, pub s: Integer, pub a: Integer, pub x: Integer, pub y: Integer }

/// An ECPP primality certificate: MAGMA's `PrimalityCertificate` (ch. 18 §18.9.1). Not related to the
/// trait `rustmath_constants::PrimalityCertificate` (Pratt certificates); import with an alias if both are
/// needed. No steps iff n < 2^64.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EcppCertificate { n: Integer, steps: Vec<EcppStep> }
impl EcppCertificate {
    /// An unchecked container; `verify` decides.
    pub fn from_steps(n: Integer, steps: Vec<EcppStep>) -> Self;
    /// The number this certificate is about. `verify` proves `n()` prime, so a caller that wants a proof
    /// for some m must also check `n() == m` (a certificate for q_1 is a valid certificate, of q_1).
    pub fn n(&self) -> &Integer;
    pub fn steps(&self) -> &[EcppStep];
    /// MAGMA `IsPrimeCertificate`: true iff the certificate proves n() prime (section 2.3.3). Never panics.
    pub fn verify(&self) -> bool;
    /// PARI/GP format: the integer for n < 2^64, else "[[N, t, s, a, [x, y]], ...]".
    pub fn to_pari(&self) -> String;
    /// Parses the PARI/GP format (output of gp `primecert`). All ASCII whitespace (spaces, tabs, line
    /// breaks, as in gp's wrapped output) is ignored; n is the first step's N. Err(ParseError) if malformed.
    pub fn from_pari(text: &str) -> Result<Self>;
}

/// Outcome of a primality proof (MAGMA `IsPrime` with `Proof := true`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Primality {
    /// A certificate that `verify` has accepted.
    Prime(EcppCertificate),
    NotPrime(NotPrimeWitness),
    /// n passed BPSW, but the prover found no certificate; nothing is decided. The string says why.
    Unresolved(String),
}

/// BPSW, then the ECPP prover for n >= 2^64 (a certificate with no steps below 2^64). Every `Prime` carries
/// a certificate that `verify` accepted. For 2^64 <= n < psi_12 the result can be `Unresolved` although the
/// 12 strong tests prove n prime (no ECPP certificate was found); `try_is_prime` and `is_prime` decide those
/// n by the strong tests and never use ECPP there.
pub fn prove_primality(n: &Integer) -> Primality;
```

In `rustmath-integers/src/prime.rs` (public paths kept, behaviour as described):

```rust
pub use crate::ecpp::{prove_primality, EcppCertificate, EcppStep, Primality};
pub use crate::primality::{is_prime_u64, is_probable_prime, NotPrimeWitness};

/// MAGMA `IsPrime(n)` (Proof := true) as a Result: Ok(true) proven prime, Ok(false) proven not prime,
/// Err(MathError::NotSupported(..)) when ECPP is unresolved (rates and times in the doc, from 2.3.7).
pub fn try_is_prime(n: &Integer) -> Result<bool>;
/// MAGMA `IsPrime(n)`. A proof. Panics when `try_is_prime` is Err (message names n and the reason).
pub fn is_prime(n: &Integer) -> bool;
/// MAGMA `NextPrime(n)`: the least proven prime > n. Err when the least probable prime > n is unresolved
/// (it cannot be skipped: the answer must be the least prime).
pub fn try_next_prime(n: &Integer) -> Result<Integer>;
pub fn next_prime(n: &Integer) -> Integer;                   // try_next_prime, panics on Err
/// MAGMA `PreviousPrime(n)`: Ok(None) for n <= 2.
pub fn try_previous_prime(n: &Integer) -> Result<Option<Integer>>;
pub fn previous_prime(n: &Integer) -> Option<Integer>;       // panics on Err
/// The primes in [start, stop), each proven. Err if one probable prime in the range is unresolved (the list
/// must be complete).
pub fn try_prime_range(start: &Integer, stop: &Integer) -> Result<Vec<Integer>>;
pub fn prime_range(start: &Integer, stop: &Integer) -> Vec<Integer>;   // panics on Err
/// A random proven prime in [a, b) (MAGMA `RandomPrime`, Proof := true). Ok(None): no prime in [a, b).
/// An unresolved or refuted pick is dropped and another is drawn; Err only if probable primes exist and
/// none of them could be proven.
pub fn try_random_prime(a: &Integer, b: &Integer) -> Result<Option<Integer>>;
pub fn random_prime(a: &Integer, b: &Integer) -> Option<Integer>;     // panics on Err
/// MAGMA `IsProbablePrime(n: Bases := rounds)` with the bases fixed to the first max(rounds, 1) primes,
/// each used once.
pub fn is_prime_miller_rabin(n: &Integer, rounds: u32) -> bool;
/// MAGMA `IsPrimePower(n)` with its extra return values: Some((p, k)) iff n = p^k, p proven prime.
pub fn prime_power(n: &Integer) -> Option<(Integer, u32)>;
pub fn is_prime_power(n: &Integer) -> bool;   // prime_power(n).is_some()
```

`prime_power` and `is_prime_power` prove p with `is_prime`, so they panic when that proof is unresolved
(documented; no caller in the workspace passes numbers of that size).

Behaviour summary of `is_prime` and `try_is_prime`:

| n | Method | Kind of answer |
|---|---|---|
| n < 2 | | false |
| n < 2^64 | `is_prime_u64` | proof |
| 2^64 <= n < psi_12 | trial division by the primes below 1000, strong tests to the 12 bases 2..37 | proof [Sorenson-Webster 2017] |
| n >= psi_12 | trial division, BPSW; on BPSW pass: cache, then `prove_primality` | proof, or Err / panic when unresolved |

`Integer::is_prime()` keeps delegating to `prime::is_prime`.

### 2.2 BPSW (strong Lucas)

References: R. Baillie, S. S. Wagstaff Jr., *Lucas pseudoprimes*, Math. Comp. 35 (1980) 1391-1417 (strong
Lucas test, Selfridge's method A); C. Pomerance, J. L. Selfridge, S. S. Wagstaff Jr., *The pseudoprimes
to 25·10^9*, Math. Comp. 35 (1980) 1003-1026; R. Baillie, A. Fiori, S. S. Wagstaff Jr., *Strengthening the
Baillie-PSW primality test*, Math. Comp. 90 (2021) 1931-1955 (no BPSW pseudoprime below 2^64; none known
at all). MAGMA chapter digest §18.9 [Jae93] for the strong test.

`bpsw(n)` for odd n > 1 with no prime factor below 1000 (crate-internal, `pub(crate)` for tests):
1. Strong test to base 2 (n - 1 = d 2^s; 2^d = 1 or 2^(d 2^r) = -1 for some r < s). Fail: `MillerRabin(2)`.
2. If n is a perfect square (`num_integer::Roots::sqrt` on the BigInt; I3 may change `Integer::sqrt`, so do
   not depend on it): `Square`.
3. Selfridge: D = 5, -7, 9, -11, 13, ... (D_{k+1} = -D_k - 2 sign(D_k)), the first with Jacobi (D/n) = -1
   (`Integer::kronecker_symbol`). If (D/n) = 0 with |D| < n: `Divisor(gcd(|D|, n))`. For a non-square n
   such a D exists, so the loop ends (in practice after 1 to 3 tries). Then Q = (1 - D)/4 and
   g = gcd(n, |Q|): if g > 1 (then g < n, since |Q| < n), return `Divisor(g)`. This keeps the contract of
   `StrongLucas` (gcd(n, 2DQ) = 1). It cannot trigger for |D| < 4000, because n has no prime factor below
   1000 and |Q| <= (|D| + 1)/4. The gcd checks and steps 4-5 for a given D are a separate
   `pub(crate) fn lucas_check(n: &BigInt, d: i64) -> Result<bool, NotPrimeWitness>` (Ok(true): strong Lucas
   probable prime), so a test can reach the gcd branch with a chosen D.
4. P = 1, n + 1 = d 2^s with d odd. Left-to-right binary over the bits of d below the top bit, starting from
   (U, V, Q^k) = (1, P, Q):
   doubling: U <- U V, V <- V^2 - 2 Q^k, Q^k <- (Q^k)^2;
   on a 1 bit: (U, V) <- ((P U + V)/2, (D U + P V)/2), Q^k <- Q^k Q;
   all mod n, halving as multiplication by (n + 1)/2.
5. Strong Lucas probable prime iff U_d = 0, or V_(d 2^r) = 0 for some 0 <= r < s
   (V <- V^2 - 2 Q^k, Q^k <- (Q^k)^2 between r and r + 1). Fail: `StrongLucas(D)`.

`is_probable_prime(n)`: n < 2 false; n < 2^64 `is_prime_u64`; a prime factor p < 1000 gives n == p;
n < psi_12: the 12 bases; else `bpsw(n)`.

Cost (debug, `$X/ecpp_rs.rs bpsw`): u64 10 µs; psi_12 0.9 ms; 89- and 127-bit primes 0.43 ms; 256-bit
3.8-4 ms; 521-bit 7.5 ms. HEAD `is_prime` on the same primes: about 1 ms (u64), 20 ms (127-bit), 90 ms
(256-bit), 0.7 s (521-bit). Review 1 checked the prototype's BPSW against PARI `ispseudoprime` on 202 numbers
(primes, semiprimes, 40 base-2 strong pseudoprimes above 2^64, psi_12, psi_13): no mismatch.

### 2.3 ECPP (Atkin-Morain)

Chosen over APR-CL (no checkable certificate) and over the n - 1 / n + 1 methods of Brillhart, Lehmer and
Selfridge (they need N - 1 or N + 1 factored to N^(1/3), which fails for most numbers above about 60 digits).
ECPP is what MAGMA uses ([AM93]), it produces a certificate, and the certificate format of PARI/GP 2.15
lets the tests use PARI as an independent checker in both directions.

References: [AM93] A. O. L. Atkin, F. Morain, *Elliptic curves and primality proving*, Math. Comp. 61 (1993)
29-68 (digest §18.9, bibliography line 548); S. Goldwasser, J. Kilian, *Primality testing using elliptic
curves*, J. ACM 46 (1999) 450-472; H. Cohen, *A Course in Computational Algebraic Number Theory*, GTM 138,
§9.2 (the elliptic curve test: Goldwasser-Kilian and Atkin's test), §7.6 (computing Hilbert class
polynomials by evaluating j at the reduced forms), §5.3 (reduced forms, the class number as their count),
Algorithm 1.5.1 (Tonelli-Shanks), Algorithm 1.5.3 (modified Cornacchia, x^2 + |D| y^2 = 4p), §1.6.1 (roots
mod p); R. Crandall, C. Pomerance, *Prime Numbers: A Computational Perspective*, 2nd ed., §7.6; H. Cohen,
A. Miyaji, T. Ono, *Efficient elliptic curve exponentiation using mixed coordinates*, ASIACRYPT 1998, LNCS 1514,
51-65 (Jacobian doubling); D. Hankerson, A. Menezes, S. Vanstone, *Guide to Elliptic Curve Cryptography*,
Springer 2004, §3.2 (mixed Jacobian-affine addition, EFD `madd-2004-hmv`); D. A. Cox, *Primes of the form
x^2 + ny^2*, §3 (genus characters); A. Enge, *The complexity of class polynomial computation via floating
point approximations*, Math. Comp. 78 (2009) 1089-1107 (precision of the floating-point method);
N. Yui, D. Zagier, *On the singular values of Weber modular functions*, Math. Comp. 66 (1997) 1645-1662
(Weber's j = (f_2^24 + 16)^3 / f_2^24); M. Watkins, *Class numbers of imaginary quadratic fields*, Math. Comp.
73 (2004) 907-938 (the list of all imaginary quadratic fields with h <= 100 is complete: 42,272 fields);
T. M. Apostol, *Modular Functions and Dirichlet Series in Number Theory*, GTM 41, ch. 3 (Euler's pentagonal
number theorem for the eta product); PARI/GP 2.15.4 documentation of `primecert`, `primecertisvalid` and
`polclass`. The Cohen algorithm and section numbers (1.5.3, §1.6.1, §5.3, §7.6) are from memory: the
implementer checks them against the book before citing (the HANDOFF already lists unchecked Cohen numbers
elsewhere).

#### 2.3.1 Theorem used

Let N > 1 with gcd(N, 6) = 1, a, b integers with gcd(4a^3 + 27b^2, N) = 1, P = (x, y) with
y^2 = x^3 + a x + b (mod N), and integers s >= 1, q with q > (N^(1/4) + 1)^2, q prime. Suppose that for every
prime p | N the point T_p = [s](P mod p) of E_p: y^2 = x^3 + a x + b over F_p is not O and [q] T_p = O. Then
N is prime.

Proof: T_p has order q, so q <= #E_p(F_p) <= p + 1 + 2 sqrt(p) = (sqrt(p) + 1)^2 (Hasse). If N were composite it
would have a prime factor p <= sqrt(N), giving q <= (N^(1/4) + 1)^2, a contradiction. (This is the
Goldwasser-Kilian theorem in the form of PARI's `primecert` documentation, with m P = O replaced by its
equivalent [q]([s]P) = O.)

#### 2.3.2 Order check in Jacobian coordinates (no inversion per step)

The verifier must establish the hypothesis for every prime p | N without knowing p. Affine arithmetic with
Lenstra's pseudo-addition does that but needs a modular inversion per group operation, which made the
prototype slow (232 µs per bit of the scalar at 127 bits, debug). The design uses Jacobian coordinates and
the following argument instead.

Jacobian coordinates: (X : Y : Z) with Z != 0 is the affine point (X/Z^2, Y/Z^3); Z = 0 stands for O.
Formulas (doubling `dbl-1998-cmo-2`, Cohen-Miyaji-Ono 1998; mixed addition with Z2 = 1 `madd-2004-hmv`,
Hankerson-Menezes-Vanstone 2004; review 1 corrected the name of the second, the formulas are unchanged):
- DBL(X1, Y1, Z1): XX = X1^2, YY = Y1^2, ZZ = Z1^2, S = 4 X1 YY, M = 3 XX + a ZZ^2, X3 = M^2 - 2S,
  Y3 = M (S - X3) - 8 YY^2, Z3 = 2 Y1 Z1.
- MADD((X1, Y1, Z1), (x2, y2)): Z1Z1 = Z1^2, U2 = x2 Z1Z1, S2 = y2 Z1 Z1Z1, H = U2 - X1, R = S2 - Y1,
  HH = H^2, HHH = H HH, V = X1 HH, X3 = R^2 - HHH - 2V, Y3 = R (V - X3) - Y1 HHH, Z3 = Z1 H.

Facts over a field F_p (p >= 5), for an input with Z1 != 0 representing T:
1. DBL gives Z3 = 2 Y1 Z1. If T is not 2-torsion (Y1 != 0), Z3 != 0 and the output represents 2T; otherwise
   Z3 = 0.
2. MADD gives Z3 = Z1 H. If T != +-P2 (H != 0), Z3 != 0 and the output represents T + P2; otherwise Z3 = 0.
3. Z = 0 is absorbing: DBL and MADD of an input with Z1 = 0 have Z3 = 0.

The formulas are polynomials, so computing over Z/NZ and reducing mod p is the same as computing over F_p.
For `mul(k, x, y)` = left-to-right double-and-add over the bits of k below the top bit, starting from
(x : y : 1) (k = 1 returns the base point): if the final Z is nonzero mod p, then by fact 3 every
intermediate Z was nonzero mod p, so by facts 1 and 2 every step computed the true multiple, and the final
triple represents [k](P mod p), which is not O.

`order_check(N, a, x, y, s, q)`:
1. T = mul(s, x, y). If gcd(Z_T, N) = N (Z_T = 0 mod N): `Retry` (pick another point). If
   1 < gcd(Z_T, N) < N: `Factor(g)`.
2. Normalize: z = Z_T^(-1) mod N, (tx, ty) = (X_T z^2, Y_T z^3).
3. U = mul(q - 1, tx, ty). If 1 < gcd(Z_U, N) < N: `Factor(g)`. If gcd(Z_U, N) = N: `Reject`.
4. Accept iff X_U = tx Z_U^2 and Y_U = -ty Z_U^3 (mod N), that is U = -T; else `Reject`.

Soundness: on Accept, for every p | N, Z_T and Z_U are units mod p, so T_p = [s]P_p != O and U_p = [q - 1]T_p
(both computed correctly), and the congruences give U_p = -T_p as affine points, so [q]T_p = O. That is the
hypothesis of 2.3.1. Completeness for prime N: if P has order dividing m = s q and [s]P has order q, the
computation of [q - 1]T never meets O or T = +-base (in left-to-right double-and-add an addition always
follows a doubling, so the accumulator before an addition is jT with 2 <= j <= q - 2), and no point is
2-torsion (odd prime order), so Z stays nonzero and the check accepts. Review 1 re-derived both arguments.

Measured cost of the whole check (prototype, debug): about 31 µs per bit at 128 bits; the verification
of a 3-level certificate for 2^127 - 1 takes 9 ms, of the 6-level PARI certificate for nextprime(2^200)
45 ms.

#### 2.3.3 Certificate verification (`EcppCertificate::verify`)

```
verify(cert):
  if cert.steps is empty: return cert.n < 2^64 and is_prime_u64(cert.n)
  cur = cert.n
  for st in steps:
     require st.n == cur, cur > 1, gcd(cur, 6) == 1
     require t^2 < 4 cur                      (PARI's condition; implies m = cur + 1 - t > 0)
     require s >= 1 and s | m; q = m / s
     r = iroot4_ceil(cur)                      (least r with r^4 >= cur)
     require q > (r + 1)^2                     (implies q > (cur^(1/4) + 1)^2)
     a, x, y reduced mod cur; b = y^2 - x^3 - a x mod cur
     require gcd(4 a^3 + 27 b^2 mod cur, cur) == 1
     require order_check(cur, a, x, y, s, q) == Accept
     cur = q
  return cur < 2^64 and is_prime_u64(cur)
```

Every arithmetic step reduces mod `cur` first, so arbitrary input (negative, huge, s = 0) gives `false`
and never a panic. The verifier uses neither the discriminant list, nor any class polynomial, nor any
randomness, so a wrong polynomial can make the prover fail but cannot make a certificate valid. The leaf
convention (q_last < 2^64, proved by `is_prime_u64`) is PARI's ("q_ell <= 2^64 is prime"), so certificates are
exchangeable. The verifier is iterative (no recursion).

`iroot4_ceil(N)`: r = floor(sqrt(floor(sqrt(N)))) (= floor(N^(1/4))) with `num_integer::Roots::sqrt` on
BigInt; if r^4 < N then r + 1.

Checked in the prototypes: the four PARI certificates (2^127 - 1, nextprime(10^40), nextprime(2^64),
nextprime(2^200)) are accepted; each with x + 1 in the first step is rejected. PARI accepts the prototype's
certificates of 2^89 - 1, 2^127 - 1, the P-256 prime and 2^521 - 1 (`primecertisvalid` = 1 for 1, 3, 10-12
and 25 levels), and review 1 had PARI accept five more.

#### 2.3.4 Prover

Input N: BPSW-probable prime, N >= 2^64, gcd(N, 6) = 1 (BPSW and the trial division ensure it).

Parameters (constants in `ecpp/mod.rs`, documented):
- `TRIAL_BOUND = 2^16`: m is stripped of the primes below 2^16 (6542 primes, batched into products below
  2^64, one `BigInt % u64` per batch; the prototype's `Batches`).
- `H_TOP = 100`: the top level (depth 0) scans the discriminant list up to h = 100, and so does every frame
  whose N has more than `H_INNER_BITS = 2048` bits; `H_INNER = 20`: every other frame scans up to h = 20. Below
  2048 bits a failing inner level is cheaper to replace (the parent tries its next candidate) than to extend
  (the rows with h in 21..100 are 23 times as many and need slower root finding); the top level cannot
  backtrack. Above 2048 bits an h <= 20 scan finds no candidate for 50 % (3072 bits) to 83 % (4096 bits) of the
  primes, so a parent would lose most of its candidates to failing children; those frames scan to h = 100.
  Section 2.3.7 has the measurements behind these numbers.
- `MAX_FAILED_SUBTREES = 64` (total, per call): after that many failed inner levels, return `Unresolved`.
  Measured: at most 10 in one proof at 2048 bits (65 in 12 proofs), at most 3 at 1536 bits (8 in 5 proofs), 3
  in the 3072-bit proof.

The four limits are fields of a `pub(crate) struct ProverParams` with a `const DEFAULT`; `prove_primality`
calls `prove_chain(n, &ProverParams::DEFAULT)`, and unit tests pass other values (`ecpp_driver_backtracks`).

Rows are scanned in the list order (h ascending, then |D| ascending). One level ("frame") works as follows.

Frame state: N; bound = (iroot4_ceil(N) + 1)^2; hmax (H_TOP or H_INNER); next_row (index into the list);
cands (the candidates of the current row, sorted by q ascending, consumed from the front).

`advance(frame)` returns `Found(step, q)`, `Composite(witness)` or `Exhausted`:
1. While cands is empty: if next_row is past the end or its h > hmax, return `Exhausted`. Otherwise take the
   row (D, h, prime discriminants of D) and:
   a. Skip unless Kronecker (D/N) = 1.
   b. Genus filter: skip unless (p*/N) = 1 for every prime discriminant p* of D (D = prod p*, p* in
      {-4, 8, -8} and (-1)^((p-1)/2) p for odd p; Cox §3). N is the norm of an element of the maximal order
      only if every genus character is 1, so the filter only removes rows without a solution.
   c. Cornacchia (Cohen 1.5.3): x0 = sqrt(D mod N) by `sqrt_mod_prime_checked` (made `pub(crate)` in
      integer.rs); if x0 and D differ in parity, x0 = N - x0; Euclid on (2N, x0) until the remainder is
      <= floor(sqrt(4N)); with b the last remainder, require |D| | 4N - b^2 and c = (4N - b^2)/|D| a square:
      (u, v) = (b, sqrt c). An `Err` from the square root means N is composite: find a strong-test witness
      (section 2.3.6) and return Composite.
   d. Traces: t in {u, -u}; D = -4: also {2v, -2v}; D = -3: also {+-(u + 3v)/2, +-(u - 3v)/2}.
   e. For each t: m = N + 1 - t; (k, q) = strip(m); keep if k >= 2, q > bound and `is_probable_prime(q)`.
      k >= 2 makes q <= m/2 < N, so the chain ends.
   f. Sort the kept candidates by q ascending into cands.
2. Pop the first candidate (q, m, t, k, row):
   - H_D = `hilbert::class_polynomial(D)` (2.3.5; cached). None (the rounding check failed at every
     precision): next candidate. A degree different from h: next candidate (cannot happen with the checked
     list).
   - A root j of H_D mod N (2.3.5). A proper gcd met while inverting: Composite(Divisor). No split after 64
     attempts: next candidate.
   - Curves: D = -3: (a, b) = (0, g^i), i = 0..5, with g a non-square that is not a cube
     (g^((N-1)/3) != 1). D = -4: (a, b) = (g^i, 0), i = 0..3, g a non-square. Else c = 1728 - j,
     (a, b) = (3 j c, 2 j c^2) and the twist (a g^2, b g^3), g a non-square. (j(y^2 = x^3 + 3k x + 2k) = j0
     for k = j0/(1728 - j0); scaling by c gives the quadratic twist by c, so the pair covers both twists.)
     g: the least g >= 2 with these properties, searched up to 2^20 (none found: next candidate).
     With e = gcd(4a^3 + 27b^2, N): e = N skips the curve, 1 < e < N is Composite(Divisor(e)).
   - For each curve, up to 4 random points: x uniform mod N, f = x^3 + a x + b; f = 0 mod N: new x;
     (f/N) = 0 otherwise: Composite(Divisor(gcd(f, N))); require (f/N) = 1 (up to 64 tries, else next
     curve), y = sqrt(f) (`sqrt_mod_prime_checked`; Err: Composite via 2.3.6).
     `order_check(N, a, x, y, k, q)`: Accept: return Found(EcppStep { n: N, t, s: k, a, x, y }, q); Retry:
     next point; Reject: next curve (wrong twist); Factor(g): Composite(Divisor(g)).
   - No curve accepted: next candidate (for a prime N and a correct H_D this does not happen; it is not a
     proof of anything, so no claim is made). Then go back to 1.

Iterative driver (review 1, m11; no recursion, so the depth is not limited by the thread stack):

```
prove_chain(N) -> Ok(steps) | Composite(w) | Unresolved(why):
  stack = [Frame::new(N, H_TOP)]; steps = []; failed = 0
  (hmax(q) = H_TOP if bits(q) > H_INNER_BITS, else H_INNER)
  loop:
    match advance(stack.last_mut()):
      Found(step, q):
        steps.push(step)
        if q < 2^64: return Ok(steps)          (is_probable_prime(q) is is_prime_u64(q) there: exact)
        stack.push(Frame::new(q, hmax(q)))
      Composite(w):
        if stack.len() == 1: return Composite(w)            (N itself)
        stack.pop(); steps.pop()                            (q was a BPSW pseudoprime; parent goes on)
      Exhausted:
        if stack.len() == 1: return Unresolved("no curve found with class number <= 100 for N")
        stack.pop(); steps.pop(); failed += 1
        if failed == MAX_FAILED_SUBTREES: return Unresolved("64 failed subtrees")
```

Invariant: `steps[i]` is the accepted step of `stack[i]`, and its q is `stack[i + 1].n`; so on `Ok` the steps
form a chain from N to a q below 2^64. Memory: one frame per level (a few BigInts and the current row's
candidates); 120 frames at 2048 bits, 186 at most at 3072 bits.

`prove_primality(n)`, in this order: n < 2: NotPrime(BelowTwo); n is one of the primes below 1000: Prime
(no steps); a prime p < 1000 divides n: NotPrime(Divisor(p)); n < 2^64: `is_prime_u64(n)` gives Prime (no
steps), otherwise NotPrime(MillerRabin(a)) with the first failing base a in 2..37 (one exists: Sorenson-
Webster); 2^64 <= n < psi_12: the 12 bases (a failing base is the witness), then `prove_chain`; n >= psi_12:
BPSW, a failure gives its witness, then `prove_chain`. An Ok of `prove_chain` gives
cert = EcppCertificate { n, steps }; then `cert.verify()` (always, review 1 m12; measured 2 to 7 % of the
proof time: 1.4 s of 21 s at 1536 bits, 3.3 to 4.1 s of 60 to 115 s at 2048, 16 s of 957 s at 3072). If it is false, that is a prover bug:
`debug_assert!` fails, and release returns `Unresolved("internal error: the ECPP certificate failed
verification")`, never Prime. So `Primality::Prime` always carries a verified certificate.

Randomness: a SplitMix64 generator seeded with (N mod 2^64) xor 0x9E3779B97F4A7C15, so proofs and
certificates are reproducible, and whether a given N is proven does not depend on earlier calls.
Random residues mod N: concatenate ceil(bits/64) + 1 words, reduce mod N.

Arithmetic: internals work on `num_bigint::BigInt` (the crate's existing dependency; `Integer` wraps it),
converting at the API boundary. No new wrapper types.

Optional speed-up (not required; the measured numbers do not use it): in step 1c, compute sqrt(p*) mod N
once per frame for each prime discriminant met, and take sqrt(D) as their product (the FastECPP trick of
Franke, Kleinjung, Morain, Wirth). The scan of rows (1a-1e) is 60 % of the time at 2048 bits.

#### 2.3.5 Discriminant list and Hilbert class polynomials at run time

**The list.** File `rustmath-integers/data/ecpp_discriminants.txt`, generated by the committed script
`rustmath-integers/data/ecpp_discriminants.gp` (the content of `$R1/dlist.gp`; it starts with
`default(parisizemax, 2^30)` and takes 83 s): line h (h = 1..100) is "h |D_1| |D_2| ...", the absolute values of
all fundamental discriminants D < 0 with h(D) = h, ascending. 42,272 discriminants, 273,422 bytes, largest
|D| = 2,383,747. Completeness for h <= 100 is Watkins' theorem (the list of all imaginary quadratic fields
with h <= 100); the script prints the counts per h, and their total 42,272 is Watkins' number. The class
numbers come from PARI `qfbclassno`; `$R1/dcheck.rs` checks them independently and exactly, by counting the
primitive reduced forms of every discriminant with |D| <= 2.4·10^6: all 100 lines agree. Embedded with
`include_str!`. Each line is parsed on first use (a `OnceLock` per line; the prime discriminants of each D
are computed then by trial division, at most 1544 for |D| <= 2,383,747), so a proof that never leaves
h <= 20 parses 9.3 KB.

Size options measured (bytes of the list up to h): h <= 20: 9,302; 32: 25,219; 48: 59,310; 64: 107,890; 100:
273,422. Binary cost: every test binary that reaches `is_prime` carries the list, so 228 x 0.27 MB = 62 MB of
`target/` (was 300 MB with the polynomial table). h <= 64 would save 38 MB but lose the coverage measured
in 2.3.7 for h in 65..100 (one of the 20 primes at 3072 bits has its least top-level h at 68, so it would be
unresolved with h <= 64); the design keeps h <= 100.

**Hilbert class polynomial H_D** (`hilbert::class_polynomial(d: i64) -> Option<Arc<Vec<BigInt>>>`,
coefficients low to high, monic of degree h(D)):
1. D = -3: X. D = -4: X - 1728.
2. Reduced forms (Cohen §5.3): all (a, b, c) with b^2 - 4ac = D, |b| <= a <= c, b >= 0 if |b| = a or a = c
   (they are primitive because D is fundamental; the code checks gcd(a, b, c) = 1 anyway). Their number is
   h(D); if it differs from the list's h, return None.
3. Precision: est = sum over the forms of (pi sqrt|D| / (a ln 2) + 3.5) (f64; an upper bound for
   log2 of prod (1 + |j_i|), which bounds every coefficient, because |j(tau) - 1/q| <= 2079 for
   Im tau >= sqrt(3)/2 (the sum 744 + 196884 |q| + 21493760 |q|^2 + ... of the absolute q-expansion
   coefficients of j at |q| = e^(-pi sqrt 3)) and |1/q| = e^(pi sqrt|D| / a) >= e^(pi sqrt 3) = 230.7). Working precision
   P = ceil(est) + guard, guard = 2 floor(sqrt(est)) + 64 + 2 ceil(log2(h + 1)).
4. Fixed-point complex numbers: a pair of BigInt (re, im) standing for (re + i im) / 2^P. Products are
   (x y) >> P, quotients (x << P) / y, so each operation errs by at most a few units in the last place.
   Constants: pi by Machin's formula 16 atan(1/5) - 4 atan(1/239) (Gregory series, 16 guard bits);
   sqrt|D| as isqrt(|D| << 2P).
5. For each form with b >= 0 (the form (a, -b, c) gives the complex conjugate value):
   w = 1/q = exp(pi sqrt|D| / a + i pi b / a) by `cexp` (z halved k times until |z / 2^k| < 2^-s with
   s = floor(sqrt(P)/2) + 1, Taylor series, then k squarings); q = 1/w;
   E(q) = sum over k in Z of (-1)^k q^(k(3k-1)/2) = prod (1 - q^n) (Euler's pentagonal theorem; terms
   generated with q^(3k+1) and q^k, stopped when below 2^-P); R = E(q^2)/E(q); R24 = R^24 (two products,
   three squarings);
   j = (1 + 256 q R24)^3 w / R24.
   This is j(tau) for tau = (-b + sqrt D)/(2a): with g = Delta(2 tau)/Delta(tau) = q R^24 and Weber's
   f_2^24 = 2^12 g, j = (f_2^24 + 16)^3 / f_2^24 = (256 g + 1)^3 / g. Computing 1/g as w / R24 keeps the relative
   precision of the huge values (|j| up to 2^9215 in the measured cases); q itself only enters through
   correction terms below 1.
6. H_D = product of (X - j) over the real values (b = 0, b = a or a = c) and of (X^2 - 2 Re(j) X + |j|^2)
   over the conjugate pairs, in fixed point.
7. Rounding check: each coefficient c is rounded to the nearest integer r, and |c - r 2^P| must be below
   2^(P - 30); the leading coefficient must be 1 and the degree h. On failure, double the guard and retry
   (at most two retries), then return None.
8. Cache: a process-wide `Mutex<Vec<(i64, Arc<Vec<BigInt>>)>>` (`Vec::new` is const), at most 256 entries,
   oldest dropped first, locked with `lock().unwrap_or_else(PoisonError::into_inner)`.

Correctness: when the rounding check passes, the error of every coefficient is below 1/2 in practice
(Enge 2009 bounds the error of the same evaluation), so the integers are H_D. But nothing depends on that for
soundness: H_D only proposes curves, and the order check and the verifier decide. A wrong H_D makes the prover
waste a candidate, never accept a wrong step. The prototype's polynomials were compared with PARI: 54
discriminants, h = 1 to 100, |D| up to 1,856,563 (the first, middle and last |D| of 16 class numbers), all equal
to `polclass(D)` (`$R1/hpcheck.gp`). The tests check a sample (section 4).

Cost (prototype, `polytime`): debug h = 20: 17 ms (|D| = 455) to 109 ms (|D| = 58507, P = 1654 bits);
release h = 40: 11 to 45 ms; h = 64: 33 to 340 ms; h = 100: 0.22 to 1.3 s (|D| = 1,856,563, P = 9851 bits,
largest coefficient 9215 bits). Only the rows that have a candidate are computed, and the measured
`t_poly` is at most 0.04 s per 2048-bit proof.

**Root of H_D mod N** (Cohen §1.6.1; H_D splits into distinct linear factors mod N when N is prime and
4N = u^2 + |D| v^2, except when N divides its discriminant): f = H_D mod N, monic; while deg f > 1:
delta random, g = gcd((x + delta)^((N-1)/2) - 1 mod f, f); if 0 < deg g < deg f, replace f by the factor of
smaller degree. The power is computed left to right with squarings mod f; multiplying by x + delta is linear.
Dense polynomials over `Vec<BigInt>`, private to `ecpp/polymod.rs` (rustmath-polynomials depends on
rustmath-integers, so it cannot be used). Lazy reduction: a product accumulates the unreduced sums and reduces
each coefficient once; the remainder by a monic f subtracts multiples of f without reducing and reduces only
the coefficient it divides by. Measured (`rootbench`, release): 1024-bit N: h = 10 0.10 s, h = 20 0.33 s,
h = 40 1.5 s; 2048-bit N: 0.59 s, 2.3 s, 9.1 s (about h^2 log N). Kronecker substitution would be faster
for large h; it is not needed for the measured times.

#### 2.3.6 Composite verdicts inside the prover

Every NotPrime carries a checkable witness. When the prover meets an impossibility for a prime modulus
(Tonelli-Shanks bound exceeded, a square root that does not square back, Cornacchia inconsistency), N is
composite; the prover then searches a strong-test witness among the bases 3, 5, 7, ... (at least 3/4 of all
bases are witnesses for an odd composite, Rabin 1980) for up to 64 bases and returns `MillerRabin(a)`; if
none is found it returns `Unresolved("composite (prover contradiction), no witness found")`. A proper gcd
gives `Divisor(g)` directly. In practice none of this runs (no BPSW pseudoprime is known); unit tests reach
it with a crafted call of `prove_chain` on a composite past the BPSW gate in a `#[cfg(test)]` helper (for
example N = psi_12).

#### 2.3.7 Cost and coverage (prototype `$R1/ecpp_rt.rs`, H_TOP = 100, H_INNER = 20, H_INNER_BITS = 2048)

Full proofs of random BPSW primes (`stats`; every certificate verified; release and debug builds as in
`$R1/build.sh`):

| Bits | Build | Proven / tried | Mean (max) | Levels | Inner failures | Largest top-level h |
|---|---|---|---|---|---|---|
| 128 | release / debug | 10 / 10 | 0.024 s (0.038) / 0.065 s (0.094) | 3.6 | 0 | 2 |
| 256 | release / debug | 10 / 10 | 0.070 s (0.13) / 0.36 s (0.52) | 9.6 | 0 | 5 |
| 512 | release / debug | 8 / 8 | 0.41 s (0.57) / 3.2 s (4.5) | 26 | 0 | 1 |
| 768 | release | 6 / 6 | 1.7 s (2.4) | 40 | 0 | 10 |
| 1024 | release | 10 / 10 | 3.3 s (4.3) | 58 | 2 | 10 |
| 1024 | debug | 2 / 2 | 38 s (41) | 57 | 0 | 5 |
| 1536 | release | 5 / 5 | 20.6 s (27.8) | 85 | 8 | 14 |
| 1536 | debug | 1 / 1 | 6.4 min | 84 | 2 | 4 |
| 2048 | release | 12 / 12 | 75.8 s (115.3) | 120 | 65 (at most 10 in one proof) | 25 |
| 2048 | debug | 1 / 1 | 15.9 min | 119 | 1 | 12 |
| 3072 | release | 2 / 2 | 957 s (1031) | 182 | 8 (3 and 5) | 4 (inner frames up to 69) |

Debug is about 8 to 15 times slower than release (512, 1024, 1536 and 2048 bits; the larger ratios were
measured under load). Times are upper bounds: all runs after the first shared the 4-core machine with up to
four other jobs. Up to 2048 bits the proof time grows like (log N)^4.5: x 6.2 from 1024 to 1536 bits, x 3.7
from 1536 to 2048. Within the 2048-bit proofs, the row scan
takes 62 %, root finding 27 %, curves and points 10 %, the class polynomials 0.04 %. Verification adds 3.3 to
4.1 s (5 %).

Top-level coverage (`count` and `minh` modes, random primes; a prime with no top-level candidate up to a
limit is `Unresolved` with that limit):

| Bits | Primes | Mode | No top-level candidate with h <= 20 | h <= 32 | h <= 48 | h <= 64 | h <= 100 | Largest least h | Mean candidates, h <= 20 / h <= 100 |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 100 | minh | 3 (3 %) | 0 | 0 | 0 | 0 | 24 | |
| 1536 | 40 | count | 2 (5 %) | 1 (2.5 %) | 1 (2.5 %) | 0 | 0 | 61 | 3.6 / 16.3 |
| 2048 | 40 | count | 6 (15 %) | 1 (2.5 %) | 0 | 0 | 0 | 37 | 2.3 / 11.5 |
| 3072 | 20 | minh | 10 (50 %) | 6 (30 %) | 2 (10 %) | 2 (10 %) | 1 (5 %) | 68 | |
| 4096 | 12 | minh | 8 (67 %) | 5 (42 %) | 2 (17 %) | 1 (8 %) | 1 (8 %) | 56 | |

(`count` lists every top-level candidate up to h = 100; `minh` stops at the first one, which is all that
coverage needs and is much faster. Logs: `$R1/count1536.*`, `count2048.*`, `minh1024.*`, `minh3072.*`,
`minh4096.*`; `$R1/table.py` builds this table. A count run at 4096 bits takes about 19 min per prime, so 4096
has only the minh run.) The candidates of a prime are not independent: a prime whose small Kronecker
symbols (p*/N) are mostly -1 loses most discriminants to the genus filter at once, which is why the
no-candidate rates fall more slowly than a Poisson model with the mean predicts.

Old static table for comparison (review 1 and the first version, h <= 20 without 3 | D above h = 6): 3 % at
1024 bits, 6 of 16 at 2048 bits.

What happens at 4096 bits and above: proofs above 2048 bits run with the same code (frames above 2048 bits scan to
h = 100, 2.3.4). Measured in release: two 3072-bit proofs took 882 s and 1031 s
(182 levels on average, root finding 38 % of the time, because frames above 2048 bits use rows up to h = 69).
From 2048 to 3072 bits the time grew by 12.6, like (log N)^6, so a 4096-bit proof should take about 1.5 hours
in release and a day in debug. A 4096-bit run (log `$R1/stats4096_rel.err`) had reached depth 21 (3876 bits)
after 31 min, with one frame at 3861 bits that scanned all 42,272 rows without a candidate (its parent went on
to the next candidate); with a per-level cost growing like bits^5, the remaining levels cost about 1.5 times
the first 21, which gives the 1.5 hours. The run was stopped there. 8192 bits would take days. The top level has no
candidate with h <= 100 for about 1 prime in 16 at 3072 and 4096 bits (table above), and the share grows with
the size, since the number of candidates falls like 1/log N. For those primes `prove_primality` is
`Unresolved`, `try_is_prime` is `Err` and `is_prime` panics; no answer is ever wrong. No size limit is
imposed: a caller that needs a fast answer uses `is_probable_prime` (MAGMA `IsPrime(n: Proof := false)`) or
bounds the size itself, as the REPL does (J1a).

### 2.4 `is_prime`, `try_is_prime` and the cache

```
try_is_prime(n):
  n < 2: Ok(false); n < 2^64: Ok(is_prime_u64(n))
  a prime p < 1000 divides n: Ok(false)
  n < psi_12: Ok(strong tests to 2..37 all pass)
  !bpsw(n): Ok(false)
  cache contains n: Ok(true)
  match prove_primality(n): Prime(_) => insert n in the cache, Ok(true); NotPrime(_) => Ok(false);
        Unresolved(why) => Err(NotSupported(format!("is_prime: {n} is a BPSW probable prime, but ECPP found
        no proof ({why}); use prime::prove_primality or prime::is_probable_prime")))
is_prime(n) = try_is_prime(n).unwrap_or_else(|e| panic!("{e}"))
```

The cache: `static PROVEN_PRIMES: Mutex<VecDeque<Integer>> = Mutex::new(VecDeque::new());` (both
constructors are `const`), at most 64 entries, only primes >= psi_12 whose certificate `verify` accepted
(that is every `Prime` of `prove_primality`), oldest dropped first, lock with
`lock().unwrap_or_else(PoisonError::into_inner)`. It holds only what was proven, so it cannot change an
answer; it makes repeated contract checks on one large prime (the finite-field square roots, Berlekamp,
Cantor-Zassenhaus and the drivers call `is_prime(p)` per call) cost one proof per process.

Docs of `is_prime` and `try_is_prime` state the cost table of section 0 and the `Unresolved` rates of 2.3.7
("no unresolved prime up to 2048 bits in the experiments; at 3072 and 4096 bits 2 of 32 primes had no
top-level candidate and were unresolved").

Why panic and not a guess: `is_prime` returns `bool`; the honest options for an undecided input are a panic
or an `Err`. Every caller that returns `Result` or `Option` and can see a large prime uses the `try_`
functions after I1 (section 3.1). The panic is left to callers whose result type cannot carry it (`bool`
predicates such as `ZMod::is_field`, `Integer::is_prime`), to callers that already panic on a non-prime
(asserts), to the convenience wrappers `next_prime`, `previous_prime`, `prime_range`, `random_prime` (each has
a `try_` version), and to the complete factorization drivers (I2 gives them a `Result`).

### 2.5 Sieve, prime functions and Miller-Rabin rounds (S6)

New module `rustmath-integers/src/sieve.rs` (C. Bays, R. H. Hudson, *The segmented sieve of Eratosthenes
and primes in arithmetic progressions to 10^12*, BIT 17 (1977) 121-127; J. B. Rosser, L. Schoenfeld,
*Approximate formulas for some functions of prime numbers*, Illinois J. Math. 6 (1962) 64-94, (3.13):
p_n < n (ln n + ln ln n) for n >= 6; MAGMA ch. 18 §18.9.2 `PrimesUpTo`, `PrimesInInterval`, `NthPrime`):

```rust
/// Calls f(p) for each prime lo <= p < hi in ascending order until f returns ControlFlow::Break.
pub fn for_each_prime(lo: u64, hi: u64, f: impl FnMut(u64) -> std::ops::ControlFlow<()>);
pub fn primes_in_range(lo: u64, hi: u64) -> Vec<u64>;   // [lo, hi)
pub fn primes_up_to(b: u64) -> Vec<u64>;                 // MAGMA PrimesUpTo(B): [2, b]
pub fn primes_in_interval(b: u64, e: u64) -> Vec<u64>;   // MAGMA PrimesInInterval(b, e): [b, e], e = u64::MAX allowed
pub fn prime_count(x: u64) -> u64;                       // pi(x)
pub fn nth_prime_u64(n: u64) -> Option<u64>;             // None for n = 0 or p_n >= 2^64
pub(crate) fn small_primes() -> &'static [u32];          // primes below 2^16, OnceLock (ECPP, trial division)
```

Algorithm: odd-only byte sieve over segments of 2^15 odd numbers (32 KiB), base primes up to
L = min(isqrt(hi - 1), 2^20) from a plain sieve; for each base prime p, start at the first odd multiple of p
that is >= max(lo, p^2). If isqrt(hi - 1) > 2^20 (hi > 2^40), the survivors are confirmed with
`is_prime_u64` (exact). No float in the bounds: `isqrt` by integer correction.

Overflow-free arithmetic near 2^64 (review 1, m2): the inclusive interval [b, e] with e <= u64::MAX is the
internal form (`primes_in_range(lo, hi)` is [lo, hi - 1] for hi > lo); segment ends are computed as
`seg_lo.saturating_add(2 * SEG - 1).min(e)`, the loop stops after the segment that contains e (no
`seg_lo + 2 SEG` past it); p^2 is compared as `p <= e / p` (or in u128); the first multiple is computed in u128
and skipped when it exceeds e; indices into the segment are `(m - seg_lo) / 2` with m <= e. Prototype
`$X/sieve_rs.rs`: `prime_count(10^8)` = 5761455 in 0.75 s debug (0.14 s release), `nth_prime(10^7)` = 179424673
in 1.5 s debug, `nth_prime(10^6)` 0.11 s, primes in [10^12, 10^12 + 100) correct.

Rewired functions in `prime.rs` (signatures unchanged unless stated):
- `nth_prime(n: usize) -> Integer`: n = 0 panics as before; otherwise `nth_prime_u64` with the upper bound
  max(15, n (ln n + ln ln n) + 3) for n >= 6 (Rosser-Schoenfeld); p_n >= 2^64 (n > pi(2^64) =
  425656284035217743, OEIS A007053) panics with that message (HEAD never returned there). MAGMA promises
  primes up to 10^10; `nth_prime(455052511)` (p_n just below 10^10) is about 75 s in debug.
- `prime_pi(x: &Integer) -> usize`: 0 for x < 2; `prime_count` for x <= u64::MAX; above it panics with
  "prime_pi: x = ... exceeds 2^64; the sieve cannot reach it and sublinear counting (Lagarias-Miller-
  Odlyzko) is not implemented" (HEAD never returned there either).
- `try_prime_range(start, stop)` (half-open, as now): the part below 2^64 by `primes_in_range` (exact); the part
  above 2^64 by windows of 2^16 odd candidates pre-sieved with `small_primes()` (residues of the window start
  by `BigInt % u32`); each survivor `is_probable_prime`; each probable prime `try_is_prime`: Ok(true) listed,
  Ok(false) dropped (a BPSW pseudoprime), Err returned (the list must be complete; MAGMA PrimesInInterval
  semantics). So the range proves only the primes it returns, and composites cost one BPSW. `prime_range`
  panics on Err.
- `primes_first_n(n)`: like `nth_prime`, collecting.
- `try_next_prime(n)`: n < 2 gives 2 (adjacent fix); n < 18446744073709551557 (the largest u64 prime): scan odd
  candidates with `is_prime_u64`; above: windows of 2^12 odd candidates pre-sieved by the primes below 2^12
  (MAGMA: "candidates are pre-sieved by small primes"), each survivor `is_probable_prime`, the first probable
  prime through `try_is_prime`: Ok(true) returned, Ok(false) next candidate, Err returned. MAGMA
  `NextPrime(n)` is the least prime > n (gp `nextprime(n + 1)`). `next_prime` panics on Err.
- `try_previous_prime(n)`: Ok(None) for n <= 2 (unchanged), the mirror image, crossing from big to u64 at 2^64.
- `try_random_prime(a, b)` (review 1, m6): Ok(None) if a >= b. Collect the probable primes of [a, b) (below 2^64
  the sieve, exact; above, the windows of `try_prime_range` with `is_probable_prime` only, no proofs). Loop:
  none left: Err(NotSupported("random_prime: the k probable primes in [a, b) could not be proven")) if a pick was
  dropped as unresolved, else Ok(None); otherwise draw one uniformly (`rand::thread_rng`, as now), `try_is_prime`:
  Ok(true) return it, Ok(false) or Err drop it and draw again. The result is uniform over the primes of
  [a, b) that the prover proves (all of them in every experiment up to 2048 bits). The doc says so.
  `random_prime` panics on Err. The tracker remark "enumerates every prime" stays true for the probable primes
  (section 7).
- `is_prime_miller_rabin(n, rounds)`: n <= 3 and even n as now; then one strong test to each of the first
  max(rounds, 1) primes (2, 3, 5, ...; taken from `small_primes()`, so at most 6542 bases) that are below n;
  composite on the first failure. `x <- x^2` by multiplication, not `mod_pow(&2)`. Docs: MAGMA
  `IsProbablePrime(n: Bases := rounds)` with fixed instead of random bases; a proof for n < psi_12 when
  rounds >= 12; `false` always a proof. HEAD's `is_prime_miller_rabin(3215031751, 4)` is `false` because all 12
  bases run; after the fix it is `true` (3215031751 = 151 * 751 * 28351 is a strong pseudoprime to 2, 3, 5,
  7), which is the documented meaning of a 4-base probable-prime test.
- `prime_power(n)` / `is_prime_power(n)` (adjacent; HEAD factors n by trial division, tracker row
  `ch018.IsPrimePower` "slow for large n"): n <= 1: None; `Integer::is_perfect_power` gives n = c^m with c
  not a perfect power: Some((c, m)) iff `is_prime(c)`; otherwise Some((n, 1)) iff `is_prime(n)`.
- `pollard_p_minus_1(n, bound)` (`:650-653`, review 1, B3): stage 1 runs over every prime p <= bound, from
  `sieve::for_each_prime(2, bound as u64 + 1, ..)`; for each p the largest power p^e <= bound, as now. No other
  change, so the function finds exactly what HEAD finds, without one `is_prime` per integer below `bound`.

`fast_arith::prime_range(start, end)` (public, re-exported at the crate root) becomes
`sieve::primes_in_range(start, end)`; its tests keep their values. `ArithInt::compute_small_primes` and
`ArithLLong::compute_small_primes` become `sieve::primes_up_to(n)` conversions.

### 2.6 Word-size dedup (fast_arith)

- Move `is_prime_u64` from `fast_arith.rs:65-99` to `primality.rs` (public). Keep `pow_mod_u64` in
  fast_arith (`pub(crate)`), imported by primality.rs and integer.rs (`integer.rs:1265` changes its import).
- `ArithLLong::is_prime(n)`: `n >= 2 && is_prime_u64(n as u64)`; delete `ArithLLong::miller_rabin`.
- `ArithInt::is_prime(n)`: `n >= 2 && is_prime_u64(n as u64)`. Removes the i32 overflow (I3 tests the same
  function, `$I3/t3_fast.rs`; if I3 already patched the loop, this replaces the patched loop).
- `prime::is_prime` uses `is_prime_u64` below 2^64.

### 2.7 Square roots: `try_sqrt_mod_prime` and one primality test

`rustmath-integers/src/integer.rs`:

```rust
/// Square root modulo p without testing p for primality (for callers that already proved p prime).
/// Precondition: p is prime. Ok(Some(r)): 0 <= r < p and r^2 = self (mod p), checked. Ok(None): the Jacobi
/// symbol (self/p) is -1, a proof that self is not a square mod p (for any odd p). Err(InvalidArgument):
/// p < 2, p even and > 2, p a perfect square (checked first), or the computation proved p composite
/// (gcd(self, p) > 1 with p not dividing self, the Tonelli-Shanks order bound ran out, or the root did not
/// square back). For a composite p that is not a square the call still ends: a quadratic non-residue
/// modulo p (Jacobi symbol -1) exists below p, and the search stops at the first one; there is no time
/// bound beyond that for composite p.
pub fn try_sqrt_mod_prime(&self, p: &Self) -> Result<Option<Self>>;
```

It is `sqrt_mod_prime_checked` behind the argument checks, including the perfect-square check (review 1,
m5: for a square p every Jacobi symbol is 0 or 1, so the non-residue search at `integer.rs:1180-1186` would run
to p). `sqrt_mod_prime_checked` becomes `pub(crate)` (ECPP uses it on BPSW probable primes, which are never
squares).

`rustmath-finitefields/src/integer_mod.rs:651` becomes:
```rust
pub fn square_root_mod_prime(a: &Integer, p: &Integer) -> Option<Integer> {
    assert_prime("square_root_mod_prime", p);   // the only primality test (a proof)
    match a.try_sqrt_mod_prime(p) {             // reduces a mod p, one Kronecker symbol inside
        Ok(r) => r,
        Err(e) => panic!("square_root_mod_prime: p = {p} is not prime ({e})"),
    }
}
```
`sqrt_mod_prime_checked` reduces `a` first and returns `a` for p = 2 before any Kronecker symbol, which is
the point of the current comment at `:654-655` (3 is a square mod 2 although (3/2) = -1). The
should-panic message "p = 15 is not prime" still comes from `assert_prime`.

`unit_sqrts_odd` (`integer_mod.rs:814-815`, review 1, m7) calls `u.try_sqrt_mod_prime(p)` instead of
`square_root_mod_prime(u, p)`: `Ok(None)` gives no roots, `Ok(Some(y))` lifts y as now, `Err(e)` panics with
"internal error: p = {p} was proven prime but the square root failed ({e})" (unreachable for a prime).
Its callers reach it only after `checked_prime_power_args` proved p prime, so the prime-power functions test p
once.

Other residue functions in integer.rs (MAGMA §18.7 `LegendreSymbol`: "a fast probabilistic primality test is
applied to m"; §18.12.1 `Modsqrt`):
- `legendre_symbol` (`:702`), `is_quadratic_residue_prime` (`:833`), `sqrt_mod_prime` (`:881`),
  `quadratic_residues` (`:907`): `is_probable_prime` instead of `is_prime`. Their answers already verify
  themselves (a root is squared back, a Jacobi -1 is a proof for any odd modulus), so a probable prime is the
  MAGMA contract and keeps them cheap. Docs: "BPSW; exact below psi_12; no BPSW pseudoprime is known".
- `binomial_mod` (`:1000`): `try_is_prime(p)?` (a proof; Lucas' theorem gives a wrong value for a composite
  modulus, so a probable prime is not enough). An unresolved p gives `Err(NotSupported)`.

### 2.8 Factorization drivers (I1 changes, I2 interface)

I1 changes:
- Loop guards and divisor checks keep `is_prime` (`prime.rs:829`, `:833`; `quadratic_sieve.rs:572`;
  `ecm.rs:523`, `:534`): every returned factor is now proven prime, and psi_12 is split
  (`factor_pollard_rho(psi_12)` = [(399165290221, 1), (798330580441, 1)]: Floyd finds 399165290221 after
  188311 iterations, 4.6 s in debug; Brent alone 3.4 s, `$X/t11_rho.rs`).
- Early returns use `is_probable_prime`: `pollard_rho` (`:556`), `pollard_p_minus_1` (`:641`), `ecm_factor`
  (`ecm.rs:441`), `quadratic_sieve_factor` (`quadratic_sieve.rs:450`). Their docs: "None if n is a probable
  prime (BPSW) or no factor was found".
- `split_odd_composite`'s `debug_assert` (`:871`) becomes `debug_assert!(m.is_odd() && *m > Integer::one())`
  (review 1, m4). Its doc gains: "m is composite (the caller's `is_prime` said so). If m is a BPSW pseudoprime
  (none is known), `pollard_rho(m)` returns `None` at once through its `is_probable_prime` shortcut, and
  `split_brent_and_trial_division` (complete, it always returns a divisor) splits m."
- Docs of the three complete drivers (`prime.rs:727-730`, `quadratic_sieve.rs:519-523`, `ecm.rs:486-489`):
  "Every returned factor is proven prime by `is_prime` (strong tests below psi_12, BPSW and ECPP above).
  `is_prime` panics when ECPP is unresolved (no unresolved prime up to 2048 bits in the experiments; about
  1 in 16 at 3072 and 4096 bits); such a factor makes the driver panic." (I2 may move the drivers to `Result`;
  see the interface below.)

Interface for I2 (the capped variant with MAGMA's third return value, the QS fix, `trial_division_with_primes`):
- Deciding whether to split a cofactor m: `prime::is_probable_prime(&m)`. `false` proves m composite (split it);
  `true`: stop splitting.
- Certifying each final factor p: `prime::prove_primality(&p)`. `Prime(cert)`: a proven factor (the
  certificate is verified and can be kept); `NotPrime(w)`: p is composite after all (a BPSW pseudoprime; `w` may
  be `Divisor(d)`): go back to splitting; `Unresolved(_)`: report p as an unproven probable prime (a fourth list
  next to MAGMA's unfactored composites, or an `Err`), never as a prime.
- The complete drivers that return `Vec<(Integer, u32)>` may keep `is_prime` (panics on unresolved) or move to
  a `Result` with `try_is_prime`; that is I2's choice.
- Do not call `is_prime` on a value that is only used for a splitting decision: it proves primes.
(Review 1 checked this against the I2 design, lines 393-399 and 693-705: consistent. The certificate type's
rename to `EcppCertificate` does not touch I2, which never names the type.)

### 2.9 Downstream code changes forced or made honest by the new semantics

- `rustmath-polynomials/src/factorization.rs:172-196` (`integer_constant_is_irreducible`): replace the
  `MR_PROOF_BOUND` logic by `prove_primality(|c|)`: Prime -> Ok(true), NotPrime -> Ok(false), Unresolved ->
  Err(NotSupported). Doc `:122-127`. Test `tests/ch23_is_irreducible.rs:596-598`: the constant 2^89 - 1 is
  now `Ok(true)` (gp `isprime(2^89-1)` = 1); psi_12 `Ok(false)`. This is P1's crate; it is done here because
  the code exists only to work around the old `is_prime`, and P1 is implemented after I1.
- `rustmath-polynomials/src/berlekamp.rs:74` (`check_prime`, returns `Result`): `!try_is_prime(p)?` (P1's design
  does not cover this site; its `check_prime` at `fp_factor.rs:504` is a different function).
- `rustmath-finitefields`: `finite_field.rs:147` (`FiniteField::new`), `:185` (`with_modulus`),
  `galois_ring.rs:78` (`GaloisRing::new`), `:106` (`with_modulus`): `if !p.is_prime()` becomes
  `if !rustmath_integers::prime::try_is_prime(&p)?`; the messages for a composite p are unchanged, an
  unresolved p is `Err(NotSupported)`. F1's design leaves these lines alone (it rewrites other parts of
  `FiniteField::new`), so the edits do not collide; F1 rebases on them.
- `rustmath-crypto/src/prng_number.rs:69-104` (`random_prime_bits`, `random_prime_3mod4_bits`): the candidate
  test `is_prime(&cand)` becomes `matches!(try_is_prime(&cand), Ok(true))`: a composite costs trial division
  and one BPSW (inside `try_is_prime`), a probable prime one proof, and a candidate whose proof is unresolved
  (`Err`) is skipped like a composite. Doc of both functions and of `rsa_modulus`, `rsa_modulus_with_exponent`,
  `random_sequence_rsa`, `bbs_modulus`, `blum_blum_shub`: "primes are proven (MAGMA `RandomPrime`,
  Proof := true; ch. 158 digest line 54); a candidate whose ECPP proof is unresolved is skipped, so the prime
  is the least proven prime after the random start, a slight change of distribution only when such candidates
  occur (none up to 2048 bits in the experiments; about 1 in 16 at 3072 and 4096 bits). Cost per prime: one
  ECPP proof, for example about 3 s in release for 1024 bits and 76 s for 2048 bits." No unit owns this crate
  in this run.
- `rustmath-rings` (R1's crate; one-token edits, `rustmath_core::Result` there): `padics/eisenstein.rs:82`,
  `padics/capped_relative.rs:106`, `:175`, `:341`, `padics/unramified.rs:82`, `:159`,
  `padics/newton_polygon.rs:123`: `!prime.is_prime()` becomes `!try_is_prime(&prime)?`;
  `number_field/s_unit_solver.rs:412` (returns `SUnitError`): `match try_is_prime(p)` with `Ok(false)` as now and
  `Err(e)` mapped to `SUnitError::InvalidSUnit(e.to_string())`. R1's design does not change these lines (it
  edits other parts of `unramified.rs`), so R1 rebases on one-line edits.
- `rustmath-constants/src/certificates.rs:72-90` (`PrattCertificate::generate`, returns `Option`): `match
  try_is_prime(p) { Ok(true) => {}, _ => return None }`, doc "None when p is not prime or its primality could
  not be proven". (The Pratt generator factors p - 1 by trial division, so it never reaches a size where ECPP
  is unresolved; the change only removes the panic path.)
- `rustmath-jupyter/src/repl.rs` (no `catch_unwind` in the REPL, so a panic ends the session):
  - `is_prime` arm (`:914`): `try_is_prime`; Err becomes `EvalError::new("ValueError", ..)`.
  - `next_prime` arm (`:2680`): `try_next_prime`, Err to ValueError.
  - `primes` / `prime_range` arm (`:2725`): `try_prime_range`, Err to ValueError. J1a replaces this arm later with
    its limits (J1a design §5, it also uses `try_is_prime`).
  - `prime_pi` arm (`:2710`): x > u64::MAX is a ValueError ("prime_pi: x exceeds 2^64") instead of the new
    panic; J1a adds its tighter limit (x <= 6·10^9).
  - `nth_prime` arm (`:2696`): n > 425656284035217743 (pi(2^64)) is a ValueError instead of the new panic.
  - `sqrt_mod_report` (`:10915-10917` constant, `:10941` doc, `:10964`, `:10978-10996`): the prime test becomes
    `match try_is_prime(n)`, Err to ValueError; delete the psi_12 caveat (it would now be false); in the composite
    branch, for odd n, return "no square root: b is not a square mod n" when `b.kronecker_symbol(n) == -1`
    before calling `IntegerMod` (a proof, and it avoids trial-division factoring in that case). J1a rewrites
    this function later and keeps these outputs.
  - Test `test_sqrt_mod_probable_prime_caveat_and_refusal` (`:11746`) becomes
    `test_sqrt_mod_large_prime_and_strong_pseudoprime`: 2^127 - 1 gives the same roots without
    "probable prime"; `sqrt_mod(41, psi_12)` gives "no square root: 41 is not a square mod 3186...461" (gp
    `kronecker(41, psi_12)` = -1); `is_prime(318665857834031151167461)` gives `false`. The 4 and 73 mod
    psi_12 cases are dropped: on the composite path they need psi_12 factored, which `IntegerMod` does by
    trial division (about 2·10^11 divisions; HANDOFF already lists this as the REPL sqrt_mod follow-up, J1a).
- `rustmath-finitefields/tests/prime_residues.rs`: `square_root_mod_prime_pseudoprime_never_false_none`
  (`:296-329`) becomes `square_root_mod_prime_refuses_strong_pseudoprime`: `!is_prime(psi_12)` and
  `square_root_mod_prime(a, psi_12)` panics with "is not prime" for a in [2, 4, 41, 43, 73];
  `square_root_mod_prime_power(73, psi_12, 1)` panics too. Header comments `:13`, `:30-32`, `:73`, `:135`.

## 3. Callers and behaviour changes (whole workspace, grep of `is_prime(`, `.is_prime()`, `prime::`, the moved names)

Semantics after I1: `is_prime` is a proof everywhere, faster below 2^64 (10 µs instead of about 1 ms), slower
for primes above psi_12 (one ECPP proof, then cached; section 0), and can panic for an unresolved prime
(no unresolved prime up to 2048 bits in the experiments; at 3072 and 4096 bits 2 of 32 primes had no
top-level candidate and were unresolved).

### 3.1 Callers that return `Result` or `Option` and can see an `Integer` of any size

| Site | Returns | Decision |
|---|---|---|
| integers `integer.rs:1002` `binomial_mod` | Result | switched in I1: `try_is_prime(p)?` |
| integers `prime.rs` `previous_prime`, `random_prime` | Option | switched in I1: `try_previous_prime`, `try_random_prime` behind them; the `Option` wrappers panic only when a needed proof is unresolved (`previous_prime` cannot skip the largest prime below n), documented |
| finitefields `finite_field.rs:147`, `:185`; `galois_ring.rs:78`, `:106` | Result | switched in I1 (2.9) |
| finitefields `cantor_zassenhaus.rs:143` `check_prime_characteristic` | Result | handed to F1 (F1 design, "check_prime_characteristic ... uses I1's try_is_prime", moved to `ff_poly`) |
| finitefields `integer_mod.rs:880` `assert_prime` and its callers | Option/Vec | already panic on a composite p (documented contract); unchanged |
| polynomials `factorization.rs:185` | Result | switched in I1 (2.9, `prove_primality`) |
| polynomials `berlekamp.rs:74` `check_prime` | Result | switched in I1 (2.9) |
| crypto `prng_number.rs:78`, `:97` (behind `rsa_modulus`, `rsa_modulus_with_exponent`, `random_sequence_rsa`, `bbs_modulus`, `blum_blum_shub`, all Result) | Result | switched in I1: skip an unresolved candidate (2.9) |
| rings `padics/eisenstein.rs:82`, `capped_relative.rs:106`, `:175`, `:341`, `unramified.rs:82`, `:159`, `newton_polygon.rs:123` | Result | switched in I1 (2.9) |
| rings `number_field/s_unit_solver.rs:412` | Result (SUnitError) | switched in I1 (2.9) |
| constants `certificates.rs:75`, `:88` `PrattCertificate::generate` | Option | switched in I1 (2.9) |
| jupyter `repl.rs:914`, `:2680`, `:2696`, `:2710`, `:2725`, `:10964` | Result (EvalError) | switched in I1 (2.9); J1a later replaces the `primes` arm and `sqrt_mod_report` |
| ellipticcurves `rootnumber.rs:121` `local_root_number` | Result | already `assert!(is_prime(p))`, a panic on a composite p by contract; unchanged |
| rings `valuation/maclane.rs:272`, `padics/newton_lift.rs:406`; curves `igp_closure.rs:39`; coding `reed_solomon_generic.rs:451` | Result | argument is an i64 or u64, so `is_prime_u64` answers; no panic path; unchanged |

### 3.2 All other callers

rustmath-integers:
| Site | Change |
|---|---|
| `prime.rs` is_prime, is_prime_miller_rabin, next/previous_prime, nth_prime, prime_range, primes_first_n, prime_pi, random_prime, is_prime_power | rewritten (2.4, 2.5) |
| `prime.rs:556`, `:641` | `is_probable_prime` |
| `prime.rs:651` (p-1 stage primes) | `sieve::for_each_prime(2, bound + 1)` |
| `prime.rs:871` | `debug_assert!(m odd, m > 1)` (2.8) |
| `prime.rs:829`, `:833` | unchanged call, proof now; docs |
| `quadratic_sieve.rs:87`, `:182`, `:679` (u64) | unchanged (faster) |
| `quadratic_sieve.rs:450` | `is_probable_prime` |
| `quadratic_sieve.rs:572` | unchanged; docs `:516-530` |
| `ecm.rs:441` | `is_probable_prime` |
| `ecm.rs:523`, `:534` | unchanged; docs `:481-496` |
| `integer.rs:702`, `:835`, `:883`, `:909` | `is_probable_prime`; docs `:693-700`, `:816-819`, `:860-866` |
| `integer.rs:978` (`Integer::is_prime`) | unchanged; doc `:975` ("Miller-Rabin" -> proof) |
| `integer.rs:36-38` module doc | BPSW / ECPP references |
| `integer.rs:1265` | import `is_prime_u64` from primality |
| `sage_wrapper.rs:111`, `:348`, `:461` | doc only |
| `fast_arith.rs` | 2.6 |
| `factorint.rs:858` (test) | unchanged |
| `lib.rs` | `pub mod ecpp; pub mod primality; pub mod sieve;` |
| tests `ch18_factorization_drivers.rs:112`, `ch18_signs_roots.rs:565-570` | unchanged (comment at `:566` updated) |
| tests `ch18_residues.rs:621-660` | 4 |

Other crates (bool-returning or u64-sized callers of `rustmath_integers::prime::is_prime` / `Integer::is_prime`):
| Site | p size in practice | Change |
|---|---|---|
| finitefields `zmod.rs:79` (`ZMod::is_field`, bool) | any | unchanged: a proof, cached; panics on unresolved (doc) |
| finitefields `prime_field.rs:183` | doc | the psi_12 sentence goes |
| finitefields `integer_mod.rs:651`, `:815` | any | 2.7; docs `:630-636`, `:912-918`, `:958-962` |
| polynomials `fp_factor.rs:505` (p <= MAX_P, u64) | small | doc `:502` |
| rings `category_methods.rs:153`, `:172`, `:272`, `:313` (u64) | small | doc `:146-151` |
| rings `abc.rs:244` (`is_field`, bool) | any | unchanged (as `zmod.rs:79`) |
| ellipticcurves `curve.rs:330`, `tate.rs:358` (asserts) | small | unchanged (already panic on a non-prime) |
| numbertheory `bernoulli.rs:324`, `:703`; constants `sequences.rs:147`; quadraticforms `local_densities.rs:39,279` | small (u64) | unchanged (faster) |
| crypto tests `prng_number.rs:310`, `:321` | 20 bits | unchanged |
| benchmarks `bench_integers.rs:93-114` | 2^127 - 1 | doc comment: the first call proves, later calls hit the cache |
| py `integers.rs:128-130` (bool; crate excluded from the workspace, built with maturin) | | doc only: a Rust panic reaches Python as pyo3's `PanicException` |
| jupyter `repl.rs:25` imports `next_prime`, `nth_prime`, `prime_pi`, `prime_range` | | the arms move to the `try_` versions (3.1); values unchanged, much faster; `next_prime(0)` and `next_prime(1)` now 2 |
| rustmath-numbertheory `lib.rs:10` `pub use rustmath_integers::prime::*` | | gains the new names (no collision) |

Local `fn is_prime` definitions (not this API, unchanged): algebras `steenrod_algebra.rs`, numberfields
`abext.rs`, groups `libgap_mixin.rs`, combinatorics `hadamard.rs`, rings `algebraic_closure.rs`,
ellipticcurves `modular.rs`, interfaces `gap_functions.rs`, and the ideal/knot `is_prime` methods.

Behaviour changes a user can see:
- `is_prime(psi_12)`, `is_prime(psi_13)`: true -> false. Drivers split psi_12.
- `next_prime(0)`, `next_prime(1)`, `next_prime(-5)`: 3 -> 2.
- `is_prime_miller_rabin(n, k)` with k < 12 can now accept more strong pseudoprimes (documented meaning).
- `ArithInt::is_prime` no longer panics near 2^31.
- `legendre_symbol`, `sqrt_mod_prime`, ... reject psi_12 up front (`Err` "must be ... prime") instead of an
  `Err` from inside the residue code.
- `binomial_mod`, the finite-field and p-adic constructors, `berlekamp`, the crypto generators and the REPL
  give an `Err` (or skip the candidate) for an unresolved huge prime instead of answering from 240 Miller-Rabin
  rounds.
- Large-prime contract checks cost one ECPP proof per process per prime (section 0 table).
- `prime_pi(x)` for x > 2^64 and `nth_prime(n)` for n > pi(2^64) panic with a message (they never returned).

## 4. Tests

New integration test file `rustmath-integers/tests/ch18_primality.rs` (header: MAGMA ch. 18 §18.9, digest
path, [AM93], BPSW references, PARI/GP 2.15.4). Every expected value has its gp command next to it; gp
commands that build certificates or class polynomials are run after `default(parisizemax, 2^30)` (review 1,
m13: `primecert(nextprime(2^200))` and `polclass(-23, 1)` overflow the default stack). "HEAD" says whether the
test fails at HEAD (F), is impossible at HEAD (API missing, N), or passes (P).

The PARI certificates are string constants in the test file (review 1, m15), each with its command:
`C127` (`primecert(2^127-1)`, 383 bytes), `C40` (`primecert(nextprime(10^40))`, 591), `C65`
(`primecert(nextprime(2^64))`, 94; with spaces after the commas), `C200` (`primecert(nextprime(2^200))`, 1115;
no spaces). The implementer copies them from `$X/c*.txt`; one of them is also stored with its line broken in
the middle of a number and indented (`C200_WRAPPED`), to test the whitespace rule of `from_pari`.

| Test | Input -> expected | gp | HEAD |
|---|---|---|---|
| `is_prime_rejects_strong_pseudoprimes_psi12_psi13` | `is_prime`, `is_probable_prime`, `try_is_prime` of 318665857834031151167461 and 3317044064679887385961981: false / false / Ok(false) | `isprime(318665857834031151167461)` = 0, `factor(..)` = [399165290221, 1; 798330580441, 1]; `factor(3317044064679887385961981)` = [1287836182261, 1; 2575672364521, 1] | F |
| `prove_primality_witnesses` | psi_12, psi_13 -> NotPrime(StrongLucas(-7)) (both pass the base-2 strong test) and `confirms`; 561 -> NotPrime(Divisor(3)); 2047 -> NotPrime(Divisor(23)); 1000036000099 = 1000003·1000033 (no factor below 1000) -> NotPrime(MillerRabin(2)); 0, 1, -7 -> NotPrime(BelowTwo); 2^90 -> Divisor(2); every witness `confirms` its n and does not confirm a prime (e.g. `MillerRabin(2).confirms(2^127-1)` is false) | `factor(561)`, `factor(2047)`, `factor(1000036000099)`, `Mod(2,1000036000099)^1000036000098 != 1`; Selfridge D of psi_12 and psi_13 is -7 (first D in 5, -7, ... with `kronecker(D, n) == -1`) | N |
| `is_prime_mersenne_and_neighbours` | 2^89-1, 2^107-1, 2^127-1, 2^521-1 true; 2^89+1, 2^127+1 false. The 2^521 - 1 proof is about 4 s in debug and is counted in section 5 (review 1, m9) | `isprime(2^89-1)` ... = [1, 1, 1, 1, 0, 0] | P (now proven) |
| `bpsw_rejects_strong_base2_pseudoprimes` | the odd composites n < 10^6 that pass the base-2 strong test (found in the test with a u64 base-2 test and the sieve) number 46 (first 2047, 3277, 4033, 4681, 8321, last 983401; OEIS A001262), and the internal BigInt `bpsw` rejects each | gp script `$X/t16_sp2.gp` (`sp2(n)` = strong base-2 test; counts 46) | N (unit test in `primality.rs`) |
| `strong_lucas_pseudoprimes_below_20000` | the odd composites n < 20000 that pass the internal strong Lucas test are exactly [5459, 5777, 10877, 16109, 18971] | OEIS A217255 (Selfridge method A); matches `$X/t5.py` and review 1's gp | N (unit test) |
| `bpsw_gcd_with_q` (unit) | Selfridge's D never gets large enough to reach this branch for an n without prime factors below 1000, so the test calls the internal `lucas_check(n, d)` (the steps 3-5 with a given D) directly: n = 1032247 = 1013 · 1019, D = 4053, Q = (1 - 4053)/4 = -1013 -> `Divisor(1013)`; and `bpsw(psi_12)` is still `StrongLucas(-7)` | `gcd(1013*1019, (1-4053)/4)` = 1013, `4053 % 4` = 1, `isprime(1013)`, `isprime(1019)` | N |
| `is_probable_prime_and_is_prime_agree_with_pari` | for k = 64, 80, 100, 128, 160, 200: 2^k + d_k is prime with d_k = 13, 13, 277, 51, 7, 235 (the least prime above 2^k); 2^k + d_k + 2 and 2^k + 1 are composite; 10^30 - 11 is prime. Both `is_probable_prime` and `is_prime` give these answers (the 200-bit `is_prime` is one proof, about 0.25 s) | `nextprime(2^k) - 2^k`, `isprime(nextprime(2^k)+2)` = 0, `isprime(2^k+1)` = 0, `precprime(10^30)` = 10^30 - 11 | P (values), the proofs are new |
| `ecpp_accepts_pari_certificates` | `EcppCertificate::from_pari(C).verify()` true for C127, C40, C65, C200 and C200_WRAPPED, and `n()` is the number in each gp command | `primecert(...)` as above; `primecertisvalid` = 1 each | N |
| `ecpp_rejects_tampered_certificates` | from C200 (6 steps): s doubled, x + 1, t + 2 in step 1; N + 2 in step 1 (chain broken); step 3 of 6 removed (chain broken: step 4's N is not step 2's q); `from_steps(N, steps[1..])` (step 1 removed, n kept: step 2's N is not N); the leaf replaced by a composite; s = 0; negative N: all `verify` false; garbage text: `from_pari` Err. And `from_pari` of steps 2..6 alone is a valid certificate whose `n()` is step 1's q, not N (review 1, m3) | gp `primecertisvalid` of the tampered C200 variants = 0 (`$X/t15_certs.gp`, with `default(parisizemax, 2^30)` added); of `c[2..6]` = 1 | N |
| `ecpp_roundtrip_and_pari_shape` | for 2^89-1, 2^127-1, nextprime(10^40), nextprime(2^200): `prove_primality` is `Prime(c)`, `c.verify()`, `c.n()` is the input, `from_pari(c.to_pari()) == c`, last q < 2^64, every step has s >= 2 | design check: PARI accepted the prototype's certificates (`primecertisvalid` = 1) | N |
| `ecpp_small_inputs` | `prove_primality(p)` for p = 2, 3, 2^61-1, 2^64-59: Prime with no steps, `verify` true; `to_pari()` is the decimal p | `isprime(2^64-59)` = 1 | N |
| `ecpp_cornacchia_traces` (unit) | D = -23, N = 1000000000933: (u, v) with u^2 + 23 v^2 = 4N and \|u\| = 1967590, \|v\| = 74772; D = -3, N = 2^127 - 1: u^2 + 3 v^2 = 4N | `qfbsolve(Qfb(1,1,6), 1000000000933)` = [-1021181, 74772] (u = 2x + y); `qfbsolve(Qfb(1,1,1), 2^127-1)` | N |
| `hilbert_polynomials_match_pari` (unit, `ecpp/hilbert.rs`) | `class_polynomial(-3)` = X, `(-4)` = X - 1728, `(-15)` = [-121287375, 191025, 1], `(-23)` = [12771880859375, -5151296875, 3491750, 1], `(-71)` = the 8 coefficients of gp; for D = -455 (h 20), -58507 (h 20), -1271 (h 40), -7991 (h 100): degree h, monic, the largest coefficient has 471, 1434, 1039, 3336 bits, and H_D(12345) mod 2^61 - 1 = 1161148812500064316, 1599249951824458418, 621737880862793894, 2255316476449223031 | `default(parisizemax, 2^30); polclass(D)`; `lift(Mod(subst(polclass(D), x, 12345), 2^61-1))`; `exponent(c) + 1` of the largest coefficient (`$R1/hpref.gp`) | N |
| `hilbert_roots_give_j` (unit) | the roots of `class_polynomial(-23)` mod 1000000000933 are exactly {120712464354, 933014773255, 946269272507} | `lift(polrootsmod(polclass(-23), 1000000000933))` | N |
| `discriminant_list_shape` (unit) | 100 lines; line h starts with h; 42,272 discriminants; counts per h begin [9, 18, 16, 54, 25, 51, 31, 131, 34, 87] and the h = 100 line has 1736 entries; each line ascending; every \|D\| fundamental (D = 1 mod 4 squarefree, or D = 4m with m = 2, 3 mod 4 squarefree); for every row with h <= 20 (1817 rows, |D| <= 58507, about 2 s in debug) the number of reduced forms equals h, and for 20 rows spread over h = 21..100 too (the exact check of all 42,272 rows is `$R1/dcheck.rs`, 13 s in release, run once by the implementer) | Watkins 2004; the counts printed by `ecpp_discriminants.gp`; `$R1/dcheck.rs` | N |
| `ecpp_order_check_matches_affine` (unit) | on y^2 = x^3 + 2x + 3 over F_97 and a 61-bit prime, `mul(k, P)` agrees with an affine reference for k = 1..200; order_check accepts (s, q) = (#E/q, q) and rejects wrong q | gp `ellmul(ellinit([2,3], 97), [x,y], k)`, `ellcard` | N |
| `ecpp_driver_backtracks` (unit) | `prove_chain` with H_INNER forced to 1 on nextprime(2^200): every inner level that has no h = 1 candidate fails, the driver pops and tries the next candidate; the result is still `Prime` with a verified certificate, or `Unresolved` with a failed-subtree count; never a panic; the stack depth equals the chain length | design check | N |
| `is_prime_miller_rabin_counts_bases` | 3215031751: rounds 4 -> true, rounds 5 -> false; 3825123056546413051: rounds 11 -> true, rounds 12 -> false; 2^127-1 rounds 20 true | `factor(3215031751)` = 151·751·28351 (a strong pseudoprime to 2, 3, 5, 7, not to 11); `factor(3825123056546413051)` = 149491·747451·34233211 (a strong pseudoprime to 2..31, not to 37) | F (rounds 4 and 11 give false at HEAD) |
| `handbook_h18e4_safe_primes` (MAGMA H18E4: 10-digit safe primes with `IsPrime`) | the first five p >= 10^9 with p and (p-1)/2 prime: [1000000007, 1000000403, 1000000787, 1000001447, 1000001819], found with `next_prime` and `is_prime` | `$X/t17_handbook.gp` | P (much faster now) |
| `handbook_h18e6_probable_repunits` (MAGMA H18E6: probable repunit primes with `IsProbablePrime`) | n in 2..=320 with `is_probable_prime((10^n - 1)/9)`: exactly [2, 19, 23, 317]; `is_prime` of R_19 and R_23 true (both below psi_12, a proof) | gp `select(n->ispseudoprime((10^n-1)/9), [2..320])` = [2, 19, 23, 317]; `isprime((10^23-1)/9)` = 1 | N (function new) |
| `next_previous_prime_values` | next_prime(-5), (0), (1) = 2; (2) = 3; (2^64) = 18446744073709551629; (2^127) = 170141183460469231731687303715884105757; (10^40) = 10^40 + 121; previous_prime(3) = 2; (2^64) = 18446744073709551557; (2^127) = 2^127 - 1; `try_next_prime` and `try_previous_prime` give the same as `Ok` | gp `nextprime(n+1)`, `precprime(n-1)` | F (n < 2) |
| `random_prime_proves_only_the_pick` | `try_random_prime(10^30, 10^30 + 300)` is Ok(Some(p)) with p - 10^30 in {57, 99, 211, 231, 271} and `is_prime(p)`; `try_random_prime(24, 29)` = Ok(None); `(10, 20)` in {11, 13, 17, 19}; `(20, 10)` Ok(None) | gp `primes([10^30, 10^30+299])` | N |
| `nth_prime_and_prime_pi_by_sieve` | nth_prime(1, 10, 100, 10^5, 10^6) = 2, 29, 541, 1299709, 15485863; prime_pi(10^6) = 78498, prime_pi(10^7) = 664579 | `prime(10^5)`, `prime(10^6)`, `primepi(10^6)`, `primepi(10^7)` | P in value, but HEAD needs about 20 min for nth_prime(10^6) |
| `prime_range_windows` | [10^12, 10^12+100) = [1000000000039, 1000000000061, 1000000000063, 1000000000091]; [2^64-100, 2^64) = [18446744073709551521, 18446744073709551533, 18446744073709551557]; [2^64, 2^64+100) = [..629, ..653, ..667, ..697, ..709]; `try_prime_range` gives the same as `Ok` | `primes([10^12, 10^12+99])`, `primes([2^64-100, 2^64-1])`, `primes([2^64, 2^64+99])` | P |
| `sieve_matches_is_prime_u64` (unit, sieve.rs) | `primes_in_range(lo, lo + 10^4)` equals the `is_prime_u64` filter for lo in {0, 10^9, 2^32 - 5000, 2^40, 2^50}; for lo = 2^64 - 10^4 (where lo + 10^4 = 2^64 does not fit, review 1 m2) `primes_in_interval(lo, u64::MAX)` equals the filter and ends with 18446744073709551557; `primes_in_interval(u64::MAX - 1, u64::MAX)` is empty; `for_each_prime(u64::MAX - 100, u64::MAX, ..)` visits 18446744073709551521, ..533, ..557 | self-check against the deterministic test; gp `precprime(2^64-1)` = 18446744073709551557 | N |
| `pollard_p_minus_1_bound_above_2_16` (review 1, B3) | n = 1835038000112854837 = 917519 · 2000000000123: `pollard_p_minus_1(&n, 70000)` = Some(917519) (917519 - 1 = 2·7·65537 and 65537 > 2^16); `pollard_p_minus_1(&n, 65535)` = None | `$R1/pm1.gp`: `factor(917519-1)` = [2,1; 7,1; 65537,1], `factor(2000000000122)` = [2,1; 1000000000061,1], stage-1 gcd 917519 for B = 70000 and 1 for B = 65535 | P (HEAD tests every p <= bound), fails under a truncation to the primes below 2^16 |
| `prime_power_values` | 2^61 -> Some((2, 61)); 3^40 -> Some((3, 40)); (2^61-1)^3 -> Some((2^61-1, 3)); (2^61-1)·(2^31-1) -> None; 2^89-1 -> Some((2^89-1, 1)); psi_12 -> None; 1, 0, -8 -> None | gp `ispower(3^40,,&r)` = 40 with r = 3; `ispower((2^61-1)^3,,&r)` = 3 with r = 2^61-1; `isprime(2^61-1)` = 1 | N (function new); `is_prime_power((2^61-1)^3)` does not finish at HEAD |
| `ecpp_q_bound_edges` (unit) | `iroot4_ceil(10^20)` = 10^5, `(10^20 + 1)` = 10^5 + 1, `(10^20 - 1)` = 10^5; the bound test "q > (r + 1)^2" is false at q = (r + 1)^2 and true at (r + 1)^2 + 1 | exact integer arithmetic (10^20 = (10^5)^4) | N |
| `try_is_prime_cache_is_transparent` | two calls on 2^127-1 give Ok(true); the second is at least 10x faster (soft check: only assert it is Ok(true) both times; print the timings) | | N |
| `try_sqrt_mod_prime_refuses_squares` (unit, integer.rs; review 1, m5) | `try_sqrt_mod_prime(2, 1000003^2)` and `(3, (2^61-1)^2)` are Err within milliseconds; `(4, 7)` = Ok(Some(2)), `(3, 7)` = Ok(None), `(2, 15)` is Err (15 = 3 mod 4 and (2/15) = 1, so the candidate root is 2^4 = 1 mod 15, which does not square back to 2) | `isprime(1000003)` = 1, `kronecker(2, 15)` = 1, `Mod(2, 15)^4` = 1 | N |
| `arith_int_is_prime_no_overflow` (fast_arith unit) | `ArithInt::is_prime(2147483647)`, `(2147483629)` true; `(2147395601)` false; `ArithLLong::is_prime(i64::MAX)` false, `(9223372036854775783)` true | `isprime(2^31-1)`, `isprime(2147483629)`, `factor(2147395601)` = 13·165184277, `factor(2^63-1)`, `isprime(9223372036854775783)` | F (panic) |

Changed tests:
- `rustmath-integers/tests/ch18_factorization_drivers.rs`: new `psi12_is_split_by_every_driver`:
  `factor_pollard_rho(psi_12)`, `quadratic_sieve_factor_complete(psi_12)`, `ecm_factor_complete(psi_12, 1000,
  10000, 0)` all equal [(399165290221, 1), (798330580441, 1)] (gp `factor(318665857834031151167461)`). F at HEAD.
  Expected time about 5 s each in debug (Floyd's rho); if the three together exceed 15 s, keep
  `factor_pollard_rho` and the zero-curve ECM call (same path) and move QS to an `#[ignore]` test with a note.
- `rustmath-integers/tests/ch18_residues.rs::residues_modulo_strong_pseudoprime_are_honest`: assert
  `!n.is_prime()` and that `sqrt_mod_prime`, `is_quadratic_residue_prime`, `legendre_symbol` of 4, 43 and
  399165290221 modulo psi_12 are all `Err` (refused up front); the Jacobi values stay. P at HEAD only through the
  `else` branch.
- `rustmath-integers/src/prime.rs` tests: `test_is_prime` gains 2147483647, 2^61-1, psi_12;
  `test_trial_division_window_rejects_prime` and the rest unchanged.
- `rustmath-integers/src/fast_arith.rs` tests: `test_is_prime_u64_*` move with the function to primality.rs.
- `rustmath-finitefields/tests/prime_residues.rs`, `rustmath-jupyter` REPL test, `rustmath-polynomials`
  `ch23_is_irreducible.rs`: section 2.9. All three fail at HEAD after the change unless edited (the REPL one
  would hang).
- `rustmath-finitefields`: a test that `FiniteField::new(psi_12, 1)` and `GaloisRing::new(psi_12, 2, 1)` are
  `Err(InvalidArgument)` (they were `Ok` at HEAD, since `is_prime(psi_12)` was true; F at HEAD).
- `rustmath-crypto` `prng_number.rs` tests: unchanged (20- to 32-bit moduli); a new doc test of the skip rule is not
  possible without an unresolved prime, so the doc states it and the code path is the same `match` as for a
  composite.
- `rustmath-jupyter`: `is_prime(318665857834031151167461)` gives false; `next_prime(1)` gives 2;
  `prime_pi(2^70)` and `nth_prime(10^18)` give ValueErrors (F at HEAD: they hang).

Doctest updates: `prime.rs` examples of `nth_prime`, `prime_range`, `prime_pi` keep their values; add a doctest
for `prove_primality` (2^127 - 1, `verify()`), for `is_probable_prime(psi_12) == false`, and for
`EcppCertificate::from_pari` with C65.

## 5. Cost

- `is_prime` below 2^64: about 10 µs debug (HEAD about 1 ms for a prime). Between 2^64 and psi_12: 12 strong
  tests, about 1-2 ms. Composite above psi_12: one strong test in almost all cases (BPSW stops there).
- Prime above psi_12: the table in section 0 (debug: 65 ms at 128 bits, 0.36 s at 256, 3.2 s at 512, 38 s at
  1024; release: 3.3 s at 1024, 21 s at 1536, 76 s at 2048); repeated calls on the same prime: cache
  lookup.
- Memory: the discriminant list (273 KB, in the binary), parsed lines (at most 42,272 rows of a few words), the
  Hilbert polynomial cache (at most 256 entries; an h = 100 polynomial is at most 115 KB), the small primes (6542
  u32), segment buffers (32 KiB), one prover frame per level. No large allocation per proof.
- Test-suite time added: ch18_primality about 15-25 s in debug (the 2^521 - 1 proof about 4 s, the 2^200 proof,
  the sieve to 10^7, the Hilbert polynomials of the PARI comparison: 2.1 s in debug for D = -7991 with h = 100,
  under 0.1 s for the others; the list-shape test about 2 s), the psi_12 driver test 5-15 s. Workspace tests: no slowdown expected; many get faster (u64 `is_prime` 100x).

## 6. Plan

File headers (CLAUDE.md: cite the chapter digest and the paper in each new file):
- `primality.rs`: MAGMA ch. 18 §18.9.1 (`IsProbablePrime`, `IsPrime` with `Proof := false`), digest path,
  [Jae93], Sorenson-Webster 2017, Baillie-Wagstaff 1980, Pomerance-Selfridge-Wagstaff 1980,
  Baillie-Fiori-Wagstaff 2021.
- `ecpp/mod.rs`: MAGMA ch. 18 §18.9.1 (`IsPrime`, `PrimalityCertificate`, `IsPrimeCertificate`), [AM93],
  Goldwasser-Kilian 1999, Cohen GTM 138 §9.2, Cohen-Miyaji-Ono 1998, Hankerson-Menezes-Vanstone 2004, the
  PARI/GP 2.15.4 certificate format, the soundness argument of 2.3.2 in prose, the iterative driver.
- `ecpp/hilbert.rs`: MAGMA ch. 18 §18.9.1 [AM93] §7; Cohen GTM 138 §5.3 and §7.6; Enge 2009; Yui-Zagier 1997
  (Weber's f_2 formula); Apostol GTM 41 ch. 3 (pentagonal theorem); Watkins 2004 (completeness of the list);
  the list's sha256 and its generator.
- `ecpp/polymod.rs`: Cohen GTM 138 §1.6.1 (roots mod p, Cantor-Zassenhaus splitting with (x + delta)^((p-1)/2)).
- `sieve.rs`: MAGMA ch. 18 §18.9.2 (`PrimesUpTo`, `PrimesInInterval`, `NthPrime`), Bays-Hudson 1977,
  Rosser-Schoenfeld 1962.
- `data/ecpp_discriminants.gp`: what the list is, Watkins 2004, PARI version, `default(parisizemax, 2^30)`.

Order (each step keeps `cargo test -p rustmath-integers` green):

1. `rustmath-integers/data/ecpp_discriminants.gp` and `.txt` (generate with gp 2.15.4, 83 s; check the printed
   counts and the sha256 `33c5463dc8c737aa15a2bbb3db5ba3d9d0c4dd8f9f606b223077d7854132ee3b`; optionally run
   `$R1/dcheck.rs`; record the sha256 in the header of `ecpp/hilbert.rs`).
2. `src/sieve.rs` (+ unit tests), `src/lib.rs` (`pub mod sieve`).
3. `src/primality.rs`: `is_prime_u64` moved from fast_arith, BPSW (with the gcd(n, Q) check), `NotPrimeWitness`
   (+ unit tests, the u64 tests moved from fast_arith). `src/fast_arith.rs`: remove `is_prime_u64`, delegate
   `ArithInt::is_prime`, `ArithLLong::is_prime`, drop `miller_rabin`, `compute_small_primes` via the sieve,
   `prime_range` via the sieve. `src/integer.rs:1265` import.
4. `src/ecpp/hilbert.rs` (list loader, reduced forms, fixed-point complex functions, `class_polynomial`, cache;
   + unit tests against the PARI values), `src/ecpp/polymod.rs` (polynomials mod N, root finding; + unit tests),
   `src/ecpp/mod.rs` (Jacobian `mul` and `order_check`, `EcppStep`, `EcppCertificate` (verify, PARI I/O), the
   frame, `advance`, `prove_chain`, `Primality`, `prove_primality` with the final `verify`; + unit tests).
   `src/integer.rs`: `sqrt_mod_prime_checked` -> `pub(crate)`, new `try_sqrt_mod_prime` (with the square check).
5. `src/prime.rs`: `try_is_prime`, `is_prime`, cache, re-exports, `is_prime_miller_rabin`, the sieve-backed
   functions, the `try_` functions and their wrappers, `prime_power` / `is_prime_power`, `pollard_p_minus_1`
   over the sieve, driver early returns, `split_odd_composite` assert, docs.
6. `src/integer.rs` residue functions and `binomial_mod`, docs; `quadratic_sieve.rs`, `ecm.rs` early returns and
   docs; `sage_wrapper.rs` docs; `tests/ch18_primality.rs`; `tests/ch18_factorization_drivers.rs`;
   `tests/ch18_residues.rs`.
7. Downstream (section 2.9): `rustmath-finitefields` (`integer_mod.rs` square roots and `unit_sqrts_odd`,
   `finite_field.rs:147`, `:185`, `galois_ring.rs:78`, `:106`, `prime_field.rs:183`, `cantor_zassenhaus.rs`
   docs, `tests/prime_residues.rs`, the new constructor test); `rustmath-polynomials` (`factorization.rs`,
   `berlekamp.rs:74`, `fp_factor.rs` doc, `tests/ch23_is_irreducible.rs`); `rustmath-crypto/src/prng_number.rs`;
   `rustmath-rings` (the seven p-adic sites, `s_unit_solver.rs:412`, `category_methods.rs:146-151` doc);
   `rustmath-constants/src/certificates.rs`; `rustmath-jupyter/src/repl.rs` (six arms and the test);
   `rustmath-py/src/integers.rs:128` doc; `rustmath-benchmarks/src/bench_integers.rs:93` doc.
8. Trackers: `PROJECT_TRACKER.md` rows `prime.rs:32` (Miller-Rabin), `:137` (next_prime family), `:808`
   (driver), `fast_arith.rs:315`, `integer.rs:701` (Legendre), `integer.rs:999` (binomial_mod),
   `integer_mod.rs:937` (square roots), `factorization.rs:146`, REPL `sqrt_mod`; new rows for BPSW, ECPP +
   certificate, Hilbert class polynomials, segmented sieve; the "Baillie-PSW" and "Meissel-Lehmer"
   missing-algorithm bullets (`:436`, `:441`). `docs/magma/status_overrides.csv`: ch018.IsPrime, IsPrime.2,
   PrimalityCertificate (now `rustmath-integers/src/ecpp/mod.rs`, type `EcppCertificate`; the constants Pratt code
   remains as noted), IsProbablePrime, IsPrimePower, NextPrime, PreviousPrime, PrimesUpTo, PrimesInInterval,
   NthPrime, RandomPrime; then `python3 tools/magma_tracker/build.py`. `docs/HANDOFF.md`: remove the five items;
   add the measured ECPP coverage (sections 0 and 2.3.7, the residual at 2048 bits and above) and section 7's
   follow-ups.

Acceptance commands:
```
cargo test -p rustmath-integers --no-fail-fast
cargo test -p rustmath-finitefields --no-fail-fast
cargo test -p rustmath-polynomials --no-fail-fast
cargo test -p rustmath-crypto -p rustmath-constants --no-fail-fast
cargo test -p rustmath-rings --no-fail-fast
cargo test -p rustmath-jupyter --no-fail-fast sqrt_mod
cargo test -p rustmath-jupyter --no-fail-fast prime
cargo test -p rustmath-numbertheory -p rustmath-ellipticcurves -p rustmath-coding -p rustmath-quadraticforms -p rustmath-curves --no-fail-fast
cargo check --workspace --tests
cargo clippy -p rustmath-integers
```
plus, once, outside the test suite: run gp `primecertisvalid` on the certificates the implementation produces
for 2^127-1, nextprime(10^40), nextprime(2^200) and one random 256-bit prime (print with `to_pari`; gp after
`default(parisizemax, 2^30)`), and record the result in the commit message; and one release run of a random
1024-bit proof (`cargo test --release -p rustmath-integers -- --ignored ecpp_1024_release`, an `#[ignore]` test
that prints the time) to confirm the 3 to 5 s of section 0.

Risks:
- Jacobian soundness: the argument in 2.3.2 depends on the exact formulas (Z3 = 2 Y1 Z1 and Z3 = Z1 H). A
  different formula set needs its own check of facts 1-3. The verifier must not use the prover's knowledge
  (list, D, j).
- Off-by-one in `iroot4_ceil` or in the q bound would weaken soundness; `ecpp_q_bound_edges` tests both at
  exact fourth powers, and the bound is a separate function so the test reaches it.
- Fixed-point precision: a too small guard makes the rounding check fail (the row is skipped) or, in the worst
  case, passes with a wrong H_D; either only costs a candidate (the verifier is independent). The PARI
  comparison test catches a systematic error.
- Debug-build time of the new tests (the 2^521 - 1 proof is the largest; do not prove 768-bit or larger numbers
  in debug tests).
- `is_prime` panics on unresolved inputs: only bool-returning callers and asserting callers keep that path
  (3.1); the rates are in the docs.
- Overlap with I3 in `fast_arith.rs` (I3 changes gcd, lcm, add_mod, mul_mod; I1 changes is_prime and the
  sieve helpers) and `integer.rs` (I3 changes sqrt and friends; I1 does not use `Integer::sqrt`). Rebase on I3.
- Overlap with P1 in `rustmath-polynomials/src/factorization.rs` (only `integer_constant_is_irreducible` and its
  doc) and `berlekamp.rs:74` (one line); with F1 in `integer_mod.rs` (square roots), `finite_field.rs` and
  `galois_ring.rs` (one line each); with R1 in seven p-adic lines and `s_unit_solver.rs:412`; with J1a in the
  REPL arms it later replaces. All are implemented after I1 and rebase on it.

## 7. Out of scope here, and hand-offs

- ECPP coverage above 2048 bits: the speed and the residual `Unresolved` rate above 2048 bits (2.3.7). This is
not an item of the list (the item asks to state which numbers are proven within what time, and 2.3.7 does);
it is the measured limit of the chosen algorithm with num-bigint. What would improve it: class numbers beyond
100 at the top level (lists past Watkins' range need not be complete for this use), with root finding below
O(h^2 log N) (Kronecker substitution for the polynomial products, or the decomposition along the class group
of G. Hanrot, F. Morain, *Solvability by radicals from an algorithmic point of view*, ISSAC 2001); FastECPP's
shared square roots and larger smoothness bound by batch remainder trees (J. Franke, T. Kleinjung, F. Morain,
T. Wirth, *Proving the primality of very large numbers with fastECPP*, ANTS VI, LNCS 3076 (2004) 194-207;
F. Morain, *Implementing the asymptotically fast version of the elliptic curve primality proving algorithm*,
Math. Comp. 76 (2007) 493-505); a faster multiplication for operands of 3000 bits and more. The honest interim
behaviour is that of 2.4 (`Unresolved`, `Err`, or a panic in `is_prime`). HANDOFF gets it as a follow-up with
the measured rates.
- F1: `cantor_zassenhaus::check_prime_characteristic` (`:138-148`, returns `Result`) uses `try_is_prime` (F1
  design, moved to `ff_poly`).
- I2: the interface in 2.8; `prime_divisors` (`prime.rs:469`) and `factor` (trial division) are I2's.
- J1a: the REPL `primes`, `prime_pi`, `sqrt_mod` limits (S15) replace I1's minimal guards in those arms.
- Owners of rustmath-modular, ellipticcurves, galois: their private `is_prime_u64` copies can call
  `rustmath_integers::prime::is_prime_u64`.
- `random_prime(a, b)` still enumerates the probable primes of the whole range (tracker row); MAGMA's
  `RandomPrime(n)` form (random n-bit prime without enumeration) is missing. Not in this unit's list.
- `rustmath-constants/src/certificates.rs` (Pratt certificates, tracker: verify unsound) is a separate crate;
  MAGMA's `PrimalityCertificate` row moves to the ECPP certificate here.
