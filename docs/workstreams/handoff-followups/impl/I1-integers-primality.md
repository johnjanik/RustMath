# Implementation report: Unit I1-integers-primality

Design: `fu2/design/I1-integers-primality.md` (Revision 1 body), reviews 1 and 2 (approved).
Base: HEAD `80130e7` (I3 committed on top of `c79a582`). The I3 report has no handoff to I1;
I3 had already made `ArithInt::is_prime` / `ArithLLong::is_prime` call `is_prime_u64` and deleted
`ArithLLong::miller_rabin` (review 2 n10), so only the move of `is_prime_u64` was left. Nothing is
committed (the orchestrator commits). All five items are implemented; nothing is declared out of
scope. No `rustmath-misc/output.txt` or `.tmp_dir_*` change.

## Per item: what changed

### Item A: a proving `is_prime` (BPSW + ECPP with certificates)

- `rustmath-integers/data/ecpp_discriminants.gp` (new): the generator, with header (MAGMA ch. 18
  §18.9.1, [AM93], Watkins 2004, PARI 2.15.4, `default(parisizemax, 2^30)`). Run here in an empty
  directory (79 s): counts total 42,272, output sha256
  `33c5463dc8c737aa15a2bbb3db5ba3d9d0c4dd8f9f606b223077d7854132ee3b` (the design's value);
  committed as `data/ecpp_discriminants.txt` (273,422 bytes). `dcheck.rs` was not re-run (the
  design and review 2 ran it); the unit test re-counts all 1817 rows with h <= 20 by reduced forms.
- `src/primality.rs` (new, 626 lines): `is_prime_u64` (`:42`, moved from fast_arith, now public),
  `NotPrimeWitness` (`:77`) with `confirms` (`:97`; validates its data: `StrongLucas(D)` needs
  D = 1 mod 4, review 2 n8), `PrimeBatches` (batched `BigInt % u64` trial division and the ECPP
  strip), `jacobi_small` (`:223`), `strong_probable_prime` (`:275`), `lucas_check` (`:300`, the
  gcd(n, Q) check, review 1 m8), `selfridge_d`, `bpsw` (`:371`; documented correct for every odd
  n > 1, review 2 n1), `probable_prime_big`, `is_probable_prime` (`:428`).
- `src/ecpp/mod.rs` (new): certificate types `EcppStep` (`:65`), `EcppCertificate` (`:84`;
  `from_steps`, `n`, `steps`, `verify` `:252`, `to_pari`, `from_pari` `:324`, whitespace-agnostic
  parser), `Primality` (`:91`); Jacobian `jdbl`/`jmadd`/`jmul` and `order_check` (`:180`, the
  2.3.2 argument in the module header); `iroot4_ceil`, `q_bound`; `ProverParams::DEFAULT`
  (100 / 20 / 2048 / 64); SplitMix64 seeded from N; frames, `cornacchia` (`:606`), `scan_row`,
  `try_candidate`, `advance`, iterative `prove_chain` (`:845`) returning `ChainResult` with
  `failed_subtrees` and `max_depth`; `prove_primality_with`, `prove_primality` (`:997`, always
  verifies the certificate before `Prime`; release builds return `Unresolved("internal error
  ...")` on a failed verification, debug asserts).
- `src/ecpp/hilbert.rs` (new): list loader (`OnceLock` per line), `prime_discriminants`,
  `reduced_forms`, fixed-point complex arithmetic (Machin pi, Taylor exp, pentagonal eta, Weber
  j), `class_polynomial` (`:390`) with the rounding check and a 256-entry cache. Header explains
  why fixed point on BigInt and not `rug` (review 2 n9).
- `src/ecpp/polymod.rs` (new): dense polynomials mod N with lazy reduction, `find_root`
  (Cantor-Zassenhaus splitting, 64 attempts).
- `src/prime.rs`: module header; re-exports (`prove_primality`, `EcppCertificate`, `EcppStep`,
  `Primality`, `is_prime_u64`, `is_probable_prime`, `NotPrimeWitness`); proven-prime cache
  (`:87`, 64 entries, poison-safe); `try_is_prime_with` (`:108`), `try_is_prime` (`:160`, cost and
  unresolved-rate table in the doc), `is_prime` (`:173`, panics with the `Err` message).
- `src/integer.rs`: module header (BPSW/ECPP), `Integer::is_prime` doc; `kronecker_bigint` and
  `sqrt_mod_prime_checked` are `pub(crate)`.

### Item S6: distinct bases, sieve-backed prime functions

- `src/sieve.rs` (new): segmented odd-only byte sieve (32 KiB segments, per-prime next multiple in
  u128, survivors confirmed by `is_prime_u64` only above 2^40 when isqrt(e) > 2^20, overflow-free
  up to `u64::MAX`): `for_each_prime` (`:189`), `primes_in_range`, `primes_up_to`,
  `primes_in_interval`, `prime_count`, `PRIME_PI_2_64`, `first_n_primes` (crate), `nth_prime_u64`
  (`:286`), `small_primes()` (crate, primes below 2^16).
- `prime.rs`: `is_prime_miller_rabin` (`:65`) runs the first max(rounds, 1) primes once each;
  `try_next_prime` (`:252`, n < 2 gives 2), `next_prime`, `try_previous_prime` (`:302`),
  `previous_prime` (pre-sieved windows of 2^12 above 2^64); `prime_power` (`:357`) /
  `is_prime_power` via `is_perfect_power`; `nth_prime` (`:405`, panics above pi(2^64));
  `try_prime_range` (`:462`) / `prime_range`; `primes_first_n`; `prime_pi` (`:542`, panics above
  2^64); `try_random_prime` (`:570`) / `random_prime` (proves only the pick);
  `pollard_p_minus_1` (`:854`) stage 1 over `sieve::for_each_prime(2, bound + 1)`.
- `fast_arith.rs`: `prime_range` is `sieve::primes_in_range`; both `compute_small_primes` use
  `sieve::primes_up_to`.

### Item: duplicate word-size tests

- `is_prime_u64` moved to `primality.rs` (public, also `prime::is_prime_u64`); its two unit tests
  moved with it. `fast_arith.rs` and `perfect_power.rs` import it; `pow_mod_u64` stays
  `pub(crate)` in fast_arith. I3's renamed test comment (`ch18_signs_roots.rs:1101`) and
  `ch18_residues.rs:816` now name `primality.rs`. The design's `arith_int_is_prime_no_overflow`
  is merged into I3's `fast_arith_int_is_prime_near_i32_max` (`ch18_residues.rs:914`): added
  2147395601 false, `ArithLLong::is_prime(i64::MAX)` false, 2^63 - 25 true (gp `factor`,
  `isprime`), and its comment no longer calls `Integer::is_prime` fixed-base Miller-Rabin.

### Item: `square_root_mod_prime` tests primality twice

- `integer.rs:971` new `Integer::try_sqrt_mod_prime` (no primality test; refuses p < 2, even
  p > 2 and perfect squares first, then `sqrt_mod_prime_checked`).
- `rustmath-finitefields/src/integer_mod.rs:653` `square_root_mod_prime`: `assert_prime` (the
  only test, a proof) then `try_sqrt_mod_prime`; `unit_sqrts_odd` (`:814`) uses
  `try_sqrt_mod_prime` (no second test); docs of both prime-power functions and `assert_prime`.
- Residue functions (`legendre_symbol` `:759`, `is_quadratic_residue_prime` `:892`,
  `sqrt_mod_prime` `:941`, `quadratic_residues` `:1003`) use `is_probable_prime`; the
  `legendre_symbol` doc keeps the Jacobi-symbol caveat (review 2 n11). `binomial_mod` (`:1098`)
  uses `try_is_prime(p)?`.

### Item: factorization drivers and docs

- Early returns use `is_probable_prime`: `pollard_rho` (`prime.rs:770`), `pollard_p_minus_1`,
  `ecm_factor` (`ecm.rs`), `quadratic_sieve_factor_with_params` (`quadratic_sieve.rs`); their
  "None if ..." docs say "probable prime (BPSW) or no factor was found".
- Loop guards keep `is_prime` (a proof now). Docs of `factor_pollard_rho`,
  `quadratic_sieve_factor_complete`, `ecm_factor_complete` rewritten (proven factors, psi_12 split,
  `# Panics` when an ECPP proof is unresolved). `split_odd_composite` (`prime.rs:1102`):
  `debug_assert!(m odd, m > 1)` and the BPSW-pseudoprime paragraph (review 1 m4).
- `sage_wrapper.rs`: two doc lines.

### Downstream (design 2.9)

- finitefields: `finite_field.rs:147`, `:185`, `galois_ring.rs:78`, `:106` use `try_is_prime(&p)?`;
  docs in `prime_field.rs`, `cantor_zassenhaus.rs` (module doc and `check_prime_characteristic`
  doc; its code is F1's). CRLF line endings of `integer_mod.rs` preserved.
- polynomials: `factorization.rs:176` `integer_constant_is_irreducible` uses `try_is_prime` (review
  2 n3; not `prove_primality`), `MR_PROOF_BOUND` deleted, doc of `is_irreducible`;
  `berlekamp.rs:75` `check_prime` uses `try_is_prime(p)?` (+ docs of `factor` and
  `is_irreducible`); `fp_factor.rs` doc.
- crypto `prng_number.rs:91`, `:111`: `matches!(try_is_prime(&cand), Ok(true))`; module doc
  section *Primes* and one sentence in each of the five public functions.
- rings: `padics/eisenstein.rs:82`, `capped_relative.rs:106`, `:175`, `:341`, `unramified.rs:82`,
  `:151`, `newton_polygon.rs:123` use `try_is_prime(..)?`; `number_field/s_unit_solver.rs:412`
  matches `try_is_prime` (Err to `SUnitError::InvalidSUnit`); `category_methods.rs` doc (CRLF
  preserved).
- constants `certificates.rs:77`, `:90`: `matches!(try_is_prime(p), Ok(true))`, doc.
- jupyter `repl.rs`: `is_prime` arm via `checked_is_prime` (`:10962`), new `is_probable_prime`
  arm (`:925`), `next_prime` via `try_next_prime` with `check_proof_bits`, `nth_prime` ValueError
  above pi(2^64), `prime_pi` ValueError above 2^64, `primes` via `try_prime_range`, help text;
  `PROOF_BITS = 1024` (`:10947`); `sqrt_mod_report` (`:11003`): `try_is_prime` up to 1024 bits,
  `try_sqrt_mod_prime`, the psi_12 caveat and `SQRT_MOD_PSI12` removed, a caveat only for an
  unproven probable prime above 1024 bits, Jacobi shortcut on the composite path.
- `rustmath-benchmarks/src/bench_integers.rs` and `rustmath-py/src/integers.rs`: docs only
  (rustmath-py is outside the workspace and not compiled).

## Tests and reference sources

New `rustmath-integers/tests/ch18_primality.rs` (18 tests + 2 ignored), all with the gp command
next to the value (PARI/GP 2.15.4, `/usr/bin/gp`; certificates after
`default(parisizemax, 2^30)`): psi12/psi13 rejection, witnesses (StrongLucas(-7), Divisor(3),
Divisor(23), MillerRabin(2), BelowTwo, Divisor(2)), Mersenne numbers including 2^521 - 1,
nextprime(2^k) agreement, the four PARI certificates C127, C40, C65, C200 plus C200_WRAPPED,
tampered certificates (s doubled, x + 1, t + 2, N + 2, s = 0, step 3 removed, last step removed,
first step removed with n kept, negative N, no-step composite leaf, garbage text, and the valid
tail certificate of q_1), prover round trip and PARI shape for 2^89-1, 2^127-1, nextprime(10^40),
nextprime(2^200), small inputs, `is_prime_miller_rabin` base counts, H18E4 safe primes, H18E6
repunits [2, 19, 23, 317], next/previous prime values, `try_random_prime`, nth_prime/prime_pi,
prime_range windows across 2^64, p - 1 above 2^16, `prime_power`, cache transparency; ignored:
`ecpp_1024_release`, `print_certificates_for_pari`.

Unit tests: `sieve.rs` (isqrt, small primes, `sieve_matches_is_prime_u64` incl. the 2^64 edge,
intervals, counts); `primality.rs` (word tests, `jacobi_small` against `kronecker_symbol`, the 46
base-2 strong pseudoprimes below 10^6, six base-2 strong pseudoprimes without factors below 1000
that must give `StrongLucas` (review 2 n1; gp `factor`, `ispseudoprime` = 0), the strong Lucas
pseudoprimes below 20000 = [5459, 5777, 10877, 16109, 18971] with the A217255 enumeration stated,
`bpsw_gcd_with_q`, witness data validation incl. `StrongLucas(-8)`, batches); `ecpp/hilbert.rs`
(`discriminant_list_shape`, `hilbert_polynomials_match_pari` for -3, -4, -15, -23, -71, -455,
-58507, -1271, -7991, `hilbert_roots_give_j`); `ecpp/polymod.rs`; `ecpp/mod.rs`
(`ecpp_order_check_matches_affine`, `order_check_accepts_and_rejects`, `ecpp_q_bound_edges`,
`ecpp_cornacchia_traces`, `verify_accepts_c127_and_rejects_x_plus_one`, `ecpp_driver_backtracks`,
`prove_chain_on_strong_pseudoprime`); `prime.rs` (`test_is_prime` gains 2^31-1, 2^61-1, psi_12;
`test_try_is_prime_unresolved_is_err` and `test_is_prime_panics_when_unresolved`, review 2 n7);
`integer.rs:1926` `try_sqrt_mod_prime_refuses_squares`. Every reference value was recomputed with
gp in this session (scripts in `scratchpad/i1impl/*.gp`).

Changed tests: `ch18_factorization_drivers.rs:209-222` three psi12 tests (one per driver, all
enabled: 2.2 s for the whole file in debug); `ch18_residues.rs:627`
`residues_modulo_strong_pseudoprime_are_honest` (asserts `!is_prime(psi_12)`, refusal up front for
4, 43, 399165290221; Jacobi values kept; the dead `if n.is_prime()` branch removed);
`rustmath-finitefields/tests/prime_residues.rs:301`
`square_root_mod_prime_refuses_strong_pseudoprime` and new `:330`
`finite_field_constructors_refuse_strong_pseudoprime`; `rustmath-polynomials/tests/
ch23_is_irreducible.rs` (2^89 - 1 now `Ok(true)`, psi_12 `Ok(false)`); REPL
`test_sqrt_mod_large_prime_and_strong_pseudoprime` (`:11813`) and new
`test_prime_commands_with_proofs` (`:11844`).

Assertions changed because they pinned the old wrong behaviour (reference value in each case):
- `prime_residues.rs`: the old test asserted `is_prime(psi_12)` is true. gp
  `isprime(318665857834031151167461)` = 0. Its other assertions (`None` for 2 and 41, a root for
  4, a panic for 43 and 73) described the old acceptance of psi_12; now every call panics with
  "is not prime", so no false `None` is possible.
- REPL: the old test listed two roots of 4 mod psi_12 with a probable-prime caveat (gp: 4 has
  four roots mod psi_12) and took the prime path for 73. Those two rows are dropped, as the design
  says (the composite path needs psi_12 factored by trial division); J1a restores them. The
  2^127 - 1 row is now an exact equality without the caveat (stronger).
- `ch23_is_irreducible.rs`: `Err` for the constant 2^89 - 1 becomes `Ok(true)` (gp
  `isprime(2^89-1)` = 1); the honest `Err` is replaced by a proven answer.
No other assertion was changed or weakened.

## API and behaviour changes (callers checked)

New public API: `prime::{try_is_prime, try_next_prime, try_previous_prime, try_prime_range,
try_random_prime, prime_power, is_probable_prime, is_prime_u64, prove_primality, EcppCertificate,
EcppStep, Primality, NotPrimeWitness}`, modules `primality`, `sieve`, `ecpp`;
`Integer::try_sqrt_mod_prime`; REPL command `is_probable_prime`. Behaviour: `is_prime(psi_12)` and
`is_prime(psi_13)` false; the drivers split psi_12; `next_prime(n)` = 2 for n < 2;
`is_prime_miller_rabin(n, k)` uses k distinct bases (3215031751 passes k = 4);
`ArithInt::is_prime` no longer panics near 2^31 (I3 had fixed it already); the residue functions
refuse psi_12 up front; `binomial_mod`, the finite-field and p-adic constructors, Berlekamp, the
s-unit solver, the crypto generators and the REPL give `Err` (or skip) for an unresolved proof;
`nth_prime` above pi(2^64) and `prime_pi` above 2^64 panic with a message (they never returned).
Callers: the design's tables 3.1 and 3.2, re-grepped (`is_prime(`, `.is_prime()`, `next_prime(`,
`prime_range`, `random_prime`, the numbertheory glob re-export): no caller passes n < 2 to
`next_prime` or relies on `next_prime(1) = 3`; `cargo check --workspace --tests` passes.

## Deviations from the design (with evidence)

1. REPL proof-size guard (review 2 n5, the "add a limit in I1" option): `is_prime`, `next_prime`
   and `sqrt_mod` prove only up to `PROOF_BITS = 1024` bits (3.4 s measured in release, see
   below; 1536 bits is about 21 s). `is_prime` above it answers `False` when BPSW refutes n and
   is a ValueError naming `is_probable_prime` otherwise; `next_prime` above it is a ValueError;
   `sqrt_mod` above it computes the roots of a probable prime with the caveat (as J1a plans).
   A new `is_probable_prime` command gives the BPSW answer.
2. `nth_prime` in the REPL: the int_arg range stays `1..=usize::MAX` (so
   `test_int_arg_ranges_named` keeps its message); a separate ValueError covers n > pi(2^64).
3. `ecpp_order_check_matches_affine`: the design's "agree for k = 1..200" on F_97 is false for
   points of small order (MADD of T and P2 = T gives Z = 0, the exceptional case of fact 2; first
   failure [7]P for the point (3, 6) of order 5, gp `ellorder` = 5). The test now checks, for all
   99 affine points, soundness (a nonzero final Z gives the true multiple) and exactness for
   k < ord(P)/2, plus full agreement for k = 1..200 on a curve over 2^61 - 1.
4. `ecpp_driver_backtracks` (review 2 n6): with H_INNER = 1, nextprime(2^200) has no failed
   subtree, so the test pins both 2^200 + 235 (0 failed) and the next prime 2^200 + 697 (1 failed,
   gp `nextprime(2^200+236) - 2^200` = 697), each with a verified certificate, plus a starved run
   (H_INNER = 0, limit 3) that must end `Unresolved`. `max_depth >= steps.len()` (a deeper branch
   may have been explored and popped).
5. `prove_chain_on_strong_pseudoprime` is pinned: psi_12 gives `Composite(MillerRabin(41))` (gp:
   the strong test passes for the twelve bases and fails for 41).
6. `ecpp_cornacchia_traces` calls the factored-out `cornacchia` directly and checks the traces of
   a D = -3 scan (t^2 < 4N, q | m).
7. "Leaf replaced by a composite" is tested as the last step removed (leaf not below 2^64) and as
   a no-step certificate of a composite word and of psi_12.
8. `discriminant_list_shape` samples 40 rows above h = 20 (first and last of every h = 1 mod 4)
   instead of 20.
9. `pollard_p_minus_1(2, B)` is now `None` (it returned `Some(2)`, not a proper divisor; the doc
   says `None` for primes): the probable-prime check now comes before the even check.
10. `class_polynomial` also caches a failed (`None`) computation.
11. `fast_arith` imports `is_prime_u64` privately (no new `fast_arith::is_prime_u64` path).
12. A test I added on my own (ECM on (2^61-1)(2^89-1) with zero curves) was removed: the rho
    fallback needs about 2^30 steps there.
13. `try_is_prime_cache_is_transparent` uses 10^45 + 9 (gp `nextprime(10^45)`), which no other
    test in that binary proves, so the first call really proves.

## Review notes: folded in / skipped

Review 2: n1 folded (contract and tests), n2 folded (a failed Cornacchia requirement skips the
row; only a square-root `Err` or a proper gcd is a contradiction; `Advance::Composite(None)` gives
`Unresolved` at depth 0 and pops deeper), n3 folded (`try_is_prime` in
`integer_constant_is_irreducible`; the I2 sentence is in the handoffs), n4 folded (the full
`cargo test -p rustmath-jupyter` was run), n5 folded (deviation 1), n6 folded (deviation 4), n7
folded, n8 folded, n9 folded (hilbert.rs header), n10 folded (only the move was left; test merged),
n11 folded, n12 folded (the test comment says about 5 s; section 5 of the design is the design's).
Nothing skipped.

## Measurements

- gp check of the prover's own certificates (`print_certificates_for_pari`, then
  `primecertisvalid` after `default(parisizemax, 2^30)`): 2^127 - 1 (3 levels), nextprime(10^40)
  (4), nextprime(2^200) (6), a 256-bit prime (12): all valid = 1. Tampering the last step's x
  gives 0 for three of them; for nextprime(2^200) it stays valid in PARI and in `verify` alike
  (the last step has a = 0, so x + 1 gives another sextic twist of the same order). Record this
  in the commit message.
- Release, `cargo test --release -p rustmath-integers --test ch18_primality -- --ignored
  ecpp_1024_release`: a 1024-bit prime (2^1023 + 3^640 + 14) proven in 3.41 s, 54 levels (design:
  3.3 s mean). `target/release` (80 MB) was deleted afterwards.
- Debug test time: ch18_primality 4.6 s, integers lib 2.5 s, ch18_factorization_drivers 2.2 s.

## Final test counts (debug, `--no-fail-fast`)

| command | passed | failed | ignored |
|---|---|---|---|
| `cargo test -p rustmath-integers` (after the final fmt/clippy edits) | 349 | 0 | 5 |
| `cargo test -p rustmath-finitefields` | 262 | 0 | 0 |
| `cargo test -p rustmath-polynomials` | 377 | 0 | 4 |
| `cargo test -p rustmath-crypto -p rustmath-constants` | 198 | 0 | 0 |
| `cargo test -p rustmath-rings` | 1914 | 0 | 82 |
| `cargo test -p rustmath-jupyter` (full, review 2 n4) | 64 | 0 | 0 |
| `cargo test -p rustmath-numbertheory -p rustmath-ellipticcurves -p rustmath-coding -p rustmath-quadraticforms -p rustmath-curves` | 660 | 0 | 13 |

`cargo check --workspace --tests`: passes. `cargo clippy -p rustmath-integers --tests`: no
warning in the new or rewritten code (the remaining ones are pre-existing: `prime.rs` thread_local
`rand_int`, a test `vec!`, quadratic_sieve.rs, integer.rs). rustfmt was applied to the new files
and to the touched files that were rustfmt-clean at HEAD (perfect_power.rs, the three ch18 test
files, prime_residues.rs, berlekamp.rs, ch23_is_irreducible.rs); other touched files were not
reformatted. Disk: 134 stale superseded test binaries in `target/debug/deps` (older than 36 h,
each with a newer binary of the same name) were removed to free 5.9 GB (free space had dropped to
1.7 GB); cargo rebuilds any that a later run needs.

## Left undone

- Full `cargo test --workspace` (the orchestrator runs it).
- rustmath-py is not compiled (outside the workspace).
- `dcheck.rs` was not re-run (exact check of all 42,272 rows, done in the design and review).
- No edits to docs/*, PROJECT_TRACKER.md or docs/magma/*.

## Handoffs

- I2 (factorization): use `prime::is_probable_prime` for splitting decisions and
  `prime::try_is_prime` to certify factors (it caches and decides 2^64 <= p < psi_12 by the
  twelve strong tests); use `prime::prove_primality` only to keep a certificate, and not for
  p < psi_12 (it can be `Unresolved` there). `sieve::for_each_prime` exists with the I2 contract:
  no survivor test below 2^40, survivors above 2^40 confirmed by `is_prime_u64`. The QS case of
  the psi12 driver check is not ignored (three tests `psi12_is_split_by_*`). `prime::factor` and
  `prime_divisors` are unchanged (trial division). `fast_arith::pow_mod_u64` is `pub(crate)`.
- F1 (finitefields): `cantor_zassenhaus::check_prime_characteristic` still calls
  `Integer::is_prime` (doc updated only); F1 switches it to `try_is_prime`. `Integer::try_sqrt_mod_prime`,
  `prime::is_probable_prime` and `prime::prove_primality` exist with the design's names. The four
  constructor lines already use `try_is_prime(&p)?`. `integer_mod.rs` has CRLF line endings.
- P1 (polynomials): `berlekamp::check_prime` uses `try_is_prime`; the two `internal` error texts
  "composite modulus that passed the primality test" are untouched (P1's B.7). `prime::is_prime_u64`
  is public (also `primality::is_prime_u64`).
- R1 (rings): the seven p-adic lines and `s_unit_solver.rs:412` are switched; rebase on them.
  `category_methods.rs` has CRLF line endings.
- J1a (REPL): new `PROOF_BITS` (1024), `check_proof_bits`, `checked_is_prime`, the
  `is_probable_prime` arm; `primes` uses `try_prime_range` (J1a replaces it), `prime_pi` refuses
  x > 2^64 (J1a adds 6·10^9), `nth_prime` refuses n > pi(2^64), `sqrt_mod_report` is rewritten as
  described (J1a replaces it and restores the 4 and 73 mod psi_12 rows). `PROOF_BITS` sits next to
  `sqrt_mod_report`, not in `repl_limits`; J1a may move it and align it with
  `SQRT_MOD_PROOF_BITS`.
- Trackers (orchestrator, design Plan step 8): PROJECT_TRACKER rows `prime.rs:32`
  (Miller-Rabin), `:137` (next_prime family), `:808` (driver), `fast_arith.rs:315`,
  `integer.rs:701` (Legendre), `integer.rs:999` (binomial_mod), `integer_mod.rs:937` (square
  roots), `factorization.rs:146`, REPL `sqrt_mod`; new rows for BPSW (`primality.rs`), ECPP and
  certificate (`ecpp/mod.rs`), Hilbert class polynomials (`ecpp/hilbert.rs`), segmented sieve
  (`sieve.rs`); the "Baillie-PSW" missing-algorithm bullet is done, "Meissel-Lehmer" stays
  missing. `docs/magma/status_overrides.csv`: ch018.IsPrime, IsPrime.2, PrimalityCertificate (now
  `rustmath-integers/src/ecpp/mod.rs`, type `EcppCertificate`; MAGMA's `IsPrimeCertificate` is
  `EcppCertificate::verify`), IsProbablePrime, IsPrimePower, NextPrime, PreviousPrime, PrimesUpTo,
  PrimesInInterval, NthPrime, RandomPrime. HANDOFF: remove the five I1 items; add the measured
  coverage (no unresolved prime up to 2048 bits in the design's experiments; 2 of 32 at 3072 and
  4096 bits; 1024 bits 3.4 s release here) and these follow-ups: faster ECPP above 2048 bits
  (design §7), `random_prime(a, b)` enumerates the whole range and MAGMA `RandomPrime(n)` is
  missing, `prime_pi` above 2^64 (LMO) missing, the Pratt certificate verifier in
  rustmath-constants (tracker: unsound) is separate, and the private `is_prime_u64` copies in
  rustmath-modular, rustmath-ellipticcurves and rustmath-galois can call
  `rustmath_integers::prime::is_prime_u64`.
