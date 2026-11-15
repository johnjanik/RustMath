# Review 1 of the F1 implementation (rustmath-finitefields, with the core and polynomials hooks)

Reviewer scratch: `$R` = `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/F1-finitefields.rev1/`
(`accept.sh` and `acc_<step>.log`, `summary.log`; gp scripts `ref1.gp`, `ref2.gp`, `ref3.log` with output
`ref1.log`, `ref2.log` (PARI/GP 2.15.4); probes `probe1.rs` .. `probe3.rs` with `.log` output, built by `rc.sh`
with rustc against the current rlibs; clippy `clippy_all.log`). Probe binaries deleted.

Verdict: **approved**. No blockers. Every acceptance command passes. Four minor notes below, all docs.

## What I read

The whole uncommitted diff (17 modified files) and the five new files in full: `src/group_order.rs`,
`tests/field_sqrt.rs`, `tests/unit_group_orders.rs`, `tests/unit_group_orders_capped.rs`,
`rustmath-polynomials/tests/unbound_constants.rs`. The design (revision 1), both design reviews, the F1
implementation report, and the handoffs of I3, I1, I2 and P1. Surrounding code: `ring_hooks.rs`,
`FiniteField::new`/`with_modulus`, `GaloisRing::with_modulus`, `PrimeField::new`/`is_quadratic_residue`,
`Integer::is_quadratic_residue_prime`, I2's `LimitedFactorization`/`FactorizationLimits` contract, every
writer of the term maps in `multivariate.rs` and `laurent.rs`, and the workspace callers of the changed APIs.

## Acceptance commands (design 9.2), run by me on the final tree

| Command | Passed | Failed | Ignored |
|---|---|---|---|
| `cargo check --workspace --tests` | green | | |
| `cargo test -p rustmath-core --no-fail-fast` | 50 | 0 | 1 |
| `cargo test -p rustmath-polynomials --no-fail-fast` | 434 | 0 | 4 |
| `cargo test -p rustmath-finitefields --no-fail-fast` | 316 | 0 | 0 |
| `cargo test -p rustmath-finitefields --test unit_group_orders_capped -- --test-threads=1` | 1 | 0 | 0 |
| `cargo test -p rustmath-galois --no-fail-fast` | 85 | 0 | 5 |
| `cargo test -p rustmath-matrix --no-fail-fast` | 295 | 0 | 1 |
| `cargo test -p rustmath-rings --no-fail-fast` | 1914 | 0 | 82 |
| `cargo test -p rustmath-coding -p rustmath-crypto -p rustmath-curves -p rustmath-ellipticcurves -p rustmath-algebras -p rustmath-modules -p rustmath-powerseries -p rustmath-schemes --no-fail-fast` | 1999 | 0 | 27 |
| `cargo test -p rustmath-jupyter --no-fail-fast` | 64 | 0 | 0 |
| `cargo clippy -p rustmath-finitefields -p rustmath-polynomials -p rustmath-core` | exit 0 | | |

The counts equal the implementation report's table. Extra: `cargo clippy -p rustmath-finitefields -p
rustmath-polynomials --all-targets --no-deps` gives 68 lint locations, none on a line this unit added
(filtered with the implementer's `added_lines.py`). Every crate with a `rustmath-finitefields` dependency
that is a workspace member is in the list (`rustmath-py` is excluded from the workspace).

Timings (debug, my run): `unit_group_orders` 2.24 s, `unit_group_orders_capped` 0.79 s, `field_sqrt`
1.79 s, finitefields unit tests 1.60 s. The report's numbers hold; no `#[ignore]` is needed. The
rustmath-galois `descent` test takes 20.2 s, against 24.3 s to 27.9 s in earlier logs (`p1impl/acc/galois.log`,
`test_all.log`), so the `GFPN_VALID` lookup in every `Gfpn::new` is no regression.

## Reference values (re-run by me with gp)

`$R/ref1.log`: all 12 `znorder` rows of `integer_mod_orders_match_pari`; the `PrimeField` rows (2000000000123,
20000000000002643, p30, all prime by `isprime`); `fforder` of w and w+1 in GF(5^30), GF(3^40), GF(2^127),
GF(101^12), GF(125), with `polisirreducible` = 1 for every default modulus; `factor(fforder(w+1))` in
GF(5^30); P43: `isprime`, `factor(P43-1)`, the orders mod P43 and P43^2; the λ-fallback modulus
m = 7 · s (`factor(m)`, `factor(m-1)`, `Mod(a,m)^(m-1) == 1`, `znorder` = 6, `factor(s-1)`); λ(341),
λ(2^10), λ(561); `znorder(Mod(2, 12*2^64+1))`; 648 units of GR(27, 2); `Mod(3,91)^90 = 1`.
`$R/ref2.log`: all 12 `lr` rows of `prime_field_sqrt_matches_pari`; every `show` row of GF(9), GF(125),
GF((2^61-1)^2), GF((2^80+13)^2) and both GF(256) roots; `kronecker(3, 2^61-1) = -1`; irreducibility of the
six pinned `random_irreducible` outputs; the Gfpn rows (`t^2+2` mod 3 reducible, `t*t = 2` mod `2t^2+1`,
`t^2+t+1` irreducible mod 5 and `(t+3)(t+5)` mod 7, the three `from_int` residues); the DDF rows; every
container value of `unbound_constants.rs`, `sentinel_eq.rs` and `galois_ring_sentinel.rs`, including the
Laurent `subst` form; the two Z/8 values of `polynomial_hooks_use_commutativity_of_finite_rings`. All match.

## Per item

- **Item 1 (DDF and squarefree checks).** Both public DDFs check in the designed order (characteristic for
  the lava one, then f = 0, then squarefree) and make f monic. Internal callers use the unchecked bodies:
  `cz::factor` calls `squarefree_prime_char`, `factor_squarefree_prime_char` calls `ddf_squarefree`,
  canonical `factor` checks each part once and calls `ddf_squarefree`. Probe: over a `Gfpn` GF(9) the lava
  squarefree factorization of `(y+w)^2 (y+1)` gives multiplicities [1, 2], the DDF of it panics with
  "f is not squarefree", and the DDF of `(y+w)(y+1)` gives one degree-1 block.
- **Item 2 (Gfpn is a field).** `try_new` validates in the documented order, rejects p < 2 before any
  division (review 2 m1), normalizes the modulus, and memoizes only pairs that passed all checks. Probe: a
  valid modulus over p = 3 followed by a reducible one over the same p is still rejected; `[5, -1, 4]` mod 3
  is stored as `[2, 2, 1]`; a degree-1 non-monic modulus and `[1, 1, 0]` are rejected as not monic. The
  memo is bounded (review 2 m2). The only callers outside the crate (rustmath-galois) pass irreducible
  moduli; their tests pass.
- **Item 3 (generators, `is_irreducible_fp`, `from_int`).** Both stacks draw every coefficient with the exact
  `uniform_below`, check p first, and fall back as designed. `FiniteField::new`, `with_modulus`,
  `GaloisRing::with_modulus`, `irreducible_polynomial` and `Gfpn::try_new` call the unchecked bodies only
  after `try_is_prime` proved p prime (I read each site). `Gfpn::from_int` is O(1); the trait default is
  double-and-add, tested through a delegating newtype that keeps the default.
- **Item 4 (orders).** `group_order.rs` follows Cohen Alg. 1.4.3, with the `x^h = 1` check always on.
  Honesty: I2's `primes` are proven by its contract; `probable_primes` are used only after `try_is_prime`
  returns `Ok(true)` (Capped, at most 256 bits) and make Complete fail; `composites` always fail; only
  proven results are cached. The λ fallback of review 2 m3 uses a failed BPSW test only to choose a path.
  `carmichael_lambda` and `cyclotomic_value` are correct (λ(2), λ(4), λ(2^e), exact Möbius quotient). Every
  hook maps a failure to `Undecided`; the inherent methods panic only on an unresolved proof. Probes: with
  p = 2q + 1, q a 261-bit prime, the three capped hooks answer `Undecided` in 5 ms (PrimeField, IntegerMod)
  and 0.36 s (GF(p), most of it the ECPP proof of p in `FiniteField::new`); the inherent order of 3 is
  (p - 1)/2 in 0.53 s (gp `znorder(Mod(3,p)) == (p-1)/2` = 1), after which the hook is `FiniteExceedsU64`
  (the documented cache effect). For an RSA-like 79-digit modulus the IntegerMod hook answers `Undecided`
  in 43 ms. GR(9, 2): -1 has order 2 without factoring.
- **Item 5 (IsSquare, Sqrt).** Euler's criterion decides; Tonelli-Shanks matches Cohen Alg. 1.5.1 step by
  step; the e = 1 and characteristic-2 roots are right; every root is squared back; the non-square scan is
  bounded by q. Probe: 200 random elements in each of GF(1000003^2), GF(101^3), GF(65537^2), GF(7^6),
  GF(13^4): every root squares back and every `None` is a non-square. `PrimeField::is_square` and `sqrt`
  both test the modulus with BPSW only (I read `is_quadratic_residue_prime`), so review 2 m4 is moot, as
  the report says.
- **Item 6.** `square_root_mod_prime` runs `assert_prime` once and then `try_sqrt_mod_prime`; I1's
  `square_root_mod_prime_refuses_strong_pseudoprime` exists and passes.
- **Item 7 (containers).** `UnivariatePolynomial::new` is still the only constructor and no method mutates
  `coeffs`. In `MultivariatePolynomial` and `LaurentPolynomial` the term maps are written only by the
  one-term constructors, `add_term`, `bind_all`, and the Laurent `unit_inverse`, which inserts the nonzero
  coefficients of a univariate inverse (bound or unbound as a whole, by invariant U), so invariant M holds.
  Every arithmetic operation goes through `add_term` (I read `Add`, `Sub`, `Neg`, `Mul`, `scalar_mul`,
  `monomial_mul`, `relabel`, `divide`, `divide_multiple`, `partial_derivative`). The three hooks satisfy the
  core contract in all five types and in both container impls. No polynomial type implements `Hash`
  anywhere in the workspace. The P1 handoff (`static_is_commutative() = Some(true)` on the five types) is
  done and tested.

## Checks of the project rules

- No `unsafe`, no em-dash in an added line, no new `num-*` wrapper. CRLF endings of `integer_mod.rs` kept
  (1472 of 1472 lines).
- New file headers cite the MAGMA chapter and the papers (`group_order.rs`: ch. 18, 19, 21, 48, [Coh93],
  [Car10], [BLSTW02]; the three finitefields test files and `unbound_constants.rs` as designed, review 2 m6).
- Removed assertions: only inside the replaced `polynomials_from_static_constructors_keep_unbound_coefficients`
  (degree 7, `q != qb`, the unbound coefficients 8..8, `q_bound`). That test asserted the defect; gp
  `lift(Mod(1,8)*((x+1)^8-x^8))` = `4*x^6 + 6*x^4 + 4*x^2 + 1` proves degree 6. The new test keeps the
  rest (`coeff(0) = 1 + w`, the unbound all-Z part with its Z coefficients, the `scalar_mul` binding) and
  the report says so. No other assertion was removed or weakened.
- New tests fail on HEAD (new APIs, panics where HEAD answered, HEAD's degree 7, or HEAD's O(m) loops).
- Working tree: only this unit's files; no `rustmath-misc/output.txt` or `.tmp_dir_*` change; no
  `target/release`.

## Review 2 minor notes

All eight are folded in as the report states; I checked each in the code (m1 order of checks and the
p = 0, 1, -7 rows; m2 the 4096-pair bound; m3 the λ fallback and its test with gp values; m4 moot after
I1; m5 the handoff wording; m6 the headers; m7 the `use` in `cantor_zassenhaus.rs`; m8 "for a prime
modulus").

## Blockers

None.

## Minor issues

- **n1 (doc, `rustmath-polynomials/src/univariate.rs:69-77`, `PartialEq`).** The doc says that equal lengths
  give "the same answer as binding first" because the coefficient `==` coerces. That holds for scalar
  coefficients and for univariate nesting, not for a coefficient type whose zero is a parentless sparse
  map. Probe (`$R/probe1.log`): over `UnivariatePolynomial<LaurentPolynomial<IntegerMod>>`,
  f = `new([8 (unbound), 3 mod 8])` is `[0 (empty map), 3]`, g = `new([8, 3])` (all unbound) stays
  `[8, 3]`, and `f == g` is false, although binding g gives f. This is the documented sparse-zero
  consequence of `LaurentPolynomial`, so no value is wrong by the stated representation, but the univariate
  sentence overstates. **Change:** restrict the sentence to coefficient types whose zero carries a parent
  (or add: "over `LaurentPolynomial` coefficients an empty-map coefficient compares as in that type").
- **n2 (doc, `rustmath-finitefields/src/prime_field.rs:159`, `is_generator`).** The doc cites MAGMA
  `IsPrimitive` (ch. 19 §19.7), which for Z/mZ asks whether n generates (Z/mZ)^*. For a composite m with a
  cyclic unit group the two differ: 2 mod 9 has order 6 = φ(9) (gp `znorder(Mod(2,9))` = 6,
  `eulerphi(9)` = 6), so MAGMA's `IsPrimitive` is true and `is_generator` is false. The doc already says
  "for a composite modulus m this is always false", which is honest and the HEAD behaviour. **Change:**
  "MAGMA `IsPrimitive` for a prime modulus" in the citation.
- **n3 (doc, the `FiniteFieldElement` and `GaloisRingElement` hooks).** The hook docs describe the cost cap
  of the factorization only. After it, Alg. 1.4.3 needs about ω(q - 1) + 1 exponentiations in GF(q), and in
  debug these dominate: the capped hook of the generator of GF(2^n) takes 0.45 s, 2.6 s, 5.0 s, 12.4 s,
  36.6 s for n = 64, 100, 128, 160, 200, and the same on a second (cached) call; one full exponentiation in
  GF(2^200) takes 2.5 s and ω(2^200 - 1) = 18 (`$R/probe2`, `probe3.log`; gp `omega`). This is no
  regression (HEAD runs the same exponentiations after its trial division), and the design's cost table
  says "dominated by `pow_int`". **Change (suggested):** one sentence in the hook `# Cost`: "the cap bounds
  the factorization; the order then costs about ω(q - 1) + 1 exponentiations in GF(q)".
- **n4 (doc, `finite_field.rs:649`, `:663`, `multiplicative_order`, `factored_multiplicative_order`).** `# Panics` says
  "for an unbound element", but an unbound zero returns `None` (zero is tested before the field is needed,
  as at HEAD and as deviation 6 says for `is_primitive`). **Change:** "for an unbound nonzero element".

## Observations (no change needed in this unit)

- `FiniteField::new(2, 256)` did not return within several minutes in my probe (debug). That is the
  existing non-Conway search with debug Rabin tests (design §8 finding 1 covers the O(p) case for large p;
  this is the small-p, large-n case of the same search). The orchestrator may add it to that HANDOFF line.
- The capped hooks recompute a failed capped factorization on every call (only successes are cached). The
  cost stays within the cap per call; this is the designed behaviour.
