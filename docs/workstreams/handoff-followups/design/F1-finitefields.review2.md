# Review 2 of design F1 (rustmath-finitefields, with the core and polynomials hooks)

Reviewer scratch: `$R` = `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/F1-finitefields/review2/`
(`rc.sh` builds against the designer's frozen HEAD rlibs in `../lib`; programs `r2_head.rs`, `r2_sqf.rs`,
`r2_p0.rs`, `r2_p3.rs`; Python `r2_order.py`; gp scripts `ref2.gp` to `ref5.gp` with `.log` output, PARI/GP 2.15.4).

Verdict: **approved**. No blockers. Revision 1 fixes all three blockers and all eight minors of review 1.
Eight minor issues below. Each is a small edit the implementer can make in place.

## Reproduced at HEAD c79a582 (`$R/r2_head.rs`)

| Item | Claim | My check | Result |
|---|---|---|---|
| 1 | Both DDFs accept non-squarefree input | lava DDF of `(x+1)^2` mod 5 gives `[(x+1, 1), (x+1, 1)]` | confirmed |
| 1 | Routing `poly_factor::factor` through a part check cannot fire on valid input | exhaustive: every monic polynomial over GF(2) to degree 11, GF(3) to 7, GF(4) to 6, GF(5) to 5, GF(7) to 4 (19,538 inputs); canonical `square_free_factorization` parts are squarefree, pairwise coprime, and multiply back to f (`$R/r2_sqf.rs`) | 0 failures; stronger than the designer's random test |
| 2 | `Gfpn::new` checks nothing; non-monic modulus gives silent wrong products | `t*t` in F5[t]/(2t^2+1) = `[4, 0]`; gp `lift(lift(Mod(Mod(1,5)*x, Mod(1,5)*(2*x^2+1))^2))` = 2 | confirmed |
| 2 | Every constructor path | grep `Gfpn {` / `Gfpn::new` over the workspace: literals only in `ff_poly.rs`, each copies `p`/`modulus` from an existing value; fields private; no serde/Default/From; callers `galois_ctx.rs:213`, `:585`, `relative_invariant.rs:300`, `:379`, `:382`, in-crate tests. Every in-tree modulus is irreducible (GF(4) `[1,1,1]`, GF(9) `[1,0,1]`, GF(8) `[1,1,0,1]`, galois factors from `fp_factor`, `generate_gfpn_modulus`) | confirmed, complete |
| 3 | `is_irreducible_fp(x^2+1, 15)` answers | `true` | confirmed |
| 3 | Pinned uniform draws | reran the designer's `p3_randirr.rs` (`$R/r2_p3.rs`): the same eight pinned polynomials and the range counts 65/128, 67/128, 128/128; its sampler and SplitMix64 are identical to `ff_poly.rs:146` and `poly_factor.rs:534`; gp `polisirreducible` = 1 for all six pinned rows (`$R/ref5.gp`) | confirmed |
| 4 | O(m) order loop | `IntegerMod(2, 1000003).multiplicative_order()` 0.98 s (design 1.08 s) | confirmed |
| 4 | Default moduli in the order tests | `FiniteField::new` gives `t^30+t^3+t+3` (5^30, 1.09 s at HEAD, not Conway), `t^40+t+2`, `t^12+t+11`, `t^127+t+1`; Conway `t^2+2t+2`, `t^3+3t+3`, `t^8+t^4+t^3+t^2+1` | match the test inputs |
| 5 | No root API; sentinel convention | `PrimeField` sentinel 2 `is_quadratic_residue` = false (Z semantics, as `IntegerMod::sqrt` of the sentinel) | the designed sentinel `sqrt` follows the HEAD convention |
| 6 | HEAD fallback | `Integer::sqrt_mod_prime` (`integer.rs:883-893`) tests `is_prime` once and maps Jacobi -1 to "Not a quadratic residue"; `sqrt_mod_prime_checked` reduces a and returns a for p = 2 (`:1160-1164`) | the fallback snippet is correct on HEAD's API |
| 7 | Univariate and multivariate GR(8,2) keep 8 terms, degree 7 | `(x+1)^8 - x^8 + w`: univariate degree 7; `MultivariatePolynomial` 8 terms, degree 7 | confirmed |

## Algorithms and references (re-derived or re-run)

- **Order from a factored multiple (Cohen Alg. 1.4.3) with the Fermat multiple m - 1 or Carmichael lambda,
  plus the ±1 shortcuts**: independent Python implementation (`$R/r2_order.py`) against brute force on every
  unit mod m < 400: 48,517 units agree, 1,526 of them Fermat liars on a composite m. The values 10, 30, 40, 80, 6
  for 2 and 3 mod 341, 2 and 5 mod 561, 3 mod 91 match gp `znorder`.
- **gp reference values I re-ran** (`$R/ref2.log`, `ref3.log`, `ref4.log`): all 12 `znorder` rows of
  `integer_mod_orders_match_pari`; `Mod(3,341)^340` = 56; lambda(341), lambda(2^10), lambda(561) = 30, 256, 80;
  `isprime(P43)` = 1 and `factor(P43-1)` = 2·43·C41 with 41 digits; the three P43 orders in Z/P43 and the three in
  Z/P43^2; p16, p30 and 2000000000123 rows; `factor(fforder(w+1))` in GF(5^30) and GF(125); all 12
  `prime_field_sqrt_matches_pari` values; the GF(9), GF(125), GF(256), GF((2^61-1)^2) and GF((2^80+13)^2) roots
  with the designed canonical choice (smaller of Σ c_i p^i for r and -r); the item 7 values `4x^6+6x^4+4x^2+1` and
  `2yx`; the `from_int` residues mod 2^61-1; irreducibility of `t^2+t+1` mod 5 (1) and 7 (0), `t^2+2` mod 3 (0),
  `t^2-3` mod 2^61-1 (1); `factormod(x^5+x^4+x+1, 5)`; the nested value `(x+1)^4 y + 3y^2` mod 8.
- **Citations** checked in the digests: ch. 21 §21.4.4 `IsSquare` returns the root and `IsPrimitive`; §21.4.7
  `Order`, `FactoredOrder`, `Sqrt`; §21.5 `RandomIrreduciblePolynomial` ("random polynomial with irreducibility
  testing"); ch. 19 `Order` (0 for non-units); ch. 23 §23.8.1 DDF needs a squarefree f. Ch. 22 is Nearfields, as
  the design says.
- **I1/I2 APIs** named in §0.1 exist with the stated semantics in the current I1 and I2 designs: `try_is_prime`
  (n < 2 gives Ok(false)), `is_probable_prime`, `prove_primality`/`Primality`, `try_sqrt_mod_prime`;
  `factorization_with_limits`, `FactorizationLimits { trial_division_limit, pollard_rho_limit, mpqs_limit, proof }`
  (default 10000, 8191, None, true; `Some(x >= 25)` sieves up to x digits), `LimitedFactorization { sign, primes,
  probable_primes, composites }`, `is_complete()`. With `mpqs_limit: None` the engine always completes (sieve,
  then the complete rho fallback), so `Effort::Complete` fails only on an unresolved ECPP proof, as the design says.
  I2's debug sieve time at 40 digits is 0.46 s, so the P43 tests cost about half a second each.
- **Capped test determinism**: `factor_proven(P43-1, Capped)` must fail: C41 has 41 digits (above the 30-digit
  sieve limit) and its factors have 21 digits (8191 rho steps succeed with probability about 1e-12). All four
  types key the cache on P43 - 1. `GaloisRing::new` and `FiniteField::new(P43, 1)` factor nothing. The residues
  of `1 + P43` and `2·P43 - 1` are 1 and -1, so their inherent (Complete) orders in phase 1 factor nothing
  either. One test per file makes the phases ordered.
- **Item 7 rules**: `new` is the only univariate constructor (one struct literal, no `&mut self` on `coeffs`). In
  `MultivariatePolynomial` and `LaurentPolynomial` every operation goes through `add_term` except the one-term
  literals. No polynomial type has `Hash` or serde derives. `MultivariatePolynomial` implements no `Ring`. Invariant
  M holds by the case analysis of §7.2. The cross-length univariate `==` argument (a shortening top coefficient
  already compares unequal) is right. No name collision with the three new `Ring` items in the workspace.
- **Existing tests that must stay green**: every `Undecided` assertion in `ring_predicates.rs` is on a sentinel or on
  the `GrDefaults` wrapper; every decided order there has a group exponent that the capped factorization handles
  (`p - 1 = 2^66·3`, residue fields GF(2), GF(9)). `matrix_forms_gfp.rs` compares equal-length polynomials. No
  crate outside finitefields calls the order, primitivity, generator or generator-search APIs (grep).

## Blockers

None.

## Minor issues

- **m1 (item 2, `Gfpn::try_new` with p = 0).** §2.3 builds the memo key first ("length check; reduce the modulus;
  ... look up; on a miss run the checks"). The reduction `Gfpn::redp` calls `div_rem(p).unwrap()`, and HEAD
  `Integer::div_rem` by 0 is `Err(DivisionByZero)`. So `try_new(.., 0, ..)` would panic before the prime check
  (`$R/r2_p0.rs`: HEAD panics "called `Result::unwrap()` on an `Err` value: DivisionByZero" at `ff_poly.rs:575`).
  **Change:** reject p < 2 with the "characteristic {p} is not prime" error before the reduction. Add p = 0, 1 and
  -7 to `gfpn_rejects_composite_characteristic`.
- **m2 (item 2, memo growth).** `GFPN_VALID` is never evicted. A long rustmath-galois batch (one prime and up to
  deg f factor moduli per polynomial) grows it without bound. The order cache already clears at 4096 entries.
  **Change:** give `GFPN_VALID` the same bound. Revalidating is always correct.
- **m3 (item 4, Fermat path first).** For a Carmichael modulus every unit passes a^(m-1) = 1, so `unit_order_mod`
  always factors m - 1. When m has only small prime factors and m - 1 has a composite part above 30 digits,
  the capped hook answers `Undecided` although lambda(m) would decide at once. The inherent method also takes the
  harder factorization. This is honest but avoidable. **Change (suggested):** with `Effort::Capped`, when the
  Fermat-path factorization fails, try `carmichael_lambda(m, Capped)` before answering `Undecided`. Or state the
  case in the `# Cost` docs.
- **m4 (item 5, `is_square` vs `sqrt` on `PrimeField`).** `is_square` is an alias of `is_quadratic_residue`, and
  that calls `is_prime` (an ECPP proof, which panics when unresolved). `sqrt` uses only BPSW. So "is_square and
  sqrt agree" fails when the proof of a large modulus is unresolved: `is_square` panics and `sqrt` answers. The cost
  also differs (one proof against none). **Change:** say so in both docs. Or define `is_square` as
  `self.sqrt().is_some()` for bound nonzero elements, as `IntegerMod::is_square` does.
- **m5 (item 7 hand-off, severity).** §7.6 says that for a mixed `Matrix` "is_zero, rank and determinant ... can see
  those integers". For `SparseMatrix` it says an unbound zero "is stored as nonzero". Both mean that wrong values,
  not only an `Err`, remain possible after F1. **Change:** the HANDOFF line and a tracker row should call this a
  correctness defect with that scope, not a caveat.
- **m6 (CLAUDE.md headers).** §7.5 gives no header for `rustmath-polynomials/tests/unbound_constants.rs`. §4.6 gives
  the header of `tests/unit_group_orders_capped.rs` only as the reason for its one test. **Change:** cite ch. 48
  §48.2.3 (`R ! a`) and ch. 23 in the first, and ch. 19 §19.7, ch. 21 §21.4.7 and Cohen Alg. 1.4.3 in the second.
- **m7 (plan step 3.2).** Moving `check_prime_characteristic` from `cantor_zassenhaus.rs` to `ff_poly.rs` in step
  3.2 breaks `cantor_zassenhaus.rs` until step 3.3, unless step 3.2 also changes its call sites (or keeps a
  `use`). **Change:** add that to step 3.2 so every step builds.
- **m8 (doc precision, item 5).** For a BPSW pseudoprime modulus (none known), `PrimeField::sqrt` returns
  min(r, m - r), which need not be "the least root in [0, p)". **Change:** add "for a prime modulus" to that
  sentence.

## Checked and fine

- Item 1: eager checks with private unchecked bodies. Internal callers skip repeated tests. The messages keep the
  existing should-panic texts ("f is not squarefree", "characteristic 15 is not a prime"). The p > usize::MAX
  argument for `pth_root` holds.
- Item 2: validation order, normalization (the B1 fix uses an irreducible modulus), the unreachable `invert`
  branch kept as an `Err`, and the panic text through `Display` ("Invalid argument: Gfpn: ...").
- Item 3: exact sampler, `check_char` before any division, the bounded canonical search ends in
  `Err(NotSupported)` (decides nothing), and the lava fallback is exhaustive. `is_irreducible_fp` callers are
  complete (in-crate only; the numberfields and rings functions of that name are private copies).
  `from_int` is O(1) on `Gfpn` and O(log p) in the default.
- Item 4: only complete, proven factorizations are used or cached. Hooks map failures to `Undecided`. The cache
  semantics and the separate test binary resolve review 1's B2. The GR residue step has the ±1 shortcut, and
  `UnitOrder::Undecided` panics only in the inherent method. `factored_multiplicative_order(1) = Some(vec![])`.
- Item 5: Euler's criterion decides. The non-square scan only finds a helper and always ends. Every root is
  squared back. The canonical root agrees with `IntegerMod::sqrt` for n = 1.
- Item 6: I1 takes it (I1 §2.7). The HEAD fallback is correct.
- Item 7: hooks with inert defaults; `const` gating compiles out every branch for other coefficient types. The
  matrix, series and p-adic hand-off is explicit, as review 1 required. M1b (which edits rustmath-matrix later) knows
  the hook and does not depend on HEAD container behaviour.
- Plan: the order builds green step by step, apart from m7. Acceptance covers every reverse dependency of
  rustmath-finitefields that has tests, plus `cargo check --workspace --tests`.
