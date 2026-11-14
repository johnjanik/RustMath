# Implementation report: Unit F1 (rustmath-finitefields, with the core and polynomials hooks)

Branch `claude/gracious-gates-e6b6l7`, on top of HEAD `2406bbe` (I3, I1, I2, P1 committed). Design
`fu2/design/F1-finitefields.md` (revision 1, approved by review 2). Nothing committed (the
orchestrator commits). Logs: `fu2/impl/F1-logs/` (`accept_summary.log`, `acc_<step>.log`,
`check_ws.log`, `clippy*.log`); helper scripts and gp scripts: `fu2/impl/F1-tools/`.

Line numbers below are in the working tree after this unit.

## Per item

### Step 1: rustmath-core hooks (design 7.2, 4.4)

- `rustmath-core/src/traits.rs:207-241`: `Ring::HAS_UNBOUND_CONSTANTS` (default `false`),
  `is_unbound_constant` (default `false`), `bind_to_parent_of` (default `None`), with the
  container wording of design 7.2 (a zero container counts as unbound).
- `traits.rs:259-265`: new doc of `MultiplicativeOrder::Undecided` (cost cap, cached work can
  decide later, both answers honest); `:94-97` the predicate-section comment says the same.
- Test `default_hooks_are_inert` (core unit tests): `i32`, `i64`. Deviation: the design names
  `Integer` too; rustmath-core does not depend on rustmath-integers, so `Integer` is checked in
  `rustmath-polynomials/tests/unbound_constants.rs::default_hooks_are_inert_for_containers`.

### Step 2: rustmath-polynomials containers (design 7.2, 7.3)

- `univariate.rs`: type docs "Unbound constants" (invariant U, coercing `==`); `PartialEq`
  (`:69`) coerces across lengths when exactly one side has a bound coefficient; private
  `bound_model` (`:93`, a scan) and `bind_coefficients` (`:101`); `new` (`:117`) binds before
  trimming; `coeff` (`:174`) returns a bound zero past the degree; `impl Ring` (`:1186`)
  `HAS_UNBOUND_CONSTANTS = R::...`, `is_unbound_constant` (all coefficients unbound),
  `bind_to_parent_of`.
- `multivariate.rs`: derive `PartialEq` replaced by a manual coercing impl (`:308`); type docs
  (invariant M, the sparse-zero consequence); `bound_model` (`:364`), `bind_all` (`:373`);
  `add_term` (`:385`) binds an unbound coefficient entering a bound map, or binds the whole
  unbound map once when a bound coefficient enters; `coefficient` (`:419`) gives a bound zero.
  `MultivariatePolynomial` has no `Ring` impl, so no hooks.
- `laurent.rs`: the same (`PartialEq` `:51`, `impl<R: Ring + Eq> Eq`, `bound_model` `:81`,
  `bind_all` `:90`, `add_term` `:131`, `coeff` `:168`), plus the three hooks in `impl Ring`
  (`:583`). `add_term` still creates no `R::zero()` (p-adic safety, P1's note).
- New `tests/unbound_constants.rs` (header: ch. 48 §48.2.3, ch. 23; review 2 m6) with a test
  sentinel `Zs`: the 10 tests of design 7.5 table 1. Reference values gp (in the header and at
  each test): `lift(Mod(1,8)*((x+1)^8 - x^8 + 1))`, `Vecrev((x+1)^8 - x^8)`,
  `lift(Mod(1,8)*((x+1)^4*y + 3*y^2))`, and for the Laurent row
  `subst(lift(Mod(1,8)*((y+1)^8 - y^8 + 1)), y, 1/q)` = `(2*q^6 + 4*q^4 + 6*q^2 + 4)/q^6` (the
  design's direct `lift(Mod(1,8)*((1/q+1)^8 - q^-8 + 1))` fails in gp with "impossible inverse",
  so the substitution form is quoted).

### Item 3 part A, item 2, item 1, item 3 part B (plan steps 3.1 to 3.5)

- `poly_factor.rs:173` `check_char` (I1 `try_is_prime`; "characteristic {p} is not prime"; an
  unresolved proof is returned as is). `:190` `is_irreducible_fp` checks and panics
  "poly_factor::is_irreducible_fp: {e}"; `:199` `is_irreducible_fp_unchecked` (the old body).
  `finite_field.rs` `with_modulus` and `galois_ring.rs` `with_modulus` call the unchecked body
  (p proven above).
- `ff_poly.rs:196` `pub(crate) check_prime_characteristic` (moved from `cantor_zassenhaus.rs`,
  now `try_is_prime`; `Ok(false)` "characteristic {c} is not a prime", `Err(e)` "the primality of
  the characteristic {c} is not decided ({e})"); `cantor_zassenhaus.rs` imports it in the same
  step (review 2 m7).
- `ff_poly.rs:67` trait default `from_int`: double-and-add over the bits of `n mod p`
  (O(log p) additions). `:788` `Gfpn::from_int` builds `[n mod p, 0, ...]` directly.
- `ff_poly.rs:656` `Gfpn::try_new`: degree >= 1; `p < 2` rejected before any reduction (review 2
  m1); reduce the modulus (no allocation when already reduced); memo lookup; on a miss
  `try_is_prime`, monic mod p, Rabin (`is_irreducible_fp_unchecked`); messages exactly as in the
  design. `:701` `Gfpn::new` panics with the error's `Display`. `:603` memo `GFPN_VALID`
  (`RwLock<HashMap<Integer, HashSet<Vec<Integer>>>>`, borrowed lookup, poison recovered), bounded
  at 4096 pairs (cleared beyond; review 2 m2). `:920` the unreachable `invert` branch keeps an
  `Err` "internal: gcd with the defining polynomial is not 1". Type docs: a `Gfpn` is a field by
  construction.
- `cantor_zassenhaus.rs`: module docs rewritten (every public entry checks the characteristic;
  a `Gfpn` is a field by construction; the DDF checks squarefree); `:136` `squarefree_check`
  (shared by `block_contract` and `factor_squarefree_prime_char`, same message);
  `:363` `squarefree_factorization` checks the characteristic, body in `:376`
  `squarefree_prime_char` (with the `p_usize = 0` argument in its doc); `:464` checked
  `distinct_degree_factorization` (characteristic, f = 0, squarefree, in that order), body in
  `:488` `ddf_squarefree`; `factor` calls `squarefree_prime_char` (`:635`) and
  `factor_squarefree_prime_char` calls `ddf_squarefree`, so no test runs twice. The "a proof
  below psi12" wording in three `# Panics` sections is gone.
- `irreducible.rs`: module docs ([Rab80], ch. 21 §21.5); `:63` `is_irreducible` checks the
  characteristic (panic "irreducible::is_irreducible: characteristic 15 is not a prime"), body
  `:72` `rabin_irreducible`; `:142` `find_irreducible` / `:194` `random_irreducible` /
  `:225` `generate_gfpn_modulus` give `None` when p is not proven prime; `random_irreducible`
  draws `c_0..c_{n-1}` with `ff_poly::uniform_below` from `SplitMix64::new(seed)`, 64 n
  candidates, then `find_irreducible_unchecked` (`:150`).
- `poly_factor.rs:242` `find_irreducible` (checked) and `:252` `find_irreducible_unchecked`
  (used by `FiniteField::new` and `irreducible_polynomial`); `:303` `random_irreducible`
  (checked) and `:312` `random_irreducible_unchecked` with `uniform_below` (both seedings kept);
  `:681` checked DDF ("f = 0 is not squarefree", "f is not squarefree"), `:699`
  `squarefree_check` (also used by `fq_edf_contract`), `:712` `ddf_squarefree`; `:946` `factor`
  checks each square-free part ("... fails the squarefree check (...); this is a bug").
- Tests (all with gp values in comments):
  `poly_factor`: `is_irreducible_fp_rejects_non_prime_p` (should_panic, 15),
  `is_irreducible_fp_rejects_p_below_two` (1, 0, -7), `is_irreducible_fp_unchanged_for_primes`,
  `random_irreducible_draws_the_whole_field` (the four pinned outputs and the range counts),
  `generators_reject_non_prime_p`, `ddf_rejects_non_squarefree` (four rows plus `(x+w)^2` over
  GF(4)), `ddf_rejects_zero`, `ddf_squarefree_matches_pari`.
  `ff_poly`: `gfpn_rejects_reducible_modulus` and its should_panic twin
  `gfpn_new_panics_on_reducible_modulus`, `gfpn_rejects_composite_characteristic` (15, 0, 1, -7),
  `gfpn_rejects_non_monic_modulus` (also a leading 5 mod 5), `gfpn_rejects_degenerate_modulus`,
  `gfpn_normalizes_the_modulus` (p = 5, `[6,1,1]`, plus a negative entry and a degree-1 modulus),
  `gfpn_from_int_is_constant_time`, `default_from_int_is_logarithmic` (test newtype
  `DefaultFromInt`).
  `cantor_zassenhaus`: `ddf_rejects_non_squarefree`, `ddf_rejects_zero`,
  `ddf_rejects_composite_characteristic`,
  `squarefree_factorization_rejects_composite_characteristic` (15 and 4),
  `squarefree_factorization_rejects_the_sentinel`, `ddf_squarefree_matches_pari` (also
  non-monic input).
  `irreducible`: `random_irreducible_draws_the_whole_field` (the three pinned lava outputs and
  the range counts), `generators_reject_non_prime_p` (15, 0, 1, 4, -7),
  `is_irreducible_rejects_composite_characteristic`.
  The pinned outputs equal the design prototype (`p3_randirr.log`) bit for bit.

### Item 4: orders (plan steps 3.6, 3.7)

- New `src/group_order.rs` (crate-private, `lib.rs:30`), header ch. 19 §19.7, ch. 21 §21.4.4,
  §21.4.7, ch. 48, ch. 18 §18.10.1, [Coh93] Alg. 1.4.3/1.4.4, [Car10], [BLSTW02].
  `Effort::{Complete, Capped}` (`:36`), the four cap constants (`:51-54`), `Factored`,
  the cache (`:63`, keyed by the factored integer, bounded at 4096, poison recovered, no lock
  while factoring), `factor_proven` (`:74`), `factor_proven_uncached` (`:95`),
  `carmichael_lambda` (`:169`), `field_group_order` (`:233`, cyclotomic split with exact Möbius
  quotients), `product`, `order_from_exponent` (`:260`, checks x^h = 1),
  `unit_order` (`:289`, ±1 shortcuts), `unit_order_mod` (`:313`, Fermat path first, else λ).
- `IntegerMod::multiplicative_order` (`integer_mod.rs:244`) and its hook (`:633`);
  `PrimeField::multiplicative_order` (`prime_field.rs:129`, now `None` for every non-unit of a
  composite modulus), `is_generator` (`:166`), hook (`:572`);
  `FiniteFieldElement::order_factored` (`finite_field.rs:612`), `multiplicative_order` (`:649`),
  new `factored_multiplicative_order` (`:663`, MAGMA `FactoredOrder`), `is_primitive` (`:676`,
  x^((q-1)/r) test, no order computed), hook (`:1142`); `GaloisRingElement`
  `order_by_residue_lifting(effort)` (`galois_ring.rs:1196`) through the residue's
  `order_factored`, new `UnitOrder::Undecided(String)` (`:1125`): the hook maps it to
  `Undecided`, the inherent method (`:1156`) panics with the reason. All `# Cost` / `# Panics`
  docs as designed. `use rustmath_integers::prime::factor` removed from `finite_field.rs`.
  `tests/ring_predicates.rs:237` comment updated.
- Tests: `group_order` unit tests (`order_from_exponent_any_multiple`,
  `order_from_exponent_checks_the_multiple`, `unit_order_shortcuts`,
  `unit_order_mod_fermat_and_lambda_paths` (341, 561, and brute force on every unit mod m < 120
  for both efforts), `carmichael_lambda_matches_pari`, `field_group_order_is_p_to_the_n_minus_one`,
  `factor_proven_capped_and_complete_on_p43_minus_one` (uncached), `factor_proven_small_values`,
  `capped_fermat_path_falls_back_to_lambda`); new `tests/unit_group_orders.rs` (six tests of
  design 4.6) and `tests/unit_group_orders_capped.rs` (one test, phases 1 and 2, header explains
  why it is alone). gp script: `fu2/impl/F1-tools/orders_ref.gp` (all design values re-run and
  matched; quoted in the test headers).
- Timing (debug): `unit_group_orders` 2.3 s for the whole file (the P43 sieve included);
  `unit_group_orders_capped` 0.8 s; `group_order` unit tests 0.6 s. No `#[ignore]` needed.

### Item 5: IsSquare / Sqrt (plan step 3.8)

- `FiniteFieldElement::is_square` (`finite_field.rs:699`), `sqrt` (`:725`, char 2 inverse
  Frobenius, a^((q+1)/4), Tonelli-Shanks after Cohen Alg. 1.5.1; always-on `r^2 = a` check),
  private `non_square` (`:786`, the scan of design 5.2) and `integer_key` (`:829`, canonical
  root). `PrimeField::is_square` (`prime_field.rs:241`, alias) and `sqrt` (`:263`: zero, the
  sentinel, BPSW precheck, `try_sqrt_mod_prime`, `min(r, m - r)`; "for a prime modulus" in the
  doc, review 2 m8). `ExtensionField::is_square` / `sqrt` (`extension_field.rs:73`, `:79`).
- New `tests/field_sqrt.rs` (header ch. 21 §21.4.4, §21.4.7, Cohen Alg. 1.5.1, the gp functions
  `key`, `canon`, `show`, `lr`): the eight tests of design 5.5, with the composite-modulus panic
  as its own should_panic test (`prime_field_sqrt_rejects_composite_modulus`). Values from the
  design's `t6_ref.log` (PARI/GP 2.15.4). 1.8 s in debug.

### Item 6 (plan step 3.10)

Verified, no change: `integer_mod.rs:696` `square_root_mod_prime` runs `assert_prime` (the only
primality test, a proof) and then `Integer::try_sqrt_mod_prime`, which tests nothing; I1's
`tests/prime_residues.rs:301` `square_root_mod_prime_refuses_strong_pseudoprime` exists and
passes (13/13 in that file). The HEAD fallback of design §6 was not needed.

### Item 7 in rustmath-finitefields (plan step 3.9)

- The three hooks in the five types: `IntegerMod` (`integer_mod.rs:585`, `make`), `PrimeField`
  (`prime_field.rs:523`, `make`), `FiniteFieldElement` (`finite_field.rs:1093`, `bind_into`),
  `GaloisRingElement` (`galois_ring.rs:1056`, `bind_into`), `ExtensionField`
  (`extension_field.rs:247`, delegates). Each returns `Some` exactly when self is unbound and
  the model bound.
- Docs (design 7.3): the Galois-ring "containers" bullet (`galois_ring.rs:371`) rewritten; the
  same sentence in the type docs of `FiniteFieldElement` (`finite_field.rs:420`), `PrimeField`
  (`prime_field.rs:35`) and `IntegerMod` (`integer_mod.rs:60`).
- `tests/galois_ring_sentinel.rs`: `polynomials_from_static_constructors_keep_unbound_coefficients`
  is replaced in the same step by `polynomials_from_static_constructors_bind_on_construction`
  (review 1 m3). This is the one test that asserted the old behaviour (degree 7, `q != qb`); the
  reference value proving it wrong for GR(8, 2)[x] is gp
  `lift(Mod(1,8)*((x+1)^8-x^8))` = `4*x^6 + 6*x^4 + 4*x^2 + 1` (degree 6), already in the file
  header as `lift(lift(((x+1)^8 - x^8)*Mod(Mod(1,8),D8) + w8))`. The new test keeps every other
  assertion of the old one (`qb` value, `scalar_mul` binding, the unbound coefficients
  `[1, 8, 28, 56, 70, 56, 28, 8]` of the all-unbound part) and adds `q == qb`, degree 6, a bound
  `coeff(9)`, and the cross-length `==`. New `sparse_polynomials_from_static_constructors_bind`
  (the reviewer's test: 4 terms, degree 6, both orders, bound zero for the absent x_0^1, ^3, ^5,
  ^7, equal to the bound construction; the Laurent value).
- `tests/sentinel_eq.rs`: `containers_bind_unbound_constants` (GF(3) zero, Z/8
  `[2,0,4,0,6,0,4]`, GF(4) `1 + w`, multivariate GF(3) single term `2xy`; gp lines in the doc)
  and `nested_polynomials_bind`.

### Handoff from P1 folded in

P1's handoff to F1: the finite-ring element types now return `static_is_commutative() =
Some(true)` (`IntegerMod`, `PrimeField`, `FiniteFieldElement`, `GaloisRingElement`,
`ExtensionField`; each a commutative ring, and each sentinel an integer). Effect: the
`UnivariatePolynomial` hooks over them use the commutative rules. New test
`tests/ring_predicates.rs::polynomial_hooks_use_commutativity_of_finite_rings` (1 + 2t over Z/8 is
a unit with inverse 1 + 6t + 4t^2, 2 + 4t is nilpotent; gp
`lift(Mod(1,8)*(1+2*t)*(1+6*t+4*t^2))` = 1, `lift(Mod(1,8)*(2+4*t)^3)` = 0). No existing test
changed.

## Review 2 minor notes

All eight folded in:
- m1: `p < 2` rejected before the reduction; p = 0, 1, -7 in `gfpn_rejects_composite_characteristic`.
- m2: `GFPN_VALID` bounded at 4096 pairs.
- m3: implemented (cheap and decides more): when the factorization of `m - 1` fails and `m` is
  not a BPSW probable prime, `unit_order_mod` tries `λ(m)` before giving up (both efforts).
  Test `capped_fermat_path_falls_back_to_lambda` with m = 7 · 952841793122185456008103211101,
  m - 1 = 6 · 1045908801259457 · 1062854387785943 (31-digit cofactor: no sieve, rho fails) and
  the Fermat liar a = 6207317463790804616919148922629 of order 6 (gp script
  `F1-tools/liar_ref.gp`, search `F1-tools/search_liar.gp`).
- m4: moot after I1. `Integer::is_quadratic_residue_prime` (behind `PrimeField::is_square`) now
  tests the modulus with BPSW only, like `PrimeField::sqrt`, so both have the same cost and
  agree; the `is_square` doc says so.
- m5: the hand-off below calls the matrix and sparse-matrix interim behaviour a correctness defect.
- m6: headers of `unbound_constants.rs` (ch. 48 §48.2.3, ch. 23) and
  `unit_group_orders_capped.rs` (ch. 19 §19.7, ch. 21 §21.4.7, Cohen Alg. 1.4.3).
- m7: `cantor_zassenhaus.rs` imports the moved `check_prime_characteristic` in the same step.
- m8: "for a prime modulus" in the `PrimeField::sqrt` doc.

## Deviations from the design (with evidence)

1. `Effort::Capped` proves its probable primes (at most 256 bits) with I1's `try_is_prime`
   instead of `prove_primality` directly. Same proof (BPSW, then ECPP via
   `prove_primality_with(DEFAULT)` above psi_12, and the I2 engine only reports probable primes
   >= psi_12), plus I1's process-wide cache of proven primes; I1's handoff recommends
   `try_is_prime` to certify factors.
2. The λ fallback of review 2 m3 (above) is an addition.
3. Core test without `Integer` (see step 1).
4. The Laurent gp reference uses `subst(..)` (see step 2).
5. `PrimeField::sqrt`'s composite-modulus panic is its own should_panic test, not part of
   `prime_field_sqrt_sentinel_and_composite`.
6. `FiniteFieldElement::is_primitive` returns `false` for an unbound zero without needing the
   field (as HEAD did); for other unbound elements it panics as before.

## API and behaviour changes, callers checked

- New public API: `Ring::{HAS_UNBOUND_CONSTANTS, is_unbound_constant, bind_to_parent_of}`;
  `Gfpn::try_new`; `FiniteFieldElement::{factored_multiplicative_order, is_square, sqrt}`;
  `PrimeField::{is_square, sqrt}`; `ExtensionField::{is_square, sqrt}`. Crate-private:
  `group_order`, `check_char`, `is_irreducible_fp_unchecked`, `find_irreducible_unchecked`,
  `check_prime_characteristic` (moved).
- New panics (each replaces a wrong value or an unrelated panic): lava DDF, squarefree
  factorization and `is_irreducible` with a composite characteristic; canonical
  `is_irreducible_fp` with a non-prime p; both DDFs on non-squarefree input or 0; `Gfpn::new` on
  an invalid modulus or p; `PrimeField::sqrt` with a composite modulus; the inherent orders only
  on an unresolved ECPP proof.
- Generators: `poly_factor::{find,random}_irreducible` return `Err(InvalidArgument)` for a
  non-prime p; the lava ones and `generate_gfpn_modulus` return `None`. The random draws changed
  for most seeds (no test pinned them before); deterministic per seed as before.
- Containers: values that mix bound and unbound coefficients over the five sentinel types now
  follow the bound ring (degree, `is_zero`, term counts, `==`). Values with only unbound
  coefficients are unchanged. Other coefficient types: every new branch is behind a `const`
  test and compiles out.
- Callers checked: `Gfpn::new` in rustmath-galois (`galois_ctx.rs`, `relative_invariant.rs`),
  `generate_gfpn_modulus` (`relative_invariant.rs:472`), every user of univariate,
  multivariate and Laurent polynomials over the sentinel types (design 7.4 list), and every
  reverse dependency in the acceptance list; all green. `matrix_forms_gfp.rs` unchanged and green.

## Acceptance commands (design 9.2) and results

All run on the final tree (logs `F1-logs/acc_<step>.log`, summary `accept_summary.log`), debug:

| Command | Passed | Failed | Ignored |
|---|---|---|---|
| `cargo test -p rustmath-core --no-fail-fast` | 50 | 0 | 1 |
| `cargo test -p rustmath-polynomials --no-fail-fast` | 434 | 0 | 4 |
| `cargo test -p rustmath-finitefields --no-fail-fast` (rerun after the last test was added, `acc_ff2.log`) | 316 | 0 | 0 |
| `cargo test -p rustmath-finitefields --test unit_group_orders_capped -- --test-threads=1` | 1 | 0 | 0 |
| `cargo test -p rustmath-galois --no-fail-fast` | 85 | 0 | 5 |
| `cargo test -p rustmath-matrix --no-fail-fast` | 295 | 0 | 1 |
| `cargo test -p rustmath-rings --no-fail-fast` | 1914 | 0 | 82 |
| `cargo test -p rustmath-coding -p rustmath-crypto -p rustmath-curves -p rustmath-ellipticcurves -p rustmath-algebras -p rustmath-modules -p rustmath-powerseries -p rustmath-schemes --no-fail-fast` | 1999 | 0 | 27 |
| `cargo test -p rustmath-jupyter --no-fail-fast` | 64 | 0 | 0 |
| `cargo check --workspace --tests` | green, 0 errors | | |
| `cargo clippy -p rustmath-finitefields -p rustmath-polynomials -p rustmath-core` | exit 0 | | |

Clippy: with `--no-deps` (and `--all-targets` for polynomials and finitefields) no lint points
at a line this unit added (four `manual_is_multiple_of` lints in `group_order.rs` were fixed).
`cargo clippy -p rustmath-core --all-targets` fails on a pre-existing deny lint
(`approx_constant`, `rustmath-core/src/analytic.rs:708`, untouched). `cargo doc --no-deps` on the
three crates: no warning on an added public doc line after escaping `\[Rab80\]` in
`irreducible.rs`; `target/doc` deleted. No release build was needed (no timing check in the
plan). Total of the table: 5158 passed, 0 failed, 120 ignored.

## Left undone

- The full `cargo test --workspace` (the orchestrator runs it).
- Trackers and docs (`PROJECT_TRACKER.md`, `docs/*`, `docs/magma/*`): not edited, see Handoffs.
- The existing finitefields and polynomials files were not rustfmt-clean at HEAD, so only the
  new files were formatted (`group_order.rs`, `field_sqrt.rs`, `unit_group_orders.rs`,
  `unit_group_orders_capped.rs`, `unbound_constants.rs`).

## Handoffs

- **Later units (M1a, M1b, R1 and rustmath-powerseries)**: the hook call that fixes the generic
  containers outside rustmath-polynomials is the rule of `UnivariatePolynomial::new` (dense:
  take the first entry e with `!e.is_unbound_constant()` as the model and replace every unbound
  entry c by `c.bind_to_parent_of(&e)`, in every constructor and setter, when
  `R::HAS_UNBOUND_CONSTANTS`) or the sparse `add_term` rule of `MultivariatePolynomial`. Until
  then these are a correctness defect with this scope (review 2 m5), not a caveat:
  | Type | Where | Interim behaviour |
  |---|---|---|
  | `Matrix<R>`, `Vector<R>` | rustmath-matrix `matrix.rs`, `vector.rs` | an entry that met only unbound constants stays an integer in Z, so `is_zero`, rank and determinant of a mixed matrix can see those integers (wrong values possible) |
  | `SparseMatrix<F>` | rustmath-matrix `sparse.rs` | an unbound value that is 0 in the field is stored as nonzero (wrong values possible) |
  | `PowerSeries<R>`, `LaurentSeries<R>` | rustmath-powerseries `series.rs`, `laurent.rs` | valuation and trimming see integers in Z (no in-tree caller over a sentinel type) |
  | `PowerSeriesPari<R>`, `PowerSeriesPoly<R>`, `LaurentSeries<R>`, `LaurentSeriesElement<R>` | rustmath-rings | same |
  | `PadicInteger`, `PadicRational`, `PadicExtensionElement` (R1) | rustmath-rings padics | once R1 gives them unbound sentinels they must implement `HAS_UNBOUND_CONSTANTS = true`, `is_unbound_constant` (`!is_bound()`), `bind_to_parent_of` (MAGMA `L ! c` at the model's precision); also `static_is_commutative() = Some(true)` (P1's note) |
- **J1 (REPL)**: the remaining trial-division sites of the crate (`IntegerMod::sqrt`,
  `all_sqrts`, `num_sqrts`, `is_nilpotent`, `Zmod::factored_modulus`,
  `ring_hooks::is_nilpotent_mod`) can switch to `group_order::factor_proven(.., Effort::Complete)`
  (a one-line change each; the function is crate-private).
- **Trackers (orchestrator; design plan step 4)**: `PROJECT_TRACKER.md` rows 805
  (cantor_zassenhaus: drop "Not checked: a Gfpn ...; squarefree ... and DDF check nothing"), 807
  (DDF now checked in both stacks), 809, 812 (generators uniform via `uniform_below`, p checked;
  `is_irreducible_fp` checked), 814 (orders: `group_order`, Cohen Alg. 1.4.3, Fermat multiple
  then λ, cyclotomic split, capped hooks with their cost bound, process cache), 817
  (`PrimeField::is_square`/`sqrt` with the root), 822 (`from_int` O(1) on `Gfpn`, O(log p)
  default), 823; a new row for `Gfpn::try_new` (validation, memo); the rows of
  `UnivariatePolynomial`, `MultivariatePolynomial`, `LaurentPolynomial` (binding of unbound
  constants, coercing `==`); the core three-valued predicates row (new hooks; finite-ring types
  now `static_is_commutative = Some(true)`). `docs/magma/status_overrides.csv`: ch021.IsSquare
  and ch021.SquareRoot (partial to done: `tests/field_sqrt.rs`), ch021.FactoredOrder (missing to
  done: `factored_order_matches_pari`), ch021.Order, ch021.IsPrimitive, ch019.Order (notes:
  `group_order.rs`, tests `unit_group_orders*.rs`), ch021.RandomIrreduciblePolynomial (still
  partial: F_p only, now uniform), ch023.DistinctDegreeFactorization and
  ch023.SquarefreeFactorization (notes: checked); then `python3 tools/magma_tracker/build.py`.
  `docs/HANDOFF.md`: remove the seven F1 items; add the container hand-off above (one line:
  "Generic containers outside rustmath-polynomials (table in the F1 design §7.6 and report): bind
  unbound constants with `Ring::bind_to_parent_of`; a correctness defect for mixed matrices and
  sparse matrices; R1's p-adic sentinels implement the three hooks"), and the findings of design
  §8: (1) `FiniteField::new(p, n)` without a Conway polynomial can take O(p) Rabin tests
  (`FiniteField::new(2^80+13, 3)` does not return), a fix changes default moduli; (2) canonical
  `poly_factor::roots(0)` and `factor(0)` return `[]`; (3) the trial-division sites listed for J1.
  New finding: `GaloisRing::with_modulus` runs the primality and Rabin tests twice (its own, then
  `FiniteField::with_modulus` again); cheap for small p, an ECPP proof is cached by I1.
