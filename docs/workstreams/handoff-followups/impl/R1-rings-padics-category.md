# Implementation report R1: rings, p-adics, category

Branch `claude/gracious-gates-e6b6l7`, on HEAD `4b7f034` (I3, I1, I2, P1, F1 landed). Design
`design/R1-rings-padics-category.md` (Revision 1 body), reviews 1 and 2. Nothing committed.
Work files: `impl/R1-work/` (edit scripts, `cb.sh` filtered cargo, `r1chk.gp` + `.log`,
`r1chk.g` + `.log`: the PARI/GAP references re-run here). Logs: `impl/R1-logs/`.

Uncommitted changes at start: none (clean tree).

## Adapting to the earlier units

* I1: `prime::is_prime_u64`, `prime::try_is_prime` exist, so the 0.2 helpers use them and no
  fallback was written. `cheap_primality(n)` = `try_is_prime` below psi_12 (never `Err` there),
  `None` at once above; `proven_primality(n)` = `try_is_prime(n).ok()`.
* P1: `Ring::static_is_commutative` exists; the p-adic types return `Some(true)`, the series
  types delegate, and the commutative nilpotency rules are on.
* F1: the core hooks `HAS_UNBOUND_CONSTANTS`, `is_unbound_constant`, `bind_to_parent_of` exist.
  The three p-adic types implement them (F1 handoff, review 2 N2); the core hook doc
  (`rustmath-core/src/traits.rs:205-213`) now names the p-adic types too.
* F1's container hand-off for `PowerSeries`/`LaurentSeries` binding is not R1's (the F1 design
  hands it to the orchestrator); not done, listed below.

## Items, by design section

### 0.2 Primality helpers (`rustmath-rings/src/category_methods.rs:36-131`)
`psi_12` :43, `prime_u64` :49, `cheap_primality` :57, `proven_primality` :71, `minimal_root` :79,
`is_prime_power_u64` :86, `PrimePowerStatus` :96 (`PrimePower{q,a}` / `NotPrimePower` /
`Undecided`), `padic_base_prime_power` :109, `padic_base_is_prime_power` :123. Test
`test_primality_helpers` :1476 (gp: `isprime(2^64-59)`=1, `isprime(561)`=0, `ispower(64,,&r)`=6
r=2, `ispower(36,,&r)`=2 r=6, `isprimepower(4)`=2, psi_12 = 399165290221*798330580441, composite).
rustmath-category got private copies `prime_u64`, `is_prime_power_u64`, `proven_primality`
(`ring_category.rs:198-224`).

### A. p-adic sentinels
* `PadicInteger` (`padic_integer.rs`, rewritten): module header cites Ch. 47 §47.7.1, §47.8.1-3,
  Ch. 17 §17.5.2, §17.5.5, Sage; type docs carry the four MAGMA departures of 0.1. `unbound`
  :127, `is_bound` :137, `bind_like` :143, `coerce_pair` :151, accessors panic on unbound
  (`prime`, `precision`, `valuation`, `residue`), `lift`/`truncate`/`hensel_lift_linear`/
  `hensel_lift_root` return `Err(InvalidArgument)`, `inverse` :290 (±1 itself, 0
  `NotInvertible`). Value `PartialEq` :422; `Add`/`Sub`/`Mul` with the exact-zero rules :442-;
  `Ring` :522 (`zero`/`one` unbound, `pow` :543, hooks, `static_characteristic` 0,
  `static_is_commutative`, F1 hooks :641-650).
* `PadicIntegerRing::adjust_precision` binds an unbound element (`factory.rs:196`).
* `PadicRational` (`padic_rational.rs`, rewritten): `lies_in_power` :59 (the integer test of
  review 2 N1), `from_rational(0)` = (0 + O(p^n), 0) :78, `bind_integer` :138, `normalized` :168,
  `representative` :187 (moved from extension.rs), `absolute_precision` :206,
  `from_rational_to_absolute_precision` :216, `bound_inverse` :309, `checked_div` :341, value
  `PartialEq` :387, `Ring` :406 (`is_one` :423 value-based and false when weakly zero, `pow`,
  hooks), `Field` :559 with `inverse` and the `divide` override :582, `Add` :609, `Mul` :674,
  `Div` :701 (panics with the `checked_div` message).
* `newton_lift::padic_to_int` :351: an unbound c gives `c mod m`.
* `PadicExtensionElement` (`extension.rs`): `extension: Option<Arc<..>>` :673, `new` binds
  unbound coefficients :690, `unbound` :708, private `ext()` :721, `bind_into` :740, `norm`/
  `trace` `Err` on unbound, `inverse` :939 (E.6), value `PartialEq` :1048, `Add` :1069, `Mul`
  :1131, `Ring` :1246 (hooks, `pow`, F1 hooks), `PadicEmbedding::apply` binds :1478.
* `PadicExtension` gains four private values (review 2 N7a): `prime_power`, `field_certificate`,
  `reduced`, `field_status` (`OnceLock`) :287-296, computed in `new`; `FieldCertificate` :197,
  `certify_field` :213, `field_certificate` :378, `is_certified_field` :383, `is_reduced` :390,
  `field_status` :408 / `compute_field_status` :412. `FieldCertificate` exported in
  `padics/mod.rs:33`.

Tests (all fail or panic at c79a582 unless noted; references in comments):
`padic_integer.rs`: `ring_zero_one_are_unbound`, `generic_horner_through_sentinels` (gp 162),
`pow_zero_is_bound_one` (gp `Mod(7,5^4)^5` = 557), `equality_common_precision`,
`zero_is_exact_in_sums_and_products`, four `#[should_panic]` accessor tests,
`unbound_results_refuse`, `unbound_constant_hooks`, `hooks_in_z5`.
`padic_rational.rs`: `sum_one_plus_four_equals_five`, `zero_at_valuation_two_equals_ring_zero`,
`ring_one_binds_with_full_precision`, `is_one_unnormalised`, `weakly_zero_is_not_one` (with the
N1 case p = 6, 3·6^-1 + O(6^0): not one, order not `Finite(1)`),
`equality_is_not_fooled_by_add_precision_loss`, `inverse_of_unnormalised_element` (gp
`1/(5+O(5^10))` = `5^-1 + O(5^8)`), `divide_binds_unbound` (gp `lift(Mod(3,5^10)/2)` =
4882814), `unbound_quotient_panics`, `division_by_unbound_zero_panics`,
`from_rational_zero_has_valuation_zero`, three `#[should_panic]`, `unbound_rational_refuses`,
`pow_and_hooks_for_containers`, `hooks_in_q5`.
`factory.rs:743` `adjust_precision_binds_unbound`; `newton_lift.rs` `padic_to_int_of_unbound`
(gp `(-3)%25` = 22).
`extension.rs` :2364-: `ring_constants_bind`, `new_binds_unbound_coefficients`,
`product_of_generators_is_five`, `one_after_dividing_by_five_is_one`,
`unbound_extension_element_refuses`, `unbound_extension_element_has_no_degree`,
`embedding_binds_unbound`, `unbound_constant_hooks`, `field_certificates`,
`field_status_decides`, `hooks_in_extensions`, `inverse_in_certified_fields`,
`inverse_errors_and_low_precision`. Every value and precision of design E.5 is asserted; gp
references (`r1chk.gp.log`): the six exact inverses, PARI's polmod inverse precisions,
`factorpadic` counts [1, 2, 2, 1] for x^2-125/5, x^2-1/5, x^2-2/7, x^2-5/2.
rustmath-padics `basic_padic` extended (`Ring::one() + x`).
rustmath-schemes `src/elliptic_curves/tests.rs`: the six p-adic tests un-ignored, the stale
comment replaced, new `test_padic_doubling_matches_pari` :468 (gp `ellmul`, `elladd`,
`valuation`), all seven pass.

### B. `RingMorphismMethods` (`category_methods.rs:578`, `:590`)
`domain_one() -> Option<Domain>` (default `None`), `is_nonzero() -> Option<bool>` evaluates
`apply` only on it. Tests: `DoubleMap`/`IdZ` gain `domain_one` and assert `Some(true)`; new
`test_is_nonzero_without_domain_one_is_undecided`, `test_is_nonzero_on_padic_domain`,
`test_is_nonzero_to_zero_ring`. Callers of `is_nonzero`: only these tests (grep).

### C. Real parents
* `rustmath-rings/src/category_parents.rs` (new, `pub mod` in `lib.rs`): `IntegerRingParent`,
  `RationalFieldParent`, `PolynomialRingParent<P>`, impls for `Zmod`, `FiniteField`,
  `GaloisRing`, `PadicIntegerRing`, `PadicField`, `PadicExtension`, R[x] (lines 69-410). Tests
  (GAP `r1chk.g.log`, PARI): `integer_ring_parent`, `rational_field_parent`, `zmod_parents`
  (incl. 2^64+13 and 3·2^64), `finite_field_parents`, `galois_ring_parents`, `padic_parents`
  (Z_6, Q_6, Z_4, Q_4; gp idempotent 24151041), `padic_extension_parents`,
  `polynomial_ring_parents`; `category_methods.rs:1440` `test_wrappers_agree_with_real_impls`.
* `rustmath-category/src/ring_parents.rs` (new, private `mod`): `Zmod`, `FiniteField`,
  `GaloisRing`, `MatrixAlgebra<F>`, `BigFloatField`, all with `CommutativeRingParentMethods`.
  Cargo: rustmath-integers moved to `[dependencies]`, plus finitefields, matrix, reals.
* `rustmath-numberfields/src/category.rs` (new, `pub mod category`): `defining_polynomial_is_irreducible`
  :31, `impl Parent for NumberField` :45, the category impls :72-89; test
  `number_field_category` (PARI `polisirreducible`, GAP). Cargo: + rustmath-category.
* `rustmath-algebras/src/algebra_with_parent.rs:72-96`: `DownUpAlgebra` impls; test
  `down_up_algebra_category` (GAP U(sl_2) not commutative; `d*u != u*d`).
* N4: `rustmath-manifolds/src/modules.rs:197-225` `DiffScalarFieldAlgebra` (char 0,
  commutative, dim >= 1 not a domain/field, dim 0 `None`); test
  `scalar_field_algebra_ring_predicates`. Cargo: + rustmath-category (no cycle: the category
  closure is {combinatorics, complex, core, finitefields, graphs, groups, integers, matrix,
  polynomials, powerseries, rationals, reals, typesetting}, `cargo tree`).
* `rustmath-matrix/src/matrix_algebra.rs:319` new `MatrixAlgebra::is_full()` (review 2 N3a).

### D. Composite characteristic and cardinality rules
* rustmath-rings trait: `characteristic` :434 adds the prime-order rule; `is_integral_domain`
  :362 adds the non-prime-power-order rule; `is_prime_field` and `residue_ring_modulus` use
  `prime_u64`. `test_prime_order_alone_decides` now asserts `Some(7)` (GAP
  `Characteristic(ZmodnZ(7))` = 7) and adds `OrderSix`.
* rustmath-category trait (`ring_category.rs:226-341`): `characteristic_rules_out_domain`,
  `residue_ring_modulus`, the new `characteristic`, `is_commutative`, `is_integral_domain`
  defaults and docs. Tests `test_composite_characteristic_is_not_a_domain` :854,
  `test_residue_ring_by_cardinality` :865, `test_non_prime_power_cardinality_is_not_a_domain`
  :880; the four existing tests unchanged and passing.

### E. Hooks
* p-adic: E.2 for the three types (see A). Extension hooks use `field_status`, `is_reduced`,
  the certified valuation data.
* Series (MAGMA Ch. 49 §49.4.4, Ch. 50): `PowerSeries` (`series_core_impls.rs:79-170`, module
  doc with the facts, the M10 precondition and the N6 weak-predicate departure), `LaurentSeries`
  (`laurent.rs:281-375`), `PuiseuxSeries` (`puiseux.rs:265-295`), `LazyPowerSeries`
  (`lazy.rs:288-352`). New `rustmath-powerseries/tests/ring_predicates.rs` (6 tests, gp
  references in the header).
* E.4 test updates (Plan step 3): `test_defaults_on_padic_integer_do_not_panic` renamed
  `test_padic_integer_hooks_decide` (`None` -> `Some(true)` unit, `Some(false)` nilpotent, inverse
  8138021: gp `lift(Mod(6,5^10)^-1)`); `test_parent_carrying_elements_do_not_panic`: Z_p
  characteristic `None` -> `Some(0)`, zero ring `None` -> `Some(false)`, finite `None` ->
  `Some(false)` (Sage `Zp(5).characteristic()` = 0). These replace "undecided" by proven
  answers; no assertion was weakened.

### Comment-only edits outside the unit
`rustmath-core/src/traits.rs:97-100`, `:802`, `:900`; `rustmath-finitefields/tests/ring_predicates.rs:492`;
`rustmath-polynomials/tests/ring_predicates.rs:457`; `rustmath-algebras/src/algebra_with_parent.rs:46`,
`:363`. `rustmath-schemes/.../tests.rs` keeps its CRLF line endings (the five LF-only lines of
HEAD were `#[ignore]` lines, now removed).

## Review 2 minor notes

| Note | Done | How |
|---|---|---|
| N1 composite-p valuation | yes | `lies_in_power` integer test; N1 case in `weakly_zero_is_not_one` |
| N2 F1 hooks | yes | all three p-adic types; core hook doc updated |
| N3a structural "full" | yes | `MatrixAlgebra::is_full` (rustmath-matrix, additive) |
| N3b `Field` trust | yes | documented on the impl; finding in Handoffs |
| N4 `DiffScalarFieldAlgebra` | yes | implemented in R1 |
| N5 prime-power p | partly | done: `field_status` step 5 runs OM at q (test `new(4, x^2-5)` is `Some(true)`, gp one factor), the `PadicInteger` order rule uses q, `PadicRational::unit_status` says `Some(true)` for a prime-power p. Skipped: `field_certificate`, the E.6 inverse and the extension order rule stay for prime p, because `PadicRational::from_rational` cannot build coefficients such as 1/2 in Q_4 (`Err`); stated in the `field_certificate` doc (`extension.rs:374-379`) |
| N6 series departure | yes | `series_core_impls.rs` module doc, Laurent/lazy comments |
| N7a four fields | yes | listed on the struct |
| N7b `try_prime_power` in rustmath-integers | no | optional; out-of-unit crate; the local helpers cover it |

## Deviations from the design (with evidence)

1. `MatrixAlgebra` rows: the "dim = 1" row is claimed only for `full(1)`. A `generated`
   algebra of computed dimension 1 gets `None`, because over an inexact F the computed
   dimension can be wrong (N3). Test: `generated(2, [])` (= Q·I) gives `None`.
2. rustmath-algebras depends on rustmath-category optionally, through feature `full`
   (`algebra_with_parent` and `down_up_algebra` compile only with `full`; rustmath-jupyter enables
   it, so the workspace run sees it). Acceptance used `--features full` in addition.
3. `PadicInteger::coerce_pair` returns `(Self, Self)` (clippy `result_large_err` on the design's
   `Result<(Self, Self), (Self, Self)>`); the unbound-unbound case is handled before the call.
4. `test_wrappers_agree_with_real_impls` checks agreement wherever the wrapper decides: the
   GF(25) wrapper says domain `None`, the real impl `Some(true)`.
5. `PadicExtensionElement::unit_inverse` is `inverse()` exactly; ±1 are not special-cased for a
   non-certified K (an element 1 + O(p^n) need not be invertible when g is not integral).
6. `PadicExtension::generator()` doc notes the degree-1 defect (F.4), not fixed.

## Behaviour and API changes, callers checked

* `Ring::zero()`/`one()` of the three p-adic types no longer panic (unbound). Callers: the
  schemes tests and generic EC code (`rustmath-ellipticcurves/src/generic.rs`), `Ring::pow`.
* `==` is value equality to the common precision (`PadicInteger`, `PadicRational`, extension
  elements). Callers: extension coefficient comparisons, EC `Point ==`, the Galois-group tests
  (still pass), tests. No hashing (grep).
* `from_rational(0)` valuation 0 (callers: factory, newton_lift: never 0, tests).
* `PadicRational` inverse/division through `normalized()`; `Field::divide` override.
* `RingMorphismMethods::is_nonzero` returns `Option<bool>` (callers: tests only).
* New public items: `is_bound` (three types), `PadicRational::absolute_precision`,
  `PadicExtensionElement::inverse`, `FieldCertificate`, `PadicExtension::{field_certificate,
  is_certified_field, is_reduced, field_status}`, `category_parents::*`,
  `NumberField::defining_polynomial_is_irreducible`, `impl Parent for NumberField`,
  `MatrixAlgebra::is_full`.
* Method resolution: no file imports both `RingParentMethods` traits (grep); inherent
  `zero/one/is_field/characteristic/order/is_prime_field` keep winning (`cargo check
  --workspace --tests` green).
* New dependency edges (Cargo.lock updated): category -> {integers, finitefields, matrix,
  reals}; numberfields -> category; algebras -> category (optional, `full`); manifolds ->
  category.

## Acceptance (logs in `impl/R1-logs/`)

| Command | passed | failed | ignored |
|---|---|---|---|
| `cargo check --workspace --tests` | green (re-run after the last edits) | | |
| `cargo test -p rustmath-core` | 50 | 0 | 1 |
| `cargo test -p rustmath-rings` | 1971 | 0 | 82 |
| `cargo test -p rustmath-padics` | 22 | 0 | 0 |
| `cargo test -p rustmath-category` | 189 | 0 | 0 |
| `cargo test -p rustmath-powerseries` | 77 | 0 | 0 |
| `cargo test -p rustmath-numberfields` | 167 | 0 | 1 |
| `cargo test -p rustmath-algebras` | 729 | 0 | 1 |
| `cargo test -p rustmath-algebras --features full` | 869 | 0 | 1 |
| `cargo test -p rustmath-schemes elliptic_curves` | 125 | 0 | 7 |
| `cargo test -p rustmath-ellipticcurves` | 188 | 0 | 0 |
| `cargo test -p rustmath-modules` | 202 | 0 | 0 |
| `cargo test -p rustmath-graphs` | 875 | 0 | 2 |
| `cargo test -p rustmath-curves -p rustmath-affineschemes -p rustmath-quantumgroups` | 409 | 0 | 16 |
| `cargo test -p rustmath-finitefields` | 316 | 0 | 0 |
| `cargo test -p rustmath-matrix` (added: `is_full`) | 295 | 0 | 1 |
| `cargo test -p rustmath-manifolds` (added: N4) | 290 | 0 | 11 |
| `cargo test -p rustmath-polynomials --test ring_predicates` (comment edit) | 10 | 0 | 0 |

After the clippy fixes (`coerce_pair`, `rational_solve`, `?Sized`) I re-ran
`cargo test -p rustmath-rings --lib` (1783 passed), `-p rustmath-category` (189),
`-p rustmath-schemes padic` (7) and `cargo check --workspace --tests`: green.

Clippy: the command as written exits 101 on pre-existing deny lints in rustmath-combinatorics
(as P1/F1 reported). With `--no-deps --all-targets` on rings, category, powerseries,
numberfields, no lint points at a line this unit added (fixed: `result_large_err`,
`needless_range_loop`, `needless_maybe_sized`); the remaining ones in touched files are on HEAD
lines, and `--all-targets` still stops on pre-existing deny lints (rings `divisor.rs`,
`function_field/*`, `real_mpfi.rs`; manifolds `catalog.rs:625`; core `analytic.rs:708`).
rustfmt: the four new files are fmt-clean; edited existing files were not reformatted. No
release build was needed. `target/` holds only `debug` (no release dir). No
`rustmath-misc/output.txt` or `.tmp_dir_*` changes.

## Left undone

* The full `cargo test --workspace` (orchestrator).
* Trackers and docs (design Plan step 13): not edited, see Handoffs.
* N5 for the certificate / E.6 inverse / extension order rule at prime-power p (reason above).
* The extension inverse in a field certified only by OM (F.3): `Err(NotImplemented)`.

## Handoffs

* **Trackers (orchestrator; design Plan step 13).** PROJECT_TRACKER rows :361 (series and
  p-adic hooks decided), :819 (the p-adic types return unbound sentinels), :846 (shim), :1046
  (category: composite and cardinality rules, real impls), :1108 (🔴 -> 🟢: value equality and
  `is_one` for `PadicRational` and extension elements; the lossy `Add` stays a note), :1148
  (the extension inverse exists for certified fields; `Mul`'s reduction note stays), :1165
  (`is_nonzero` fixed via `domain_one`, real impls); new rows for `category_parents.rs`,
  `rustmath-category/src/ring_parents.rs`, `rustmath-numberfields/src/category.rs`, the
  `DownUpAlgebra` and `DiffScalarFieldAlgebra` impls, `MatrixAlgebra::is_full`, the series hooks
  and `FieldCertificate`/`field_status`. `docs/magma/status_overrides.csv`:
  ch017.Characteristic, ch017.IsCommutative, ch017.IsField, ch034.IsField,
  ch047.Characteristic, ch047.IsUnit, ch049.IsUnit, ch049.IsNilpotent; then
  `python3 tools/magma_tracker/build.py`. `docs/HANDOFF.md:318-320` and the "series and
  p-adics" words at :331-332.
* **New HANDOFF follow-ups** (design F.2, F.3, F.4 and new findings):
  1. `SeriesRing` carries no base ring, so no series parent has ring predicates.
  2. `SimpleAlgebra` has no product: not a ring, no impl.
  3. `nf_elem::NumberField` (no `PartialEq` on `NfElem`): copy `category.rs`.
  4. Extension inverse when only OM certifies the field (needs an integral basis), and the
     certificate/inverse for a proper prime-power p (needs a `PadicRational` constructor for
     denominators sharing a factor with p).
  5. `PadicRational::add` drops known digits (sound but lossy).
  6. `PadicInteger::lift` raises the precision without new digits.
  7. `PadicExtension::generator()` returns 0 for degree 1.
  8. `rustmath-ellipticcurves/src/generic.rs:79-82` stale doc on `PrimeField` zero/one.
  9. `QuotientRingGeneric::is_integral_domain` always false (`quotient_ring.rs:203`).
  10. `PadicExtension::new` accepts any f under any label; `ramification_index`/`residue_degree`
      come from the label (OM's `ramification_data` could check it).
  11. New: `impl Field for PadicRational` holds for every p > 1, but Q_p is a field only for a
      prime-power p (Q_6 ≅ Q_2 × Q_3); `MatrixAlgebra::<PadicRational>::full(1)` trusts it.
  12. New: `PadicRational::from_rational` rejects 1/2 in Q_4 (denominator not coprime to p).
  13. New: the `algebra_with_parent`/`down_up_algebra` tests of rustmath-algebras run only with
      feature `full`; a plain `cargo test -p rustmath-algebras` skips them.
  14. F1's container hand-off still open for `PowerSeries`/`LaurentSeries` (rustmath-powerseries)
      and the rustmath-rings series types: they do not bind unbound constants.
* **MF1** (rustmath-manifolds, next units): `modules.rs` now has the `DiffScalarFieldAlgebra`
  ring-predicate impl (:197-225) and a rustmath-category dependency; rebase on it.
* **M1a/M1b** (rustmath-matrix): new `MatrixAlgebra::is_full` (:319).

## Fix-up round 1

Review: `impl/R1-rings-padics-category.review1.md` (1 blocker, 4 minor issues). I fixed all
five and skipped none. Every change is in rustmath-rings. Work files are in `impl/R1-work/fix1/`:
the reviewer's probes `p1.rs`, `p4.rs`, `p5.rs` and `p6.rs`, re-run here, and `../fix1_char.g`.
Logs are in `impl/R1-logs/fix1/`. Nothing is committed. The tree has the same 28 modified and
4 new files as before.

### B1 (blocker): a constant defining polynomial
* `PadicExtension::new` (`extension.rs:329-336`) now refuses `deg f = 0` with the existing
  message, `InvalidArgument("Polynomial must have positive degree")`. The check is
  `.degree().filter(|&d| d >= 1)`.
* Q_p[x]/(c) for a nonzero constant c is the zero ring, so the parent rows and
  `static_characteristic` would be wrong for it.
* `unramified(p, 0, n)` and `eisenstein` already refuse a constant. `new` builds every
  extension (grep finds no other `PadicExtension {` literal).
* The doc of the parent impl (`category_parents.rs:328-333`) now says why its rows hold: deg
  f >= 1, so Q_p embeds in K.
* Test `test_constant_polynomial_rejected` (`extension.rs:2362`): f = 3, f = 5 and f = 0 over
  Q_5 each give `Err(InvalidArgument)`.
* The reviewer's probe `p5.rs` now stops at its `unwrap` with that `Err`, and `p1.rs` prints
  "deg0 rejected".

### m1: the cost of `PadicRational` equality grew with the valuation
I fixed this as the review proposed, rather than adding a HANDOFF note.
* `divides_shifted_difference(p, e, (a, s), (b, t))` (`padic_rational.rs:66`) tests whether
  `p^e | a p^s - b p^t` on integers. It drops a term whose shift is at least e. It replaces
  `lies_in_power`, which built `p^(m+k)` and a rational difference.
* **Two bound operands** (`PartialEq`, `:433`). Let `v0 = min(v_x, v_y)` and let m be the
  smaller absolute precision. The test is `p^(m - v0) | u_x p^(v_x - v0) - u_y p^(v_y - v0)`.
  `m - v0` is at most the relative precision of the operand of valuation v0. So every power
  formed is at most `p^n`, with n the larger relative precision. This meets the bound of
  design A.6, O(M(n log p)).
* **A bound operand and an integer c** (`value_equals_integer`, `:323`). This case is used by
  `==` with an unbound operand and by `is_one`.
  * For `v <= 0` the test is `p^n | u - c p^-v`. The c term is dropped when `-v >= n`.
  * For `v > 0`, `p^v | c` is necessary. It is refused at once, from bit lengths, when
    `p^v > |c|`. Otherwise the test is `p^n | u - c / p^v`.
  * The powers formed are at most `p^n`, or about `c^2`.
* **Correctness.** The membership test is the same as before: an integer D lies in `p^e Z_p`
  iff `p^e | D` in Z, and multiplying by `p^w` maps `p^j Z_p` onto `p^(j+w) Z_p`. So it also
  holds for a composite p.
  * New test `equality_matches_the_definition` (`:919`) draws 2400 random pairs for p = 5, 6,
    12 and 4, on unnormalised storage. y is either random, x plus a multiple of p^k at another
    precision, or the value of x stored as `(u p^j, v - j)`.
  * It checks `x == y`, `y == x`, `x == c` and `c == x` for an unbound c, and `is_one`. The
    reference is the definition "`v_q(d) >= a_q m` for every `q^a_q || p`". The test requires
    at least 300 equal pairs.
  * A mutation of the code (`m - v0 - 1`) makes the test fail.
  * The reviewer's probe `p6.rs` (9000 cases) finds 0 mismatches.
* **Cost.** Test `equality_cost_does_not_grow_with_the_valuation` (`:987`) uses valuations
  `±10^9`. It compares against bound and unbound elements, uses unnormalised storage, and
  calls `is_one` and the order hook.
  * Probe `p1.rs`, debug build, x = 5^v: `x == 5` and `x.is_one()` take 2.8 µs at v = 10^3,
    0.7 µs at 10^5 and 1.3 µs at 10^7. The review measured 0.09 ms, 53 ms and 48 s.
  * Probe `p4.rs` at v = 10^5: `x == x` takes 3.3 µs and `unit_status` 8 µs.
* **Not changed.** `Add` is still linear in the valuation difference. `p4.rs` measures 1 ms,
  10 ms and 104 ms at v = 10^3, 10^4 and 10^5, as at HEAD. This is HEAD's code. HANDOFF item 5
  ("`PadicRational::add` drops known digits") should also record this cost.

### m2: the convention also covers `is_one`
* The type docs now say that the predicates are proofs "under the convention that `is_zero`
  and `is_one` are exact". This is in `PadicInteger` (`padic_integer.rs:31-35`),
  `PadicRational` (`padic_rational.rs:23-26`) and `PadicExtensionElement`
  (`extension.rs:669-674`).
  * Example in the docs: `1 + O(5^10)` has order `Finite(1)`, although the lift `1 + 5^10`
    has infinite order.
  * The extension doc names the three rules for ±1: unit, inverse and order.
* `PadicExtensionElement::unit_inverse` (`extension.rs:1343`) now returns the element itself
  for ±1 when `inverse()` is undecided. That happens when there is no certificate, or when the
  precision is too low. The core default does the same.
  * Where `inverse()` decides (a certified K), its precision-tracked result is kept.
  * So `unit_status` (`Some(true)`) and `unit_inverse` now agree for ±1 in every K.
* I withdraw deviation 5 of this report. Its argument ("1 + O(p^n) need not be invertible")
  contradicts the convention that `unit_status` already used.
* Test: `hooks_in_extensions` (`:2606`) checks ±1 in four rings. Three have no certificate:
  x^2 - 1 over Q_5, (x - 1)^2 over Q_5, and x^2 - 5 over Q_6. The fourth, Q_5(√5), is
  certified. In each, `unit_status` is `Some(true)` and `unit_inverse() == ±1`. The existing
  `NotImplemented` assertions for 1 + g and for the OM generator still hold.

### m3: the wrapper test compared only "characteristic == 1"
* `test_wrappers_agree_with_real_impls` (`category_methods.rs:1466`) now asserts the
  characteristic itself for all four parents, for the wrapper and for the real impl: GR(8, 2)
  is 8, Z/6 is 6, Z/7 is 7 and GF(25) is 5.
* Reference values:
  * GAP: `[Characteristic(ZmodnZ(6)), Characteristic(ZmodnZ(7)), Characteristic(GF(25))]`
    gives `[ 6, 7, 5 ]` (`R1-work/fix1_char.g`).
  * GR(p^a, d) has characteristic p^a (MAGMA Ch. 48).
* The file keeps its CRLF line endings.

### m4: the `field_status` cache was per clone
I fixed this rather than documenting it.
* The cache is now `Arc<OnceLock<Option<bool>>>` (`extension.rs:296-299`, `:373`). Every
  clone shares it, including the clone inside each element made by `generator()`.
* Sharing is sound because the value depends only on p and f, and no method changes them. The
  only `&mut self` method is `set_generator_name`.
* I did not fill the cache before cloning in `generator()`. That would run the OM
  factorisation on every `generator()` call, even when nobody asks for the field status.
* The `field_status` doc says the cache is shared.
* Test `field_status_cache_is_shared_by_clones` (`:2564`; gp `#factorpadic(x^2-125,5,10)[,1]`
  gives 1): `ext.generator().unit_status()` fills the cache of `ext`, and a clone points to
  the same cache.

### Hygiene
* The new test code is rustfmt-formatted. Edited existing files were not reformatted.
  `category_parents.rs` passes `rustfmt --check`.
* Added lines contain no em-dash and no `unsafe`.
* No release build. The probe binaries are deleted. There are no `rustmath-misc/output.txt`
  or `.tmp_dir_*` changes.

### Acceptance (fix-up round 1, logs in `impl/R1-logs/fix1/`, `summary.txt`)

| Command | passed | failed | ignored |
|---|---|---|---|
| `cargo check --workspace --tests` | green | | |
| `cargo test -p rustmath-core` | 50 | 0 | 1 |
| `cargo test -p rustmath-rings` | 1975 (+4 new) | 0 | 82 |
| `cargo test -p rustmath-padics` | 22 | 0 | 0 |
| `cargo test -p rustmath-category` | 189 | 0 | 0 |
| `cargo test -p rustmath-powerseries` | 77 | 0 | 0 |
| `cargo test -p rustmath-numberfields` | 167 | 0 | 1 |
| `cargo test -p rustmath-algebras` | 729 | 0 | 1 |
| `cargo test -p rustmath-algebras --features full` | 869 | 0 | 1 |
| `cargo test -p rustmath-schemes elliptic_curves` | 125 | 0 | 7 |
| `cargo test -p rustmath-ellipticcurves` | 188 | 0 | 0 |
| `cargo test -p rustmath-modules` | 202 | 0 | 0 |
| `cargo test -p rustmath-graphs` | 875 | 0 | 2 |
| `cargo test -p rustmath-curves -p rustmath-affineschemes -p rustmath-quantumgroups` | 409 | 0 | 16 |
| `cargo test -p rustmath-finitefields` | 316 | 0 | 0 |
| `cargo test -p rustmath-matrix` | 295 | 0 | 1 |
| `cargo test -p rustmath-manifolds` | 290 | 0 | 11 |
| `cargo test -p rustmath-polynomials --test ring_predicates` | 10 | 0 | 0 |

Clippy: the design's command exits 101, as before, on deny lints in rustmath-combinatorics
that predate this unit. `cargo clippy --no-deps -p rustmath-rings --all-targets` stops on deny
lints that predate this unit (`divisor.rs`, `function_field/*`, `real_interval_absolute.rs`,
`real_mpfi.rs`). `cargo clippy --no-deps -p rustmath-rings` exits 0. In the touched p-adic
files it warns only on HEAD lines (`extension.rs` 145, 531 and 1454; they are 156, 341 and 859
at HEAD). No warning points at a line from this round.

### Handoff changes
* HANDOFF item 5 (`PadicRational::add` drops known digits) should add that `Add` takes time
  linear in the valuation difference (measured above). `==` and `is_one` no longer do.
* The tracker row for `PadicRational` value equality (:1108) can cite O(M(n log p)) for two
  bound operands, as design A.6 says. Against an unbound integer c, the cost also depends on
  the size of c.
* Deviation 5 of this report is withdrawn (see m2).
