# Review 1 of design R1 (rings, p-adics, category)

Design: `R1-rings-padics-category.md` (903 lines, no Revision section). HEAD `c79a582`.
Scratch: `R1-rings-padics-category/review1/` (`rc.sh` builds against the designer's HEAD rlib copies,
read only; `v1.rs`, `v2.rs`, `e2_copy.rs`, `e3_copy.rs`, `r1.gp`, `r2.g`, `r3.gp`).

**Verdict: not approved. 2 blockers, 14 minor issues.** Items A, B, D and most of C and E are well
designed and well evidenced. Both blockers are out-of-scope claims that the rules do not allow: one the
design itself calls feasible, one whose justification is wrong.

## What I reproduced

* All five defects at HEAD. `v1.rs`: `<PadicInteger as Ring>::zero()`, `<PadicRational as Ring>::one()`,
  `<PadicExtensionElement as Ring>::one()` panic; `from_rational(0)` is `(u=0, v=-1, n=10)`
  (`Integer::zero().valuation(5) = 4294967295`, cast to -1). `e2_copy.rs`, `e3_copy.rs` (designer's
  programs, rebuilt): `ReduceZ5.is_nonzero()` panics; category char 4 gives `None` for domain, field,
  PID, UFD, Euclidean; Z/6 domain `None`; cardinality 7 gives `None` throughout; `bare::<PadicInteger>`
  char/finite `None`; `O(x^0)` unit `Some(false)` nil `Some(true)`; lazy `1+x^25` order `Finite(1)` and
  `unit_inverse` returns itself; lazy `x^30` nilpotent `Some(true)`.
* Value equality does not break the Galois-group tests (`v1.rs` computes it on HEAD values): degree 2:
  `sg != g`, `ssg == g`; degree 3: `sg`, `ssg`, `g` pairwise unequal. The `assert_eq!` sites can only
  get easier to satisfy.
* The lossy `Add` case of A.3.2: `0 + O(5^10)` minus `5^5 + O(5^8)` is `(0, 0, n=3)` at HEAD, while
  value equality says they differ. Confirmed.
* gp (`r1.gp`, PARI 2.15): 268, 8138021, Teichmüller 6139557 with `^4 == 1`, `Mod(2,5^10)^4 = 16`,
  `1/(5+O(5^10)) = 5^-1 + O(5^8)`, the two `== 1` checks, `ellmul` = `[202904/15625, 91751917/1953125]`
  with valuations -6, -9, `elladd` = `[4,-9]`, all six series inverses of E.5 (including
  `-1 - x^2 - x^4`, `1 + 2x` over Z/4, `x^4 + O(x^8)`, `Mod(0,4)`, the two Laurent inverses), lazy
  coefficients -1 and 1, `factorpadic(x^2-1,5,5)` two factors, `isprime(2^64+13) = 1`,
  `isprime(2^64-59) = 1`, psi_12 = 318665857834031151167461 composite and not a pseudoprime, 561
  composite, Horner 162, `Mod(26,5^5) == Mod(1,5^5)` = 0. All match the design.
* GAP (`ref3.g` rerun, and `r2.g`): every value of C.5 matches; `IsFinite(PolynomialRing(ZmodnZ(2^64+13),1))`
  = false (used in M2 below).
* Algorithms re-derived: Z_p unit iff p does not divide u; mu(Q_p) = mu_(p-1) for odd p and {±1} for
  p = 2, so `u^(p-1) != 1 mod p^n` proves infinite order; normalised inverse precision n - j; the
  irreducible-reduction certificate (Gauss's lemma, unit leading coefficient keeps the degree); the
  power-series right-inverse recurrence (a right inverse of a unit is its inverse); A-M Ex. 1.5(ii) for
  the commutative nilpotency rule; Laurent unit rule; R[x] rules (Hilbert, A-M Ex. 5.9, dim + 1); the
  composite-characteristic rule. All correct.
* Callers: my greps agree with A.2, B.3, C.4, D.3, E.4, with the additions in M3 and M13.
* Citations: Ch. 17 §17.4.2-3, §17.5.2 (`R ! a`), §17.5.5; Ch. 47 §47.7.1; Ch. 49 §49.4.4
  (`IsWeaklyZero`, `IsWeaklyEqual` "not transitive"); Ch. 50 `IsUnit` all exist in the digests.
* Cost: `is_prime_miller_rabin(M521, 1)` 29 ms, `(M521, 12)` 396 ms in debug (`v2.rs`), see M7.

## Blockers

### X1. E.6: `PadicExtensionElement::unit_inverse` is declared out of scope although the design calls it feasible

E.6 says the honest inverse "is feasible (about 150 lines with tests)" and gives the algorithm, then
hands it off because "it is a new `Field` implementation, not a hook". The rules put a feasible
algorithm in scope, and `unit_inverse` is one of the hooks of the item ("implement what can be
decided"). The objection that it touches `Mul`'s precision semantics does not hold: the algorithm uses
the exact rational representatives and `multiplication_matrix_rational`, not `Mul`.

Required change: implement `unit_inverse` (and an inherent `inverse`) for a nonzero element of a
certified field. Solve `M_x y = e_1` exactly over Q (Gaussian elimination, as `rational_det` already
does). Let v_K be the valuation of K normalised by v_K(p) = e. For the two certified shapes,
v_K(sum c_i g^i) = min(e·v_p(c_i) + i) (Eisenstein, g = π) or min v_p(c_i) (irreducible reduction with a
unit leading coefficient). Bound v_K(ε) from below by the same formula applied to the coefficients'
absolute precisions. If v_K(ε) <= v_K(x_rep), return an undecided `Err` (x is not known well enough).
Otherwise v_K(y - y_rep) >= v_K(ε) - 2·v_K(x_rep) (the design's own derivation, which I re-derived), and
each coefficient of y gets absolute precision ceil((B - i)/e) (Eisenstein) or B (unramified). Tests:
in Q_5(√5), `(1 + g)^-1 = -1/4 + (1/4)g` (gp `lift(Mod(1,x^2-5)/(1+x))` = `1/4*x - 1/4`) with the
computed precisions, `g^-1 = g/5`, and `x * x^-1 == 1`; one unramified case (gp `lift(Mod(1,f)/x)` for
the `unramified(5,2)` modulus). Non-certified extensions keep `Err(NotImplemented)`.

### X2. C / F.1: the number-field parent is declared out of scope on a wrong argument

The computed task lists number fields among the parents to cover. F.1 argues that
`NumberFieldElement` is not a `Ring` (true, so the rustmath-rings trait is closed) and that "the orphan
rule forbids implementing core's `Parent` for a rustmath-numberfields type in rustmath-rings or
rustmath-category, so the category trait is closed too". The second half does not follow:
`rustmath_category::RingParentMethods` needs only core `Parent`, whose `Element` must be
`Clone + PartialEq`; `NumberFieldElement` derives `Clone` and implements `PartialEq`
(`rustmath-numberfields/src/lib.rs:83`, `:156`). The owning crate may implement `Parent` for
`NumberField`, and rustmath-category may then implement its own trait for it (new dependency on
rustmath-numberfields; no cycle, only rustmath-modules depends on rustmath-category). The decision
procedure exists: `rustmath_polynomials::factorization::is_irreducible_over_integers`
(`factorization.rs:467`, `Err` on recombination failure).

Required change: implement it. `impl Parent for NumberField` in rustmath-numberfields (`contains`: the
element's minimal polynomial equals the field's; `is_finite` `Some(false)`), and
`impl rustmath_category::RingParentMethods for NumberField`: characteristic `Some(0)`, commutative
`Some(true)`, `is_integral_domain` = `is_field` = irreducibility of the primitive part of the minimal
polynomial (`Ok(b)` gives `Some(b)`, `Err` gives `None`; `NumberField::new` does not check it,
`lib.rs:222`), plus `CommutativeRingParentMethods` with the defaults. Tests with PARI
`polisirreducible(x^2-2)` = 1 (field, PID) and `polisirreducible(x^2-1)` = 0 (Q[x]/(x^2-1) ≅ Q × Q, not a
domain). If the orchestrator rules that rustmath-numberfields is outside this unit, F.1 must instead
state that obstacle (a crate boundary, not the orphan rule) and the actual hand-off (a core `Parent`
impl, not a `Ring` impl).

## Minor issues

* **M1. `Field::divide` with an unbound divisor, and unbound / unbound.** `Field::divide` is the core
  default `self * other.inverse()?`, so `x.divide(&two)` (two = `R::one() + R::one()`) returns
  `Err(InvalidArgument)` although `x / two` binds. Override `divide` for `PadicRational` (bind the
  unbound operand into the bound one first). Specify `/` and `divide` with two unbound operands: the
  exact quotient when d divides c, otherwise a panic or `Err` that says "bind first"; never "Division by
  zero" for d != 0 (the prototype's `Div` prints that message for unbound 2).
* **M2. `is_zero_ring` is not overridden.** For Z/m, GF(q), GR with m, q >= 2^64, `characteristic` and
  `order` are `None`, so the default `is_zero_ring` is `None`, and `PolynomialRingParent` over such a base
  gives `is_finite` `None` and Krull `None`. Override `is_zero_ring = Some(false)` for every listed
  parent (all constructors reject the zero ring) and add Z/(2^64+13)[x]: finite `Some(false)` (GAP
  `IsFinite(PolynomialRing(ZmodnZ(2^64+13),1))` = false), Krull 1.
* **M3. Callers that receive an unbound element but know the parent.** `PadicIntegerRing::adjust_precision`
  (`factory.rs:193`) calls `valuation()`/`precision()` and would panic; it should bind into the ring.
  `PadicExtensionElement::new` and `from_base_field` (`extension.rs:489`, `:528`) should bind unbound
  coefficients with `bind_integer`. `newton_lift::padic_to_int` (`newton_lift.rs:349`, `Result`-valued)
  calls `valuation()` and would panic on a user seed built from `PadicRational::one()`. Add them to A.4.
* **M4. Plan order.** Step 2 updates `test_defaults_on_padic_integer_do_not_panic` and
  `test_parent_carrying_elements_do_not_panic` to the decided values, but the hooks and
  `static_characteristic` arrive in steps 3 to 5, so the crate is red between steps. Move those two test
  updates after step 5.
* **M5. MAGMA semantics not cited.** Ch. 47 §47.8.2 disables `eq` for free precision, §47.8.3 says
  `IsZero`/`IsOne` always return false there and `IsUnit(x)` is "valuation 0". The design adopts Sage's
  weak semantics and the Ch. 17 unit notion (every nonzero element of Q_p is a unit) without saying that
  it departs from these lines. State the departure and the reason (the `Ring` trait needs `PartialEq` and
  `is_zero`; HEAD already tests known digits) in 0.1 and in the type docs. Likewise note that
  `PrecisionModel::FixedModulus` is read as Sage's fixed-mod Z_p (characteristic 0), not MAGMA's
  `pAdicQuotientRing` (characteristic p^k).
* **M6. Elements that are both zero and one.** Under the value `is_one`, an element of absolute precision
  <= 0 is both: `from_rational(1/5, 5, 1) - from_rational(1/5, 5, 1)` is `(0, -1, n=1)`, `is_zero` true and
  `v_5(0 - 1) = 0 >= 0` so `is_one` true. The `PadicExtensionElement` order hook keeps the core default,
  which tests `is_one` first, so it would say `Finite(1)` while the new `unit_status` says `Some(false)`.
  Make `is_one` false when `is_zero`, or override every hook to test zero first.
* **M7. `cheap_primality` cost.** `is_prime_miller_rabin(n, 12)` runs the 12 bases 12 times (396 ms vs
  29 ms for one round on M521, debug). Every hook only tests `== Some(true)`, so above psi_12 the run is
  wasted on each order or nilpotency query. Return `None` at once above psi_12 in the hook helper, and
  use one round (or I1's `is_probable_prime`) where `Some(false)` matters.
* **M8. GF(q) and GR rows depend on I1.** `FiniteField::new` and `GaloisRing::new` test p with
  `Integer::is_prime`, a probable-prime test above psi_12 at HEAD. With I1 it is a proof. Say so in the
  table, or use `proven_primality(p)` for those rows in the fallback.
* **M9. Decidable cases left undecided.** For any p > 1, Z_p = lim Z/p^n is a finite product of complete
  DVRs: Noetherian, Krull dimension 1, reduced, so a nonzero `PadicInteger` or `PadicRational` is never
  nilpotent (the design restricts these to prime p). Unbound -1 has order 2 in every Z_p (the design
  reuses the finitefields sentinel table, which says `Undecided`). In a non-certified extension a nonzero
  element is not nilpotent when f is squarefree over Q (gcd(f, f') = 1). A degree-1 extension is a
  field. In both category traits, a known cardinality that is not a prime power proves "not a domain"
  (a finite domain is a field), and a prime cardinality p gives `characteristic() = Some(p)` (the
  defaults prove Z/p but still report characteristic `None`). `PolynomialRingParent::krull_dimension`
  should also require `base.is_commutative() == Some(true)` (the dim + 1 theorem is commutative).
* **M10. Weak-zero coefficients.** The series rule "lowest nonzero coefficient non-nilpotent implies not
  nilpotent" (used without commutativity) and the Laurent unit rule read coefficient `is_zero()`. With a
  weak coefficient type they hold only under the "is_zero is exact" convention: over Z_5,
  `O(5^3)·x^-1 + 1` is not a unit of Z_5((x)) if the weak zero is 5^3. Document the precondition in E.3.
* **M11. Other ring parents with a core `Parent` impl are not mentioned:** `MatrixAlgebra<F>`
  (`rustmath-matrix/src/matrix_algebra.rs:351`), `BigFloatField` (`rustmath-reals/src/bigfloat.rs:1174`),
  `SimpleAlgebra`, `DownUpAlgebra` (`rustmath-algebras`). Implement the category trait (for M_n(F):
  commutative iff n <= 1, domain iff n = 1 and F a domain) or list them as hand-offs.
* **M12. `0 * x` loses precision.** Binding the exact 0 gives a zero of absolute precision v_x + n (for
  example `a1·x·y` in `is_on_curve` lowers the precision of the left side for integral points). Unbound 0
  times anything can stay the exact unbound 0, as `x + 0` already keeps x.
* **M13. Acceptance list.** rustmath-graphs depends on rustmath-powerseries; add
  `cargo test -p rustmath-graphs` (no hook use found, so low risk).
* **M14. Small spec gaps.** `test_is_nonzero_on_padic_domain` writes `PadicInteger::one(5, 10)?` inside a
  `fn domain_one(&self) -> Option<_>` (use `.ok()`). The non-transitive `PartialEq` breaks the formal
  Rust contract; the precedent (`PowerSeries`, `GaloisRingElement`) should be cited in the type docs.

## Checks that passed

No facade or silent truncation is introduced: every new `Some`/`Finite`/`Infinite`/`Ok` is a theorem
under the stated convention, bounded searches (lazy series) never decide on failure, and unknowns stay
`None`, `Undecided` or a non-`NotInvertible` `Err`. The sentinel design matches `029f7ab`, and value
equality is needed for it to compare correctly (A.3.4 is right). The tests fail on HEAD (panics,
`None`, or missing impls) and use PARI, GAP or Sage values that I reproduced. The plan keeps each crate
green apart from M4, and the dependency additions (rustmath-category on rustmath-finitefields and
rustmath-integers) create no cycle.
