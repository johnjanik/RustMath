# Review 2 of design R1 (rings, p-adics, category)

Design: `R1-rings-padics-category.md` (1463 lines, with "Revision 1"). HEAD `c79a582`.
Scratch: `R1-rings-padics-category/review2/` (`rc.sh` builds against the designer's HEAD rlib copies in
`../lib`, read only; `composite_val.rs`, `r2.gp`, `r3.g`, `r4.gp`, copies of the designer's programs,
`meta.json` from `cargo metadata`).

**Verdict: approved. No blockers, 7 minor issues.** Both blockers of review 1 are fixed: the extension
inverse (E.6) is implemented with a sound precision bound, and the number field parent (C.4) is in scope
with the correct crate-boundary argument. All 14 minor issues of review 1 are answered. The minor issues
below should be folded into the implementation; none changes the structure of the plan.

## What I reproduced

* **Defects at HEAD.** I rebuilt `e1_panics.rs`, `e2_morph_cat.rs`, `e3_series.rs`: the three p-adic
  `zero()`/`one()` and `x.pow(0)` panic; `(5,0) != from_rational(5)`; `Field::inverse(5,0)` is
  `Err(NotInvertible)`; `from_rational(0)` has valuation -1; `ReduceZ5.is_nonzero()` panics; the
  category trait gives `None` for characteristic 4, Z/6 and cardinality 7; `bare::<PadicInteger>` char
  `None`; `O(x^0)` unit `Some(false)` nil `Some(true)`; lazy `1+x^25` order `Finite(1)`, lazy `x^30`
  nilpotent `Some(true)`. All as the design states.
* **Prototypes.** `proto_ext_inverse.rs`: every value and precision of E.5 (V, E, B, P_i), and
  `x * x^-1` equals 1 in value. `om_cost.rs`: the ten OM factorisations, 0.2 to 49 ms (design: 0.2 to
  46 ms). `sqfree.rs`: the squarefree answers. `cost_prim.rs`: `is_prime` 7.6 ms and 14.9 ms,
  `is_prime_miller_rabin(n, 1)` 0.37 ms and 0.75 ms, `is_perfect_power` 21 µs, minimal roots 36 -> (6, 2),
  64 -> (2, 6), 6 -> None (design: 8.5, 17.7, 0.42, 0.97 ms; same order).
* **PARI/GP 2.15** (`r2.gp`, `r4.gp`), all matching the design: A.5 (`1`, `5^-1 + O(5^8)`, 162, 0, 1);
  E.5 (268, 8138021, 6139557 with `^4 == 1`, `Mod(16, 5^10)`, 1037, `Mod(256, 1296)`); the six exact
  inverses `1/4*x - 1/4`, `1/5*x`, `1/20*x - 1/20`, `-1/2*x - 2`, `-1/10*x - 2/5`, `x - 1`; PARI's own
  p-adic polmod inverses with precisions O(5^10), O(5^9), O(5^9), O(5^10), O(2^8) (the E.6 bounds are
  equal to PARI's in every case); `factorpadic` factor counts 1, 2, 2 and `factorpadic(x^2-5,2,10)` = 1;
  `ellmul` = `[202904/15625, 91751917/1953125]` with x valuation -6, `elladd` = `[4,-9]`; all series
  inverses of E.5 (Z, Q, Laurent, Puiseux, Z/4), lazy coefficients -1 and 1; `isprime(2^64+13)`,
  `isprime(2^64-59)` = 1, 561 and psi_12 composite and not pseudoprime; the Z_6 idempotent 24151041;
  `polisirreducible` 1, 1, 0, 1; `isprimepower(4)` = 2.
* **GAP 4.12** (`r3.g`): `[0, true, true, false, false]` for Integers; `IsField(ZmodnZ(2^64+13))` true,
  `IsField(ZmodnZ(3*2^64))` false; `IsFinite` of the two polynomial rings false; full matrix algebra
  `[0, false]`, degree 1 commutative; U(sl_2) not commutative; the prime-power and `IsField` values of
  D.4; `IsField(AlgebraicExtension(Rationals, x^2-2))` true. All match.
* **Algorithms re-derived.** The sentinel contract (Z -> Z_p is injective, so unbound arithmetic in Z is
  exact); value equality and why `(x - y).is_zero()` is not used; the normalised inverse (relative
  precision n - j); Z_p ≅ ∏ Z_q for any p > 1 and the parent rows built on it; the three field
  certificates and their valuation formulas (the Eisenstein residues i mod d are distinct, so the
  minimum is attained once; for an irreducible reduction the scaled reduction is nonzero in
  F_p[x]/(f̄)); the E.6 bound v_K(x^-1 - y_rep) >= E - 2V and its translation into coefficient
  precisions, checked by hand on 5^10 + g in Q_5(√5) (true error valuations 9 and 18, bound 9 and 9);
  `field_status` steps 2 to 5 (F(y) = lc^(d-1) f(y/lc) is monic integral and squarefree iff f is); the
  order rules (μ(Q_p) = μ_(p-1) for odd p, {±1} for p = 2, and p^(mk) is never a unit); the power-series
  right-inverse recurrence; the Laurent unit rule and the (Z/4)((x)) counterexample 2 + x (inverse
  x^-1 - 2x^-2); R[x] rules; the composite-characteristic and non-prime-power-cardinality rules; the
  DU(α, β, γ) non-commutativity. All correct, apart from issue N1.
* **Callers (my own greps).** p-adic types: only the files of A.2 (no container over a p-adic type, no
  hashing, `MPowerSeries<PadicRational>` needs `From<i32>`). `RingParentMethods` (both traits):
  implementors only in tests; no file outside the two defining modules imports either trait, so the new
  impls cause no method ambiguity at existing call sites. `RingMorphismMethods` / `.is_nonzero()`: only
  the tests of B.3. Core `Parent` impls: exactly the list of C.1. Ring impls among p-adic and series
  types: exactly the seven of E. Hook callers: as E.4. Changed existing assertions: only
  `test_prime_order_alone_decides` and the two tests of E.4; the `assert_ne!` Galois tests stay unequal
  (equality only gets weaker, and the step-3 intermediate equality implies the final one).
* **Dependencies** (`cargo metadata`, resolve graph): rustmath-category's closure is exactly the list in
  C.3 and contains neither rustmath-numberfields nor rustmath-algebras nor rustmath-manifolds; its only
  direct reverse dependency is rustmath-modules. No cycle from any new edge.
* **Plan order.** Each step compiles on the previous one; the E.4 test updates sit in the step that
  changes them (review 1 M4); the extension steps depend only on steps 3 and 4.

## Minor issues

* **N1. Value equality and `is_one` for composite p need the unreduced valuation.** A.3.2 defines
  `x == y` as "v_p(d) >= min(absolute precisions)" and `is_one` as "v_p(representative - 1) >=
  absolute_precision". The natural implementation is `Rational::valuation(p)`, which works on the
  reduced fraction (`rational.rs:142`). For a composite p that is wrong: the reduced denominator can
  contain part of p. Reproduction on the HEAD API (`review2/composite_val.rs`): p = 6,
  x = `from_padic_integer(3 + O(6))` * `from_rational(1/6, 6, 1)` = 3·6^-1 + O(6^0). Its
  representative is 1/2; `Rational::valuation(1/2 - 1, 6)` = 0 >= absolute precision 0, so the formula
  says `is_one() == true`, and then the order hook answers `Finite(1)` and `unit_inverse` returns x.
  But every lift is 1/2 + Z_6, whose Q_2 component has valuation -1, so x is never 1 (its order is even
  infinite). The failure needs a composite p and a common absolute precision <= 0 (for m >= 1 the
  reduced form cannot hide a factor), exactly the region of the M6 test. Required change: specify
  "d ∈ p^m Z_p" as an integer test: with k = max(0, -min(v_x, v_y)), A = d·p^k is an integer and the
  condition is p^(m+k) | A (always true when m + k <= 0). For prime p this equals the
  `Rational::valuation` test. Add the case above to `weakly_zero_is_not_one` or a new test
  (`is_one()` false, order not `Finite(1)`).
* **N2. F1's core unbound-constant hooks are not addressed.** F1 lands before R1 and adds to the core
  `Ring` trait `HAS_UNBOUND_CONSTANTS`, `is_unbound_constant()` and `bind_to_parent_of()` ("whether
  values of this type can be unbound constants"; containers bind through them). After R1 the three
  p-adic types do have unbound constants, so the defaults (`false`, `None`) would misstate the hook's
  contract. The effect is small (Z -> Z_p is injective, so an unbound c in a container is still the
  right element, and no container over a p-adic type exists today), which is what the design's last
  risk says. Required change: implement the three hooks for the p-adic types
  (`HAS_UNBOUND_CONSTANTS = true`, `is_unbound_constant = !is_bound()`, `bind_to_parent_of(model)` =
  bind into the model's parent at the model's precision), or state in A.3 why they opt out and update
  F1's hook doc, which lists only the rustmath-finitefields types.
* **N3. MatrixAlgebra rows: "full" must be structural, and the degree-1 field claim trusts `F: Field`.**
  (a) C.3 does not say how "full, n >= 2" is detected. If it is read from `dimension() == n^2` of a
  `generated` algebra, an inexact F (f64, RealMPFR) can inflate the computed dimension through rounding
  in `reduce_into_basis`, and the claim "not commutative" would then be made about a commutative
  algebra. Decide "full" only for algebras built by `MatrixAlgebra::full` (`generators.is_empty()`).
  (b) The dim = 1 row answers field `Some(true)` because `F: Field`. `PadicRational` implements `Field`
  for every p > 1, while C.2 itself says Q_6 is not a field, so `MatrixAlgebra::<PadicRational>` with
  p = 6 and dim 1 would be called a field. Record `impl Field for PadicRational` with composite p as an
  adjacent finding in F.4 and say in the C.3 doc that the row relies on the `Field` impl being true.
* **N4. The `DiffScalarFieldAlgebra` hand-off has no taker.** F.2 hands it to MF1, but the MF1 design
  (`MF1-manifolds.md`) does not mention `RingParentMethods` or `DiffScalarFieldAlgebra`. The impl is
  small and feasible here (rustmath-manifolds → rustmath-category creates no cycle; the manifold
  dimension is known), the same kind of out-of-unit additive change as C.4 and C.5. Required change:
  implement it in R1 with the values F.2 already states, or make sure Plan step 13 writes it to HANDOFF
  as an open follow-up (not "handed to MF1").
* **N5. Prime-power p is decidable but left `None`.** When p = q^a with q proven prime, Z_p = Z_q and
  Q_p = Q_q (the design uses this for the parent rows). The extension certificates, the OM step, the E.6
  inverse and the `PadicInteger` order rule can then all run with q. The test `field_status_decides`
  asserts `None` for `new(4, x^2 - 5, ..)`, which is decidable: gp `factorpadic(x^2-5,2,10)` has one
  factor, so K is a field. Required change: in `field_certificate`, `field_status` step 5 and the order
  rules use the minimal root q when `padic_base_is_prime_power(p) == Some(true)`, and change that test to
  `Some(true)`; or keep `None` and state the reason in F.
* **N6. Series: the departure from MAGMA's strict predicates is not documented.** The Ch. 49 digest
  (§49.4.3-4) says `eq`, `IsZero`, `IsOne` are strict for series; RustMath's `PowerSeries` and
  `LaurentSeries` use the weak versions (HEAD). The new hooks build on them (`O(x^5)` nilpotent
  `Some(true)`, `1 + O(x^5)` order `Finite(1)`). 0.1 documents the matching departures for the p-adic
  types; E.3 should state this one too, in the module doc of the four hooks.
* **N7. Small spec gaps.** (a) A.3.3 says `PadicExtension` gains "three private fields", but the text
  needs four values (the certificate, `is_reduced`, the proven primality of p that `field_status`
  step 5 reads again, and the `OnceLock` cache); list them. (b) The 0.2 helpers are copied into two crates;
  with I1 landed, a non-panicking `try_prime_power` next to I1's `prime_power` in rustmath-integers would
  serve both, since I1's `prime_power` panics when a proof is unresolved. Optional.

## Checks that passed

* Every item is covered; no item is declared out of scope. The residual limits of F (series parents
  without a base ring, `SimpleAlgebra` without a product, the inverse in an extension certified only by
  OM, the internal `nf_elem::NumberField`) are honest `None`/`NotImplemented` with what would be needed.
  The F.3 limit is justified: without an integral basis, v_K(δ) >= B gives no coefficient bound.
* No facade or silent truncation: every new `Some`, `Finite`, `Infinite` or `Ok` is a theorem under the
  stated weak-zero convention (with N1 fixed); the lazy series rules never decide from the 20-term
  window; an undecided extension inverse is `Err(NumericalError)`, never `NotInvertible`; OM `Err`
  gives `None`.
* The MAGMA digests support the citations (Ch. 17 §17.4.2-3, §17.5.2, §17.5.5; Ch. 47 §47.7.1,
  §47.8.2-3, §47.3.1; Ch. 49 §49.4.4; Ch. 25, 34, 48, 83), and 0.1 states each p-adic departure.
* Tests fail on HEAD (panics, `None`, `NotImplemented`, or missing impls) and their reference values
  come from PARI/GP and GAP, which I re-ran on a broad sample.
* The acceptance list covers every crate whose `src/` changes and their direct dependents;
  `cargo check --workspace --tests` covers the rest.
