# Design R1: rings, p-adics, category (rustmath-rings, rustmath-padics, rustmath-category, rustmath-core, rustmath-powerseries)

HEAD `c79a582` (same code as `9c24389`). Experiments and logs:
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/R1-rings-padics-category/`
(`$X` below). `rc.sh` builds a standalone program against the HEAD rlibs copied to `lib/`. `e1_panics.rs`,
`e2_morph_cat.rs`, `e3_series.rs` reproduce the defects. `proto_padic.rs` is a working prototype of the sentinel
part of the fix, run through `rustmath_ellipticcurves::generic::EllipticCurve`. `ref1.gp` to `ref4.gp` and
`ref3.g` hold the reference values of the first version. Revision 1 work is in `$X/rev1/`:
`proto_ext_inverse.rs` (the extension inverse of E.6 on the HEAD API), `om_cost.rs` (OM factorisation results and
times), `cost_prim.rs` (primality helper costs), `sqfree.rs` (the squarefree test over Z), `rev1.gp`, `rev1b.gp`, `rev1c.gp`, `rev1.g` (new PARI/GP and GAP
references); each output is in the matching `.log`. The pre-revision design is kept as `$X/rev1/design_v0_backup.md`.

## Revision 1

Review 1 (`R1-rings-padics-category.review1.md`) found 2 blockers and 14 minor issues. All are accepted. Two
are answered in a slightly different way than proposed, with the reason in the table (M6, M9 on degree-1
extensions). While revising I also found more that can be decided (the OM factorisation of the crate decides
whether a non-certified extension is a field) and two more parents with a core `Parent` impl.

| Issue | What changed | Where |
|---|---|---|
| X1 (blocker): the extension inverse was out of scope | Implemented. `PadicExtensionElement::inverse` (inherent) and `Ring::unit_inverse` solve `M_x y = e_1` exactly over Q on the rational representatives, then bound the error with the valuation of K in the power basis: v_K(Σ c_i g^i) = min(e·v_p(c_i) + i) for an Eisenstein f, min v_p(c_i) for an irreducible reduction or degree 1. If v_K(x_rep) >= the error bound E the answer is an undecided `Err(NumericalError)`. Otherwise coefficient i has absolute precision ceil((B - s_i)/e), B = E - 2 v_K(x_rep). Prototype on the HEAD API (`rev1/proto_ext_inverse.rs`): the precisions agree with PARI's own p-adic polmod inverses in every case tried. The certificate is now an enum `FieldCertificate` (which shape) instead of a bool. | A.3.3, E.2, E.5, E.6 |
| X2 (blocker): number fields out of scope on a wrong argument | The orphan-rule argument is withdrawn: it was wrong. `rustmath-numberfields` implements core `Parent` for `NumberField` and `rustmath_category::RingParentMethods` + `CommutativeRingParentMethods` (new dependency numberfields → category; no cycle, checked with `cargo metadata`). Domain = field = irreducibility of the defining polynomial over Q (`is_irreducible_over_integers`, `Err` gives `None`). | C.4, C.7 |
| M1: `Field::divide` with an unbound operand, unbound / unbound | `PadicRational` overrides `Field::divide` (binds first, like `/`). Unbound c / unbound d: exact quotient when d divides c, else `Err(InvalidArgument("... bind an operand first"))` (`divide`) or a panic with that message (`/`); "Division by zero" only for d = 0. | A.3.2 |
| M2: `is_zero_ring` not overridden | `Some(false)` for every listed parent; `base.is_zero_ring()` for R[x]. New test Z/(2^64+13)[x]: finite `Some(false)` (GAP), Krull 1. | C.2, C.7 |
| M3: callers that receive unbound elements | `PadicIntegerRing::adjust_precision` binds into its ring; `PadicExtensionElement::new` and `from_base_field` bind unbound coefficients; `newton_lift::padic_to_int` returns c mod m for an unbound c. Tests for each. | A.3.1 to A.3.3, A.4, A.5 |
| M4: plan order | The two existing tests whose values change are updated in the step that changes them (step 3, PadicInteger), not in step 2. | Plan |
| M5: MAGMA semantics | 0.1 states each departure from Ch. 47 §47.8.2 (`eq`), §47.8.3 (`IsZero`, `IsOne`, `IsUnit`) and §47.3.1 (fixed precision = `pAdicQuotientRing`), with the reason; the type docs repeat them. | 0.1, Plan step 3 to 5 |
| M6: an element both zero and one | `PadicRational::is_one` is false for a weakly zero element; every hook also tests zero first. I keep weak `==` (such an element still compares equal to 1, like `O(x^0) == 1` for series) and document it. | A.3.2, E.2 |
| M7: `cheap_primality` cost | It returns `None` at once above psi_12. Below, it runs one pass of the strong tests (measured 0.42 ms against 8.5 ms for HEAD `is_prime` on a 64-bit prime, debug). `proven_primality` (parents, once per `PadicExtension`) is the only caller above psi_12. | 0.2 |
| M8: GF(q) and GR rows depend on I1 | The rows use `proven_primality(p)` (`Some(true)` only when proven); with I1 and F1 the constructors already prove p, so the answers are the same and cheap. | C.2 |
| M9: decidable cases left `None` | Decided now: Z_p and Q_p for every p > 1 (Z_p = ∏ Z_q: reduced, Noetherian, Krull 1; domain iff p is a prime power, decided by the minimal root of p); unbound -1 has order 2 in Z_p; unbound c with \|c\| >= 2 has infinite order in Q_p and K; a nonzero element of K is not nilpotent when f is squarefree; every K is Noetherian of dimension 0; prime cardinality gives the characteristic and a non-prime-power cardinality proves "not a domain" (both traits); `PolynomialRingParent::krull_dimension` requires a commutative base. On "a degree-1 extension is a field": it is K = Q_p, a field only when Q_p is, so its `field_status` is the prime-power test of p, not `true`. Beyond the review: `PadicExtension::field_status` decides the remaining extensions by the certified OM factorisation (`om_factorization`, MacLane), cached on first use. | C.2, D, E.2 |
| M10: weak-zero coefficients | E.3 and the hook docs state the precondition: the series rules that read coefficient `is_zero()` are proofs when that test is exact; for p-adic coefficients they hold under the 0.1 convention. | E.3 |
| M11: other ring parents | `MatrixAlgebra<F>` and `BigFloatField`: rustmath-category impls (both crates are already in its dependency closure). `DownUpAlgebra`: impl in rustmath-algebras (new dependency algebras → category, no cycle). `SimpleAlgebra` has no multiplication, so it is not a ring: no impl. `Lattice` and the module parents are not rings. `DiffScalarFieldAlgebra` (C^∞(M), rustmath-manifolds): hand-off with the decided values (F.2). | C.3, C.5, F.2 |
| M12: `0 * x` loses precision | The unbound 0 times anything (and divided by a nonzero element) is the exact unbound 0, for all three types. | A.3 |
| M13: acceptance list | `cargo test -p rustmath-graphs` added, with rustmath-numberfields and rustmath-algebras. | Plan |
| M14: small gaps | `domain_one = PadicInteger::one(5, 10).ok()`. The type docs cite the non-transitive precedents (`PowerSeries`, the sentinel corner of `GaloisRingElement`, `FiniteFieldElement`, `IntegerMod`). | B.4, 0.1 |

Other changes made while revising:
* The field certificate uses `proven_primality(p)` (computed once in `PadicExtension::new`), not `cheap_primality`.
* Every `PadicExtension` parent is now Noetherian of Krull dimension 0 (a finite Q_p-algebra), certified or not.
* Crates outside the unit list that this design now changes in `src/` (additive only): rustmath-numberfields
  (C.4) and rustmath-algebras (C.5). If the orchestrator keeps them out of R1, C.4 and C.5 become hand-offs with
  exactly that spec; nothing else in the design depends on them.
* The order of units is fixed (I1, P1 and F1 land before R1), so the primary path uses their API; each fallback
  is still stated.
* Adjacent finding: `PadicExtension::generator()` returns 0 for a degree-1 extension (F.4).

## 0. Summary

| Item (HANDOFF "Open follow-ups") | At HEAD | Fix |
|---|---|---|
| A. p-adic `zero()`/`one()` panic (`PadicInteger`, `PadicRational`, `PadicExtensionElement`) | Confirmed: all three panic, and so does `x.pow(0)` (`e1_panics.rs`). Six tests in rustmath-schemes are `#[ignore]`d because of it. | Unbound integer sentinels that bind on contact (the `GaloisRingElement` design of `029f7ab`), with the binding of MAGMA `L ! c` (Ch. 47 §47.7.1); the unbound 0 stays exact in sums and products. `PadicRational` equality and `is_one` become value comparisons to the common precision (needed for the sentinel to compare correctly; it also closes tracker row "PadicRational / PadicExtensionElement equality and is_one", PROJECT_TRACKER.md:1108). `Ring::pow(x, 0)` returns a bound 1. Callers that know the parent bind. |
| B. `RingMorphismMethods::is_nonzero` reads the static `Self::Domain::one()` | Confirmed: panics for a p-adic domain (`e2_morph_cat.rs`); for sentinel domains it feeds an unbound 1 to `apply`. | New `domain_one(&self) -> Option<Self::Domain>` (default `None`); `is_nonzero(&self) -> Option<bool>` evaluates `apply` only on that 1. |
| C. No real ring type implements the three-valued `RingParentMethods` | Confirmed: the only impls in the workspace are test parents (grep, section C.1). | rustmath-rings: Z, Q (new unit parents), R[x] (new generic parent), Z/nZ, GF(q), Galois rings, Z_p, Q_p, p-adic extensions. rustmath-category: `Zmod`, `FiniteField`, `GaloisRing`, `MatrixAlgebra<F>`, `BigFloatField`. rustmath-numberfields: `NumberField` (core `Parent` and the category traits). rustmath-algebras: `DownUpAlgebra`. |
| D. The category trait lacks the composite-characteristic rule | Confirmed: characteristic 4 and Z/6 give `is_integral_domain() = None`, a prime cardinality alone decides nothing (`e2_morph_cat.rs`). | The rustmath-rings rules move into `rustmath_category::RingParentMethods` (composite characteristic is not a domain; characteristic = cardinality, or a prime cardinality, is Z/c). Both traits gain: a prime cardinality p gives characteristic p; a cardinality that is not a prime power is not a domain. rustmath-category gains a normal dependency on rustmath-integers. |
| E. Ring hooks for series and p-adics (honest `None`) | Confirmed: every hook is `None` for p-adic and series elements, and the defaults give three wrong answers on lazy series and on `O(x^0)` (`e3_series.rs`). | Exact hooks for `PadicInteger`, `PadicRational`, `PadicExtensionElement` (including its inverse in a certified field, E.6), `PowerSeries`, `LaurentSeries`, `PuiseuxSeries`, `LazyPowerSeries`; `static_characteristic` (and P1's `static_is_commutative`) for all of them. |

No item is out of scope. Residual limits, each with its honest behaviour and what would be needed (section F):
`SeriesRing` does not carry its coefficient ring; `SimpleAlgebra` is not a ring; the inverse in an extension that
only OM certifies as a field stays `Err(NotImplemented)` (it needs an integral basis); `nf_elem::NumberField` and
`DiffScalarFieldAlgebra` are hand-offs. Adjacent findings that this unit does not fix are listed in F.4.

Dependencies on earlier units (implemented before R1; the fallback is stated for each):
* I1 (`rustmath-integers` primality): `prime::is_prime_u64`, `prime::is_prime` (a proof), `prime::try_is_prime`
  (proof or `Err`). Fallback: the HEAD tests with the psi_12 bound (section 0.2). I1 also switches
  `FiniteField::new` and `GaloisRing::new` to proven primality (I1 Revision 1, B2).
* P1 (`rustmath-polynomials`, `rustmath-core`): the hook `Ring::static_is_commutative()` (P1 design §G.3). The
  p-adic types return `Some(true)`, the series types delegate, and the commutative nilpotency rules of E.3 are on.
  Without it those rules are left out (the rest stands).

### 0.1 The model the p-adic types follow, and where it departs from MAGMA (needed by A and E)

`PadicInteger` is documented as "an element of Z_p truncated to precision n", and its parent `PadicIntegerRing`
is "Ring of p-adic integers". This design fixes the reading: **the elements approximate elements of Z_p (Q_p,
K)**, and `is_zero`/`is_one` mean "indistinguishable from 0/1 at the known precision" (Sage's `is_zero()`,
doc.sagemath.org `padic_capped_relative_element`; this is what the code already does). The `Ring` trait defaults
treat `is_zero` as exact, and so do the hooks below (the convention). Every other decision holds for every element
of Z_p (Q_p, K) that the approximation represents, so it is a proof under this convention. Two consequences:
`static_characteristic()` is `Some(0)` (Z embeds in Z_p for every p > 1), and an element such as 5 + O(5^10) is not
nilpotent (Z_5 is a domain), although its residue in Z/5^10 is.

The digest (Ch. 47) describes MAGMA's free-precision rings differently in four places. Each departure, and why:

1. **Equality** (§47.8.2: "Equality testing is only available for fixed precision (quotient) rings; it is disabled
   for free precision rings where multiple reasonable definitions of equality exist"). Here `Ring: PartialEq`
   (`rustmath-core/src/traits.rs:51`), and generic code compares with `==` (EC `Point` equality, `is_on_curve`'s
   `lhs == rhs`, `p == q` in `add_points`). So the types pick one definition: the values agree to the smaller known
   precision (Sage `is_equal_to`; MAGMA's own `IsWeaklyEqual` for series, Ch. 49 §49.4.4). It is not transitive:
   1 + O(5^2) equals both 1 + O(5^10) and 26 + O(5^10), which differ. The workspace has this precedent:
   `PowerSeries` (`rustmath-powerseries/src/series_core_impls.rs:21`) and the sentinel corner of
   `GaloisRingElement`, `FiniteFieldElement`, `IntegerMod` (`galois_ring.rs:929`, `finite_field.rs:735`,
   `integer_mod.rs:481`). No p-adic value is hashed or used as a map key (grep).
2. **`IsZero`, `IsOne`** (§47.8.3: "In free precision rings, the predicates below always return false"). Here they
   answer "indistinguishable", as at HEAD. Always `false` would make `R::zero().is_zero()` false for the sentinel and
   break every generic zero test (the `is_zero` checks of `Field::divide`, `add_points`, `double_point`).
3. **`IsUnit`** (§47.8.3: "True iff x has valuation 0"). `Ring::unit_status` is MAGMA Ch. 17 §17.5.5 `IsUnit(a)`
   ("a is a unit in its parent ring R"), shared by every `Ring` type and paired with `unit_inverse`. In Z_p the two
   agree. In Q_p and K every nonzero element is a unit of the field; the "valuation 0" test stays available as
   `x.normalized().valuation() == 0`.
4. **Fixed precision.** MAGMA's fixed-precision ring is `pAdicQuotientRing(p, k)`, the quotient Z_p/p^k Z_p
   (§47.3.1, §47.4.1), of characteristic p^k. `PrecisionModel::FixedModulus` follows Sage's fixed-mod Z_p instead:
   the elements are the same `PadicInteger`s and the parent is Z_p, of characteristic 0 (Sage
   `Zp(3, 10, 'fixed-mod').characteristic()` returns 0, doc.sagemath.org `padic_generic`). The quotient ring
   Z/p^k is `Zmod::new(p^k)` (characteristic p^k, C.2).

The doc comments of the three types (Plan steps 3 to 5) carry these four points.

### 0.2 Primality helpers (rustmath-rings, `category_methods.rs`, `pub(crate)`)

```rust
/// psi_12 = 399165290221 * 798330580441 = 318665857834031151167461 (Sorenson and Webster 2017).
fn psi_12() -> Integer;
/// Primality of a word, a proof: I1's `prime::is_prime_u64(n)`.
/// Fallback without I1: `prime::is_prime_miller_rabin(&Integer::from(n), 1)` (at HEAD one pass of the 12 bases
/// 2..37, a proof below psi_12 > 2^64).
pub(crate) fn prime_u64(n: u64) -> bool;
/// Cheap primality for the element hooks (the order rules of E.2 are its only callers):
/// `Some(false)` for n < 2; for n < psi_12 the exact answer `Some(prime::is_prime(n))` (with I1 a proof:
/// `is_prime_u64` below 2^64, the 12 strong bases above); `None` at once above psi_12, without running any test
/// (the hooks act only on `Some(true)`).
/// Fallback without I1: `is_prime_miller_rabin(n, 1)` below psi_12. I1 redefines `is_prime_miller_rabin(n, r)`
/// to use the first r prime bases once, so this fallback must not stay once I1 has landed.
pub(crate) fn cheap_primality(n: &Integer) -> Option<bool>;
/// MAGMA `IsPrime(n : Proof := true)`, for the parent predicates and `PadicExtension::new` (rare calls):
/// `cheap_primality(n)` below psi_12; above it I1's `prime::try_is_prime(n).ok()` (`Ok(false)` from a BPSW or
/// divisor witness, `Ok(true)` from a verified ECPP certificate that I1 caches, `Err` gives `None`).
/// Fallback without I1: `Some(false)` when `!is_prime_miller_rabin(n, 1)` (a witness), else `None`.
pub(crate) fn proven_primality(n: &Integer) -> Option<bool>;
/// Whether a word is q^k with q prime, k >= 1: b = the minimal root of n (`Integer::is_perfect_power`, whose base
/// is not itself a perfect power; n itself when it returns `None`), then `prime_u64(b)`. Used by D: a finite
/// domain is a field, and a finite field has prime-power order.
pub(crate) fn is_prime_power_u64(n: u64) -> bool;
/// Whether p (> 1) is a prime power: `proven_primality(b)` for the minimal root b of p.
/// `Some(true)`: p = q^a with q proven prime, so Z_p = Z_q. `Some(false)`: b is composite and not a perfect
/// power, so p has two distinct prime factors (a composite with one prime factor q is q^c, c >= 2, a perfect
/// power). `None`: undecided.
pub(crate) fn padic_base_is_prime_power(p: &Integer) -> Option<bool>;
```

psi_12 = 399165290221 · 798330580441 (python3: 318665857834031151167461; gp `isprime(...)` = 0,
`ispseudoprime(...)` = 0). The minimal root: `Integer::is_perfect_power` extracts q-th roots for each prime q in
increasing order while it can (`integer.rs:604`), so the final base is not a perfect power: 64 gives (2, 6), 36
gives (6, 2), 6 gives `None` (`rev1/cost_prim.log`; gp `ispower(36,,&r)` = 2 with r = 6).

Measured costs (debug, HEAD rlibs, `rev1/cost_prim.log`): `prime::is_prime` takes 8.5 ms on 2^64 - 59 and 17.7 ms
on 2^64 + 13 (twenty passes of the 12 bases); `is_prime_miller_rabin(n, 1)` 0.42 ms and 0.97 ms; `is_perfect_power`
25 to 29 µs. Review 1 measured 396 ms for `is_prime_miller_rabin(M521, 12)`, which no hook runs any more.

rustmath-category gets private copies of `prime_u64`, `cheap_primality`, `proven_primality` and
`is_prime_power_u64` (it needs them for D and for Z/m with m >= 2^64).

---

## A. p-adic `Ring::zero()`/`Ring::one()` panic

### A.1 Defect at HEAD

* `rustmath-rings/src/padics/padic_integer.rs:304-312`: `zero()`/`one()` are `panic!("Cannot create
  PadicInteger::zero() without parameters")`.
* `rustmath-rings/src/padics/padic_rational.rs:120-127`: same for `PadicRational`.
* `rustmath-rings/src/padics/extension.rs:794-801`: same for `PadicExtensionElement`.

Reproduction (`e1_panics.rs`, HEAD rlibs):

```
PadicInteger::zero() panics: true
PadicRational::one() panics: true
PadicExtensionElement::zero() panics: true
Ring::pow(x, 0) for PadicInteger panics: true
(5,0) == from_rational(5) = (1,1)? false
PadicRational (5,0): Field::inverse Err(NotInvertible)
from_rational(0): unit 0 valuation -1
g*g coeffs: [(Integer(5), 0), (Integer(0), 0)]      (in Q_5(sqrt 5))
```

The last four lines are defects of the same values that the sentinel must compare and invert: 1 + 4 is stored
as the unnormalised pair (5, 0), which compares unequal to `from_rational(5)` = (1, 1) (gp
`(1+O(5^10))+(4+O(5^10)) == 5+O(5^11)` gives 1); `Field::inverse` of that nonzero element of Q_5 claims "not
invertible" (gp `1/(5+O(5^10))` = `5^-1 + O(5^8)`); `from_rational(0)` stores `u32::MAX as i32 = -1` as the
valuation.

### A.2 Callers that reach these constructors (grep of the whole workspace)

`grep -rn "PadicInteger\b\|PadicRational\b\|PadicExtensionElement\b"` outside the defining files hits only
`rustmath-padics/src/lib.rs` (re-export shim), `rustmath-rings/src/category_methods.rs` (tests),
`padics/{factory,mod,newton_lift,unramified}.rs`, `valuation/maclane.rs` (a test) and
`rustmath-schemes/src/elliptic_curves/tests.rs`. No `UnivariatePolynomial<Padic..>`, matrix or power series over
a p-adic type exists (`grep "<Padic\(Integer\|Rational\|ExtensionElement\)>"`). The calls that reach the
panicking constructors are generic:

1. `rustmath-schemes/src/elliptic_curves/tests.rs:340-468`: six `#[ignore]`d tests (`test_padic_curve_creation`,
   `test_padic_point_on_curve`, `test_padic_point_addition_with_infinity`, `test_padic_point_doubling`,
   `test_padic_scalar_multiplication`, `test_padic_discriminant`), with the comment at :333 that blames
   `PadicRational::zero()/one()`. They reach `EllipticCurve::<PadicRational>::short_weierstrass`
   (`rustmath-ellipticcurves/src/generic.rs:256`, `<F as Ring>::zero()` three times), `b2`..`b8` and
   `discriminant` (:295-363, sums of `<F as Ring>::one()`), `c4` (:387), `double_point` (:563, :586). The tests
   read only `is_zero`, `is_infinity`, `is_on_curve` and `is_short_weierstrass`, never an accessor of a derived
   value, so derived values that stay unbound (`b2` = 0 here) are harmless.
2. `Ring::pow` default (`rustmath-core/src/traits.rs:75-77`): `x.pow(0)` calls `Self::one()`.
3. `RingMorphismMethods::is_nonzero` (`category_methods.rs:468`), item B.
4. Not reached: `MPowerSeries<PadicRational>` (`newton_lift.rs`) never calls `R::zero()`/`R::one()`; its
   `R::from(i32)` methods need `R: From<i32>`, which no p-adic type implements; the core defaults never call them
   (pinned by `traits.rs::test_defaults_never_call_static_constructors`).

Functions that know the parent and can receive an unbound element from a caller (review M3): A.4 lists them with
their new behaviour.

### A.3 Fix: unbound sentinels (MAGMA Ch. 47 §47.7.1 `L ! u`; Ch. 17 §17.5.2 `R ! a`)

The static constructors return an **unbound** element: an exact integer constant c awaiting the canonical map
Z → Z_p (Q_p, K). MAGMA Ch. 17 §17.5.2: "`R ! a` ... If a is an integer, always succeeds, returning a·1_R";
Ch. 47 §47.7.1: "`elt< L | u >` / `L ! u`: Coerce u into L with maximum possible precision. u may be: an
integer; ...". So an unbound constant meeting a bound element x is replaced by `x.parent ! c` at x's precision.
Since Z → Z_p is injective (characteristic 0), computing with two unbound operands in Z is exact, and the
GaloisRing caveat (eight `R::one()`s that should be 0) does not arise; the only effect of binding is that the
exact constant acquires x's precision (`unbound(5^10)` bound at precision 10 is 0 + O(5^10)).

**Common contract (doc comment on each type, modelled on `galois_ring.rs:346-402`):**

* `+`, `-`, `*` (and `/` for `PadicRational`): one unbound operand c is bound into the other operand's parent
  first, with two exceptions that keep the exact zero exact (review M12): x + 0 and x - 0 are x itself, 0 - x is
  -x; and 0 times anything (0 · x, x · 0), and 0 divided by an element that is not weakly zero, are the **unbound
  0**. Binding would give a zero of finite precision (absolute precision v_x + n), which lowers the precision of
  sums such as `a1·x·y` in `is_on_curve`; the exact zero is right for every lift. Two unbound operands combine in Z
  and stay unbound (division: A.3.2). Two bound operands behave as at HEAD.
* `Neg`: negates in Z for an unbound element.
* `==`: an unbound operand compares as the exact constant (A.3.4); two unbound operands compare in Z.
* `is_zero`/`is_one` of an unbound element: `c == 0` / `c == 1` in Z.
* `pub fn is_bound(&self) -> bool` on each type.
* Accessors that need the prime or precision (`prime()`, `precision()`, `valuation()`, `residue()`, `abs()`,
  `extension()`) panic on an unbound element with a message naming the method and saying how to bind it
  (combine it with a bound element, or use `PadicIntegerRing::zero()`, `PadicInteger::zero(p, n)`,
  `PadicExtensionElement::zero(ext)`). `Result`-valued methods (`lift`, `truncate`, `inverse`,
  `hensel_lift_linear`, `hensel_lift_root`, `norm`, `trace`) return `Err(MathError::InvalidArgument(..))`
  instead, except `inverse` of ±1 (itself) and of 0 (the type's zero error: `NotInvertible` for `PadicInteger`
  and extension elements, `DivisionByZero` for `PadicRational`, as at HEAD).
* `Display` of an unbound element prints the integer c.
* `Ring::pow(&self, n)` is overridden: for a bound element, `n == 0` returns the bound 1 at the element's
  precision (`PadicInteger::one(p, n)`, `PadicRational::from_padic_integer(PadicInteger::one(p, n))`,
  `PadicExtensionElement::one(ext)`); otherwise square-and-multiply (the default's algorithm). An unbound
  element computes c^n in Z.

#### A.3.1 `PadicInteger` (padic_integer.rs)

Representation (fields stay private and unchanged): an unbound element has `prime == 0`, `precision == 0` and
`value == c` (unreduced). A bound element always has `prime > 1` and `precision > 0` (`from_integer` rejects
anything else), so `is_bound() = !prime.is_zero()` is exact.

```rust
impl PadicInteger {
    pub fn is_bound(&self) -> bool;
    pub(crate) fn unbound(c: Integer) -> Self;            // used by PadicRational and Ring::zero/one
    fn bind_like(&self, other: &PadicInteger) -> PadicInteger; // from_integer(c, other.prime, other.precision)
    fn coerce_pair(self, other: Self) -> std::result::Result<(Self, Self), (Self, Self)>; // Err: both unbound
}
```

`value()` of an unbound element returns c (documented: "the unreduced integer constant"). `Add`/`Sub`: the
zero rule of A.3, else `coerce_pair`, then HEAD arithmetic (`precision = min`, same-prime assert) or
`unbound(a op b)`. `Mul`: an unbound 0 operand gives `unbound(0)`; else the same. `Ring::zero() = unbound(0)`,
`Ring::one() = unbound(1)`.

`PadicIntegerRing::adjust_precision(elem)` (`factory.rs:193`): an unbound elem is first bound into the ring,
`self.from_integer(c)` (its prime and default precision), then the model's rule applies. At HEAD this method reads
`valuation()` and `precision()`, which would panic on an unbound element.

**Equality (A.3.4 rule):** two bound elements are equal iff they have the same prime and their values agree
modulo p^min(n_x, n_y). At HEAD the precision was part of the identity (`padic_integer.rs:246-252`); the new rule
is Sage's `is_equal_to` and the rule `PowerSeries` already uses (0.1, point 1). An unbound c equals a bound x iff
c ≡ x mod p^n_x (it is exact, so its precision is infinite).

#### A.3.2 `PadicRational` (padic_rational.rs)

Unbound: `unit` is an unbound `PadicInteger` holding c, `valuation == 0`. `is_bound() = unit.is_bound()`.
`from_padic_integer` of an unbound `PadicInteger` is the unbound `PadicRational` with the same c.

```rust
impl PadicRational {
    pub fn is_bound(&self) -> bool;
    pub(crate) fn unbound(c: Integer) -> Self;
    /// `L ! c` (Ch. 47 §47.7.1) at relative precision `precision`: `from_rational(c)` (normalised: c = m p^k,
    /// unit m to precision n, valuation k) for c != 0; the zero 0 + O(p^n) at valuation 0 for c == 0.
    pub(crate) fn bind_integer(c: &Integer, prime: &Integer, precision: usize) -> Self;
    /// x = u' p^k with p not dividing u', u' known to relative precision n - j where j = v_p(unit value);
    /// `None` for an element that is zero to its precision. Bound elements only.
    pub(crate) fn normalized(&self) -> Option<PadicRational>;
    /// Exact rational u * p^v (the canonical representative; extension.rs:86 has the same helper, which moves here).
    pub(crate) fn representative(&self) -> Rational;
    /// valuation + precision: x is known modulo p^absolute_precision. Bound elements only.
    pub fn absolute_precision(&self) -> i64;
    /// The exact rational r known to absolute precision `absolute`, with relative precision at most `cap`:
    /// `from_rational(r, p, min(cap, absolute - v_p(r)))` when r != 0 and absolute > v_p(r); otherwise the zero
    /// 0 + O(p^cap) at valuation `absolute - cap` (absolute precision `absolute`). Used by E.6.
    pub(crate) fn from_rational_to_absolute_precision(r: &Rational, prime: &Integer, absolute: i64, cap: usize) -> Self;
}
```

Arithmetic with one unbound operand c and a bound x:
* `x + c`, `c + x`: if c == 0 the result is x itself (the exact zero must not lower x's precision: HEAD's `Add`
  aligns to the smaller valuation, so adding the bound zero (0, 0) to 25 = (1, 2) would give (25, 0) and lose
  two digits; the prototype prints the kept valuation 2). Otherwise `bind_integer(c, x.prime, x.precision) + x`.
* `x - c` = `x + (-c)`; `c - x` = `(-x) + c`.
* `x * c`, `c * x`: c == 0 gives `unbound(0)`; else bind, then the bound product.
* `x / c`, `c / x`: c == 0 in `c / x` gives `unbound(0)` when x is not weakly zero; else bind, then the bound
  quotient (below). A zero divisor (bound and weakly zero, or the unbound 0) panics "Division by zero", as at HEAD.
* Two unbound operands c, d: `+`, `-`, `*` in Z. `c / d`: d == 0 panics "Division by zero"; d divides c gives
  `unbound(c / d)`; otherwise it panics with "PadicRational: c / d is not an integer constant; bind an operand
  first (combine it with a bound element or use PadicField::from_rational)". Never "Division by zero" for d != 0
  (review M1).

**`Field` impl:**

```rust
impl Field for PadicRational {
    /// Bound: the normalised inverse (below); weakly zero: Err(DivisionByZero). Unbound c: ±1 give themselves,
    /// 0 gives Err(DivisionByZero), any other c gives Err(InvalidArgument("1/c is not an integer constant;
    /// bind it first")).
    fn inverse(&self) -> Result<Self>;
    /// MAGMA `x / y` (Ch. 47 §47.8.1), with the rules of `/` above as `Result`s: an unbound operand is bound
    /// into the other operand's field first, so `x.divide(&two)` equals `x / two`; a zero divisor gives
    /// Err(DivisionByZero); the unbound 0 over a nonzero y is the unbound 0; unbound c / unbound d gives the
    /// unbound c/d when d divides c, else Err(InvalidArgument(".. bind an operand first")).
    fn divide(&self, other: &Self) -> Result<Self>;
}
```

The core default `divide` (`traits.rs:327`) is `self * other.inverse()?`, which would return `Err` for
`x.divide(&two)` with an unbound two although `x / two` binds; the override removes that difference.

**Equality (A.3.4):** bound x, y: same prime, and the exact representatives agree to the common absolute
precision: `d = representative(x) - representative(y)`, equal iff `d == 0` or
`v_p(d) >= min(absolute_precision(x), absolute_precision(y))`. Unbound c vs bound x: `d = representative(x) - c`
against `absolute_precision(x)`. This is **not** computed as `(x - y).is_zero()`: HEAD's `Add` shifts the
higher-valuation unit at the other operand's relative precision and can drop known digits, so the difference
can be zero to its (smaller) precision while x and y differ within both precisions. The prototype shows it:
`0 + O(5^10)` vs `5^5 + O(5^8)` gives `(x - y).is_zero() = true` but the representative rule gives `false`.

**`is_one`** (bound): `!self.is_zero()` and (`representative - 1 == 0` or `v_p(representative - 1) >=
absolute_precision`); so (5, -1) is 1 (tracker row 1108 example). The `!is_zero()` part is review M6: an element
of absolute precision <= 0 such as `from_rational(1/5, 5, 1) - from_rational(1/5, 5, 1)` = (0, -1, n = 1) is
weakly zero and also within O(5^0) of 1; it is reported as zero only, so the hooks (which test zero first) see a
consistent element. Weak `==` still says it equals `Ring::one()` (the vacuous comparison of 0.1, point 1, like
`O(x^0) == 1` for series); the type doc says so. **`is_zero`** is unchanged (`unit.is_zero()`).

**`from_rational(0)`** returns (0 + O(p^n), 0) instead of (0, -1) (the `u32::MAX as i32` cast at
`padic_rational.rs:31-34`); this is the zero that `bind_integer(0, ..)` also returns.

**Inverse** (shared with item E): `Field::inverse`, `Div`, `divide` and `Ring::unit_inverse` go through
`normalized()`: x = u' p^k with u' a unit of Z_p known to n - j digits, so x^-1 = (u'^-1 mod p^(n-j)) · p^-k. gp
`1/(5+O(5^10))` = `5^-1 + O(5^8)`: unit 1 at relative precision 9, valuation -1. At HEAD `Field::inverse`
returns `Err(NotInvertible)` and `Div` (`padic_rational.rs:236-243`) panics on `unwrap` for such a divisor.
(For a non-prime p, `gcd(u', p) != 1` can happen; `inverse` then returns `Err(NotImplemented)` and `/` panics with
that message; E.2.)

`newton_lift::padic_to_int` (`newton_lift.rs:349`, `Result`-valued): an unbound c returns `Ok(c.modulo(m))` (c is
an exact integer of Z_p); bound elements as at HEAD. At HEAD it reads `valuation()`, which would panic.

#### A.3.3 `PadicExtensionElement` and `PadicExtension` (extension.rs)

Field change: `extension: Arc<PadicExtension>` becomes `extension: Option<Arc<PadicExtension>>`; `None` is the
unbound sentinel, whose `coefficients` is the one-element vector `[PadicRational::unbound(c)]`.

* `pub fn extension(&self) -> &Arc<PadicExtension>`: panics for an unbound element. Every internal use
  (`extension.rs:541-790`, 38 sites) goes through a private `fn ext(&self, op: &str) -> &Arc<PadicExtension>`
  with the same panic message.
* `coefficients()` of an unbound element is that one-element slice (documented, like `GaloisRingElement::eltseq`).
* `new`, `zero(ext)`, `one(ext)`, `from_base_field`, `generator` stay bound constructors. `new(ext, coeffs)` and
  `from_base_field(ext, value)` bind every unbound coefficient with
  `PadicRational::bind_integer(c, ext.prime(), ext.precision())` (review M3; at HEAD they store whatever they get,
  so an unbound coefficient would reach `Mul` and panic there).
* An unbound element acts as "the unbound `PadicRational` constant times 1", so the coercions delegate to
  `PadicRational`'s: `x + c` = x with `coefficients[0] = coefficients[0] + c` (c == 0 leaves x unchanged, so a
  coefficient of positive valuation keeps its precision); `x - c`, `c - x` likewise; `x * c` = every
  coefficient times c (Q_p-linear scaling, exact for c = ±1), and the unbound 0 when c == 0 (A.3); `x == c` iff
  `coefficients[0] == c` (A.3.4) and every other coefficient `is_zero()`.
* Bound `==`: `same_extension` and coefficient-wise `PadicRational ==` (now value equality), so in Q_5(√5) the
  product g·g, stored as (5, 0) by `Mul` (`extension.rs:709-781`), equals `from_base_field(5)` = (1, 1).
* `is_one` (bound): `coefficients[0].is_one()` (value; false when that coefficient is weakly zero) and the others
  `is_zero()`. So `is_zero` and `is_one` never both hold. HEAD (`extension.rs:807-811`) requires unit 1 at
  valuation 0, so (g·g)·(1/5), whose constant coefficient is (5, -1), is not one at HEAD.
* `norm`/`trace` of an unbound element: `Err(InvalidArgument)` (they depend on the degree).
* `PadicEmbedding::apply` (`extension.rs:909`): an unbound element is first bound into `self.source`
  (`from_base_field(source, PadicRational::bind_integer(c, p, prec))`), then mapped.
* `pub fn inverse(&self) -> Result<PadicExtensionElement>`: E.6.

`PadicExtension` gains three private fields, set in `PadicExtension::new` (the only constructor; the struct
literal at `extension.rs:257` is the only one in the workspace), and public accessors. `same_extension` keeps
ignoring them (they are functions of the fields it compares):

```rust
/// How `PadicExtension::new` proved that K = Q_p[x]/(f) is a field, which also fixes the valuation of K in the
/// power basis 1, g, ..., g^(d-1) (E.6). Computed from f and p only; the `ExtensionType` label is not trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldCertificate {
    /// deg f = 1: K = Q_p and v_K = v_p.
    Linear,
    /// f is Eisenstein at p (`is_eisenstein`, extension.rs:327): g is a uniformizer, e = d = deg f, and
    /// v_K(Σ c_i g^i) = min_i (d·v_p(c_i) + i), with v_K(p) = d.
    Eisenstein,
    /// lc(f) is a p-adic unit and f mod p, made monic by lc(f)^-1 mod p, is irreducible over F_p
    /// (`poly_arith::is_irreducible_mod_p`, poly_arith.rs:308): K/Q_p is unramified, O_K = Z_p[g], and
    /// v_K(Σ c_i g^i) = min_i v_p(c_i), with v_K(p) = 1.
    IrreducibleReduction,
}

impl PadicExtension {
    /// `Some(_)` iff `proven_primality(p) == Some(true)` and one of the three shapes holds (checked in the order
    /// Linear, Eisenstein, IrreducibleReduction).
    pub fn field_certificate(&self) -> Option<FieldCertificate>;
    /// `field_certificate().is_some()`.
    pub fn is_certified_field(&self) -> bool;
    /// f is squarefree over Q (`UnivariatePolynomial::<Integer>::is_square_free`, computed in `new`): then
    /// K = ∏ over the distinct irreducible factors is reduced, for every p > 1.
    pub fn is_reduced(&self) -> bool;
    /// Whether K is a field (MAGMA requires f irreducible for `ext< L | f >`, Ch. 47 §47.4; here it is decided):
    /// 1. `field_certificate()` is `Some` ⇒ `Some(true)`;
    /// 2. f not squarefree ⇒ `Some(false)` (f = g^2 h gives the nonzero nilpotent (g h)(α), see below);
    /// 3. `padic_base_is_prime_power(p) == Some(false)` ⇒ `Some(false)` (Q_p ≅ ∏_{q|p} Q_q, so K is a product of
    ///    at least two nonzero rings);
    /// 4. deg f = 1 ⇒ `padic_base_is_prime_power(p)` (K = Q_p);
    /// 5. p proven prime and p < 2^31 ⇒ `om_factorization(F, p, 1)` (om_factorization.rs:244) on the monic integral
    ///    transform F(y) = lc^(d-1) f(y / lc) (K ≅ Q_p[y]/(F) by y = lc·g): `Ok(fac)` ⇒ `Some(fac.is_irreducible())`,
    ///    `Err` ⇒ `None`;
    /// 6. otherwise `None`.
    /// Computed once, on first use, and cached in a `std::sync::OnceLock<Option<bool>>` field.
    pub fn field_status(&self) -> Option<bool>;
}
```

Correctness of the three certificates: Eisenstein's criterion (f/lc is a monic Eisenstein polynomial over Z_p) and,
for the irreducible reduction, a factorisation over Q_p descends by Gauss's lemma (Z_p is a UFD) to Z_p[x] with
unit leading coefficients, whose reduction would factor f̄ (Neukirch, *Algebraic Number Theory*, II.(5.9)); a
linear f is irreducible. Each proves f irreducible over Q_p, so K is a field. The valuation formulas: for an
Eisenstein f the root g is a uniformizer, v_K(p) = d, and the terms c_i g^i have valuations d·v_p(c_i) + i, which
are distinct mod d for 0 <= i < d, so the minimum is attained once (Serre, *Local Fields*, Ch. I §6, totally
ramified extensions). For an irreducible reduction, scale by p^-m with m = min v_p(c_i): the reduction
Σ c̄_i ḡ^i is nonzero because 1, ḡ, ..., ḡ^(d-1) is a basis of F_p[x]/(f̄), so the scaled element is a unit
(Serre, *Local Fields*, Ch. III §5, unramified extensions). The label of `new` (`ExtensionType`) is not used: an
Eisenstein polynomial passed with the label `Unramified` still gets `Eisenstein`.

Correctness of `field_status` steps 2 to 5: (2) if g^2 | f with deg g >= 1, write f = g^2 h; then (g h)(α) is
nonzero (degree < deg f) and its square is f(α)·h(α) = 0. (3) and (4): Z/p^n ≅ ∏ Z/q^(a_q n) gives
Z_p ≅ ∏_{q|p} Z_q and Q_p = Z_p[1/p] ≅ ∏ Q_q; K = Q_p[x]/(f) ≅ ∏ Q_q[x]/(f), each factor nonzero. (5) the factor
count of `om_factorization` is its DECIDED data (MacLane's completeness criterion Σ e_i f_i = deg F, re-checked
in the module; om_factorization.rs:1-30); F is monic, integral and squarefree (F is squarefree iff f is).

#### A.3.4 Why value equality is part of item A

Generic code compares with `==` (EC `Point` equality, `is_on_curve`'s `lhs == rhs`, `p == q` in `add_points`).
With the sentinel, `R::zero() == x` must be true for every x that is zero to its precision, and `R::one() + x`
must equal x + 1 as built from `from_rational`. Under HEAD's structural `PartialEq` (`padic_rational.rs:114-118`)
a zero at valuation 2 (from `(3 - 3)·25`) is unequal to the bound zero (0, 0), and the binding of 5, whether
normalised (1, 1) or not (5, 0), is unequal to the other form. So the sentinel would compare wrongly unless
equality compares values. One rule serves all three types and the sentinel: values agree to the common known
precision, an unbound constant being exact.

### A.4 Behaviour changes and callers

| Change | Callers (grep) | Effect |
|---|---|---|
| `Ring::zero()`/`one()` no longer panic | A.2 list | The six schemes tests can run (A.5). |
| `x.pow(0)` returns a bound 1 | `Ring::pow` users on p-adic types: none in the workspace besides new tests | was a panic |
| The unbound 0 times anything is the unbound 0 | generic code only (EC `b2`, `b6`, `a1·x·y` in `is_on_curve`) | exact zero instead of a bound 0 + O(p^n); no accessor is read on these values in the workspace |
| `PadicInteger ==`: common precision instead of identical precision | `PartialEq` users: `PadicRational` (its unit), tests. No `Hash`, no map keys (`derive(Clone, Debug)` only). `grep assert_eq/assert_ne` in `padics/`: none compares `PadicInteger`s of different precision | 1 + O(5^10) == 1 + O(5^5) is now true |
| `PadicRational ==`, `is_one`: value; `is_one` false when weakly zero | `PadicExtensionElement ==` (:660), `Vec<PadicRational> ==` in `extension.rs` tests :1454, :1639, :1643, Galois-group tests :1479-1521 (`assert_ne!(sg, g)`: conjugates differ in the first digits, so they stay unequal; review 1 recomputed this on HEAD values), EC `Point ==` | tracker row 1108 fixed |
| `PadicRational` inverse via `normalized()`; `Field::divide` override | `Field::inverse`/`divide` users: `EllipticCurve::j_invariant`, `add_points`, `double_point` (`divide`), `extension.rs:767` (`/ leading`) | nonzero non-normalised divisors no longer fail or panic; an unbound divisor is bound instead of giving `Err` |
| `from_rational(0)` valuation 0 | `factory.rs:300-306`, `newton_lift.rs:537` (never with 0: comment :689), tests | zero is 0 + O(p^n), not O(p^(n-1)) |
| `PadicIntegerRing::adjust_precision` binds an unbound element | `factory.rs:193`; its tests :717, :726 (bound elements, unchanged) | was a panic on `valuation()` |
| `PadicExtensionElement::new`, `from_base_field` bind unbound coefficients | `extension.rs:489`, `:528`; callers in extension.rs (`zero`, `one`, `generator`, `Mul`, `norm`, GaloisGroup) pass bound coefficients, unchanged | an unbound coefficient becomes bound at the extension's precision |
| `newton_lift::padic_to_int` accepts an unbound c | `newton_lift.rs:349`, callers :368, :643, :657, :770 (bound coefficients from the lift, unchanged) | c mod m instead of a panic |
| `PadicExtensionElement.extension` is an `Option` (private field) | only extension.rs | public API unchanged except `coefficients()` of an unbound element |
| New `PadicExtension` fields and methods | `PadicExtension::new` only | `new` now runs `is_eisenstein`, one Rabin test and one squarefree test (A.6) |
| `PadicExtensionElement::inverse` (new) | new | E.6 |
| new panics | only on unbound elements, which did not exist before | |

### A.5 Tests (rustmath-rings, inline `mod tests`; all fail or panic on HEAD unless marked)

References: `ref1.gp`, `ref4.gp`, `rev1/*.gp` (PARI/GP 2.15), Sage docs quoted in 0.1.

`padic_integer.rs`:
* `ring_zero_one_are_unbound`: `<PadicInteger as Ring>::zero()` and `one()` do not panic, `!is_bound()`,
  `is_zero()`/`is_one()` true; `Ring::zero() == PadicInteger::zero(5, 4)?` and the reverse.
* `generic_horner_through_sentinels`: `fn horner<R: Ring>(cs: &[R], x: &R) -> R` starting from `R::zero()`, with
  `cs = [R::one(), R::one() + R::one(), 3 + O(5^4)]`, x = 7 + O(5^4): result `162 + O(5^4)` (gp
  `(1+2*7+3*7^2) % 5^4` = 162).
* `pow_zero_is_bound_one`: `(7 + O(5^4)).pow(0)` is bound, equals `PadicInteger::one(5, 4)`, precision 4.
* `equality_common_precision`: `1 + O(5^10) == 1 + O(5^5)` true; `1 + O(5^10) == 26 + O(5^5)` false (gp
  `Mod(26,5^5) == Mod(1,5^5)` = 0); different primes false.
* `zero_is_exact_in_sums_and_products` (M12): with x = 7 + O(5^4), `Ring::zero() * x` and `x * Ring::zero()` are
  unbound and `is_zero()`; `x + Ring::zero()` equals x with precision 4.
* `adjust_precision_binds_unbound` (M3): for each `PrecisionModel`, `PadicIntegerRing::new(5, 20, model)?
  .adjust_precision(Ring::one())` is bound, prime 5, precision 20, `is_one()`. HEAD: panic.
* `unbound_accessors_refuse` (`#[should_panic]` each): `prime()`, `precision()`, `valuation()`, `residue()` of
  `Ring::one()`; `Ring::one().lift(5)` is `Err(InvalidArgument)`; `Ring::one().inverse()` is `Ok(one)`;
  `Ring::zero().inverse()` is `Err(NotInvertible)`.

`padic_rational.rs`:
* `sum_one_plus_four_equals_five`: `(1 + O(5^10)) + (4 + O(5^10)) == from_rational(5)` (gp: 1). HEAD: false.
* `zero_at_valuation_two_equals_ring_zero`: `(3 - 3) * 25` (stored (0, 2)) `== Ring::zero()` and reverse;
  `Ring::zero() + from_rational(25)` keeps valuation 2.
* `ring_one_binds_with_full_precision`: `Ring::one() + from_rational(7/25) == from_rational(32/25)`.
* `is_one_unnormalised`: `from_padic_integer(5 + O(5^10)) * from_rational(1/5)` (stored (5, -1)) `is_one()` (gp
  `(1/5)*(5+O(5^10)) == 1` = 1). HEAD: false.
* `weakly_zero_is_not_one` (M6): x = `from_rational(1/5, 5, 1) - from_rational(1/5, 5, 1)` (stored (0, -1, 1),
  absolute precision 0): `is_zero()` true, `is_one()` false, `unit_status() == Some(false)`, order `NotAUnit`;
  `x == Ring::one()` true (weak equality, documented).
* `equality_is_not_fooled_by_add_precision_loss`: `0 + O(5^10)` (from `from_padic_integer`) vs
  `5^5 + O(5^8)` (unit 1 at precision 3, valuation 5): `!=` (the prototype's case).
* `inverse_of_unnormalised_element`: `Field::inverse(5 + O(5^10))` is `5^-1 + O(5^8)`: unit value 1,
  valuation -1, precision 9 (gp `1/(5+O(5^10))`); `x / (1 + 4)` does not panic. HEAD: `Err(NotInvertible)` and
  a panic.
* `divide_binds_unbound` (M1): two = `Ring::one() + Ring::one()` (unbound 2), x = `from_rational(3)`:
  `x.divide(&two)? == from_rational(3/2)` and `== x / two` (gp `3/2` = `3/2`, a unit of Z_5); `x.divide(&Ring::zero())`
  is `Err(DivisionByZero)`; `Ring::zero().divide(&x)?` is unbound and zero; `(two * two).divide(&two)? == two` and
  unbound; `Ring::one().divide(&two)` is `Err(InvalidArgument)`. `#[should_panic(expected = "bind an operand")]`:
  `Ring::one() / two`. HEAD: panics in `Ring::one()`.
* `from_rational_zero_has_valuation_zero`: valuation 0, `absolute_precision() == 10`. HEAD: -1.
* `unbound_rational_refuses`: `#[should_panic]` on `valuation()`, `prime()`, `abs()` of `Ring::one()`;
  `Ring::one().lift(5)` is `Err`.

`newton_lift.rs` (its `mod tests`): `padic_to_int_of_unbound`: three = `R::one() + R::one() + R::one()`;
`padic_to_int(&three, &25, &5) == Ok(3)`; `padic_to_int(&-three, &25, &5) == Ok(22)` (gp `(-3) % 25` = 22).

`extension.rs` (Q_5(√5) = `eisenstein(5, x^2 - 5, 10)`):
* `ring_constants_bind`: `Ring::zero() + g == g` with `coefficients()` equal **and** every coefficient's
  `valuation()`/`precision()` unchanged; `g * Ring::one() == g`; `Ring::one() + g == one(ext) + g`;
  `Ring::zero() == zero(ext)`; `g * Ring::zero()` is unbound.
* `new_binds_unbound_coefficients` (M3): `PadicExtensionElement::new(ext, vec![Ring::one(), Ring::zero()])` has
  both coefficients bound (prime 5, precision 10) and equals `one(ext)`; `from_base_field(ext, Ring::one()) ==
  one(ext)`.
* `product_of_generators_is_five`: `g * g == from_base_field(from_rational(5))` (gp: `x^2-5` defines √5,
  √5·√5 = 5). HEAD: false.
* `one_after_dividing_by_five_is_one`: `(g * g) * from_base_field(from_rational(1/5))` `is_one()`. HEAD: false.
* `unbound_extension_element_refuses`: `#[should_panic]` on `Ring::one().extension()`; `Ring::one().norm()` is
  `Err(InvalidArgument)`.
* `embedding_binds_unbound`: `PadicEmbedding::into_algebraic_closure(ext).apply(&Ring::one())` is `Ok` and
  `is_one()`.
* `field_certificates`: `Eisenstein` for `eisenstein(5, x^2-5)`; `IrreducibleReduction` for `unramified(5, 2)`
  (modulus x^2 + 4x + 2, the Conway polynomial, read back in `rev1/proto_ext_inverse.log`); `Linear` for
  `new(5, 5x - 1, 10, Unramified)`; `Eisenstein` also for `new(5, x^2 - 5, 10, Unramified)` (the label is ignored);
  `None` for `new(5, x^2 - 1, ..)`, `new(5, x^2 - 125, ..)` and `new(6, x^2 - 5, ..)`.
* `field_status_decides`: `Some(true)` for the three certified fields and for `new(5, x^2 - 125, 10, ..)` (OM: one
  factor, ram (2, 1), `rev1/om_cost.log`; gp `factorpadic(x^2-125,5,10)` has one factor of degree 2);
  `Some(false)` for `new(5, x^2 - 1, ..)` (OM two factors; gp `factorpadic(x^2-1,5,10)` two factors),
  `new(7, x^2 - 2, ..)` (gp two factors), `new(5, x^2 - 2x + 1, ..)` (not squarefree; gp `issquarefree((x-1)^2)` =
  0) and `new(6, x^2 - 5, ..)` (6 = 2·3); `None` for `new(4, x^2 - 5, ..)` (4 is a prime power but not prime).
  `is_reduced()`: false only for the `(x-1)^2` case (`UnivariatePolynomial::is_square_free` on HEAD: `rev1/sqfree.log`).
* Inverse tests: E.5.

Downstream (rustmath-schemes, `src/elliptic_curves/tests.rs`; test-only change in a crate outside this unit's
list, recommended as acceptance evidence): remove `#[ignore]` from the six tests of A.2 and replace the comment at
:333-338 by one sentence citing the sentinel; add

* `test_padic_doubling_matches_pari`: E: y^2 = x^3 + 17 over Q_5 (`short_weierstrass(qp.from_int(0),
  qp.from_int(17))`, `qp = PadicField::new(5, 10, CappedRelative)`), P = (52, 375) built with `from_rational`
  (375 = 3·5^3 has valuation 3): `is_on_curve(P)`; `2P = double_point(P)`, whose x equals
  `from_rational(202904/15625)` and differs from `from_rational((202904 + 5^3·15625)/15625)`; `is_on_curve(2P)`.
  Reference: gp `ellmul(ellinit([0,17]),[52,375],2)` = `[202904/15625, 91751917/1953125]`, x has valuation -6,
  and the computed x is known to absolute precision 4 (relative precision 10 at valuation -6), matching
  gp's `P2[1] + O(5^4)`. Also `(-2,3) + (-1,4) == (4,-9)` (gp `elladd`). The prototype prints all of these
  as true; HEAD panics in `short_weierstrass`.

### A.6 Cost

Unbound arithmetic is integer arithmetic. Value equality of `PadicRational` builds two rationals and two
valuations (O(M(log p^n)) each), as cheap as a subtraction. `PadicExtension::new` adds `proven_primality(p)` (below
psi_12 one pass of the strong tests; above it one ECPP proof that I1 caches), `is_eisenstein` (O(d)), Rabin's test
(O(d^3 log p) word operations) and one squarefree test (a gcd with f' over Z). `field_status` runs OM only for
extensions without a certificate, once: 0.2 to 46 ms in debug for the ten polynomials of `rev1/om_cost.log`
(degree <= 6).

---

## B. `RingMorphismMethods::is_nonzero` reads the static `Self::Domain::one()`

### B.1 Defect at HEAD

`rustmath-rings/src/category_methods.rs:468-470`: `!self.apply(&Self::Domain::one()).is_zero()`. The doc
(:461-467) admits that it panics for a p-adic domain and that for sentinel domains `apply` receives an unbound 1.
`e2_morph_cat.rs`: `ReduceZ5` (Z_5 → Z/5, `apply` reads `x.residue()` and `x.prime()`) panics in `is_nonzero`.
After A it would no longer panic, but `apply` would get the unbound 1 and `x.prime()` would panic instead. The
static constructor is the wrong source for the domain's 1 whenever the domain has runtime data.

### B.2 Fix

```rust
pub trait RingMorphismMethods {
    type Domain: Ring;
    type Codomain: Ring;
    fn apply(&self, x: &Self::Domain) -> Self::Codomain;

    /// The multiplicative identity of the domain ring (MAGMA `One(Domain(f))`, Ch. 17 §17.5.2), when this
    /// morphism knows it. Default `None`: the trait sees only the element type, and the static
    /// `Self::Domain::one()` is the domain's 1 only for parent-free types (Z, Q, R, C); for Z/mZ, GF(q),
    /// Galois rings and the p-adic types it is an unbound integer constant. A morphism overrides this with its
    /// domain's 1 (`Some(Integer::one())` for Z, `Some(r.one())` for a parent `r`).
    fn domain_one(&self) -> Option<Self::Domain> { None }

    /// Whether this map is not the zero map: `Some(true)` proven nonzero, `Some(false)` proven zero, `None`
    /// undecided. A ring homomorphism φ is zero iff φ(1) = 0 (φ(x) = φ(x)·φ(1)); for a unital φ that happens
    /// iff the codomain is the zero ring (Sage `Rings.MorphismMethods.is_injective`: a map from a field is
    /// injective "unless the codomain is the zero ring"). Default: `apply` on `domain_one()`; `None` (and no
    /// call to `apply`) when `domain_one()` is `None`. The answer is a proof when `apply` is a ring homomorphism.
    fn is_nonzero(&self) -> Option<bool> {
        self.domain_one().map(|one| !self.apply(&one).is_zero())
    }
    // is_injective, is_surjective, is_isomorphism unchanged
}
```

Signature change `bool` → `Option<bool>`; correctness: φ(1) is computed on the true 1 of the domain, so
`φ(1) = 0` is decided exactly by the codomain's `is_zero`.

### B.3 Callers

`grep -rn "\.is_nonzero()"`: `category_methods.rs:1121` and `:1165` (tests). Implementors of
`RingMorphismMethods`: only test structs in `category_methods.rs` (`DoubleMap` :1100, `IdZ` :1129,
`ReduceMod5` :1139, `Bijective` :1170). `rustmath_rings::morphism::RingMorphism` (`morphism.rs:75`) has no
`is_nonzero`. No other crate uses the trait.

### B.4 Tests (`category_methods.rs`)

* `test_ring_morphism_basic`: `DoubleMap` gains `domain_one() = Some(1)`; `is_nonzero() == Some(true)`.
* `test_ring_morphism_defaults_and_kleene`: `IdZ` gains `Some(Integer::one())`; `Some(true)`.
* `test_is_nonzero_without_domain_one_is_undecided` (new): a morphism whose `apply` is `panic!` and no
  `domain_one`: `is_nonzero() == None` and no panic (HEAD: panics in `apply`).
* `test_is_nonzero_on_padic_domain` (new): `ReduceZ5` (`PadicInteger` → `IntegerMod`, `fn domain_one(&self) ->
  Option<PadicInteger> { PadicInteger::one(Integer::from(5), 10).ok() }`): `Some(true)` (PARI `Mod(1,5) != 0`).
  HEAD: panic.
* `test_is_nonzero_to_zero_ring` (new): Z → a test-local zero-ring element type (`ZeroElt`, `is_zero()` always
  true, `zero() = one()`): `Some(false)` (Sage: `QQ.hom([Zmod(1)(0)], check=False).is_injective()` is False, the
  zero map).

---

## C. `RingParentMethods` for the real parent types

### C.1 Defect at HEAD

`grep -rn "RingParentMethods for\|RingMorphismMethods for"`: every hit is inside a `#[cfg(test)]` module
(`rustmath-category/src/ring_category.rs:583-646`, `rustmath-rings/src/category_methods.rs:620-1170`). No
crate-root parent implements either trait, so MAGMA `IsField`, `IsDomain`, `Characteristic`, `IsCommutative`
(Ch. 17 §17.4.2-3) have no generic entry point (status_overrides `ch017.IsCommutative` is `missing`,
`ch047.Characteristic` `missing`). There is no parent type at all for Z, Q or R[x] (grep `struct IntegerRing`,
`struct RationalField`, `struct PolynomialRing`: only test structs and `rustmath-algebras/src/cached_algebra.rs`,
a cache key). `rustmath-rings/src/abc.rs:31, :37` already use the names `IntegerRing`, `RationalField` for
element marker traits, hence the `...Parent` names below.

The core `Parent` impls in the workspace (`grep "impl.*Parent for"`, non-test): `Zmod`, `FiniteField`, `GaloisRing`
(rustmath-finitefields), `MatrixAlgebra<F>` and `Lattice` (rustmath-matrix), `BigFloatField` (rustmath-reals),
`DownUpAlgebra<R>` and `SimpleAlgebra<R>` (rustmath-algebras), five module parents (rustmath-modules) and five in
rustmath-manifolds (`DiffScalarFieldAlgebra` and four modules). The rings among them are covered below or in F.2.

### C.2 New file `rustmath-rings/src/category_parents.rs` (`pub mod category_parents;` in lib.rs)

Header: MAGMA Ch. 17 §17.4.2-3 (Characteristic, IsCommutative, IsFinite, IsField, IsDomain), Ch. 18 (Z),
Ch. 19 (Z/mZ), Ch. 20 (Q), Ch. 21 (GF(q)), Ch. 23 (R[x]), Ch. 47 (p-adics), Ch. 48 (Galois rings); Atiyah and
Macdonald, *Introduction to Commutative Algebra* (Thm 7.5 Hilbert basis, Prop. 6.5, Prop. 7.1, Thm 8.5, Ex. 1.2,
5.9, 11.7); Matsumura, *Commutative Ring Theory*, Thm 15.4; Serre, *Local Fields*, Ch. II §1 (Z_p is a complete DVR).

```rust
/// Z (MAGMA `Integers()`, Ch. 18).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntegerRingParent;
/// Q (MAGMA `Rationals()`, Ch. 20).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RationalFieldParent;
/// R[x] over a base parent (MAGMA `PolynomialRing(R)`, Ch. 23); elements `UnivariatePolynomial<P::Element>`.
#[derive(Clone, Debug)]
pub struct PolynomialRingParent<P: RingParentMethods> { base: P }
impl<P: RingParentMethods> PolynomialRingParent<P> {
    pub fn new(base: P) -> Self;
    pub fn base(&self) -> &P;
}
```

Overrides only; every method not listed keeps the default. Calls go to inherent methods with explicit paths
(`Zmod::characteristic(self)`) because the inherent methods share names (`characteristic`, `order`,
`is_field`, `is_prime_field`) and win at concrete call sites; generic code reaches the trait.

Shorthand: P(n) = `proven_primality(n)`; F(p) = `Some(true)` if P(p) == `Some(true)`, else `None`;
D(p) = `padic_base_is_prime_power(p)`; FS = `ext.field_status()`.

| Parent (Element) | char | order | zero ring | finite | comm. | domain | field | prime field | int. closed | Noetherian | Krull |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `IntegerRingParent` (`Integer`) | `Some(0)` | default `None` | `Some(false)` | default `Some(false)` | `Some(true)` | `Some(true)` | `Some(false)` | default `Some(false)` | `Some(true)` | `Some(true)` | `Some(1)` |
| `RationalFieldParent` (`Rational`) | `Some(0)` | | `Some(false)` | default | `Some(true)` | `Some(true)` | `Some(true)` | `Some(true)` | default `Some(true)` | default `Some(true)` | `Some(0)` |
| `Zmod` (`IntegerMod`) | `m.to_u64()` | `m.to_usize()` | `Some(false)` | `Some(true)` | `Some(true)` | P(m) | P(m) | P(m) | `Some(true)` if P(m) = `Some(true)`, else `None` | `Some(true)` | `Some(0)` |
| `FiniteField` (`FiniteFieldElement`) | `p.to_u64()` | `q.to_usize()` | `Some(false)` | `Some(true)` | `Some(true)` | F(p) | F(p) | `Some(degree == 1)` when F(p), else `None` | F(p) | `Some(true)` | `Some(0)` |
| `GaloisRing` GR(p^a, d) (`GaloisRingElement`) | `(p^a).to_u64()` | `p^(ad).to_usize()` | `Some(false)` | `Some(true)` | `Some(true)` | a > 1: `Some(false)`; a = 1: F(p) | same | a > 1: `Some(false)`; a = 1: `Some(d == 1)` when F(p) | a = 1: F(p); a > 1: `None` | `Some(true)` | `Some(0)` |
| `PadicIntegerRing` (`PadicInteger`) | `Some(0)` | | `Some(false)` | `Some(false)` | `Some(true)` | D(p) | `Some(false)` | default `Some(false)` | `Some(true)` if D(p) = `Some(true)`, else `None` | `Some(true)` | `Some(1)` |
| `PadicField` (`PadicRational`) | `Some(0)` | | `Some(false)` | `Some(false)` | `Some(true)` | D(p) | D(p) | `Some(false)` | default (field ⇒ `Some(true)`) | `Some(true)` | `Some(0)` |
| `PadicExtension` (`PadicExtensionElement`) | `Some(0)` | | `Some(false)` | `Some(false)` | `Some(true)` | FS | FS | `Some(false)` | default | `Some(true)` | `Some(0)` |
| `PolynomialRingParent<P>` | `base.characteristic()` | `Some(1)` if `base.is_zero_ring() == Some(true)`, else `None` | `base.is_zero_ring()` | `base.is_zero_ring()` | `base.is_commutative()` | `base.is_integral_domain()` | `Some(false)` | default `Some(false)` | `base.is_integrally_closed()` when `base.is_integral_domain() == Some(true)`, else `None` | `base.is_noetherian()` | `base.krull_dimension() + 1` when `base.is_noetherian() == Some(true)`, `base.is_zero_ring() == Some(false)` and `base.is_commutative() == Some(true)`, else `None` |

Correctness, row by row:
* Z: a PID, not a field (2 has no inverse), Noetherian and integrally closed (UFD), Krull dimension 1 (the chains
  0 ⊂ (p)). Q: a field, the prime field of characteristic 0, dimension 0. Both are nonzero.
* Z/m (m > 1, `Zmod::new`, zmod.rs:53): finite and nonzero, so Artinian of dimension 0 (A-M Thm 8.5); a domain
  iff a field iff m prime; it is its own prime field when it is a field. `characteristic`/`order` are `None` above
  u64/usize (no claim, as the trait requires), while the overrides read the exact `Integer` modulus. The
  `is_zero_ring` override (review M2) keeps R[x] over Z/m decided when m >= 2^64 (the default would read the two
  `None`s and answer `None`).
* GF(q): `FiniteField::new`/`with_modulus` (finite_field.rs:146, :184) reject a non-prime p and a reducible
  modulus (Rabin test). With I1 and F1 the primality check is a proof; at HEAD it is a probable-prime test above
  psi_12 (review M8). The rows therefore read P(p) themselves: `Some(true)` only when p is proven (with I1 the proof
  is cached, so the call is cheap), `None` otherwise. Prime field iff degree 1.
* GR(p^a, d): `GaloisRing::new` (galois_ring.rs:77) rejects a non-prime p (same remark); p is nilpotent and
  nonzero when a > 1 (not a domain), and GR(p, d) = GF(p^d).
* Z_p for every p > 1 (`PadicIntegerRing::new` accepts any p > 1, factory.rs:131-134): Z/p^n ≅ ∏_{q|p} Z/q^(a_q n)
  (CRT) and inverse limits commute with finite products, so Z_p ≅ ∏_{q|p} Z_q, a finite product of complete DVRs
  (Serre II §1). Hence: characteristic 0 (Sage `characteristic()` "always 0"), infinite, nonzero, commutative,
  reduced, Noetherian (a finite product of Noetherian rings), Krull dimension 1 (Spec of a finite product is the
  disjoint union of the factors' spectra), not a field (p is a nonzero non-unit: p·y ≡ 0 mod p); a domain iff
  exactly one prime divides p, which is D(p) (0.2); then it is a DVR, integrally closed. Not a domain: no claim on
  integral closure (the notion is for domains). Review M9.
* Q_p = Z_p[1/p] ≅ ∏_{q|p} Q_q: a field iff D(p); Noetherian, dimension 0; never Q (uncountable when a field), so
  never a prime field.
* K = Q_p[x]/(f): a finitely generated module over Q_p, which is Artinian (a finite product of fields), so K is an
  Artinian ring (A-M Prop. 6.5), hence Noetherian of dimension 0 (A-M Thm 8.5), for every f (review M9). A field
  iff FS (A.3.3); not Q, so not a prime field.
* R[x]: R ⊂ R[x] gives the characteristic and, with x central, commutativity iff R is; R[x] = 0 iff R = 0, and R[x]
  is finite iff R = 0; R[x] is a domain iff R is (leading coefficients); x is not a unit and R = 0 gives R[x] = 0,
  so R[x] is never a field; Noetherian iff R is (Hilbert basis theorem A-M 7.5, and R ≅ R[x]/(x), A-M 7.1); for a
  domain R, R[x] is integrally closed iff R is (A-M Ex. 5.9, and R = R[x] ∩ Frac(R)); for a commutative Noetherian
  R, dim R[x] = dim R + 1 (A-M Ex. 11.7, Matsumura 15.4; both are commutative statements, hence the third
  condition, review M9); for a non-Noetherian base the dimension is only bounded (Seidenberg), so `None`.

`PolynomialRingParent::order`, `is_finite` and `is_zero_ring` call only the base's methods, which the
dependency-order rule permits (it constrains calls to the same parent's methods).

### C.3 rustmath-category impls (`rustmath-category/src/ring_parents.rs`, `mod ring_parents;`)

`rustmath_category::RingParentMethods: Parent` (core) can be implemented for types with a core `Parent` impl.
New `[dependencies]` of rustmath-category: `rustmath-integers` (moved from dev-dependencies),
`rustmath-finitefields`, `rustmath-matrix`, `rustmath-reals`. All four are already in its build closure through
`rustmath-groups` (`cargo metadata`: rustmath-category → {combinatorics, complex, core, finitefields, graphs,
groups, integers, matrix, polynomials, powerseries, rationals, reals, typesetting}), and none of them depends on
rustmath-category (its reverse dependencies are rustmath-modules, rustmath-algebras with feature `full`, and
rustmath-jupyter), so there is no cycle and no new compile work.

For `Zmod`, `FiniteField`, `GaloisRing`: `characteristic` (u64 or `None`), `is_commutative = Some(true)`,
`is_integral_domain` and `is_field` as in C.2, `CommutativeRingParentMethods` with the defaults (field ⇒
Euclidean ⇒ PID ⇒ UFD; a non-domain is none of them).

`MatrixAlgebra<F: Field>` (`rustmath-matrix/src/matrix_algebra.rs:227`; the full M_n(F) or the unital subalgebra
generated by matrices; n = `degree()`, dim = `dimension()`; the full algebra is dim = n^2). MAGMA Ch. 83 (matrix
algebras) and Ch. 17 §17.4.3:

| | char | commutative | domain | field |
|---|---|---|---|---|
| n = 0 (the zero algebra) | `Some(1)` | `Some(true)` | `Some(false)` | `Some(false)` |
| dim = 1 (A = F·I ≅ F; I ∈ A because A is unital) | `F::static_characteristic()` | `Some(true)` | `Some(true)` | `Some(true)` |
| full, n >= 2 | `F::static_characteristic()` | `Some(false)`: E_12 E_21 = E_11 ≠ E_22 = E_21 E_12 | `Some(false)`: E_11 E_22 = 0 | `Some(false)` |
| other subalgebras | `F::static_characteristic()` | `None` | `None` | `None` |

The characteristic is F's because F·I ⊆ A. A proper subalgebra can be commutative or a field (Q(i) inside M_2(Q)),
and with an inexact F (reals, complex) equality of products is not a proof, so it makes no claim.
`CommutativeRingParentMethods` with the defaults.

`BigFloatField` (`rustmath-reals/src/bigfloat.rs:1129`): characteristic `Some(0)`, commutative, domain and field
`Some(true)` (MAGMA Ch. 25 §25.3.2-3: `Characteristic(R)` 0, `IsCommutative` true, `IsField` true, `IsDomain`
true; Sage `RealField(53).is_field()` True). `CommutativeRingParentMethods` defaults (a field is Euclidean, PID,
UFD; MAGMA §25.3.3 `IsEuclideanDomain` true, `IsPrincipalIdealRing` true).

### C.4 Number fields (rustmath-numberfields, new file `src/category.rs`, `pub mod category;`)

Review X2: the first version said the orphan rule closes the category trait; that was wrong. The rustmath-rings
trait needs `Element: Ring`, which `NumberFieldElement` is not (no `impl Ring` in rustmath-numberfields), but the
rustmath-category trait needs only core `Parent`, whose `Element: Clone + PartialEq` holds (`NumberFieldElement`
derives `Clone`, `lib.rs:82`, and implements `PartialEq`, `lib.rs:156`). The owning crate implements both.

New dependency: `rustmath-category = { path = "../rustmath-category" }` in rustmath-numberfields' `[dependencies]`.
No cycle: rustmath-category's closure (C.3) does not contain rustmath-numberfields. rustmath-category's own
dependencies are already dependencies of rustmath-numberfields (both closures are {combinatorics, complex, core,
finitefields, graphs, groups, integers, matrix, polynomials, powerseries, rationals, reals, typesetting}), so the
crates that depend on rustmath-numberfields (affineschemes, curves, galois, igp24, jupyter, numerical, padics,
quantumgroups, rings, schemes) gain one compile unit, rustmath-category itself.

Header: MAGMA Ch. 34 §34.2.1 (`NumberField(f)`, `Check := true`: f irreducible), §34.3.4 (`Characteristic` 0),
§34.3.6 (`IsCommutative`, `IsFinite` false, `IsField`), Ch. 17 §17.4.2-3; the Zassenhaus factoriser (Zassenhaus
1969; Cohen GTM 138 §3.5).

```rust
impl NumberField {
    /// Whether the defining polynomial is irreducible over Q (PARI `polisirreducible`; MAGMA's `Check` in
    /// `NumberField(f)`). `NumberField::new` does not check it (lib.rs:222). The rational polynomial is scaled by
    /// the lcm of its denominators to an integer polynomial (the content does not matter), then
    /// `rustmath_polynomials::is_irreducible_over_integers` (complete Zassenhaus factorisation) decides:
    /// `Ok(b)` gives `Some(b)`, `Err` (recombination bound exceeded) gives `None`.
    pub fn defining_polynomial_is_irreducible(&self) -> Option<bool>;
}

impl rustmath_core::Parent for NumberField {
    type Element = NumberFieldElement;
    /// The element belongs to this field: same minimal polynomial (`x.min_poly == self.minimal_polynomial`).
    fn contains(&self, x: &NumberFieldElement) -> bool;
    fn zero(&self) -> Option<NumberFieldElement> { Some(NumberField::zero(self)) }
    fn one(&self) -> Option<NumberFieldElement> { Some(NumberField::one(self)) }
    /// Characteristic 0 and nonzero (deg f >= 1), so infinite (MAGMA §34.3.6: `IsFinite` false).
    fn is_finite(&self) -> Option<bool> { Some(false) }
    fn name(&self) -> String; // "Number field defined by <f>"
}

impl rustmath_category::RingParentMethods for NumberField {
    fn characteristic(&self) -> Option<u64> { Some(0) }        // Q ⊂ K
    fn is_commutative(&self) -> Option<bool> { Some(true) }    // a quotient of Q[x]
    fn is_integral_domain(&self) -> Option<bool> { self.defining_polynomial_is_irreducible() }
    fn is_field(&self) -> Option<bool> { self.defining_polynomial_is_irreducible() }
}
impl rustmath_category::CommutativeRingParentMethods for NumberField {}
```

Correctness: K = Q[x]/(f), deg f >= 1. If f is irreducible over Q, (f) is maximal and K is a field (so a domain,
Euclidean, PID, UFD). If f = gh with deg g, deg h >= 1, then g(α)h(α) = 0 with g(α), h(α) nonzero (degrees < deg f),
so K is not a domain and not a field. The content of f is a unit of Q and does not change K, which is why the
"content ignored" convention of `is_irreducible_over_integers` (`2x + 2` gives true, factorization.rs:461) is the
right one. The private fields `min_poly` and `minimal_polynomial` are visible from the child module `category`.

`nf_elem::NumberField` (`nf_elem.rs:156`) is a second, internal struct: F.2.

### C.5 Down-up algebras (rustmath-algebras, in `src/algebra_with_parent.rs` next to its `Parent` impl)

rustmath-algebras depends on rustmath-category only through the optional rustmath-modules (feature `full`), so it
gains `rustmath-category = { path = "../rustmath-category" }` in `[dependencies]`. No cycle: rustmath-category does
not depend on rustmath-algebras or rustmath-modules. rustmath-category cannot host this impl instead: it would need
rustmath-algebras, which with feature `full` depends on rustmath-modules → rustmath-category (a cycle).

```rust
impl<R: Ring> rustmath_category::RingParentMethods for DownUpAlgebra<R> {
    /// The zero ring over a zero base; else the characteristic of R (R·1 ⊂ DU, and 1 is a basis monomial).
    fn characteristic(&self) -> Option<u64> {
        if self.base_ring_is_zero() { Some(1) } else { R::static_characteristic() }
    }
    /// Over a nonzero base du and ud are distinct basis monomials ((du)^1 and u^1 d^1 of the normal form
    /// u^i (du)^j d^k, Benkart and Roby, "Down-up algebras", J. Algebra 209 (1998); the crate's
    /// `DownUpBasisIndex`), so DU is not commutative.
    fn is_commutative(&self) -> Option<bool> { Some(self.base_ring_is_zero()) }
    /// `IsDomain` requires commutativity (and 1 != 0), so never.
    fn is_integral_domain(&self) -> Option<bool> { Some(false) }
}
impl<R: Ring> rustmath_category::CommutativeRingParentMethods for DownUpAlgebra<R> {}
```

`base_ring_is_zero` (algebra_with_parent.rs:65) is the existing private helper of the same module (it reads the
stored α; no static constructor). `is_field` keeps the default (`Some(false)`).

### C.6 Behaviour changes and callers

Additions only. Existing calls such as `zmod.is_field()` or `ff.order()` keep resolving to the inherent methods,
and `NumberField::zero(&self)` keeps resolving to the inherent method (it has priority over `Parent::zero`).
`grep -rn "RingParentMethods"` outside the two defining files: `rustmath-category/src/lib.rs:96` (re-export),
nothing else. The test wrappers `GrParent`, `ZmodParent`, `FqParent` (category_methods.rs:680-730) stay; their
values must agree with the real impls (a new test asserts it).

### C.7 Tests

`category_parents.rs` (all new; HEAD has no such impls). References: GAP 4.12 `ref3.g`, `rev1/rev1.g` (logs next to
them), PARI.
* `integer_ring_parent`: GAP `[Characteristic, IsCommutative, IsIntegralRing, IsField, IsFinite](Integers)` =
  `[0, true, true, false, false]`; plus zero ring `Some(false)`, prime field `Some(false)`, integrally closed,
  Noetherian, Krull 1.
* `rational_field_parent`: GAP `[0, true, true, true, false]`, `IsPrimeField(Rationals)` = true; Krull 0.
* `zmod_parents`: n = 2, 3, 4, 6, 7, 8 as the existing `test_zmod_parents` (GAP sizes and IsField); n = 2^64 + 13:
  `characteristic() == None`, `order() == None`, `is_zero_ring() == Some(false)`, `is_finite() == Some(true)`,
  `is_field() == Some(true)` (GAP `IsField(ZmodnZ(2^64+13))` = true, `IsPrime(2^64+13)` = true); n = 3·2^64:
  `is_field() == Some(false)` (GAP false); Krull 0.
* `finite_field_parents`: GF(25): GAP `[5, true, true, true, true]`, `IsPrimeField` false; GF(7) prime field true;
  GF(2^64 + 13): char `None`, field `Some(true)`, prime field `Some(true)`, zero ring `Some(false)`.
* `galois_ring_parents`: GR(8, 2) (MAGMA H48E1): char 8, order 64, domain/field `Some(false)` (PARI
  `Mod(2,8)^3 == 0`), Noetherian, Krull 0, integrally closed `None`; GR(5, 3): field `Some(true)`, prime field
  `Some(false)`; GR(7, 1): prime field `Some(true)`.
* `padic_parents`: `PadicIntegerRing::new(5, 20, CappedRelative)`: char 0 (Sage `Zp(3,10,'fixed-mod')
  .characteristic()` = 0), zero ring `Some(false)`, finite `Some(false)`, field `Some(false)`, domain, integrally
  closed, Noetherian, Krull 1; `PadicField::new(5, 20, ..)`: field, prime field `Some(false)`, Krull 0.
  p = 6 (accepted by both constructors): Z_6 domain `Some(false)` (gp `rev1c.log`: e = 24151041 =
  `lift(chinese(Mod(1,2^10),Mod(0,3^10)))` has `e*(1-e) % 6^10` = 0 with e ≡ 3 and 1 - e ≡ 4 mod 6, so e and 1 - e
  are nonzero zero divisors), field `Some(false)`, Noetherian `Some(true)`, Krull `Some(1)`, integrally closed
  `None`; Q_6 field `Some(false)`, domain `Some(false)`, Krull 0, prime field `Some(false)`. p = 4: Z_4 domain
  `Some(true)` (Z_4 = Z_2; gp `isprimepower(4)` = 2), integrally closed `Some(true)`; Q_4 field `Some(true)`.
  `PadicExtension::eisenstein(5, x^2-5, 10)`: field `Some(true)`, Noetherian, Krull 0; `new(5, x^2-1, 10,
  Unramified)`: field `Some(false)` (gp `factorpadic(x^2-1,5,10)` has 2 factors); `new(5, x^2-2x+1, ..)`: field
  `Some(false)`, Noetherian `Some(true)`, Krull `Some(0)`; `new(5, x^2-125, ..)`: field `Some(true)` (gp
  `factorpadic(x^2-125,5,10)`: one factor); `new(4, x^2-5, ..)`: field `None`.
* `polynomial_ring_parents`: Z[x]: GAP `[Characteristic, IsCommutative, IsField, IsFinite]` = `[0, true, false,
  false]`; domain, Noetherian, integrally closed, Krull 2. GF(4)[x]: GAP `[2, true, false, false]`, domain,
  Krull 1. Q[x][y] (`PolynomialRingParent::new(PolynomialRingParent::new(RationalFieldParent))`): Krull 2, domain
  (GAP `PolynomialRing(Rationals, 2)` = `[0, true, false, false]`). (Z/6)[x]: char 6, finite `Some(false)` (GAP
  `IsFinite(PolynomialRing(ZmodnZ(6),1))` = false), domain `Some(false)` (PARI `Mod(2,6)*Mod(3,6)` = 0), Krull 1.
  (Z/(2^64+13))[x] (review M2): zero ring `Some(false)`, finite `Some(false)` (GAP
  `IsFinite(PolynomialRing(ZmodnZ(2^64+13),1))` = false, `rev1/rev1.g.log`), domain `Some(true)`, Krull `Some(1)`.
  Over the test zero ring (char 1): order `Some(1)`, finite `Some(true)`, field and domain `Some(false)`, Krull
  `None`. Over a test base with `is_commutative() == None`, Noetherian and Krull 0: Krull `None`.
* `wrappers_agree_with_real_impls`: for GR(8,2), Z/6, Z/7, GF(25) the test wrappers of `category_methods.rs` and
  the real impls give the same tuple.

`rustmath-category/src/ring_parents.rs` tests: `zmod_category` (n = 6: domain/field/PID/UFD/Euclidean
`Some(false)`; n = 7: all `Some(true)`; GAP `IsField(ZmodnZ(7))` = true), `finite_field_category` (GF(25) field,
Euclidean, PID, UFD), `galois_ring_category` (GR(8,2): field `Some(false)`, PID `Some(false)`; GR(5,3): field
`Some(true)`), `matrix_algebra_category` (`MatrixAlgebra::<Rational>::full(2)`: char `Some(0)`, commutative
`Some(false)`, domain `Some(false)`, field and PID `Some(false)`; GAP `[Characteristic, IsCommutative]
(FullMatrixAlgebra(Rationals,2))` = `[0, false]` and `E11*E22` = 0; `full(1)`: commutative, domain, field
`Some(true)` (GAP `IsCommutative(FullMatrixAlgebra(Rationals,1))` = true); `full(0)`: char `Some(1)`, domain
`Some(false)`; `MatrixAlgebra::<FiniteFieldElement>::full(2)`: char `None` (the element type does not know p; GAP
answers 5 for GF(5), which this parent cannot see), commutative `Some(false)` (GAP false); the diagonal subalgebra
`generated(2, [diag(1, 2)])` over Q (dimension 2): commutative `None`, field `None`), `big_float_field_category`
(`BigFloatField::new(53)`: char 0, commutative, domain, field, Euclidean, PID, UFD all `Some(true)`; MAGMA Ch. 25
table).

rustmath-numberfields `src/category.rs` tests (`number_field_category`): K = `NumberField::new(x^2 - 2)`: char
`Some(0)`, commutative `Some(true)`, domain, field, PID, UFD, Euclidean `Some(true)`, `is_finite() == Some(false)`
(PARI `polisirreducible(x^2-2)` = 1; GAP `IsField(AlgebraicExtension(Rationals, x^2-2))` = true; MAGMA §34.3.6);
`NumberField::new(x^2 - 1)`: domain, field, PID, UFD, Euclidean `Some(false)` (PARI `polisirreducible(x^2-1)` = 0;
GAP `IsIrreducibleRingElement(PolynomialRing(Rationals,1), x^2-1)` = false); `x - 3`: field `Some(true)` (PARI
`polisirreducible(x-3)` = 1); `x^2/2 - 1` (non-integral coefficients): field `Some(true)` (PARI
`polisirreducible(x^2/2-1)` = 1); `K.contains(&K.generator())` true and `K.contains(&L.generator())` false for
L = `NumberField::new(x^2 - 3)`. HEAD: none of these impls exists.

rustmath-algebras `algebra_with_parent.rs` test (`down_up_algebra_category`): DU(2, -1, -2) over Q (Benkart and
Roby: isomorphic to U(sl_2); GAP `IsCommutative(UniversalEnvelopingAlgebra(SimpleLieAlgebra("A",1,Rationals)))` =
false): char `Some(0)`, commutative `Some(false)`, domain, field, PID `Some(false)`; and `d.multiply(&u, &a) !=
u.multiply(&d, &a)` with the crate's own product.

---

## D. The category trait lacks the composite-characteristic rule

### D.1 Defect at HEAD

`rustmath-category/src/ring_category.rs:249-261`: `is_integral_domain` answers `Some(false)` for the zero ring and
`None` otherwise; the doc says the composite-characteristic rule "needs a primality test, which this crate does
not depend on". `e2_morph_cat.rs`: a characteristic-4 parent (for example (Z/4)[x]) gives `is_integral_domain`,
`is_field`, `is_pid`, `is_ufd`, `is_euclidean` all `None`; Z/6 (cardinality 6 = characteristic) gives domain
`None`; a parent of prime cardinality 7 gives characteristic, commutative, domain, field all `None`. The
rustmath-rings trait decides the first two (`category_methods.rs:152-174`, `:263-274`), but in both traits a prime
cardinality alone leaves the characteristic `None`, and a cardinality that is not a prime power (6, 12) alone
decides nothing (review M9).

### D.2 Fix (MAGMA Ch. 17 §17.4.2-3)

In `ring_category.rs`, two private helpers identical in logic to rustmath-rings':

```rust
/// c = 1 is the zero ring; for composite c = ab (1 < a, b < c), (a·1)(b·1) = c·1 = 0 with a·1, b·1 != 0.
fn characteristic_rules_out_domain(c: u64) -> bool { c == 1 || (c >= 4 && !prime_u64(c)) }
/// `Some(c)` when the parent is Z·1 ≅ Z/cZ: a known characteristic equal to the cardinality, or a prime
/// cardinality with unknown characteristic (1 != 0 has additive order dividing p, Lagrange).
fn residue_ring_modulus<P: RingParentMethods + ?Sized>(parent: &P) -> Option<u64>;
```

New default rules, in **both** traits (`rustmath_category::RingParentMethods` and
`rustmath_rings::category_methods::RingParentMethods`):
* `characteristic`: after the existing rules (rustmath-rings: `Self::Element::static_characteristic()`, then order 1
  gives 1; rustmath-category: cardinality 1 gives 1), a prime cardinality p gives `Some(p)` (1 != 0 because the ring
  has more than one element, and the additive order of 1 divides p by Lagrange). It reads only the cardinality,
  which comes first in the dependency order.
* `is_integral_domain`: zero ring ⇒ `Some(false)`; `characteristic_rules_out_domain(c)` ⇒ `Some(false)`;
  `residue_ring_modulus = Some(c)` ⇒ `Some(prime_u64(c))`; a known cardinality n >= 2 with
  `!is_prime_power_u64(n)` ⇒ `Some(false)` (a finite domain is a field, and a finite field has p^k elements); else
  `None`. In rustmath-category these are all new (it had only the zero-ring rule); in rustmath-rings only the last
  one is new.
* rustmath-category `is_commutative`: `Some(true)` for the zero ring or when `residue_ring_modulus` is `Some`
  (replaces the inline `generated_by_one` test, which it generalises).
* `is_field` unchanged (a non-domain is not a field; a finite domain is a field).
* The trait docs drop the sentence about the missing primality test and list the rules.
* rustmath-rings `is_prime_field` calls `prime_u64(n)` instead of `Integer::from(n).is_prime()` (same answer on
  u64, cheaper at HEAD).

### D.3 Callers

`rustmath_category::RingParentMethods` has no implementor outside its tests (C.1) and after C the impls of C.3 to
C.5, which override these methods. rustmath-modules depends on rustmath-category but does not use the trait (grep).
Behaviour change: test parents that previously got `None` now get decided answers; one existing assertion changes,
`rustmath-rings` `test_prime_order_alone_decides` (`category_methods.rs:950`): `OrderSeven.characteristic()` is now
`Some(7)` (was `None`).

### D.4 Tests

`ring_category.rs`:
* `test_composite_characteristic_is_not_a_domain` (new): characteristic 4, no cardinality: domain, field, PID,
  UFD, Euclidean all `Some(false)` (PARI `Mod(2,4)^2` = 0); characteristic 561: `Some(false)` (PARI
  `isprime(561)` = 0, 561 = 3·11·17, a Carmichael number); characteristic 2^64 - 59 (prime, PARI
  `isprime(2^64-59)` = 1): `None`. HEAD: `None` throughout.
* `test_residue_ring_by_cardinality` (new): cardinality 6 and characteristic 6: commutative `Some(true)`, domain
  `Some(false)`; cardinality 7 only: characteristic `Some(7)`, commutative, domain, field, Euclidean, PID, UFD
  `Some(true)` (GAP `IsField(ZmodnZ(7))` = true); cardinality 4 only: domain `None` (Z/4 and GF(4) share it; GAP
  `IsField(GF(4))` = true, `IsField(ZmodnZ(4))` = false). HEAD: domain `None` in the first case (commutativity
  was already decided there by the `generated_by_one` test), `None` for every answer of the second case.
* `test_non_prime_power_cardinality_is_not_a_domain` (new): cardinality 6 only and 12 only: domain, field, PID,
  UFD, Euclidean `Some(false)` (GAP `IsPrimePowerInt(6)`, `IsPrimePowerInt(12)` = false, `IsField(ZmodnZ(6))` =
  false); cardinality 49 only: domain `None` (GAP `IsPrimePowerInt(49)` = true; GF(49) is a field, Z/49 is not).
  HEAD: `None`.
* Existing tests unchanged (`test_ring_parent_defaults_make_no_claim`, `test_zero_ring_parent_category`,
  `test_field_parent_category`, `test_finite_domain_is_field_category`).

`category_methods.rs` (rustmath-rings): `test_prime_order_alone_decides` asserts `Some(7)` for the characteristic
and adds `OrderSix` (order 6 only): domain, field, prime field `Some(false)`. HEAD: `None` and no `OrderSix`.

---

## E. Ring hooks for series and p-adics

### E.1 Defect at HEAD

The hooks of `rustmath-core/src/traits.rs:97-190` (`unit_status`, `nilpotent_status`, `unit_inverse`,
`multiplicative_order_status`, `static_characteristic`) are not overridden by `PadicInteger`, `PadicRational`,
`PadicExtensionElement`, `PowerSeries` (`series_core_impls.rs:31-50`), `LaurentSeries` (`laurent.rs:264-278`),
`PuiseuxSeries` (`puiseux.rs:253-266`) or `LazyPowerSeries` (`lazy.rs:270-284`) (PROJECT_TRACKER.md:361:
"Polynomials, series and p-adics keep the defaults"). `e1_panics.rs` / `e3_series.rs`:

```
unit_status(7 in Z_5) = None;  unit_status(10 in Z_5) = None;  unit_inverse(7) = Err(NotImplemented)
PadicRational (5,0): unit_status None
PowerSeries 1+x over Z: unit None;  2+x: unit None, nil None;  static char Z[[x]] None
O(x^0): is_zero true unit Some(false) nil Some(true)                       <- wrong (nothing is known)
Laurent x^-2(1+3x) over Z: unit None
lazy 1+x^25: is_one true order Finite(1) unit_inverse coefficient 25 = 1   <- wrong: infinite order, coefficient -1
lazy x^30: is_zero true nilpotent Some(true)                              <- wrong: (x^30)^3 = x^90
```

The three wrong answers come from the defaults reading `is_zero`/`is_one`, which for `LazyPowerSeries` look at
`WEAK_TERMS` = 20 coefficients only (a bounded search deciding), and for `O(x^0)` are vacuously true. gp
references: `1/(1+x^25+O(x^60))` has coefficient -1 at x^25; `(1+x^25)^2 = 1 + 2x^25 + x^50`; `(x^30)^3 = x^90`.
`PadicExtensionElement` has no inverse at all (it implements `CommutativeRing` only).

### E.2 p-adic hooks (all in the type's `impl Ring`)

Under the model of 0.1. Every hook tests zero first (review M6). "prime" means `cheap_primality(p) == Some(true)`;
it is needed only by the `PadicInteger` order rule.

`PadicInteger`, bound x = u + O(p^n), any p > 1:
* `unit_status`: `Some(gcd(u, p) == 1)`. x is a unit of Z/p^n iff p and u are coprime, and a unit of Z_p iff its
  image in Z/p^n is (n >= 1), for every lift. No primality needed.
* `unit_inverse`: the existing `inverse()` (`padic_integer.rs:147`): the truncation of the Z_p inverse, correct to
  precision n; `Err(NotInvertible)` for a non-unit (a proof by the line above).
* `nilpotent_status`: zero ⇒ `Some(true)` (the convention); otherwise `Some(false)`: every lift is nonzero, and
  Z_p ≅ ∏_{q|p} Z_q is reduced (C.2) for every p > 1 (review M9; the first version required p prime).
* `multiplicative_order_status`: weakly zero or non-unit ⇒ `NotAUnit`; `is_one` ⇒ `Finite(1)`; `(-x).is_one()` ⇒
  `Finite(2)` (the same convention; -1 ≠ 1 in Z_p); for prime p > 2, `u^(p-1) mod p^n != 1` ⇒ `Infinite`
  (μ(Q_p) = μ_(p-1), so a root of unity y has y^(p-1) = 1, and every lift y ≡ u has y^(p-1) ≡ u^(p-1) ≢ 1); for
  p = 2 (here n >= 2, since for n = 1 every unit is 1, and x ≢ ±1 mod 2^n) ⇒ `Infinite` (μ(Q_2) = {±1}); else
  `Undecided` (x is congruent to a root of unity to the working precision, so it may be that root of unity or a
  nearby element of infinite order). Sage, by its convention, answers 4 for the Teichmüller lift of 2 in Q_5; this
  design does not extend the convention beyond `is_one`. References: Sage `K(2).multiplicative_order()` = +Infinity,
  `K(5)` +Infinity, `K(-1)` 2 (doc quote, `padic_generic_element`); gp `Mod(2,5^10)^4` = 16 ≠ 1;
  `lift(teichmuller(2+O(5^10)))` = 6139557 and `Mod(6139557,5^10)^4 == 1`.
* `static_characteristic`: `Some(0)`. `static_is_commutative` (P1): `Some(true)`.
* Unbound c: `unit_status` ±1 ⇒ `Some(true)`, 0 ⇒ `Some(false)`, else `None` (whether 2 is a unit depends on p);
  `nilpotent_status` `Some(c == 0)` (Z ⊂ Z_p and Z_p is reduced); `unit_inverse` ±1 ⇒ itself, 0 ⇒
  `Err(NotInvertible)`, else `Err(InvalidArgument("depends on p; bind it first"))`; order: 0 ⇒ `NotAUnit`, 1 ⇒
  `Finite(1)`, -1 ⇒ `Finite(2)` (-1 ≠ 1 in characteristic 0; review M9: the first version copied the
  finitefields table, which must say `Undecided` because there -1 can bind into characteristic 2), else
  `Undecided` (`NotAUnit` or `Infinite` depending on p).

`PadicRational`, bound x = u p^v + O(p^(v+n)):
* `unit_status`: weakly zero ⇒ `Some(false)`; else `normalized()` = u' p^k, and `gcd(u', p) == 1` ⇒ `Some(true)`
  (u' ∈ Z_p^×, p^k ∈ Q_p^×; for prime p this is every nonzero x); else `None` (only for a non-prime p).
* `unit_inverse`: the normalised inverse of A.3.2; weakly zero ⇒ `Err(NotInvertible)`; `gcd(u', p) != 1` ⇒
  `Err(NotImplemented)` (undecided, never `NotInvertible`).
* `nilpotent_status`: weakly zero ⇒ `Some(true)`; else `Some(false)` (Q_p ≅ ∏ Q_q is reduced for every p > 1).
* `multiplicative_order_status`: weakly zero ⇒ `NotAUnit`; `is_one` ⇒ `Finite(1)`; -1 ⇒ `Finite(2)`;
  `gcd(u', p) == 1` and normalised valuation k ≠ 0 ⇒ `Infinite` for every p > 1 (if x^m = 1 then p^(mk) = u'^(-m)
  would be a unit of Z_p, but p^(|mk|) ∈ pZ_p); k = 0 ⇒ the `PadicInteger` rule on u' at precision n - j; else
  `Undecided`. Sage: `K(5)`, `K(1/5)` +Infinity.
* `static_characteristic`: `Some(0)`; `static_is_commutative` (P1): `Some(true)`.
* Unbound c: `unit_status` `Some(c != 0)` (a nonzero integer c = c_1 c_2 with c_1 | p^k and gcd(c_2, p) = 1 is a
  unit of Q_p = Z_p[1/p]: 1/c_1 = (p^k / c_1)/p^k, and c_2 ∈ Z_p^×); `unit_inverse` ±1 ⇒ itself, 0 ⇒
  `Err(NotInvertible)`, else `Err(InvalidArgument("1/c is not an integer constant; bind it first"))`;
  `nilpotent_status` `Some(c == 0)`; order: 0 ⇒ `NotAUnit`, 1 ⇒ `Finite(1)`, -1 ⇒ `Finite(2)`, |c| >= 2 ⇒
  `Infinite` (Q ⊂ Q_p, and c^m = 1 in Q forces c = ±1; review M9).

`PadicExtensionElement` (K = Q_p[x]/(f)):
* `unit_status`: weakly zero ⇒ `Some(false)`; ±1 ⇒ `Some(true)`; `field_status() == Some(true)` ⇒ `Some(true)` (a
  nonzero element of a field; nonzero to its precision means every lift is nonzero); else `None`.
* `nilpotent_status`: weakly zero ⇒ `Some(true)`; `ext.is_reduced()` (f squarefree over Q) ⇒ `Some(false)` (K is a
  product of fields, review M9); else `None`.
* `unit_inverse`: `self.inverse()` (E.6), with its error kinds (`NotInvertible` only for a weakly zero element).
* `multiplicative_order_status`: weakly zero ⇒ `NotAUnit`; `is_one` ⇒ `Finite(1)`; -1 ⇒ `Finite(2)`; with a
  `field_certificate`, V < E and V ≠ 0 (V, E as in E.6) ⇒ `Infinite` (v_K(x) = V and v_K(x^m) = mV ≠ 0); else
  `Undecided`. Units of valuation 0 stay undecided: the roots of unity of K (μ_(q-1) times p-power roots of unity in
  ramified K) need more than a valuation.
* `static_characteristic`: `Some(0)`; `static_is_commutative` (P1): `Some(true)`.
* Unbound c: the unbound `PadicRational` rules (Q_p ⊂ K, deg f >= 1, so c ≠ 0 is a unit and |c| >= 2 has infinite
  order in K).

### E.3 Series hooks (rustmath-powerseries; MAGMA Ch. 49 §49.4.4 `IsUnit`, `IsNilpotent`; Ch. 50 for lazy)

The series types model R[[x]], R((x)), R⟨⟨x⟩⟩ with `is_zero` = "all known coefficients zero" (MAGMA
`IsWeaklyZero`, §49.4.4) and equality to the smaller precision (`series_core_impls.rs:21`). Facts used, for any
ring R with x central (Atiyah and Macdonald, Ch. 1, Ex. 5(i) for commutative R; the same argument works in
general): f ∈ R[[x]] is a unit iff f(0) is a unit of R, with inverse given by g_0 = f_0^-1,
g_m = -g_0 Σ_{i=1..m} f_i g_(m-i) (this builds a right inverse, which is the inverse of a unit). The lowest
nonzero coefficient a_v of f gives a_v^k as the coefficient of x^(kv) in f^k, so a non-nilpotent a_v makes f
non-nilpotent. For commutative R, f nilpotent implies every coefficient nilpotent (A-M Ch. 1, Ex. 5(ii)); these
coefficients are known exactly below the precision.

**Precondition (review M10).** The rules that read a coefficient's `is_zero()` ("the lowest nonzero
coefficient", "the leading coefficient of a Laurent series") are proofs when the coefficient type's `is_zero` is
exact (Z, Q, Z/m, GF(q), Galois rings, polynomials over those). For a weak coefficient type (the p-adic types,
series of series) they hold under the convention of 0.1, for the representative only: over Z_5,
`O(5^3)·x^-1 + 1` is reported a unit of Z_5((x)), but the lift `5^3·x^-1 + 1` = x^-1(5^3 + x) is not one (its lowest
coefficient 5^3 is not a unit). The doc comments of the four hooks below and of the module say this.

`PowerSeries<R>` (precision p = number of known coefficients):
* `static_characteristic`: `R::static_characteristic()` (R ⊂ R[[x]]). `static_is_commutative` (P1): delegate.
* `unit_status`: p = 0 ⇒ `None` (nothing is known; HEAD says `Some(false)`); else `coeff(0).unit_status()`.
* `unit_inverse`: p = 0 ⇒ `Err(NotImplemented("O(x^0)"))`; `coeff(0).unit_inverse()`: `Ok(g0)` ⇒ the recurrence
  above to p terms (exact mod x^p); `Err(NotInvertible)` ⇒ `Err(NotInvertible)` (a proof: f(0) non-unit); any
  other `Err` is passed on (undecided).
* `nilpotent_status`: p = 0 ⇒ `None`; `is_zero` ⇒ `Some(true)` (the convention); the lowest nonzero known
  coefficient has `nilpotent_status() == Some(false)` ⇒ `Some(false)`; with `R::static_is_commutative() ==
  Some(true)` (P1), any known coefficient with `Some(false)` ⇒ `Some(false)`; else `None` (never `Some(true)` for a
  nonzero series: the unknown tail could be non-nilpotent).
* `multiplicative_order_status`: p = 0 ⇒ `Undecided`; else the default (it now sees `unit_status`).

`LaurentSeries<R>` (window `coeffs` starting at exponent `val`, all exact):
* `static_characteristic`: delegate. Empty window ⇒ `unit_status`/`nilpotent_status` `None`, `unit_inverse`
  `Err(NotImplemented)`, order `Undecided` (like `O(x^0)`).
* `unit_status`: weakly zero ⇒ `Some(false)` (convention); leading coefficient c (first nonzero in the window) a
  proven unit ⇒ `Some(true)` (f = x^w·c·(1 + h), h ∈ xR[[x]], every factor a unit); else `None` (over a non-domain
  a non-unit leading coefficient can still give a unit: 2 + x = x(1 + 2x^-1) is a unit of (Z/4)((x)) because
  2x^-1 is nilpotent; over Z the answer would need "Z is a domain", which no hook states).
* `unit_inverse`: leading coefficient a proven unit ⇒ x^(-w) times the power-series inverse of the unit part,
  relative precision unchanged (generic version of `try_inverse`, laurent.rs:287, which needs `R: Field`); weakly
  zero ⇒ `Err(NotInvertible)`; otherwise `Err(NotImplemented)`.
* `nilpotent_status`: weakly zero ⇒ `Some(true)`; leading coefficient non-nilpotent ⇒ `Some(false)`; commutative
  (P1) and any known coefficient non-nilpotent ⇒ `Some(false)` (x^k f ∈ R[[x]] for some k); else `None`.
* order: default.

`PuiseuxSeries<R>`: every hook delegates to the inner Laurent series in y = x^(1/e) (`unit_inverse` maps through
`PuiseuxSeries::from_laurent(e, g)`); `static_characteristic` delegates.

`LazyPowerSeries<R>` (exact coefficients, weak `is_zero`/`is_one`; the hooks must not read them):
* `unit_status`: `coeff(0).unit_status()` (coefficient 0 is exact).
* `unit_inverse`: `coeff(0).unit_inverse()`: `Ok(g0)` ⇒ the lazy recurrence (`from_recursive`, as `try_inverse`
  lazy.rs:215 but with `unit_inverse`, so any `R: Ring`); `Err(NotInvertible)` passed on; other errors passed on.
* `nilpotent_status`: search the first `WEAK_TERMS` coefficients for the lowest nonzero one (finding one is a
  proof that it is the lowest); if it has `nilpotent_status() == Some(false)` ⇒ `Some(false)`; with commutativity
  (P1), any found coefficient `Some(false)` ⇒ `Some(false)`; otherwise `None`, including when all 20 are zero.
* `multiplicative_order_status`: `unit_status() == Some(false)` ⇒ `NotAUnit`; else `Undecided` (1 + x^25 is
  weakly one; no bounded test decides an order).
* `static_characteristic`: delegate.

### E.4 Callers of the hooks (grep `unit_status()\|nilpotent_status()\|unit_inverse()\|multiplicative_order_status()\|static_characteristic()`)

`rustmath-category/src/ring_category.rs:170-210` (`RingElementMethods` blanket), `rustmath-rings/src/category_methods.rs:70, :94, :328` (element methods and the `characteristic` default),
`rustmath-modules/src/with_basis/parent.rs:195, :272`, `rustmath-algebras/src/algebra_with_parent.rs:142`, and
the defaults in `traits.rs` itself. None of them is instantiated with a p-adic or series type in the workspace
except the tests named below. Every new answer is a theorem of E.2/E.3. Behaviour changes in existing tests (both
updated in Plan step 3, where `PadicInteger`'s hooks and `static_characteristic` land; review M4):
* `category_methods.rs:565-581` `test_defaults_on_padic_integer_do_not_panic`: 6 in Z_5 now gives
  `is_unit_element() == Some(true)`, `nilpotent_status() == Some(false)`, `inverse_of_unit() == Ok(1/6 mod 5^10)`
  (gp `lift(Mod(6,5^10)^-1)` = 8138021); rename to `test_padic_integer_hooks_decide`.
* `category_methods.rs:813-847` `test_parent_carrying_elements_do_not_panic`: `bare::<PadicInteger>()` now has
  `characteristic() == Some(0)`, `is_zero_ring() == Some(false)`, `is_finite() == Some(false)` (from
  `static_characteristic`); `is_field()` stays `None`. The `GaloisRingElement` half is unchanged.

### E.5 Tests

p-adic (inline, extending A.5): `padic_integer.rs::hooks_in_z5` (`proto_padic.rs` prints the Z_5 values):
* 7 + O(5^4): unit `Some(true)`, `unit_inverse` = 268 (gp `lift(Mod(7,5^4)^-1)` = 268);
* 10 + O(5^4): unit `Some(false)`, nilpotent `Some(false)`, `unit_inverse` `Err(NotInvertible)`, order `NotAUnit`;
* 0 + O(5^4): nilpotent `Some(true)`;
* orders at precision 10: 2 ⇒ `Infinite`; 6139557 (Teichmüller of 2) ⇒ `Undecided`; -1 ⇒ `Finite(2)`; 1 ⇒
  `Finite(1)`; 5 ⇒ `NotAUnit`; 3 in Z_2 at precision 5 ⇒ `Infinite` (3 ≢ ±1 mod 32);
* composite p = 6, precision 4: 2 + O(6^4): unit `Some(false)`, nilpotent `Some(false)` (gp `Mod(2,6^4)^8` = 256;
  Z_6 ≅ Z_2 × Z_3 is reduced); 5 + O(6^4): unit `Some(true)`, `unit_inverse` = 1037 (gp `lift(Mod(5,6^4)^-1)` =
  1037), order `Undecided` (6 is not prime);
* unbound: -1 order `Finite(2)`; 0 nilpotent `Some(true)`; 2: unit `None`, nilpotent `Some(false)`, order
  `Undecided`; `<PadicInteger as Ring>::static_characteristic() == Some(0)`.
HEAD: `None`/`Undecided`/`NotImplemented` throughout.

`padic_rational.rs::hooks_in_q5`: (5, 0) unit `Some(true)` (HEAD `None`) and inverse as A.5; 5 and 1/5 ⇒
`Infinite`; -1 ⇒ `Finite(2)`; zero ⇒ `NotAUnit`; the weakly zero (0, -1, 1) of A.5 ⇒ `NotAUnit`; 2 + O(6^4) in
Q_6: nilpotent `Some(false)`; unbound 2: unit `Some(true)`, `unit_inverse` is `Err` but not `NotInvertible`, order
`Infinite`, nilpotent `Some(false)`.

`extension.rs::hooks_in_extensions`:
* Q_5(√5): g: unit `Some(true)`, nilpotent `Some(false)`, order `Infinite` (v_K(g) = 1), `unit_inverse(g)` equals
  `g * from_base_field(from_rational(1/5))` (E.6); 1 + g: order `Undecided` (valuation 0).
* `new(5, x^2 - 1, 10, Unramified)`: 1 + g (a zero divisor: (1+g)(1-g) = 1 - g^2 = 0, gp
  `factorpadic(x^2-1,5,5)`): unit `None`, nilpotent `Some(false)` (f squarefree, gp `issquarefree(x^2-1)` = 1),
  `unit_inverse` `Err(NotImplemented)` (never `NotInvertible`: the hook does not decide that 1 + g is a non-unit).
* `new(5, x^2 - 2x + 1, 10, Unramified)`: g - 1: nilpotent `None` (it is nilpotent, (g-1)^2 = f(g) = 0, but the
  hook does not claim it; gp `issquarefree((x-1)^2)` = 0).
* `new(5, x^2 - 125, 10, Unramified)`: g: unit `Some(true)` (`field_status` by OM; gp `factorpadic(x^2-125,5,10)`
  irreducible), `unit_inverse` `Err(NotImplemented)` (no field certificate, so no power-basis formula; F.3).

Extension inverses (E.6; `rev1/proto_ext_inverse.log` runs the algorithm on the HEAD API; gp `rev1.log`,
`rev1b.log`), `extension.rs::inverse_in_certified_fields`:
* Q_5(√5), x = 1 + g (coefficients from `one(ext) + g`, both 1 + O(5^10)): V = 0, E = 20, B = 20;
  x^-1 = (-1/4 + O(5^10)) + (1/4 + O(5^10))·g (gp `lift(Mod(1,x^2-5)/(1+x))` = `1/4*x - 1/4`; PARI's own
  `Mod((1+O(5^10)) + (1+O(5^10))*x, x^2-5)^-1` has both coefficients + O(5^10)). Check: coefficient values (value
  equality with `from_rational(-1/4)`, `from_rational(1/4)`), `absolute_precision() == 10` for both, and
  `x * x^-1 == one(ext)`.
* Q_5(√5), x = g: V = 1, E = 20, B = 18; x^-1 = O(5^9) + (5^-1 + O(5^9))·g (gp `lift(Mod(1,x^2-5)/x)` = `1/5*x`;
  PARI `Mod(O(5^10) + (1+O(5^10))*x, x^2-5)^-1` = `(5^-1 + O(5^9))*x + O(5^9)`): coefficient 0 weakly zero with
  absolute precision 9, coefficient 1 equal to 1/5 with absolute precision 9; `g * g^-1 == one(ext)`.
* Q_5(√5), x = `new(ext, [from_rational(5), from_rational(5)])` (both 5 + O(5^11)): V = 2, E = 22, B = 18;
  x^-1 = (-1/20 + O(5^9)) + (1/20 + O(5^9))·g (gp `lift(Mod(1,x^2-5)/(5+5*x))` = `1/20*x - 1/20`; PARI
  `Mod((5+O(5^11)) + (5+O(5^11))*x, x^2-5)^-1` has both coefficients + O(5^9)).
* `unramified(5, 2)` (f = x^2 + 4x + 2), x = g: V = 0, E = 10, B = 10; x^-1 = (-2 + O(5^10)) + (-1/2 + O(5^10))·g
  (gp `lift(Mod(1,x^2+4*x+2)/x)` = `-1/2*x - 2`; PARI's polmod inverse has both + O(5^10)); x =
  `from_base_field(from_rational(5)) * g` (HEAD's `Mul` leaves both coefficients at absolute precision 10, so
  E = 10): V = 1, B = 8, x^-1 = (-2/5 + O(5^8)) + (-1/10 + O(5^8))·g (gp `lift(Mod(1,x^2+4*x+2)/(5*x))` =
  `-1/10*x - 2/5`).
* `eisenstein(2, x^3 - 2, 8)`, x = `one(ext) + π + π * π` (every coefficient at absolute precision 8): V = 0,
  E = 24, B = 24; x^-1 = (-1 + O(2^8)) + (1 + O(2^8))·π +
  O(2^8)·π^2 (gp `lift(Mod(1,x^3-2)/(1+x+x^2))` = `x - 1`; PARI `Mod((1+O(2^8)) + (1+O(2^8))*x + (1+O(2^8))*x^2,
  x^3-2)^-1` has every coefficient + O(2^8)); `x * x^-1 == one(ext)`.
* Linear: `new(5, 5x - 1, 10, Unramified)`, x = `from_base_field(from_rational(25))`: x^-1 equals
  `from_base_field(from_rational(1/25))` (K = Q_5).
* Undecided: Q_5(√5), x = `new(ext, [0 + O(5^0), 1 + O(5^10)])` (coefficient 0 known to absolute precision 0): V = 1,
  E = 0, so `Err(NumericalError(_))`, not `NotInvertible` (PARI raises "impossible inverse in divpp: O(5^0)" on the
  same input, `rev1b.log`). x = `new(ext, [1 + O(5^10), 0 + O(5^0)])`: V = 0, E = 1, B = 1, x^-1 = (1 + O(5)) +
  O(5^0)·g (prototype).
* Errors: `zero(ext).inverse()` is `Err(NotInvertible)`; `Ring::one().inverse()` is `Ok` and unbound; unbound 2 is
  `Err(InvalidArgument)`; in `new(5, x^2 - 1, ..)` and `new(5, x^2 - 125, ..)` any non-±1 element gives
  `Err(NotImplemented)`.
HEAD: there is no inverse.

Series: new `rustmath-powerseries/tests/ring_predicates.rs` (references `ref2.gp`, `ref4.gp`):
* `power_series_units_over_z`: 1 + x (precision 5): `Some(true)`, inverse `[1, -1, 1, -1, 1]` (gp
  `1/(1+x+O(x^5))`); 1 + 2x + 3x^2 (precision 6): inverse `[1, -2, 1, 4, -11, 10]` (gp); -1 + x^2 (precision 6):
  inverse `[-1, 0, -1, 0, -1, 0]` (gp); 2 + x: `Some(false)`, `unit_inverse` `Err(NotInvertible)` (units of
  Z[[x]] have constant term ±1); `static_characteristic() == Some(0)`. HEAD: `None`, `NotImplemented`, `None`.
* `power_series_over_z4` (`IntegerMod` mod 4): 1 + 2x (precision 6) unit with inverse 1 + 2x (gp
  `lift(1/(Mod(1,4)+Mod(2,4)*x+O(x^6)))` = 1 + 2x); 2 + x: unit `Some(false)` (gp "impossible inverse");
  2 + x nilpotent: `Some(false)` when `<IntegerMod as Ring>::static_is_commutative() == Some(true)` (P1/F1),
  else `None` (gp `(Mod(2,4)+x+O(x^6))^4` = x^4 + O(x^8) ≠ 0); 2 + 2x nilpotent `None` (as a polynomial it is
  nilpotent, gp `(Mod(2,4)+Mod(2,4)*x)^2` = 0, but the unknown tail of the series decides nothing).
* `power_series_precision_zero`: `PowerSeries::<Integer>::new(vec![], 0)`: unit `None`, nilpotent `None`,
  `unit_inverse` an `Err` other than `NotInvertible`, order `Undecided`. HEAD: `Some(false)`, `Some(true)`.
* `laurent_hooks`: `LaurentSeries::new(-2, [1, 3, 0, 0, 0])` over Z: unit `Some(true)`, inverse valuation 2 with
  `[1, -3, 9, -27, 81]` (gp `1/(x^-2*(1+3*x)+O(x^3))`); over Q, `new(-1, [3, 0, 1, 0, 0])`: inverse valuation 1
  with `[1/3, 0, -1/9, 0, 1/27]` (gp `1/(3*x^-1+x+O(x^4))`), equal to `try_inverse`; over Z the same series has
  unit `None` (documented limit); nilpotent `Some(false)`. HEAD: `None`.
* `puiseux_hooks`: 1 + x^(1/2) (`from_laurent(2, [1, 1, 0, 0, 0, 0])`): unit `Some(true)`, inverse coefficients
  `[1, -1, 1, -1, 1, -1]` in y = x^(1/2) (gp `1/(1+y+O(y^6))`).
* `lazy_hooks`: 1 + x: unit `Some(true)`, inverse first 10 coefficients `[1, -1, ..., -1]` (gp); 1 + x^25: order
  `Undecided` and `unit_inverse` coefficient 25 = -1, 50 = 1 (gp `polcoeff(1/(1+x^25+O(x^60)), 25)` = -1); x^30:
  nilpotent `None`, unit `Some(false)`; 2x^3 + x^5 over Z: nilpotent `Some(false)` (lowest coefficient 2 is not
  nilpotent). HEAD: `Finite(1)`, coefficient 25 = 1, `Some(true)`.

### E.6 The inverse of a p-adic extension element (MAGMA Ch. 47 §47.8.1 `x / y`, `x ^ -1`)

Review X1: the first version left this out of scope although it is feasible. It needs only exact arithmetic on
the representatives, not `Mul` (whose "simplified reduction", PROJECT_TRACKER.md:1148, is not used).

```rust
impl PadicExtensionElement {
    /// MAGMA `1/x` in K = Q_p[x]/(f). `Ok(y)`: y is the inverse of every element of K that x approximates, its
    /// coefficient i known to the absolute precision P_i below. Errors:
    /// * unbound c: ±1 give themselves (unbound), 0 gives `Err(NotInvertible)`, any other c
    ///   `Err(InvalidArgument("1/c is not an integer constant; bind it first"))`;
    /// * x weakly zero: `Err(NotInvertible)` (the convention of 0.1);
    /// * no `field_certificate`: `Err(NotImplemented(..))` (K may not be a field, or OM certified it without a
    ///   power-basis valuation formula; F.3);
    /// * x not known well enough (V >= E below): `Err(NumericalError(..))`, undecided, never `NotInvertible`.
    pub fn inverse(&self) -> Result<PadicExtensionElement>;
}
```

`Ring::unit_inverse` returns `self.inverse()`. `PadicExtensionElement` still does not implement `Field` (a
non-certified K need not be a field, and `impl Field` is a claim about every value of the type).

Algorithm. Let d = deg f, N = `ext.precision()`, and for each coefficient c_i: r_i = `representative(c_i)` ∈ Q,
A_i = `absolute_precision(c_i)`. The certificate fixes (e, s_i): `Eisenstein` gives e = d, s_i = i;
`IrreducibleReduction` and `Linear` give e = 1, s_i = 0. Write w_i(t) = e·t + s_i.
1. V = min over the i with r_i ≠ 0 of w_i(v_p(r_i)) (= v_K(x_rep), x_rep = Σ r_i g^i; some r_i ≠ 0 because x is
   not weakly zero). E = min over all i of w_i(A_i).
2. If V >= E: `Err(NumericalError)`.
3. Solve M y = e_1 exactly over Q, where M = `multiplication_matrix_rational()` (extension.rs:553, column j = the
   coefficients of x_rep·g^j) and e_1 = (1, 0, ..., 0): Gaussian elimination with row swaps, the elimination of
   `rational_det` (extension.rs:138) extended to a right-hand side (new private `rational_solve`). y_rep = Σ y_i g^i
   is x_rep^-1 in Q[x]/(f).
4. B = E - 2V. P_i = ceil((B - s_i)/e). Coefficient i of the result:
   `PadicRational::from_rational_to_absolute_precision(y_i, p, P_i, N)`.

Correctness. f is irreducible over Q_p (the certificate), hence over Q, so Q[x]/(f) is a field and x_rep ≠ 0 is
invertible: M is nonsingular and step 3 succeeds. Every x that the approximation represents is x_rep + ε with
ε = Σ ε_i g^i and v_p(ε_i) >= A_i, so v_K(ε) >= min_i w_i(v_p(ε_i)) >= E (the ultrametric inequality and the
valuation formula of A.3.3). Since v_K(x_rep) = V < E, v_K(x) = V and x ≠ 0. Then
x^-1 - y_rep = (x_rep - x)/(x·x_rep) = -ε/(x·x_rep) has v_K >= E - 2V = B. Write it as δ = Σ δ_i g^i. For the
three certified shapes the valuation formula is an equality, so w_i(v_p(δ_i)) >= v_K(δ) >= B for every i, that is
v_p(δ_i) >= (B - s_i)/e, and v_p(δ_i) is an integer, so v_p(δ_i) >= P_i. Hence y_i + O(p^(P_i)) contains the
coefficient i of x^-1 for every lift x. Capping the relative precision at N only drops digits.

Reference check (`rev1/proto_ext_inverse.rs`, which implements steps 1 to 4 with the HEAD public API and a local
value equality): every case of E.5 gives the stated values and precisions, `x * x^-1` equals 1 in value, and the
precisions are equal to those of PARI's own p-adic polmod inverse in all five cases where PARI answers (`rev1.log`,
`rev1b.log`). PARI refuses the undecided case too.

Cost: one d × d exact Gaussian elimination over Q (O(d^3) rational operations) and d valuations; d is the
extension degree (small).

---

## F. Residual limits and hand-offs

### F.1 (withdrawn)

The first version put the number-field parent here on a wrong argument. It is now C.4.

### F.2 Other parents

* `SeriesRing` (`rustmath-powerseries/src/precision.rs:109`) does not carry its coefficient ring, so it cannot
  decide any predicate; and rustmath-rings does not depend on rustmath-powerseries. The rustmath-rings series rings
  (`laurent_series_ring.rs`, `multi_power_series_ring.rs`, ...) have element types that do not implement `Ring`.
  Honest interim: no impl. Needed: a series parent that stores its base parent.
* `SimpleAlgebra<R>` (`rustmath-algebras/src/algebra_with_parent.rs:84`): its `SimpleElement` has no
  multiplication (no `Mul` impl; `derive(Debug, Clone, PartialEq)` only), so it is a free module, not a ring. A
  ring predicate would be a claim about an undefined product: no impl.
* `Lattice` (rustmath-matrix) and the module parents of rustmath-modules and rustmath-manifolds are not rings.
* `nf_elem::NumberField` (`rustmath-numberfields/src/nf_elem.rs:156`) is the internal arithmetic of the relative
  extension code (`rnfeq`, `kummer`). Its element `NfElem` has no `PartialEq` (core `Parent` needs it) and its
  `zero`/`one` take `self: &Arc<Self>`. Hand-off: derive `PartialEq` on `NfElem` (same field and same reduced
  coefficients) and copy C.4 (its `new` also does not check irreducibility).
* `DiffScalarFieldAlgebra` (`rustmath-manifolds/src/modules.rs:94`, C^∞(M)): hand-off to MF1 (the owner of
  rustmath-manifolds, implemented after R1), which would need a rustmath-manifolds → rustmath-category dependency.
  The values to implement: characteristic `Some(0)` (R ⊂ C^∞(M) for nonempty M), commutative `Some(true)`; for
  dim M >= 1, domain and field `Some(false)` (two bump functions with disjoint supports multiply to 0); for dim
  M = 0 the ring is R^(#points), so `None` unless the point count is known.
* `Random(R)` (`random_element`) stays `Err(NotImplemented)` for every parent; it is not one of the items.

### F.3 The inverse in an extension certified only by OM

`field_status()` can be `Some(true)` from the OM factorisation (step 5) for a field that has none of the three
shapes (for example Q_5[x]/(x^2 - 125)). There the hooks say "unit" but `inverse` returns `Err(NotImplemented)`:
the bound v_K(ε) >= E still holds with s_i = i·v_K(g) (v_K(g) = v_p(N(g))/f_K from the exact norm), but turning
v_K(δ) >= B into coefficient bounds needs the index of Z_p[g] in O_K, that is an integral basis (Okutsu bases from
the OM leaves, Guàrdia, Montes and Nart). Honest interim: `NotImplemented`. Needed: an integral basis of O_K and
the dual basis for the coefficient bounds.

### F.4 Adjacent findings (not fixed here; for the HANDOFF)

* `PadicRational::add` (`padic_rational.rs:175-212`) shifts the higher-valuation unit at the other operand's
  relative precision and so drops known digits (sound but lossy; equality no longer depends on it).
* `PadicInteger::lift` (`padic_integer.rs:106`) raises the precision without new digits (claims precision it does
  not have).
* `PadicExtension::generator()` (`extension.rs:419-434`) sets coefficient 1 only when degree >= 2, so for a
  degree-1 extension it returns 0 instead of the root -a_0/a_1 of f.
* `rustmath-ellipticcurves/src/generic.rs:79-82`: the module doc still says `PrimeField`'s `zero()`/`one()` need
  parameters (stale since `5ff8749`).
* `rustmath-rings/src/quotient_ring.rs:203` `QuotientRingGeneric::is_integral_domain` always false
  (status_overrides `ch017.IsDomain` facade).
* `PadicExtension::new` accepts any polynomial under any `ExtensionType` label; `field_status()` now records
  whether the result is a field, but the constructor still does not refuse a reducible f, and the
  `ramification_index()`/`residue_degree()` it reports come from the label, not from f (an Eisenstein f labelled
  `Unramified` reports e = 1; OM's `ramification_data` could check the label).

---

## Plan

### Files, in implementation order (each step keeps its crate build- and test-green)

1. **rustmath-core** `src/traits.rs`: doc comments only: :97-100 (the hook section: the p-adic types now return
   unbound constants like the other parent-carrying types), :609-612 (`z6_ring!` comment: "like a parent-carrying
   type without a sentinel"). P1's `static_is_commutative` is used as it is.
2. **rustmath-rings** `src/category_methods.rs`: the helpers of 0.2; `RingMorphismMethods::domain_one`,
   `is_nonzero -> Option<bool>` (B.2); the two new default rules of D.2 in the rustmath-rings trait (prime order
   gives the characteristic; a non-prime-power order is not a domain) and `prime_u64` in `is_prime_field`; doc
   fixes at :183-192, :228-235, :461-467; tests of B.4 and the rustmath-rings part of D.4
   (`test_prime_order_alone_decides` changes to `Some(7)`). The two tests of E.4 are not touched here.
3. `src/padics/padic_integer.rs`: sentinel (A.3.1) with the zero rules, equality, `pow`, hooks and
   `static_characteristic` (E.2), type docs (0.1 points 1 to 4), tests A.5/E.5. Module header cites MAGMA Ch. 47
   §47.7.1, §47.8, Ch. 17 §17.5.2, §17.5.5 and Sage. In the same step: `src/padics/factory.rs`
   `adjust_precision` binding and its test; `src/category_methods.rs` the two E.4 test updates (they depend only on
   `PadicInteger`).
4. `src/padics/padic_rational.rs`: sentinel, `bind_integer`, `normalized`, `representative` (moved from
   extension.rs:86), `absolute_precision`, `from_rational_to_absolute_precision`, value equality and `is_one`,
   `from_rational(0)`, the `Field` impl (`inverse`, `divide`), `Div` rules, `pow`, hooks, type docs, tests;
   `src/padics/newton_lift.rs` `padic_to_int` and its test.
5. `src/padics/extension.rs`: `Option<Arc<PadicExtension>>`, private `ext()`, binding in `new`/`from_base_field`,
   sentinel coercions, equality, `is_one`, `pow`, `FieldCertificate`, the three new `PadicExtension` fields and
   their accessors (`field_certificate`, `is_certified_field`, `is_reduced`, `field_status`), `inverse` with
   `rational_solve` (E.6), hooks, `PadicEmbedding::apply` binding, `PadicRational::representative` in
   `multiplication_matrix_rational`; tests. `src/padics/mod.rs`: add `FieldCertificate` to the `pub use
   extension::{..}` list (:33).
6. `src/category_parents.rs` (new) and `pub mod category_parents;` in `src/lib.rs`: C.2 and its tests.
7. **rustmath-padics** `src/lib.rs`: no code change (re-exports keep their paths); extend the smoke test with
   `<PadicInteger as Ring>::one() + x`.
8. **rustmath-category**: `Cargo.toml` (`rustmath-integers` moved to `[dependencies]`; add `rustmath-finitefields`,
   `rustmath-matrix`, `rustmath-reals`); `src/ring_category.rs` D.2 and its tests; `src/ring_parents.rs` (new, C.3)
   with `mod ring_parents;` in `src/lib.rs`.
9. **rustmath-numberfields**: `Cargo.toml` (`rustmath-category`); `src/category.rs` (new, C.4) with `pub mod
   category;` in `src/lib.rs`; its tests.
10. **rustmath-algebras**: `Cargo.toml` (`rustmath-category`); the impl of C.5 in `src/algebra_with_parent.rs`;
    its test.
11. **rustmath-powerseries**: `src/series_core_impls.rs` (PowerSeries hooks), `src/laurent.rs`, `src/puiseux.rs`,
    `src/lazy.rs` (E.3, with the precondition of M10 in the docs); new `tests/ring_predicates.rs`. File headers cite
    MAGMA Ch. 49 §49.4.4 and Ch. 50.
12. **Downstream tests and comments** (test-only or comment-only edits in crates outside the list; each keeps its
    crate green): rustmath-schemes `src/elliptic_curves/tests.rs` (un-ignore six tests, add
    `test_padic_doubling_matches_pari`, A.5); rustmath-finitefields `tests/ring_predicates.rs:492-495` and
    rustmath-algebras `src/algebra_with_parent.rs:47, :315` ("as the p-adic element types' still do" → past tense
    or "like a type without a sentinel").
13. **Trackers**: PROJECT_TRACKER.md rows :361 (series and p-adic hooks decided), :819 ("the p-adic types still
    panic" → their sentinel), :846 (shim), :1046 (category: composite rule, impls), :1108 (🔴 → 🟢: value equality
    and `is_one` for `PadicRational` and extension elements; the lossy `Add` stays as a note), :1148 (the extension
    inverse exists; `Mul`'s reduction note stays), :1165 (`is_nonzero` fixed, real impls), plus new rows for
    `category_parents.rs`, `ring_parents.rs`, the number-field parent and the series hooks;
    `docs/magma/status_overrides.csv`: `ch017.Characteristic`, `ch017.IsCommutative`, `ch017.IsField`,
    `ch034.IsField`, `ch047.Characteristic`, `ch047.IsUnit`, `ch049.IsUnit`, `ch049.IsNilpotent`;
    `python3 tools/magma_tracker/build.py`; `docs/HANDOFF.md:318-320` and the "series and p-adics" words at :331-332
    (P1 removes "polynomials"); the F.2 and F.4 items as new HANDOFF follow-ups.

### Acceptance commands

```bash
cargo test -p rustmath-core --no-fail-fast
cargo test -p rustmath-rings --no-fail-fast            # padics, category_methods, category_parents
cargo test -p rustmath-padics --no-fail-fast
cargo test -p rustmath-category --no-fail-fast
cargo test -p rustmath-powerseries --no-fail-fast
cargo test -p rustmath-numberfields --no-fail-fast    # C.4
cargo test -p rustmath-algebras --no-fail-fast        # C.5 and the comment edits
# downstream crates that depend on the changed crates
cargo test -p rustmath-schemes elliptic_curves --no-fail-fast   # the six un-ignored tests and the new one
cargo test -p rustmath-ellipticcurves --no-fail-fast
cargo test -p rustmath-modules --no-fail-fast         # depends on rustmath-category
cargo test -p rustmath-graphs --no-fail-fast          # depends on rustmath-powerseries
cargo test -p rustmath-curves -p rustmath-affineschemes -p rustmath-quantumgroups --no-fail-fast  # depend on rustmath-rings
cargo test -p rustmath-finitefields --no-fail-fast    # comment-only edit
cargo check --workspace --tests
cargo clippy -p rustmath-rings -p rustmath-category -p rustmath-powerseries -p rustmath-numberfields
```

The schemes tests pass in the prototype: every line of `proto_padic.rs` that mirrors them prints `true`. The
extension inverse cases pass in `rev1/proto_ext_inverse.rs`.

### Risks

* Value equality is weaker than identity and not transitive (0.1, point 1). No `PadicInteger`/`PadicRational` is
  hashed or used as a map key (grep); the Galois-group tests compare conjugates that differ in their first digits.
  Run the whole rustmath-rings suite.
* The unbound 0 stays unbound through products (M12). A caller that multiplies by `R::zero()` and then reads an
  accessor gets the documented panic instead of a bound zero; no such caller exists in the workspace (A.2).
* The `Option` field in `PadicExtensionElement` touches 38 sites; a missed one fails to compile, so the risk is
  only effort.
* `field_status` runs the OM factorisation on first use for an extension without a certificate (0.2 to 46 ms in
  debug for degree <= 6, `rev1/om_cost.log`); an OM `Err` (p >= 2^31, a refinement guard) gives `None`, never a
  guess. It is cached per `PadicExtension` value (a clone copies the cache).
* `is_nonzero` changes its return type; only tests call it.
* New dependency edges: rustmath-category → {finitefields, matrix, reals} (already in its closure),
  rustmath-numberfields → rustmath-category, rustmath-algebras → rustmath-category. `cargo metadata` shows no
  cycle; the only new compile-order constraint is that rustmath-numberfields and rustmath-algebras now wait for
  rustmath-category (which waits for rustmath-groups, already a dependency of rustmath-numberfields).
* Units landing earlier: if I1's `try_is_prime`/`is_prime_u64` or P1's `static_is_commutative` is missing or
  renamed, use the fallbacks of section 0; the tests that depend on P1 (`power_series_over_z4` second assertion,
  the commutative Laurent/lazy rules) assert the `None` branch in that case. If I1 has landed, the HEAD fallback
  `is_prime_miller_rabin(n, 1)` must not be used (0.2).
* Hook cost: `cheap_primality(p)` runs only in the `PadicInteger`/`PadicRational` order hooks, and only below
  psi_12 (one pass of the strong tests, or I1's word test); the parent predicates and `PadicExtension::new` use
  `proven_primality` (with I1, one ECPP proof per prime above psi_12, cached by I1).
* A polynomial built from `var()` over a p-adic type keeps unbound integer coefficients until they meet a bound
  element; since Z → Z_p is injective this changes no zero test of an exact integer, but `unbound(5^10)` becomes
  0 + O(5^10) once bound at precision 10 (documented in the type docs).
* This design changes `src/` of two crates outside the unit list (rustmath-numberfields C.4, rustmath-algebras
  C.5). Both changes are additive; if they are ruled out of R1, drop Plan steps 9 and 10 and list C.4 and C.5 as
  hand-offs with this spec.
