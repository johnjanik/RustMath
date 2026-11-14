# Review 1 of the R1 implementation (rings, p-adics, category)

Design: `design/R1-rings-padics-category.md` (Revision 1 body), reviews 1 and 2.
Report: `impl/R1-rings-padics-category.md`. Tree: HEAD `4b7f034` plus the uncommitted R1 diff
(28 modified files, 4 new files; all read in full or as diffs with their surroundings).
Scratch: `impl/R1-rings-padics-category.rev1/` (`ref.gp` + `ref.log`, `ref2.gp`, `accept.log`,
`check.log`, probes `probe/p1.rs` .. `probe/p6.rs` built against the current debug rlibs with
`probe/rc.sh`).

**Verdict: not approved. 1 blocker, 4 minor issues.** The unit is otherwise complete and
careful: every design item is implemented, the review-2 notes are folded in (N5 and N7b skipped
with a stated reason), the tests pin PARI/GAP values, and my soundness probes found no wrong
answer except the blocker below. The fix is one line plus a test.

## Blocker

### B1. A constant defining polynomial gives the zero ring, but the new parent impl says characteristic 0 and nonzero

`PadicExtension::new` (`extension.rs:327-329`) rejects only the zero polynomial: its check is
`defining_polynomial.degree().ok_or_else(.. "Polynomial must have positive degree")`, and
`degree()` of a nonzero constant is `Some(0)`. So `PadicExtension::new(5, 3, 10, Unramified)` is
`Ok` with degree 0. Then K = Q_5[x]/(3) = 0 (3 is a unit), but the new
`RingParentMethods for PadicExtension` (`category_parents.rs:332-361`) answers
`characteristic() == Some(0)`, `is_zero_ring() == Some(false)`, `is_finite() == Some(false)`,
`krull_dimension() == Some(0)`, and the new `PadicExtensionElement::static_characteristic()` is
`Some(0)` although the values of this parent lie in the zero ring. Probe `probe/p5.rs`:

```
zero coeffs 0, char Some(0), zero_ring Some(false), finite Some(false), noetherian Some(true), krull Some(0)
static char Some(0)
is_one on degree-0 element panics: true
```

The acceptance of degree 0 is pre-existing (and the elements of such a K already panic in
`is_one` on `coefficients[0]`), but the false `Some` answers are new: a decided wrong answer on a
reachable public input. `unramified(5, 0, 10)` is already refused (`unramified_modulus`), and
`eisenstein` refuses a constant (`is_eisenstein` needs v_p(c_0) = 1 with a unit leading
coefficient), so only `new` is affected.

Required change: in `PadicExtension::new`, return
`Err(InvalidArgument("Polynomial must have positive degree"))` when `degree == 0` (the message
the code already has), and add a test (`PadicExtension::new(5, [3], 10, Unramified)` is `Err`).
That also makes the doc claims "K is nonzero, characteristic 0" of `category_parents.rs:328-331`
and of `FieldCertificate` hold for every constructible K.

## Minor issues

### m1. Value equality and `is_one` of `PadicRational` cost grows with the valuation, not the precision

`representative()` builds `u·p^v` exactly and `lies_in_power` builds `p^(m+k)` and does rational
arithmetic (with gcds) on numbers of `|v|·log2 p` bits. Measured in debug (`probe/p1.rs`,
`probe/p4.rs`, x = `from_rational(5).pow(v)` in Q_5 at precision 10):

| v | `x == 5` and `x.is_one()` | `x + 5` (HEAD's `Add`, for comparison) |
|---|---|---|
| 10^3 | 0.09 ms | 1.5 ms |
| 10^5 | 53 ms | 150 ms |
| 10^7 | 48 s | (linear, about 15 s) |

Design A.6 says value equality is O(M(log p^n)) "as cheap as a subtraction". It is as cheap as
HEAD's subtraction (which also loops over the valuation difference), so this is not a new class
of cost, but `==` and `is_one` were O(1) at HEAD and `is_one` runs in every order hook. A bounded
form exists: with v0 = min(v_x, v_y) and m the smaller absolute precision, test
`p^(m - v0) | u_x p^(v_x - v0) - u_y p^(v_y - v0)` on integers, dropping a term whose shift is
at least m - v0; for an integer constant c compare v_p(c) with v first. Every exponent is then at
most the larger relative precision. Either do that or record the cost in the HANDOFF and correct
the A.6 claim in the tracker row.

### m2. The stated convention covers `is_zero` only, but the order and unit rules also treat `is_one` as exact

The type docs (`padic_integer.rs:29-33`, repeated by reference in `padic_rational.rs` and the
extension doc) say the predicates are proofs "under the convention that `is_zero` is exact". The
order hooks answer `Finite(1)` for `1 + O(5^10)` and `Finite(2)` for `-1 + O(5^10)`, and
`PadicExtensionElement::unit_status` answers `Some(true)` for ±1 in any K; the lift
`1 + 5^10` has infinite order, so these are proofs only if `is_one` is also exact (design E.2:
"the same convention"). The report's deviation 5 argues the opposite for `unit_inverse` ("an
element 1 + O(p^n) need not be invertible when g is not integral") while `unit_status` keeps
`Some(true)` for it. Required: state the convention as "`is_zero` and `is_one` are exact" in the
type docs, and either reword deviation 5 (the `NotImplemented` there is just conservative) or
let `unit_inverse` return `self.clone()` for ±1 like the core default.

### m3. `test_wrappers_agree_with_real_impls` compares only whether the characteristic is 1

`category_methods.rs` (new test): `tuple` uses `r.characteristic().map(|c| c == 1)`, so for Z/6,
Z/7 and GF(25) a wrong characteristic other than 1 would pass. Only GR(8, 2) compares the value.
Compare `characteristic()` itself for all four (cheap; the design asked for "the same tuple").

### m4. `field_status` is cached per clone, and `generator()` clones

`PadicExtension::generator()` clones the extension into a fresh `Arc`, so on an extension without a
certificate each `ext.generator().unit_status()` made before the cache of `ext` is filled runs
the OM factorisation again (0.1 to 5 ms per run in `probe/p3.rs`). Not unbounded; worth one line
in the `field_status` doc or a note in the HANDOFF.

## What I checked

* **Items.** A (sentinels, binding rules, value equality, `is_one`, `from_rational(0)`, normalised
  inverse, `Field::divide`, `pow`, `adjust_precision`, `padic_to_int`, `new`/`from_base_field`
  binding, `PadicEmbedding::apply`), B (`domain_one`, `is_nonzero -> Option<bool>`), C (Z, Q,
  R[x], Zmod, GF(q), GR, Z_p, Q_p, K in rustmath-rings; Zmod, GF(q), GR, `MatrixAlgebra`,
  `BigFloatField` in rustmath-category; `NumberField`; `DownUpAlgebra`; `DiffScalarFieldAlgebra`
  from review 2 N4), D (both traits), E (three p-adic types, four series types, E.6 inverse), the
  F1 handoff (the three unbound-constant hooks on the three p-adic types, core hook doc) and P1's
  `static_is_commutative`. All present.
* **Reference values (PARI/GP, `ref.log`, `ref2.gp`).** Every gp value quoted in the new tests
  re-run and matching: 162, `Mod(557,625)`, 268, `Mod(16,5^10)`, 6139557 with `^4 == 1`, 1037,
  `Mod(256,1296)`, 16, 8138021, 4882814, `5^-1 + O(5^8)`, the six exact inverses and PARI's
  polmod-inverse precisions, `factorpadic` counts 1, 2, 2, 1, `ellmul`/`elladd`/valuation -6, all
  series inverses, lazy coefficients -1 and 1, the primality and prime-power values, the Z_6
  idempotent 24151041, `polisirreducible` 1, 1, 0, 1.
* **Soundness probes.**
  * `probe/p2.rs`: E.6 inverse precisions. For 8 certified extensions (Eisenstein monic and
    non-monic, irreducible reduction monic and non-monic, a linear f with g = 1/7, degree 3 at
    p = 2 and 3) and 480 random elements with random coefficient precisions, the inverse of 5
    random exact lifts each (computed at precision 80) agrees with x^-1 to the claimed precision
    of every coefficient: 4945 comparisons, 0 mismatches.
  * `probe/p6.rs`: value equality and `is_one` for p = 6, 12, 4 against the definition
    "d ∈ p^m Z_p iff v_q(d) >= a_q m for every q | p": 9000 random pairs, 0 mismatches (review 2
    N1 is fixed).
  * `probe/p3.rs`: `field_status` for prime-power p (9, 4, 25, 8, 27, 49) and for OM cases up to
    degree 16: every answer matches `factorpadic` (gp: `x^6+5x^3+125` and `(x^2-5)(x^2-10)+5^10`
    over Q_5 have 2 factors, `7x^12+2` over Q_7 has 1); each run under 6 ms.
* **No undecided result returned as decided** (apart from B1): unbound constants other than 0, ±1
  never get a unit or inverse claim in Z_p; `Err(NotImplemented)` / `NumericalError` for the
  undecided inverses, never `NotInvertible`; OM `Err` gives `None`; the lazy hooks never decide
  from the 20-term window; `MatrixAlgebra` claims only for structurally full algebras.
* **No new unbounded loop.** The new loops are bounded by the degree, the precision or the
  20-term window; OM has its own round guard. Costs above psi_12 are I1's cached ECPP proof (in
  `PadicExtension::new` and the parent rows), as the design states. m1 is the one cost that grows
  with a value rather than a size.
* **Assertions.** The only changed existing assertions are the three the design names
  (`test_prime_order_alone_decides`, the two E.4 tests): `None` replaced by proven values with GAP,
  Sage and gp references. No deleted assertion; the moved `same_extension` and same-prime asserts
  are kept; six `#[ignore]`s removed in rustmath-schemes and the tests pass.
* **Callers.** `grep` of the p-adic types, `RingParentMethods` implementors and `is_nonzero` callers
  matches the report; method resolution at concrete call sites unchanged (`cargo check
  --workspace --tests` green). No hashing of p-adic values.
* **Hygiene.** No `unsafe`; no em-dashes in added lines; the four new files are rustfmt-clean and
  cite MAGMA chapters and papers; CRLF files stay CRLF throughout; no `rustmath-misc/output.txt`
  or `.tmp_dir_*` changes; trackers and docs untouched.

## Review-2 notes skipped by the implementation

* N5 (certificate, E.6 inverse and extension order rule at a proper prime-power p): skipped
  because `PadicRational::from_rational` cannot build 1/2 in Q_4. Correct reason; `field_status`
  and the integer order rule do use q. Fine as a HANDOFF item (report item 4 and 12).
* N7b (`try_prime_power` in rustmath-integers): optional, out of the unit's crates. Fine.

## Acceptance commands (this review)

Run here on the uncommitted tree (`accept.log`, `check.log`, `clippy.log`):

| Command | passed | failed | ignored |
|---|---|---|---|
| `cargo check --workspace --tests` | green | | |
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
| `cargo test -p rustmath-matrix` | 295 | 0 | 1 |
| `cargo test -p rustmath-manifolds` | 290 | 0 | 11 |
| `cargo test -p rustmath-polynomials --test ring_predicates` | 10 | 0 | 0 |
| `cargo clippy --no-deps -p rustmath-rings -p rustmath-category -p rustmath-powerseries -p rustmath-numberfields` | exit 0; no lint on a line this unit added | | |

The numbers equal the report's. The tests pass; the blocker is a missing check, which no
existing test exercises. No release build; the probe binaries were deleted.

## For the fix round

1. B1: refuse degree 0 in `PadicExtension::new`, with a test.
2. m2 and m3: a doc sentence and a stronger test assertion (cheap).
3. m1 and m4: fix, or hand to the HANDOFF with the measured numbers above.
