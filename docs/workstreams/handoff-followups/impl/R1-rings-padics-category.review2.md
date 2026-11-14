# Review 2 of the R1 implementation (rings, p-adics, category)

Design: `design/R1-rings-padics-category.md` (Revision 1 body), design reviews 1 and 2.
Report: `impl/R1-rings-padics-category.md`, including "Fix-up round 1". Previous review:
`impl/R1-rings-padics-category.review1.md` (1 blocker, 4 minor issues).
Tree: HEAD `4b7f034` plus the uncommitted R1 diff (28 modified files, 4 new files). I read the
whole diff and the four new files in full, with the surrounding code.
Scratch: `impl/R1-rings-padics-category.rev2/` (`check.log`, `rings.log`, `accept.log` and
`t_*.log`, `clippy.log`, `ref/ref.gp` + `ref.log`, `ref/ref2.gp` + `ref2.log`, `ref/ref.g` +
`ref_g.log`, probes `probe/q1.rs` .. `probe/q4.rs` with `q1.gp`, `q3.log`, `q4.log`, and the
review-1 probes `p2.rs`, `p6.rs` re-run; binaries deleted).

**Verdict: approved. No blockers, 1 minor issue, 1 adjacent finding.** The blocker and the four
minor issues of review 1 are fixed, each with a test. Every design item is implemented. My
probes found no wrong decided answer. All acceptance commands pass.

## Review 1 items

| Item | Status | Checked how |
|---|---|---|
| B1: constant f gave the zero ring with "characteristic 0, nonzero" | fixed | `PadicExtension::new` (`extension.rs:327-336`) now refuses `deg f = 0`. `test_constant_polynomial_rejected` covers f = 3, 5, 0; it fails at HEAD (f = 3 was accepted). `unramified` and `eisenstein` go through `new`; grep finds no other `PadicExtension {` literal. |
| m1: `PadicRational` `==` / `is_one` cost grew with the valuation | fixed | `divides_shifted_difference` and `value_equals_integer` form no power of p above p^n (or about c^2). I re-derived both tests, including the bit-length shortcut for `v > 0` (`2^(v·floor(log2 p)) <= p^v`). The review-1 probe `p6.rs` (9000 pairs, p = 6, 12, 4): 0 mismatches. `5^100000 == 1` takes 0.002 ms (`q2.rs`). The randomized test `equality_matches_the_definition` checks against `v_q(d) >= a_q m`. |
| m2: the convention must cover `is_one` | fixed | The three type docs now say "`is_zero` and `is_one` are exact". `unit_inverse` of ±1 returns ±1 when `inverse()` is undecided, so it agrees with `unit_status`. Tested in four rings. |
| m3: wrapper test compared only "char == 1" | fixed | The characteristic is asserted for GR(8,2), Z/6, Z/7 and GF(25). GAP `[6, 7, 5]` (I re-ran it). |
| m4: `field_status` cache per clone | fixed | `Arc<OnceLock<..>>`, shared by clones. It is sound because p and f never change after `new`; the only `&mut self` method is `set_generator_name`. Test `field_status_cache_is_shared_by_clones`. |

## Minor issue

### m1. The extension-element order hook (and the undecided `inverse`) cost grows quadratically with the valuation

`PadicExtensionElement::certified_valuation` (`extension.rs:902-944`) runs for every
`multiplicative_order_status` call on an element of a certified K, and it runs at the start of
`inverse`. It builds the exact representative `u·p^v` of each coefficient (`representative()`).
Then `rational_valuation` calls `Integer::valuation`, which divides by p once per unit of valuation.
That costs O(v^2) bit operations. Before round 1, review 1's m1 had the same cost class for
`PadicRational` `==` and `is_one`; round 1 fixed those two but not this hook. Measured in debug
(`probe/q3.rs`, `q3.log`) on Q_5(√5), `x = from_base_field(five.pow(v))`. The element is built in
0.02 ms, because `pow` only adds valuations:

| v | `is_one`, `unit_status` | `multiplicative_order_status` | `inverse()` (an undecided `Err(NumericalError)`) |
|---|---|---|---|
| 10^3 | < 0.01 ms | 0.7 ms | 0.6 ms |
| 10^4 | < 0.01 ms | 33 ms | 35 ms |
| 3·10^4 | < 0.01 ms | 306 ms | 276 ms |
| 10^5 | < 0.01 ms | 3.1 s | 2.8 s |

The cost is quadratic, so about 5 minutes at v = 10^6. At HEAD this hook was the O(1) default.
The answers are correct (`Undecided`, because the `g` coefficient is known only to O(5^10)). V and
E can be read from the stored data without any large integer: in a certified field p is prime, so
`v_p(r_i) = valuation_i + v_p(unit value_i)`, with the unit value below p^n. A_i is
`absolute_precision()`. Then the order hook costs O(n^2) in the relative precision, and `inverse`
can return its `NumericalError` before it builds a representative. The decided inverse still needs
exact representatives (E.6 solves `M_x y = e_1` over Q), as `norm` and `trace` already did at HEAD.
`PadicRational::from_rational` also has a valuation loop, linear in the number of divisions.
Required change: compute V from the stored valuations in `certified_valuation` (about 3 lines).
Alternatively, record the cost in the HANDOFF next to item 5 ("`PadicRational::add` ... linear in
the valuation difference").

## Adjacent finding (pre-existing at HEAD, for the HANDOFF)

* `PadicRational` `Mul` adds the two `i32` valuations without a check (`padic_rational.rs`, the
  `(true, true)` arm, the same line as `padic_rational.rs:228` at c79a582). So `x.pow(n)` with a
  total valuation above 2^31 panics in debug and wraps in release, which gives a wrong valuation.
  This is reachable cheaply (`five.pow(3_000_000_000)`). HEAD had the same `Mul`, and its default
  `pow` used it too. R1's square-and-multiply `pow` does not change this.

## What I checked

* **Every design item, at HEAD.**
  * A: the sentinel contract for the three types, the exact-zero rules, value equality, `is_one`,
    `from_rational(0)`, the normalised inverse, the `Field::divide` override, the `Div` messages,
    `pow`, `adjust_precision`, `padic_to_int`, binding in `new`/`from_base_field`, and
    `PadicEmbedding::apply`.
  * B: `domain_one` and `is_nonzero -> Option<bool>`.
  * C: Z, Q, R[x], Zmod, GF(q), GR, Z_p, Q_p and K in rustmath-rings. Zmod, GF(q), GR,
    `MatrixAlgebra` (the structural `is_full`) and `BigFloatField` in rustmath-category.
    `NumberField`. `DownUpAlgebra` (feature `full`; the dependency is optional, as the report
    says). `DiffScalarFieldAlgebra`.
  * D: both traits.
  * E: three p-adic types, four series types, and the E.6 inverse.
  * Design review 2 notes: N1, N2, N3, N4, N6, N7a done. N5 is partly done and N7b skipped, with a
    valid reason: `from_rational(1/2, 4, n)` is `Err`, so a certificate at a proper prime power
    would need a new constructor.
  * Earlier units' hand-offs: I1's helpers (`is_prime_u64`, `try_is_prime`) are used. I1's
    p-adic primality switches are intact. P1's `static_is_commutative` is set. F1's three
    unbound-constant hooks are on the three p-adic types. F1's container row for
    `PowerSeries`/`LaurentSeries` was handed to the HANDOFF by the F1 design (§7.6), not to R1,
    and the report lists it.
* **Correctness re-derived.**
  * The integer membership tests of `==` and `is_one`, also for composite p.
  * The order rules, with the minimal root q for prime powers (μ(Q_q) = μ_(q-1), or {±1} for
    q = 2).
  * `unit_status` for a prime-power p.
  * The three certificates and their valuation formulas.
  * `field_status` steps 2 to 5, including `F(y) = lc^(d-1) f(y/lc)`.
  * The E.6 bound.
  * The parent rows, including K's "Artinian domain ⇒ field" for `is_integral_domain = field_status`.
  * The R[x] rules.
  * The series recurrences (a right inverse, also for non-commutative R) and the Laurent, Puiseux
    and lazy rules.
  * The composite-characteristic, prime-cardinality and non-prime-power-cardinality rules.
* **Soundness probes.**
  * `p2.rs` (review 1): the E.6 inverse precisions, 4945 comparisons with random lifts, 0
    mismatches.
  * `q1.rs`: `field_status` for p = 4, 8, 9, 25, 12, for non-monic f (5x^2-1, 25x^2-1, 2x^2-2),
    for (2x+1)^2, for large coefficients (x^2-5^200, x^2-2·5^201, x^3-2·5^41) and for degree 20.
    Every answer matches `factorpadic` and `issquarefree` in gp (`q1.gp`). Each run takes 7 ms or
    less.
  * `q2.rs`, `q4.rs`: large p. `PadicExtension::new(2^127-1, x^3+2)` takes 52 ms in debug, then
    2.3 ms (the ECPP proof is cached, as design A.6 says). `field_status` is `None` because q is at
    least 2^31 (honest). The order hooks for a 2048-bit p take about 1 ms.
* **Reference values.** I re-ran them myself.
  * PARI/GP 2.15, all matching (`ref.log`, `ref2.log`): 162; `Mod(557,625)`; `5^-1 + O(5^8)`;
    4882814; 268, 8138021, 1037; 6139557 with `^4 == 1`; `Mod(256,1296)`; the six exact
    extension inverses; PARI's polmod-inverse precisions O(5^10), O(5^9), O(5^9), O(2^8);
    `factorpadic` counts 1, 2, 2, 1; `ellmul`, `elladd` and valuation -6.
  * PARI also gives `ellmul(E,[52+O(5^10),375+O(5^13)],2)[1]` = `... + O(5^4)`, the absolute
    precision 4 that the new schemes test asserts.
  * PARI, further values: `polisirreducible` 1, 0, 1, 1; the primality and prime-power values;
    the Z_6 idempotent 24151041; every series inverse; lazy coefficients -1 and 1; 22.
  * GAP 4.12, all matching (`ref_g.log`): Integers `[0, true, true, false, false]`; `IsField` of
    Z/(2^64+13) and Z/(3·2^64); the characteristics `[6, 7, 5]`; M_2(Q) `[0, false]`; M_1(Q)
    commutative; M_2(GF(5)) `[false, 5]`; `IsPrimePowerInt`; U(sl_2) not commutative; both
    polynomial rings infinite; Q(√2) a field.
* **No undecided result returned as decided.** Unbound constants other than 0 and ±1 never get a
  unit or inverse claim in Z_p. `NotImplemented` and `NumericalError` are used for undecided
  inverses, never `NotInvertible`. An OM `Err`, or q >= 2^31, gives `None`. The lazy hooks never
  decide from the 20-term window. `MatrixAlgebra` claims only for `full` algebras.
  `NumberField::new` already panics on degree 0, so the "characteristic 0, infinite" rows hold.
* **Unbounded loops and allocations.** No new loop without a bound. The one cost tied to a value
  rather than a size is m1 above. It is bounded by the `i32` valuation and gives no wrong answer.
* **Assertions.** The only changed existing assertions are the three the design names:
  `test_prime_order_alone_decides`, `test_defaults_on_padic_integer_do_not_panic` (renamed
  `test_padic_integer_hooks_decide`) and `test_parent_carrying_elements_do_not_panic`. Each
  `None` became a proven value, with GAP, gp or Sage references. No assertion was deleted.
  The moved same-prime asserts are kept. In rustmath-schemes, six `#[ignore]`s were removed
  (13 became 7), and the seven p-adic tests run and pass.
* **Callers.** `is_nonzero` and `RingMorphismMethods` are used only by tests. The new trait impls
  are used only in the four new or extended files. No file imports both `RingParentMethods`
  traits. No p-adic value is hashed.
* **Hygiene.** No `unsafe`. No em-dash in added lines. The four new files pass `rustfmt --check`
  and cite MAGMA chapters and papers. CRLF files stay CRLF in every line (`category_methods.rs`,
  `rings/src/lib.rs`, the schemes tests). No `rustmath-misc/output.txt` or `.tmp_dir_*`
  changes. Trackers and docs are untouched. `target/` holds only `debug` and `tmp`.

## Acceptance commands (this review)

Run on the uncommitted tree:

| Command | passed | failed | ignored |
|---|---|---|---|
| `cargo check --workspace --tests` | green (exit 0) | | |
| `cargo test -p rustmath-core` | 50 | 0 | 1 |
| `cargo test -p rustmath-rings` | 1975 (1787 lib + 188 doc) | 0 | 82 |
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
| `cargo clippy --no-deps -p rustmath-rings -p rustmath-category -p rustmath-powerseries -p rustmath-numberfields` | exit 0 | | |

All numbers equal the report's. In the touched files, clippy warns only on lines from HEAD:
`extension.rs` 145, 531 and 1454, and `ring_category.rs` 9 and 11. No release build was made.

## For the orchestrator

* m1 is a three-line fix in `certified_valuation`, or a HANDOFF note with the table above.
* Add the adjacent finding (unchecked `i32` valuation in `PadicRational` `Mul`/`pow`) to the
  HANDOFF list of the report.
