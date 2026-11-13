# Review 1 of design P1 (rustmath-polynomials, with the pgalois and maclane callers)

Reviewer scratch: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/P1-polynomials/review1/`
(`$R` below). Programs there link against the HEAD debug rlibs (`$R/rc.sh`, with the rlib set
core `dc51523818e444e8`, integers `93894c98d3b259d4`, rationals `76279eac0299aa30`, groups `025a50b6d7157ed0`,
polynomials `7bc299bb35ca86d4`; the working tree is clean at `c79a582`).

## Verdict

**Not approved: 2 blockers, 11 minor issues.** Most of the design is sound and well evidenced. I reproduced the
defects, reran the prototype, and checked the reference values and the order audit. Two things must change.
First, the new `unreachable!` in pgalois can be reached by valid input. Second, the G hooks break an
existing test in `rustmath-algebras`, while the design says that no such test exists.

## What I checked, item by item

| Item | Claim | Reproduced? | Notes |
|---|---|---|---|
| A | DDF accepts non-squarefree input with wrong labels | yes (`$R/r1.rs`): `(x+1)^2` mod 5 gives `[(1,[1,1]),(1,[1,1])]`; `[1,2,1,0,1,2,1]` mod 5 gives `[(1,[1,1]),(2,[1,1,0,0,1,1])]`; `[0,0,1]` mod 2 gives degree 1 twice; `[0]` gives `[]` | The fix (eager check, private `ddf_squarefree`) is correct. All 3 callers pass squarefree input over primes. Note: `rustmath_galois::sieve::cycle_type_mod_p` is public and takes `p: u64`, so a composite p or p > MAX_P now panics there (minor m9). |
| B | `factor(x^2+1, 15) = [[1,0,1]]`; pgalois wrong at composite p, p = 0, 1, non-monic, untrimmed | yes, every row of B.1. I also found `factor([1,1,5],5)`, which panics in `div_mod`, and `factor([1,5],5)`, which panics in `pth_root` (`reduce` fixes both). | gp checks of `poldisc`/`factorpadic` agree with B.1 (`$R/pg.gp`). The MR bound holds: `factor(3215031751) = 151·751·28351`, strong to 2, 3, 5, 7 but not 11; `sqrtint(2^63-1) = 3037000499`. Using I1's public `is_prime_u64` is right (I1 design §2.1 makes it public). **Blocker 1**: the pgalois `unreachable!` is reachable. |
| C | `block_contract` keeps d + 1 rungs; maclane runs Rabin again; the literal in rings | yes (`fp_factor.rs:546-553`, `maclane.rs:1687-1694`, `typed/factor.rs:155-161`) | The sparse ladder in the prototype gives the same messages as HEAD on 12 inputs (rerun). Dropping the maclane Rabin test is sound: a degree-d divisor of a certified block is irreducible. `is_irreducible_fp` is still used at `:1257`. The C.4 gp inputs check out (`$R/ord2.gp`). |
| D | Drop phase 1, sort in PARI order | prototype rerun (`$R/proto/driver.rs`): 18 rows in PARI order; 600 random cases, 0 mismatches against HEAD sets and multiplicities | My own grep of the order audit agrees with D.3. I looked at numberfields `prime_ideals` users (`s_units.rs:91`, `s_unit_gens.rs:505`, `classgroup.rs:529-534`, `artin.rs:611/639`, `kummer.rs`, `rayclass.rs`), maclane leaves (`len()==1` before `[0]`, or loops), OM `ramification_data` (sorted), galois `monic_integer_roots` (unique root used), `reducible.rs` (order-free), modular decomposition (sorted keys), symbolic factor/Risch tests (`matches!`, sorted RatLog terms), rings typed factor tests (`any`), and functionfields `factor_over_qt` (degree only). No other test depends on the order. Timing: my reruns vary by up to ±40 % between runs, in both directions (minor m12). |
| E | `Err(())` also for "no good prime"; every caller blames the subset bound | yes; `x^2 - primorial(10^5)`: gp `kronecker(P, 100003) = -1`, so the new scan stops at the first prime past 10^5, no Hensel lift, and the test is cheap | The `ZassenhausError` variants and the 12 call sites are complete (grep). The termination argument is right. Plan order issue (m4). |
| F | Cohen numbers | Sage sources quote Alg. 3.1.2 (pseudo-division), 3.2.2 (extended Euclid), 3.3.1 (sub-resultant gcd), 3.4.2 (char-p squarefree, "basically Yun"); this matches my knowledge of GTM 138 | Small citation detail for §3.2 (m8). |
| G | Polynomial hooks keep the defaults | yes (`$R/r2.rs`): `2x+1`, `x`, constant `2` give `None`; `-1` gives order `Undecided`; `static_characteristic` gives `None` | The theory in G.2 is correct: evaluation at 0 is a ring map in any ring; top coefficient `a_n^k`; Atiyah-Macdonald Ch. 1 Ex. 2; for Laurent, a non-nilpotent `c_a c_b` avoids some prime P. GAP `units.g` rerun: all values as stated. **Blocker 2**: a missed test. |
| H | Conventions | V2.19 text confirmed in `dl_magma/ch23.txt:629-635` (Quotrem), `:1039-1042` (QMatrix "degree d − 1") and `:1058-1061` (CompanionMatrix also "degree d − 1", although the matrix is d × d); Sage doctests confirmed (`is_irreducible`: `R(2*x)` False over ZZ, True over QQ; `R(1)`, `R(4)` False; `R(5)` True); HEAD values confirmed (`$R/t7.rs`) | No behaviour change; evidence-only doc edits. Agreed. |

## Blockers

### Blocker 1. `unreachable!` in `unramified_local_galois_group` is reachable (B.4)

The design argues that if p ∤ disc(f), then `deg(f mod p) >= n - 1`. It then makes `nbar < n - 1`
an `unreachable!`. That is false for **n = 1**. A linear form has discriminant 1, so p never divides it. But
`f mod p` is 0 when p divides the content.

Evidence:
- `$R/r3.rs` (HEAD): `disc(6x+6) = 1`, `disc(2x) = 1`, `disc(-2x+4) = 1`. At p = 3, 2, 2 HEAD returns
  `Some { frobenius_cycle_type: [], order: 1, transitive: false }`. The true answer is `[1]`, order 1, transitive
  (the root is rational).
- `$R/pgrand.gp` simulates the design's algorithm over 13,986 random (f, p) pairs with p ∤ disc. It compares
  against `factorpadic(f, p, 10)`. 510 violations, every one linear with p | content (`6*x+6` at 3, `-2*x+4` at 2,
  `4*x` at 2, ...). The design's `unreachable!` would panic on each of them. No violation has degree >= 2, and the
  1,240 cases with p | lc and degree >= 2 all match `factorpadic`. So the root-at-infinity rule itself is right.

The argument holds for n >= 2 because p | content forces p^(2n-2) | disc. Required change:
- Work on the primitive part: `f = primitive_part(trim(f))`, then compute disc, fbar and degs on it.
  With p ∤ content(pp f) the binary-form argument holds for every n >= 1, including n = 1. A smaller fix is to
  special-case n = 1 (`[1]`, transitive).
- The primitive part also decides `3x^2 + 3` at p = 3. HEAD and the design return `None` there ("ramified"), although
  `x^2 + 1` is unramified at 3 (gp `poldisc(x^2+1) = -4`, `factormod(x^2+1, 3)` irreducible): cycle type `[2]`,
  order 2, transitive. If pp is used, update the doc to "None iff p | disc of the primitive part".
- Apply the same normalization in `local_galois_candidates`. The square class of the disc does not change:
  `disc(c·g) = c^(2n-2) disc(g)`.
- New tests in `rustmath-groups/tests/pgalois.rs`: `6x+6` at 3 gives `[1]`, order 1, transitive; `2x` at 2 gives `[1]`
  (both fail on HEAD: `[]`, not transitive); if pp is used, `3x^2+3` at 3 gives `[2]`, order 2, transitive
  (HEAD `None`). gp: `poldisc(6*x+6) = 1`, `factorpadic(6*x+6, 3, 8)` has one factor of degree 1.
- Correct the correctness paragraph of B.4 (the binary form F has disc 1 for n = 1 and can vanish mod p).

### Blocker 2. The G hooks break `rustmath-algebras` `test_simple_algebra_finiteness`, which the design missed

G.3 says "No existing test asserts `None` for a polynomial". That claim rests on a grep of the `tests/ring_predicates.rs`
files only. But `rustmath-algebras/src/algebra_with_parent.rs:274-277` asserts:

```rust
// The polynomial element type states no characteristic: undecided, not "infinite".
let poly2: SimpleAlgebra<UnivariatePolynomial<Integer>> = SimpleAlgebra::new(2);
assert_eq!(poly2.is_finite(), None);
```

`SimpleAlgebra::is_finite` (`:139-147`) returns `Some(false)` when `R::static_characteristic() == Some(0)`. With the
G.3 delegation, `UnivariatePolynomial<Integer>::static_characteristic()` is `Some(0)`, so this test fails. The test is
in the plan's acceptance set (`cargo test -p rustmath-algebras`). The new value is correct: Z[x]^2 is infinite (GAP
`IsFinite(PolynomialRing(Integers, 1))` is false). Required change:
- List this test in G.3, G.4 and the Plan. Change it to `Some(false)` and rewrite its comment ("the polynomial element
  type has characteristic 0 from its coefficients").
- Redo the hook-consumer grep over all test code (unit tests in `src/` too), not only `tests/ring_predicates.rs`.
  I found no other site: `rustmath-modules/src/with_basis/parent.rs:195/272` and
  `rustmath-matrix/src/matrix_algebra.rs:383` have no polynomial-coefficient tests, and no `RingParentMethods`
  implementor has a polynomial `Element`.

## Minor issues

- **m1 (B.6 tests).** `factor_reads_coefficients_mod_p` uses `[7,0,6]`, which already passes on HEAD (`make_monic`
  reduces when lc ≢ 1). Add the inputs where `reduce` matters most. `factor([1,1,5], 5) = [[1,1]]` panics on HEAD in
  `div_mod`. `factor([1,5], 5) = []` panics on HEAD in `pth_root`. `equal_degree_factor([5], 1, 5)` should panic with
  "f = 0" (HEAD returns `[]`).
- **m2 (B.6 tests).** `check_modulus_boundary` lists 3215031751, but that is above MAX_P, so it tests only the range.
  To guard the primality test itself, add strong pseudoprimes below MAX_P, each expected `Err`: 2047 = 23·89
  (spsp(2)), 1373653 = 829·1657 (spsp(2,3)) and 25326001 = 2251·11251 (spsp(2,3,5)). Checked with gp
  (`$R/spsp.gp`: `isprime` 0; strong to bases [2,3,5,7] = [1,0,0,0], [1,1,0,0], [1,1,1,0]).
- **m3 (I1 overlap).** I1 is implemented first and edits `fp_factor.rs` (the `check_prime` doc, `:502`),
  `factorization.rs` (`integer_constant_is_irreducible`) and `tests/ch23_is_irreducible.rs` (P1's H.3 adds a test
  there). I1 §7 also hands P1 an optional item: `berlekamp::check_prime` should map an unresolved p to `Err` via
  `try_is_prime`. The Plan should say "rebase on I1", and accept or decline that hand-off explicitly.
- **m4 (Plan order).** Step 4 changes `zassenhaus::factor`'s error type. `rustmath-curves/src/belyi/regular_certify.rs:215`
  (`Err(()) =>`) then stops compiling until step 10. Either move that one-line fix into step 4, or say that
  `cargo check --workspace` is expected to be red between steps 4 and 10.
- **m5 (G `unit_inverse` cap).** "At most 2^16 terms" bounds the number of terms, not the work. `(-M)^k` has degree up
  to k·deg M, and over `Z/2^e` with a large e (Integer-backed `IntegerMod`) the nilpotency index is about e. So the
  loop can run for hours before it returns `Err(NotImplemented)`. Use a work bound (total coefficient operations, or
  a degree cap) or a much smaller term cap. The result stays honest either way.
- **m6 (G Laurent).** The monomial inverse `c^(-1) x^(-k)` overflows `i32` for `k = i32::MIN`. Use `checked_neg` and
  return `Err(NotImplemented)`.
- **m7 (G scope wording).** The task text names parent-level hooks (`characteristic`, `is_field`). Say explicitly that
  no polynomial parent type implements `Parent` or `RingParentMethods`. The only `PolynomialRing` struct
  (`rustmath-algebras/src/cached_algebra.rs:15`) is a caching demo with no ring structure. So the element-level
  hooks, including `static_characteristic`, are all there is to implement.
- **m8 (F).** "Primitive parts" are Cohen §3.2.3 (Polynomials over UFDs); Euclid over a UFD is §3.2.4. Cite "§3.2.3-3.2.4"
  in `univariate.rs:18-19` and `:1042`, not §3.2.4 alone.
- **m9 (callers table).** Some public entries take a caller-supplied p and now panic on a composite p or p > MAX_P,
  where they returned meaningless data before: `rustmath_galois::sieve::cycle_type_mod_p` (through DDF);
  numberfields `prime_decomposition`, `prime_ideals`, `local_decomposition`, `panayi` (through `fp_factor::factor`).
  No caller passes such a p (I grepped the literal primes: 2, 3, 5, 7 only). List them in B.5 and A.3 as behaviour
  changes.
- **m10 (constant input to pgalois).** `local_galois_candidates` on a constant f returns `([], false)` at HEAD (empty
  table). With the eager degree check it would panic. State this in B.5. It is acceptable as a documented panic.
- **m11 (DDF test).** `ddf_constant_and_squarefree` also passes on HEAD, as stated. Add one DDF test with an unreduced
  input (for example `[-4, 1]` mod 5 = `x + 1`, giving `[(1, [1,1])]`) so that the `reduce` step in DDF is covered.
- **m12 (cost wording).** Prototype timings rerun twice (`$R/bin/driver`). Deg 60 mod 5: 150/144/154 ms for HEAD,
  207/138/151 ms for the new code. Deg 40 mod 2^31-1: 500/475/414 ms for HEAD, 413/398/599 ms for the new code.
  The differences are within noise in both directions. Write "comparable within measurement noise", not "no slowdown
  beyond 7 %".

## Commands run (sample)

- HEAD reproductions: `$R/rc.sh $R/r1.rs` (fp_factor, DDF, pgalois), `$R/r2.rs` (hooks), `$R/r3.rs` (linear pgalois),
  `$R/t7.rs` (conventions).
- gp: `$R/pg.gp` (poldisc, factorpadic, MR bases, sqrtint, nextprime/precprime), `$R/pgrand.gp` (13,986-case
  randomized check of the B.4 rule), `$R/ord2.gp` (D.4 extra rows, C.4 inputs), `$R/prim.gp` (`kronecker(P, 100003)`),
  the designer's `order_z.gp` (12/12 in PARI order).
- GAP: the designer's `units.g`. Every printed value matches G.4.
- Prototype: `$R/proto/driver.rs` rebuilt and run three times (rows, 600-case differential, messages, timings).
