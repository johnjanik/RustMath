# Review 2 of design P1 (rustmath-polynomials, with the pgalois, maclane and typed-factor callers)

Reviewer scratch: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/P1-polynomials/review2/`
(`$R` below). Programs link against the HEAD debug rlibs (`$R/rc.sh`: polynomials `7bc299bb35ca86d4`, integers
`93894c98d3b259d4`, groups `025a50b6d7157ed0`, core `dc51523818e444e8`, rationals `76279eac0299aa30`; the working tree
is clean at `c79a582`). I read the whole design, including Revision 1, and review 1.

## Verdict

**Approved: no blockers, 8 minor issues.** Both blockers of review 1 are fixed. The primitive-part rule in pgalois
holds on my own randomized check. The algebras test is now in the plan. I reproduced every defect at HEAD. The gp,
GAP and prototype results agree with the design. The minor issues below should be folded in during implementation.

## What I checked, item by item

| Item | Claim | Reproduced? | Notes |
|---|---|---|---|
| A. DDF squarefree | HEAD gives wrong labels on non-squarefree input | yes (`$R/h1.rs`): `[1,2,1]` mod 5 gives `[(1,[1,1]),(1,[1,1])]`; `(x^2+1)^2` mod 3 gives degree 2 twice; `[1,2,1,0,1,2,1]` mod 5 gives the label 2 on `x^5+x^4+x+1`; `[0,0,1]` mod 2 gives degree 1 twice; `[0]` gives `[]`; `[1,0,1]` at 15 gives `[(2,[1,0,1])]`; `[-4,1]` mod 5 gives `[(1,[-4,1])]` | Fix and correctness argument are right (squarefree iff `f' != 0` and `gcd(f, f') = 1`). All three callers pass squarefree input over primes. In maclane, `sf = work / gcd(work, work')` is squarefree even for multiplicities divisible by p. galois `sieve.rs:56-73` proves squarefree before the DDF. |
| B. Prime checks | `factor(x^2+1, 15) = [[1,0,1]]`; pgalois wrong at composite p, p = 0, 1, non-monic, untrimmed and content inputs | yes, every row of B.1 (`$R/h1.rs`). gp (`$R/g1.gp`): every `poldisc` / `factorpadic` value of B.1 and B.6; 2047, 1373653, 25326001 composite and strong to [2], [2,3], [2,3,5] only; 3215031751 = 151·751·28351 strong to 2, 3, 5, 7 but not 11; `sqrtint(2^63-1) = 3037000499`; `precprime` and `nextprime` as stated | Independent randomized check of the revised pgalois rule (`$R/pgr.gp`, my own generator, not the designer's): 5,145 pairs with p ∤ disc(pp f), 3,217 with p \| content, 1,824 with p \| lc(pp f). 0 mismatches against `factorpadic(f, p, 10)` and 0 hits of the `unreachable!` branch. The binary-form argument is right for every n >= 1 once g is primitive. Using I1's public `is_prime_u64` is right (I1 §2.1 makes it public at `rustmath_integers::prime::is_prime_u64`). |
| C. EDF cost | `block_contract` keeps d + 1 rungs; maclane repeats Rabin; literal in rings | yes (`fp_factor.rs:546-553`, `maclane.rs:1687-1694`, `typed/factor.rs:156`) | Dropping maclane's Rabin test is sound: a degree-d divisor of a certified block is irreducible, and the `degree(&q) != d` check stays. gp (`$R/c4.gp`) confirms the C.4 vectors, the PARI order of the two degree-12 factors, `polisirreducible(x^6+x+1) = 1` and the A2 cubics. Commit `5ff8749` and the three `test_factor_inseparable_*` tests exist, so the maclane and residue-tower comments are stale as claimed. |
| D. Phase 1 and order | PARI order is "degree, then coefficients top down" | Prototype rerun (`$R/proto/driver.rs`): 18 rows in PARI order, 600 random cases with 0 mismatches against HEAD sets and multiplicities, EDF messages identical to HEAD on 12 inputs, timings within noise (deg 80 mod 3: 191 vs 216 ms; deg 40 mod 2^31-1: 403 vs 382 ms). My own randomized gp check (`$R/ord.gp`): 2,826 `factormod` cases over 8 primes and 1,992 `factor` cases over Z, 0 not in that order. `t4_zorder.rs` rerun: HEAD differs from PARI on 9 of 12. | My own grep of the order consumers agrees with D.3. I checked the numberfields `prime_ideals` index uses (`s_units.rs:91`, `s_unit_gens.rs:505`, `ideals.rs` tests, `artin.rs:611-650`, `kummer.rs:215-261`): every assertion is symmetric under swapping conjugate primes. Also galois `monic_integer_roots` (length and single root only), modular decomposition (sorted keys), functionfields `lib.rs:105` (any degree-1 factor works), lmfdb tests (filters and `any`), symbolic (`matches!`). One import D.3 does not list, `rustmath-functionfields/src/branch_cycles.rs:27` (`fp_factor as fp`), uses only arithmetic helpers, so it is unaffected. The X1 design's condition for this cleanup (run the numberfields, galois, curves, modular, functionfields and rings tests) is in the acceptance commands. |
| E. Zassenhaus errors | `Err(())` also for "no good prime"; callers blame the subset bound | yes (`$R/t3_zass.rs`): `x^2 - primorial(10^5)` gives `Err` in 0.28 s debug. gp: `nextprime(10^5) = 100003`, `kronecker(P, 100003) = -1`, so the new scan stops at the first prime past 10^5 with an irreducible reduction, and no Hensel lift runs | The 12 call sites are complete (my grep, including `use` aliases). Only `regular_certify.rs:215` (`Err(()) =>`) stops compiling, and the plan fixes it in step 4. No test pins any of the old messages. The termination argument is right: f is monic and squarefree in `factor_monic_squarefree`, so a bad prime divides disc(f) != 0. Behaviour change worth knowing: an input with between 1 and 29 good primes below 10^5 now scans further (to 30 good primes) instead of taking the best one found. The result is still correct. |
| F. Cohen numbers | 3.1.2, 3.3.1 right; 3.4.2 is the char-p squarefree algorithm; §3.2.3-3.2.4 | Sage sources in `dl_sage` quote 3.1.2 (pseudo-division), 3.2.2 (extended Euclid), 3.3.1 (sub-resultant gcd), 3.4.2 ("basically Yun" with p-th powers). That matches my knowledge of GTM 138 (§3.2.3 polynomials over a UFD, §3.2.4 Euclid over a UFD, §3.4.2 squarefree factorization mod p). | All four citation sites in `zx.rs` and `univariate.rs` are covered (my grep). |
| G. Ring hooks | `UnivariatePolynomial` and `LaurentPolynomial` keep the defaults; they are the only polynomial `Ring` types | yes (grep: the other `Ring for` hits are series, which belong to R1, and test types) | I rederived rules 1-5 and the Laurent rules. Rule 5: the right inverse in R[[x]] is unique because a unit has a left inverse, and the order-n recurrence makes n consecutive zeros final, so the stop rule never cuts a true inverse. The work bound `n(d+n)` holds. The M2 example: both products are I (hand check and GAP `units_rev1.g` rerun). The Z/8, Z/12 and Z/2^e rows match the prototype rerun (`unit_inverse_proto.py`). The consumer grep is complete: the only generic hook consumers are `ring_category.rs:170-192`, `category_methods.rs:70/95/328`, and the three `static_characteristic` uses. The only test that changes is `algebra_with_parent.rs:274-277`. The `static_is_commutative` hook is coordinated with F1 and R1, which both plan on top of it. R1 adds a `PolynomialRingParent` later, so "no polynomial parent at HEAD" is right for P1. |
| H. Conventions | evidence-only doc changes | V2.19 text confirmed (`dl_magma/ch23.txt`: Quotrem, QMatrix "degree d − 1", CompanionMatrix "degree d − 1" while d × d). Sage `is_irreducible` doctests confirmed in the source (`R(2*x)` False over ZZ, True over QQ; `R(1)`, `R(4)` False; `R(5)` True). HEAD values confirmed (`$R/t7_quotrem.rs`). WebFetch of doc.sagemath.org (`polynomial_integer_dense_flint`): `(x^2).quo_rem(2*x - 1) = (0, x^2)` is shown. The `(5*x+7).quo_rem(3) = (x + 2, 2*x + 1)` doctest is in the develop source (`dl_sage`, line 743) but not on the published page | Keep the design's choice. Cite the Sage develop source (GitHub raw) for the `(5*x+7)` example, since the published page does not show it yet. |

Cost claims: the D timings hold within noise (rerun above), C's memory bound is evident from the code, and E's test
costs about 0.3 s debug (measured 0.28 s for the HEAD scan).

## Minor issues

- **m1 (B, zero modulo p).** The design does not say what `factor`, `factor_with_multiplicity` and `squarefree_factor`
  do when f reduces to 0 mod p. HEAD returns `[]` for `[0]`, and panics in `pth_root` for `[5, 10]` mod 5
  (`$R/h2.rs`). With the new `reduce`, `[5, 10]` mod 5 would silently become `[]`, an empty factorization of 0
  (MAGMA `Factorization(0)` is an error, and `[]` reads as "f is a unit"). Required change: panic
  `"fp_factor::factor: f = 0 has no factorization"` (and the same in the other two entries), as DDF and EDF already
  do, and add one `should_panic` test. No caller passes 0: `igp_closure.rs:57-74`, `panayi.rs:367-372`,
  `local_field.rs:39`, pgalois (nbar >= 0), and zassenhaus (full degree) all guard first.
- **m2 (G, Laurent non-monomials).** G.3 says the non-monomial `unit_status` "follows the commutative rules". Say
  explicitly that they apply only when `R::static_is_commutative() == Some(true)`. Two decidable cases are missing.
  (a) Commutative R: f is nilpotent iff every coefficient is (`f = x^a p`). So `Some(true)` when all are proven
  nilpotent, and `Some(false)` when any coefficient, not only an extreme one, is a proven non-nilpotent. Example:
  `2x^-1 + 4x` over Z/8 is nilpotent. (b) Every ring: if both extreme coefficients are proven units, f is not a unit.
  The product with any g != 0 has nonzero terms at degrees `a + s < b + t`. Example: `I x^-1 + I x` over M2. Add both
  rules and one test row each.
- **m3 (G, order of polynomial units).** For a proven unit of degree >= 1, the order is left `Undecided`, and
  `zmod12_sage_examples` pins that. A search that finds something decides: compute `f^k` for k = 1, 2, ... under a
  work cap, and the first k with `f^k = 1` is `Finite(k)`. Exhausting the cap stays `Undecided`. Run the search only
  after `unit_status() == Some(true)`. Over Z/12 this decides `5 + 6u` (order 2: `(5+6u)^2 = 25 + 60u + 36u^2 ≡ 1`).
  Over Z/8 it decides `1 + 2t` (order 4). Either implement it, or record it in the HANDOFF as a follow-up with this
  reason. The item asks to implement the hooks that can be decided.
- **m4 (B.7 vs I1).** B.7 says that I1 §7 hands `berlekamp::check_prime` to P1. The current I1 design switches that
  line itself (I1 §2.9: "`!try_is_prime(p)?`"; §3.1 table: "switched in I1 (2.9)"), and §7 no longer lists it. After
  the rebase, P1 should keep I1's `check_prime`. It should only map the unresolved case to its own message if it
  wants the B.7 text. Then rewrite the two error texts, add the doc sentence and add the psi_12 test. Fix the wording
  of B.7 and of the "Rebase on I1" list.
- **m5 (B.6 HEAD baseline).** On HEAD `factor(&[1,0,1], 1)` and `factor(&[1,1], 1)` overflow the stack
  (`$R/h1.rs`, `$R/h3.rs`). That aborts the whole unit-test binary rather than failing one test. When the plan
  checks that the new tests fail on HEAD, run `factor_rejects_p_one...` on its own
  (`cargo test -p rustmath-polynomials --lib factor_rejects_p_one`). Also, B.5 says the `padic_factor` entries hit
  "a `div_mod` panic" at a composite p. In fact `padic_factor::cycle_type(x^2+1, 15)` returns `Some([2])` on HEAD,
  a wrong value (`$R/h3.rs`). Correct the sentence. The test `cycle_type_rejects_composite_p` still fails on HEAD.
- **m6 (G, static constructors).** G.3 says that the polynomial hooks call no static constructor. But the `f·h == 1`
  and `h·f == 1` checks through `Mul for UnivariatePolynomial` seed every coefficient with `R::zero()`
  (`univariate.rs:920-924`). For a coefficient type whose `zero()` panics (the p-adic types until R1),
  `unit_inverse` would panic. Either compute both products as convolutions seeded by the first product term (the
  recurrence itself needs no zero), or drop the claim and say which hooks use `R::zero()`.
- **m7 (B.5, Result-returning entries that now panic).** `padic_factor::padic_factor` and `ramification_type`
  return `Result<_, PadicError>`, but they will panic on a bad p. That is defensible: an `Err` there would be read
  by `ideals.rs:150-158` as `montes_resolved = false`, which hides the misuse. State it in a `# Panics` section of
  each function's doc, and say in the design why it is a panic and not a new `PadicError` variant.
- **m8 (nits).** The comment of `split_with_exhaustive_counts` (`fp_factor.rs:1346-1349`) says "phase 1"; reword
  it when phase 1 is deleted. `HenselLiftFailed` is an internal error but maps to `MathError::NotSupported` at the
  callers; `InvalidOperation` would be more accurate (optional). Add `branch_cycles.rs:27` to the D.3 table as
  "arithmetic helpers only".

## Commands run (sample)

- HEAD reproductions: `$R/rc.sh $R/h1.rs` (A, B, pgalois rows), `$R/h2.rs` (zero mod p), `$R/h3.rs` (p = 1, p > MAX_P,
  `cycle_type` at 15), `$R/t3_zass.rs` (Zassenhaus `Err` in 0.28 s), `$R/t4_zorder.rs` (9 of 12 differ from PARI),
  `$R/t7_quotrem.rs` (conventions).
- gp: `$R/g1.gp` (B.1/B.6 references, strong pseudoprimes, MAX_P bounds), `$R/pgr.gp` (5,145-case independent check of
  the revised pgalois rule), `$R/ord.gp` (4,818 random cases of the PARI order rule), `$R/c4.gp` (C.4 and D.4 rows),
  `$R/k.gp` (`kronecker(P, 100003) = -1`), `$R/order_z_q.gp` (designer's 12 Z rows: all in canonical order).
- GAP: designer's `rev1/units_rev1.g` (det, both M2 products I, Z/8 and Z/6 products).
- Python: designer's `rev1/unit_inverse_proto.py` (all rows as stated).
- Prototype: `$R/proto/driver.rs` rebuilt and run (rows, 600-case differential, messages, timings).
- WebFetch: doc.sagemath.org `polynomial_integer_dense_flint` (`quo_rem`, `content` examples).
- Greps: every caller of `fp_factor::{factor, factor_with_multiplicity, equal_degree_factor, distinct_degree_factor,
  squarefree_factor}`, `zassenhaus::factor`, `factor_over_integers`, `prime_ideals`/`prime_decomposition`,
  `padic_factor`, the pgalois entries, the hook names and wrappers, and the other unit designs (F1, R1, I1) for
  `static_is_commutative`, `maclane`, `berlekamp`.
