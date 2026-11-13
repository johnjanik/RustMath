# Review 1 of the P1 implementation (rustmath-polynomials, with the pgalois, maclane and typed-factor callers)

Reviewer scratch: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/P1-polynomials.rev1/`
(`$R` below). Base: HEAD `0ceaeaa` (I3, I1, I2 committed), working tree with the 29 modified files and the new
`rustmath-polynomials/tests/ring_predicates.rs`. I read the design (Revision 1 body), both design reviews, the
implementation report, the I1/I2/I3 reports (only I1 hands P1 an item), the whole diff and the new test file.

## Verdict

**Approved: no blockers, 3 minor notes.** Every design item (A to H) is implemented and fixes the defect for
the input classes I could construct. Every review-2 minor note is folded in, and none is skipped. The acceptance
commands pass. A randomized differential run of 4,653 cases against PARI/GP found no mismatch. A second run of
11,000 cases checked the ring hooks against the Atiyah-Macdonald rules and against brute force, with no failure.
The tests that the report could not run on HEAD fail on HEAD's code, as claimed (checked below for fp_factor).

## Acceptance commands (run by me)

| Command | Passed | Failed | Ignored |
|---|---|---|---|
| `cargo test -p rustmath-core --no-fail-fast` | 49 | 0 | 1 |
| `cargo test -p rustmath-integers --test ring_predicates` | 1 | 0 | 0 |
| `cargo test -p rustmath-rationals --test ring_predicates` | 3 | 0 | 0 |
| `cargo test -p rustmath-polynomials --no-fail-fast` | 424 | 0 | 4 |
| `cargo test -p rustmath-groups --no-fail-fast` | 668 | 0 | 4 |
| `cargo test -p rustmath-rings --no-fail-fast` | 1914 | 0 | 82 |
| `cargo test -p rustmath-galois --no-fail-fast` | 85 | 0 | 5 |
| `cargo test -p rustmath-numberfields --no-fail-fast` (782 s) | 166 | 0 | 1 |
| `cargo test -p rustmath-curves --no-fail-fast` | 305 | 0 | 13 |
| `cargo test -p rustmath-modular --no-fail-fast` | 221 | 0 | 1 |
| `cargo test -p rustmath-functionfields --no-fail-fast` | 27 | 0 | 0 |
| `cargo test -p rustmath-symbolic --no-fail-fast` | 772 | 0 | 16 |
| `cargo test -p rustmath-category -p rustmath-modules -p rustmath-algebras --no-fail-fast` | 1112 | 0 | 1 |
| `cargo test -p rustmath-algebras --features full --lib algebra_with_parent` (the changed finiteness test runs here) | 8 | 0 | 0 |
| `cargo build -p rustmath-igp24` | ok | | |
| `cargo check --workspace --tests` | ok, no warning on a changed line | | |
| `cargo clippy -p rustmath-polynomials -p rustmath-groups` | fails in untouched `rustmath-combinatorics` (5 deny lints, `binary_words.rs:317` and four more) | | |
| `cargo clippy --no-deps --all-targets -p rustmath-polynomials -p rustmath-groups` | fails in untouched `indexed_free_group.rs:181`; the 9 lints in touched files are all on committed lines (`git blame`: `a8ff703c`, `54bc719d`, `c0918962`) | | |
| `cargo doc --no-deps -p rustmath-polynomials -p rustmath-groups -p rustmath-core` | ok; no warning on an added line (153 warning sites, matched against the added-line list); `target/doc` deleted | | |

The test counts equal the report's. The clippy failure is the one the report describes, and it is not caused by
this unit.

igp24 by hand: `rustmath-igp24 '[1,0,1]' '[5,15]'` prints `igp24: p = 15 is not a prime <= 3037000499`, status 2;
`'[5,3037000507]'` the same with that number; `'[5,7]'` prints the JSON (`"cycle_types":{"5":[1,1],"7":[2]}`), status 0.

## Differential checks (my own driver, not the implementer's tests)

`$R/diff1.rs`, linked against the current debug rlibs (polynomials `3b0e1a710822df26`, groups `81506577ffaff86a`,
integers `a1ad7d0876b91f69`, core `dc51523818e444e8`). It writes gp input files, and `$R/check.gp` compares them.

| What | Cases | Reference | Result |
|---|---|---|---|
| `fp_factor::factor` and `factor_with_multiplicity`: random products of 1-4 factors of degree <= 4 with powers <= 3, coefficients shifted by -2p..2p (unreduced, negative), primes 2, 3, 5, 7, 11, 13, 101, 65537, 2^31 - 1, 3037000493 | 1,353 | gp `apply(le, factormod(f,p)[,1]~)` and `factormod(f,p)[,2]~` | 0 mismatches: same factors, same multiplicities, **same order** |
| `zassenhaus::factor`: random products over Z, non-monic and negative leading coefficients, repeated factors | 300 | gp `factor(f)`, `content(f)·sign(lc f)` | 0 mismatches (content with sign, factors, order, multiplicities) |
| `unramified_local_galois_group`: random f of degree 1-5 at p in {2,3,5,7,11}; a quarter multiplied by p, a quarter with p \| lc, an eighth untrimmed | 3,000 (2,272 decided: 693 with p \| content, 685 with p \| lc(pp f), 276 untrimmed) | `None` iff p \| `poldisc(pp f)`; else `vecsort(apply(poldegree, factorpadic(f,p,10)[,1]~))`, its lcm, and transitivity | 0 mismatches; the `unreachable!` never fired |
| `UnivariatePolynomial` hooks over test types Z/8, Z/12, Z/36, Z/7 (commutative flag) | 10,000 | Atiyah-Macdonald rules (computed independently in the driver); both products of every returned inverse; order by brute force (powers up to 2000) | all asserts hold; every `Finite(k)` equals the brute-force order; 1,389 units left `Undecided` (mostly constants such as 3 mod 8, whose coefficient type has the default order hook) |
| `LaurentPolynomial` hooks over Z/8, Z/12, Z/6 | 1,000 | both products of every returned inverse; no `Some(true)` with an `Err` inverse; for Z/8 and Z/6, every `Some(false)` checked by a brute-force search for an inverse with exponents in [-10, 12] and up to 3 terms | all hold |

HEAD baseline (report deviation 6). I compiled HEAD's `fp_factor.rs` (`git show c79a582:...`) as a module in
`$R/headfail.rs` and ran the inputs of the new fp_factor tests. Each new test fails on HEAD: DDF returns
`[(1,[1,1]),(1,[1,1])]` for `[1,2,1]` mod 5 and `[(1,[-4,1])]` for `[-4,1]`; `factor(x^2+1, 15) = [[1,0,1]]`;
`factor([1,1], 3037000507)` returns; `factor([-3,0,1], 5) = [[-3,0,1]]`; `[1,1,5]`, `[5,10]`, `[1,5]` panic with
other messages; `equal_degree_factor([5],1,5) = []`; A1 comes out `[[1,1],[0,1]]`. The pgalois and Zassenhaus
HEAD values were reproduced by both design reviews against the HEAD rlibs.

## Item by item

| Item | Implemented as designed? | Checked |
|---|---|---|
| A. DDF input check | yes: `check_modulus`, `reduce`, monic, `squarefree_check`, private `ddf_squarefree` used by `factor` on the radical | squarefree iff `f' != 0` and `gcd(f, f') = 1` over F_p is exact; the 3 callers (factor, galois `sieve`, maclane) pass squarefree input over primes; galois `sieve` tests pass |
| B. Prime checks | yes: `check_modulus` = range and I1's `is_prime_u64` (12-base strong test, exact for every u64, read at `primality.rs:42`); checked entries reduce, then panic on `f ≡ 0`; `factor_prime` unchecked for Zassenhaus only; padic entries, pgalois entries, `is_padic_square`, igp24 | pgalois: primitive part, degree, then prime check; root at infinity when `deg(g mod p) = n - 1`; I re-derived the binary-form argument (sound for every n >= 1 once g is primitive) and the differential run above agrees. gp rerun (`$R/pg.gp`): every `poldisc`/`factorpadic`/`factormod` value in `tests/pgalois.rs`; 2047, 1373653, 25326001, 3215031751 composite and strong to [2], [2,3], [2,3,5], [2,3,5,7]; `precprime(3037000499) = 3037000493`, `nextprime(3037000500) = 3037000507`; psi_12 composite with the stated factors |
| B.7 Berlekamp | yes: I1's `try_is_prime` kept, unresolved proof gives `NotSupported`, `internal` text and `no_inverse` rewritten | the three `mod_inverse` sites use `no_inverse`; I1's doc sentences are present (`berlekamp.rs:275`, `:353`) |
| C. EDF cost, rings callers | yes: sparse ladder (`X_0`, current rung, rungs `d/l`), messages in HEAD order; maclane drops the redundant Rabin clause (keeps `degree(&q) != d`); typed factor uses `fp_factor::MAX_P` | gp rerun of the C.4 inputs (`$R/c4.gp`): product vectors, `factormod` order of the two degree-12 factors, `polisirreducible(x^6+x+1) = 1`, the two cubics |
| D. Phase 1 gone, PARI order | yes: legacy items deleted, lazy certificate before try 32, `factor_order`; `zx_factor_order` in Zassenhaus | gp rerun of all 19 reference rows (`$R/ord.gp`) and of the 12 Z rows (`$R/ordz.gp`): all in the stored order; 3,000 random Z factorizations: PARI's `factor` order is "degree, then coefficients top down, signed" in every case. Cost: `$R/timing5.rs` times HEAD's fp_factor (compiled from `c79a582`) against the new one on 160 random monic inputs per row (debug, idle machine): deg 80 mod 3 +3 %, deg 60 mod 5 +3 %, deg 120 mod 2 -1 %. So the "comparable" claim holds. One fixed sample of 20 inputs mod 3 was 20 % slower in every round (`$R/timing3.log`); other samples are not, so that is the sample and not the algorithm |
| E. Zassenhaus errors | yes: typed enum, `Display` texts as designed, `From` for `MathError` (Hensel failure is `InvalidOperation`), scan to `MAX_P`, `recombine` cap, all 12 callers | termination: f monic squarefree, a bad prime divides `disc(f) != 0`, so at most `omega(disc) + 30` primes; the loop also stops at the bound; `x^2 - primorial(10^5)` now factors (test passes) |
| F. Cohen citations | yes (`zx.rs` header and Yun, `univariate.rs:28`, `:1286`) | |
| G. Ring hooks | yes, plus review 2 m2, m3, m6 | I re-derived each rule: f(0) retraction; unit leading coefficient keeps the degree; top coefficient `a_n^k` and constant `a_0^k`; A-M Ch. 1 Ex. 2 under `static_is_commutative() == Some(true)`; the recurrence `h_k = -b Σ a_i h_(k-i)` is the right inverse with b on the left (correct without commutativity), the n-zeros stop rule is exact, the "degree < k - n" message is what is proven, and both products are checked. Laurent: two unit extremes give a non-unit in every ring; `NotInvertible` from the inner polynomial inverse is a proof (a Laurent inverse of p with `p(0)` a unit lies in R[x], so it would be p's right inverse in R[[x]]); the span bound runs before `dense` allocates; `Infinite` for a unit monomial with k != 0 is right in every nonzero ring. GAP rerun (`$R/units.g`): every reference value in the new test header, and `IsFinite(PolynomialRing(Integers,1)) = false` for the changed algebras test |
| H. Conventions | yes (docs and the two new test cases) | |

No new unbounded loop or allocation: the prime scan is bounded by `omega(disc) + 30` primes and by `prime_bound`;
`unit_inverse` and both order searches are bounded by 65536 coefficient products (and allocate at most that many
coefficients); the Laurent span check precedes the dense vector; `prime_divisors` is trial division up to sqrt(d).
Zero `unsafe`; no em-dash on an added line; no `rustmath-misc/output.txt` or `.tmp_dir_*` change.

Weakened or deleted assertions: none. The changed assertions are: the algebras finiteness test (`None` to
`Some(false)`, GAP reference above, sanctioned by the design); `factor_order_is_unchanged`, which pinned HEAD's own
output and is replaced by PARI reference lists (verified above); T1 and T3, which now sort the PARI list before
comparing (same strength); T6, whose expected panic text is now longer (stronger). The deleted `assert!(d >= 0)`
sat in the deleted `random_monic_polynomial`.

Callers: I grepped every caller of the checked fp_factor entries again. `rustmath-padics/src/montes.rs:123`
(`fp::factor`) is that crate's own module, not fp_factor. The others pass primes `<= MAX_P` and nonzero inputs, as
the design lists, and their crates are green.

## Minor notes (not blocking)

1. **rustfmt.** The new code is not rustfmt-clean. `rustfmt --check` reports 9 hunks in `fp_factor.rs` (HEAD: 2),
   7 in `zassenhaus.rs` (HEAD: 1), 4 in `laurent.rs` (HEAD: 1), 5 in `tests/pgalois.rs` (HEAD: 1) and 18 in the new
   `tests/ring_predicates.rs`. This is cosmetic. Formatting whole files would also reformat older code (for
   example 21 old hunks in `univariate.rs`), so format only the new hunks, or leave it.
2. **Laurent zero.** `LaurentPolynomial::unit_status` returns `Some(false)` for the empty map, with the comment
   "0 (in a nonzero ring)". In the zero ring the zero Laurent polynomial is 1, a unit. HEAD's default gave the same
   value. The cause is `LaurentPolynomial::is_one`, which is false for that element (the report's HANDOFF item 3).
   Mention this hook in that HANDOFF item, so that the fix of `is_one` also covers it.
3. **Adjacent defect for the HANDOFF list.** `padic_factor::cycle_type` (`padic_factor.rs:76`) takes `f.len() - 1`
   as the degree. So an untrimmed input (`[1,0,1,0]` at 7) returns `None`, documented as "p divides the leading
   coefficient", which is false. pgalois had the same defect, and this unit fixed it there. It is outside this
   unit's items, so add it to the HANDOFF section I list.

## Review-2 notes folded (report, "Review 2 minor notes")

All 8 are done as the report says. I checked m1 (three `should_panic` zero tests), m2 (the two Laurent rules and
their test rows), m3 (the order search, `Finite(2)` and `Finite(4)` match GAP), m6 (`convolve`, `dense`,
`sparse_product`, `add_term`, and the `NoStatic` test, whose `zero()` and `one()` panic) and m7 (`# Panics`
sections). The implementer skipped none.
