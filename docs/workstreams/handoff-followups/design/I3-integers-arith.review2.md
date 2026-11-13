# Review 2 of design I3-integers-arith (after Revision 1)

Reviewer scratch: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/I3-integers-arith/review2/`
(`rc.sh` builds against the HEAD debug rlibs; programs `tprim.rs`, `tl100k.rs`, `tpp.rs`, `ter.rs`,
`tbin.rs`; gp script `s.gp`, GAP script `b.g`; `appA.txt`, `all166.sites` for the to_i64 audit).
Repository at HEAD `c79a582`, clean tree. All timings on this machine.

Verdict: **approved**. No blocker. Review 1's blocker B1 and minors m1 to m10 are resolved as the
Revision 1 table says. The minors below are accuracy and robustness points for the implementer.

## What I reproduced

| item | check | result |
|---|---|---|
| 1 | HEAD debug (`../t1_prim_gcd.rs` rebuilt): gcd(4,-6) = -2, lcm(-4,6) = -12, xgcd(4,-6) = (-2,1,1); lcm(2^40,2^40) and lcm(65536,65536) panic; gcd(i32::MIN,0) = MIN; gcd(MIN,-1), norm(MIN), div_rem(i64::MIN,-1) panic | as §1.1 |
| 1 | design code (`gcd_u64`, `xgcd_i128`, the u128 lcm) against HEAD `Integer` gcd/lcm/xgcd on 300,000 pairs (shifted LCG values, MIN, MIN+1, MAX, MIN/3, ±2^62, 0, ±1) (`tprim.rs`) | all equal; the only results outside i64 are the 46 with g = 2^63, as §1.2 claims |
| 1 | gp: every §1.4 value (`s.gp`: the gcd/lcm/gcdext lists, `gcdext(-2^63,3)`, `gcdext(-2^63,-1)`, `gcdext(2^63-1,-2^63)`, `gcdext(-2^31,3)`, the lcm overflow values) | all equal the design |
| 1 | callers: only `impl EuclideanDomain for i32/i64` at traits.rs:439/:483 use the defaults; `UnivariatePolynomial<i64>` content goes through `normalize_elem` | no value changes except the fixed signs |
| 2 | HEAD debug (`../t3_fast.rs`): every §2.1 panic, add_mod(-1,0,-5) = -6, and `ArithInt::is_prime` panics at fast_arith.rs:246 for 2147483647 and 2147483629 | as §2.1 |
| 2 | gp: `lcm(-2^31,-2^31+1)`, `lcm(65536,65537)`, `lcm(2^32,2^32+1)`, `(-1+0)%5`, `(-2^31)^2%(2^31-1)`, `(-2^63)^2%(2^63-1)`, `(-2^64)%7`, isprime of the three ArithInt values, `factor(2146654199)` | as §2.4 |
| 2 | callers (`grep ArithInt\|ArithLLong`) | lib.rs re-export and integers tests only; rustmath-py does not use them |
| 3 | gp `sqrtint(3^63093+12345) % (2^127-1)`, `#binary(..)` | 130117212609866437730867252961009611028, 50001 |
| 4 | Lehmer prototype in debug on 100,000 / 99,993-bit operands (`tl100k.rs`) | gcd 0.135 s, xgcd 0.288 s (design: 0.14 s, 0.32 s) |
| 4 | digest ch. 18 :14, :223-:231 | Weber accelerated gcd, Lehmer XGCD [Knu97 pp. 345-348], half-gcd [Sch71] crossovers 32,768 / 6,000 bits; cited correctly |
| 5, 6 | `ppow.rs` prototype in debug (`tpp.rs`): pp(2^1000000+1) = None 0.136 s, pp(3^30030) = (3, 30030), pp(3^30030+2) = None, pp(2^100000) = (2, 100000); all four with input construction 0.26 s. HEAD `is_perfect_power(2^1000000+1)` 12.4 s | as §6.4 (HANDOFF quotes 12 s) |
| 5 | `exact_root` prototype in debug (`ter.rs`): the four `3^100003+2` cases 27 ms; (2^61-1)^331, -3^100055, (2^64·5)^7, 1234567^288 (k with even part 2^5), +1 neighbour, 2^3000 and 2^3001 with k = 3; all 2,486 rows of `gp/ispower_root.csv` (prototype and HEAD `is_power`) | all correct |
| 5, 6 | gp `ispower` values of §5.4 / §6.4 | [1,2], [1,-2], [1,2], 0, [1,1], [0,0,0,0], [0,30030,0,100000] |
| 5, 6 | 2-adic Newton argument re-derived (precision doubling from j = 1, kinv from z = k correct mod 8, uniqueness of the odd k-th root mod 2^K, bits(c) = ceil(bits(m)/k)) | correct; every None is a proof, so no Decided/Unresolved split is needed |
| 7 | prototype `binom.rs` in debug (`tbin.rs`): C(524288,262144) 0.14 s, C(524288,262143) 0.19 s, C(10^6,500000) 0.38 s, total 0.71 s; residues equal `gp/binom_big.csv` | as §7.4 |
| 7 | gp re-run: `binomial(524288,262144)` residue and 524279 bits; `binomial(10^30+10,1500)` residue equals the fixture row; `binomial(2^33,3)`; `binomial(1000,500) % (2^127-1)` | all equal |
| 7 | size bound of §7.2 re-derived: each factor (N-i)/(K-i) >= N/K >= 2, log2(N/K) > bits(N)-1-bits(K), bits(C) > LB | correct; never refuses a result of at most 2^32 bits |
| 7 | callers: `Integer::binomial` only in integers tests and `rustmath-jupyter/src/repl.rs:10652`; the REPL refuses when its upper bound K·(bits(N)-bits(K)+3) exceeds 2^22 | the REPL never reaches the new panic |
| 8 | HEAD debug (`../t4_sigma.rs`): the three §8.1 calls panic at integer.rs:398; k = 0 returns through `num_divisors` before the loop | as §8.1; the pre-pass in §8.2 fixes the order (review 1 m1) |
| 8 | gp `sigma(2^10*3^5,3)`, `sigma(7^40,5) % (2^127-1)`, `#binary` | 18285256324908244, 108698265444746923986228897305478117662, 562 |
| 9 | independent audit: 166 `.to_i64()` lines; `comm` against Appendix A | Appendix A lists exactly 145 of them (G 50, T 23, R-fix 50, R-keep 12, D 2, N 8); the other 21 are Option-typed (BigInt in chabauty.rs and verma_module.rs, generic `T`/`R` in integer_range.rs, hilbert.rs, q_bernoulli.rs, MPFR/BigFloat/mpc asserts, ratfunc.rs:290/:293, the definition integer.rs:90). 137 Integer sites in 24 crates, every crate in the step-9/10 list |
| 9 | other forms: no `Integer::to_i64` path call, no `.map(Integer::to_i64)`, no doc test or Markdown example; no `From<Integer>` / `TryFrom<..Integer>` impl that would conflict; no test pins the panic text; every crate is edition 2021 (TryFrom in the prelude) | as §9.1 / §9.2 |
| 9 | R-fix sites read at HEAD (all 50 contexts): series.rs, integrate.rs, radical.rs (Rational::new is `T: Into<Integer>`), limits.rs, solve.rs, substitute.rs:124 (same 0..=u32::MAX range), lseries.rs:402 and bsdratio.rs (functions return `Result<_, String>`), unramified.rs, factorization.rs:702, numberfields lib.rs:419 | the replacements are exact; no site turns "does not fit" into a value |
| 9 | `expr_to_rational` / `expr_to_integer` (solve.rs:1720/:1747) | no caller anywhere in the workspace; dead |
| 9 | `polynomial_degree` (integrate.rs:673) callers | only the recursion and integrate.rs:722/:899, all `?`; no external caller |
| 9 | `class_number_quadratic` callers | only `NumberField::class_number`, called only by numberfields lib.rs tests (rings re-exports it; its test is `#[ignore]`) |
| 9 | gp `quadclassunit(-20).no`, `(-3)`, `(-4)`, `(-15)`, `poldisc(x^2+5)`, `poldisc(x^2+x+1)`, `isprime(2^89-1)`, `(2^89-1)%4`, `polisirreducible(Mod(1,2^89-1)*(x^2+1))` | 2, 1, 1, 2, -20, -3, 1, 3, 1 |
| 10 | gp list, GAP list (`b.g`), Sage doc (doc.sagemath.org `sage.arith.misc.binomial`: "If m < 0, return 0"; `binomial(3,-1)` = 0, `binomial(-5,-2)` = 0, `binomial(-2,5)` = -6) | [0,0,0,-6,0,0,0] in gp and GAP; Sage as quoted |
| tests | `within` helper at rustmath-polynomials/tests/ch23_quotrem_gcd.rs:66-84 | exists as described |

## Blockers

None.

## Minor

### m1. The Lehmer speed test does not have the claimed 10x margin
§4.5 `gcd_and_xgcd_of_100k_bit_operands` runs under `within(5, ..)` and also checks "equality with
Stein's gcd on the magnitudes". Stein's gcd on these operands takes 0.90 s in debug (`tl100k.rs`),
so the whole test as specified takes 1.34 s (gcd 0.135 s, xgcd 0.288 s, Stein 0.90 s, identity
checks 0.016 s). That is 3.7x under the bound, not the "at least 10 times" of §3.4 and the Risks.
Required change: put only the Lehmer gcd and xgcd inside the `within(5, ..)` closure (0.42 s, 12x)
and do the Stein comparison and the identity checks after it, outside the bound. HEAD still fails
(its gcd alone takes 9.7 s in debug).

### m2. Binomial memory figure undercounts
§7.2 ("the sieve takes at most 4 GiB") and the cost paragraph count only the K-byte sieve of
`prime_range`. `binomial_nonneg` also keeps four `Vec<u64>` with one entry per prime up to K
(the primes, the Legendre exponents, n mod p, the removal counts): about 32·π(K) bytes, so about
6.5 GB at K = 2^32 and 3.4 GB at K = 2^31, on top of the sieve and the result. Required change:
state the real figure in §7.2 and the Risks ("an admitted call near the cap needs about 10 GB").
Optional: keep fewer per-prime arrays (one signed count instead of `exps` and `got`), or lower the
cap. Not a blocker: the only non-test caller (the REPL) stops at 2^22 bits, where K <= 2^22.

### m3. The class-number note misses an exact algorithm already in the crate
§11 says an exact algorithm for imaginary quadratic fields still has to be written (Cohen Alg.
5.3.5). `rustmath-numberfields/src/hcf.rs:88` `class_number_imag_quadratic(d)` already counts the
reduced primitive forms of a fundamental d < 0 (`reduced_forms`, hcf.rs:42). Required change:
the HANDOFF / tracker note for row 1799 names hcf.rs:88 as the exact path for d < 0 (after
reducing the polynomial discriminant to the field discriminant). Optional: delegate the
imaginary case in the same step-10 commit, so that x^2 + 3 keeps `Ok(1)` (gp
`quadclassunit(-3).no` = 1) instead of becoming `Err(NotImplemented)`, with x^2 + 5 -> 2 and
x^2 + 15 -> 2 as tests (gp `quadclassunit(-20).no`, `quadclassunit(-15).no`).

### m4. The debug_assert in `lehmer::gcd` taxes every Rational operation in debug builds
The Risks propose `debug_assert!(a % g == 0 && b % g == 0)` in `lehmer::gcd`. `Integer` gcd runs
inside every `Rational` normalisation, so every debug test in the workspace pays two extra
BigUint remainders per gcd. Several crates have time-bounded tests (`within(..)`). Recommended:
drop it (the differential tests and the gp fixtures already cover correctness) or apply it only
above a size threshold (for example both operands above 2^128, where the remainders are cheap
next to the gcd).

### m5. The huge-rational inequality test depends on an unchecked path
§9.4 `constant_inequality_with_huge_values` leaves to the implementer whether
`Inequality::new(Expr::Rational(r), LessThan, Expr::from(0))` reaches
`solve_constant_inequality` as a bare `Expr::Rational`. If `to_standard_form` leaves a
`Binary(Sub, ..)`, the new code returns `Unknown` and the test fails for a reason unrelated to
to_i64. Recommended: also call the private `solve_constant_inequality` directly from the
`solve.rs` test module with `Expr::Rational(r)` and `Expr::Integer(10^30)`, so the
sign-test fix is pinned whatever the standard form does.

### m6. rustmath-py stays unchecked (accepted)
The two rustmath-py sites are edited without a build, as §9.3 states. pyo3 is not in the local
registry cache, and a separate target directory would cost disk. Accepted as stated; the commit
message should say it.

## Summary for the orchestrator

Every item (1 to 10) is covered and nothing is declared out of scope; the half-gcd part of item 4
is a measured "not now" with an estimated crossover, which is what the item asks for ("decide
with numbers"). I reproduced the HEAD defects of items 1, 2 and 8, the cost of item 6 (HEAD 12.4 s
in debug vs 0.26 s for the prototype on the test inputs), the Lehmer and binomial timings in debug,
the gp/GAP/Sage reference values, and the item 9 call-site audit (exactly 145 lines, 137 Integer
sites in 24 crates, no missed site). The design code for item 1 matches HEAD's Integer gcd/xgcd on
300,000 extreme pairs. The algorithms (truncated xgcd in i128, Lehmer, the 2-adic exact root,
Goetgheluck splitting and its size bound, the sigma pre-pass) are correct. No facade or silent
truncation is introduced. Five minor fixes: a too-thin time bound in one speed test, a memory figure,
a class-number note that misses hcf.rs:88, a debug_assert cost, and one test path to pin directly.
