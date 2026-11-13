# Review 1 of design I3-integers-arith

Reviewer scratch: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/I3-integers-arith/review1/`
(programs `xg.py`, `tsqrt.rs`, `trat.rs`, `stress.rs`, `stress2.rs`, gp scripts `s1.gp` to `s4.gp`,
binaries in `bin/`). Repository at HEAD `c79a582`, clean tree. All runs on this machine.

Verdict: **not approved**. One blocker (item 9: the to_i64 migration misses 7 Integer call sites,
one of them a public trait impl in another crate). Everything else is sound and was reproduced;
the remaining findings are minor.

## What I reproduced (claims that hold)

| item | check | result |
|---|---|---|
| 1 | gp values of every §1.4 expectation (`s1.gp`): gcd/lcm/gcdext grid cases, `gcdext(-2^63,3)`, `gcdext(2^63-1,-2^63)`, `gcdext(-2^31,3)`, `lcm(-2^61,3)`, `lcm(-2^62,3)`, `lcm(-2^30,3)`, `gcd(-2^63,0)` | all equal the design |
| 1 | the i128 truncated-Euclid recurrence (`xg.py`) on 324 x 324 extreme pairs per width (MIN, MIN+1, MAX, MIN/k, MAX/k, random) | cofactors always fit i32 / i64; only g = 2^31 / 2^63 can leave the type, as claimed; values equal the §1.4 table |
| 1 | callers: every `impl EuclideanDomain for`, every generic `Polynomial<i64>`/`Matrix<i64|i32>` user, every explicit `EuclideanDomain::gcd(` | no test or code pins a negative i32/i64 gcd; `gcd_lcm_over_i64` is unaffected |
| 2 | fast_arith.rs:147-470 read; gp `lcm(-2^31,-2^31+1)`, `lcm(65536,65537)`, `lcm(2^32,2^32+1)`, `(-1+0)%5`, `(-2^31)^2%(2^31-1)`, `(-2^63)^2%(2^63-1)`, `(-2^64)%7`, `isprime` of the three ArithInt cases, `factor(2146654199)` = 46327 * 46337 | all as in §2.1 / §2.4; `ArithInt::is_prime` overflow at :246 confirmed by reading (`i * i` with i up to 46341) |
| 3 | `tsqrt.rs` (debug, HEAD rlib) | 10,000 bits: HEAD 761 ms vs `BigUint::sqrt` 0.55 ms; 30,001 bits: 9.17 s vs 4.2 ms (design: 768 ms, 9.3 s). num-bigint 0.4.8 `sqrt` (biguint.rs:453) is Newton from an f64 estimate with recursive scaling, exact floor |
| 3 | gp `sqrtint(3^63093+12345) % (2^127-1)`, `#binary(..)` | 130117212609866437730867252961009611028, 50001 |
| 4 | `lehmer.rs`, `lehmer_x.rs` rebuilt in release | Euclid / Lehmer / Stein at 1k, 3k, 10k, 30k, 100k bits: 161 us / 15 us / 17 us, ..., 965 ms / 15 ms / 71 ms; xgcd 100k bits Lehmer 43 ms vs 1.44 s. 3,000 + 6,000 random pairs agree |
| 4 | extra stress (`stress2.rs`): 40,413 structured signed pairs (2^x-1 vs 2^y-1, equal leading 63 bits, common factors, 2^x+1 vs 2^x-1, Fibonacci) against `num_integer::extended_gcd`, with the matrix entries instrumented to assert abs < 2^63 and u, v >= 0 | all agree, no entry out of range (the bound abs(entry) <= uh holds) |
| 4 | §4.2 point 2 (Lehmer cofactors equal HEAD's truncated Euclid on signed inputs) | proof re-derived (truncated remainders keep the dividend's sign, so abs(q_i) are Euclid's quotients on magnitudes); confirmed by the runs above and by `verify.rs` on `xgcd_lehmer.csv` (HEAD and Lehmer both equal gp on all 24 rows) |
| 5, 6 | 2-adic Newton: precision doubling (binomial terms C(k,i)(e/k)^i have valuation >= 2j since k is odd), uniqueness of the odd k-th root mod 2^K for odd k, size argument bits(c) = ceil(bits(m)/k), even part k divides v | all re-derived, correct; every None is a proof, so no Decided/Unresolved split is needed |
| 5, 6 | `ppow.rs` rebuilt in release | 2^20000+1 0.48 ms vs 9.3 ms; 2^100000+1 2.4 ms vs 127 ms; 2^1000000+1 21 ms; 3^20011 0.9 ms vs 10.4 ms; 3001^1999 0.33 ms vs 22 ms; exact_root(3^100003+2, 9999) 30 us vs is_power 776 ms. Small range -20000..20000 and 400 random agree with HEAD |
| 5, 6 | extra stress (`stress.rs`, debug assertions on): 3,000 random b of 1 to 4 words times 2^(0..69), k in {2,3,4,5,6,8,9,12,15,21,33,63,64,65,101}, negative b for odd k, b^k and b^k-1, +1, +2, cross k' in {2,3,5,7}, and is_perfect_power on every 10th | all agree with HEAD `is_power` / `is_perfect_power`; exact roots recovered exactly |
| 5, 6 | gp `ispower(16,4,&r)`, `(-8,3)`, `(64,6)`, `ispower(-64,6)`, `ispower(2^(64*7)*5^7,7,&r)`, the four `ispower(3^100003+2,k)`, `ispower(2^1000000+1)`, `ispower(3^30030)`, `ispower(3^30030+2)`, `ispower(2^100000)` | all equal §5.4 / §6.4 |
| 7 | Goetgheluck splitting re-derived (K! is K-smooth, gcd(T, K!) = 1, so got_p >= v_p(K!) and the u64 subtraction cannot underflow); `binom.rs` rebuilt in release | C(10^4,5000) 0.18 ms vs 4.0 ms; C(10^5,5*10^4) 2.5 ms vs 375 ms; C(524288,262144) 16 ms vs 10.8 s; C(4194304,2097152) 0.31 s; C(10^30+7,10^5) 0.71 s. Grid n in -60..60, k in -5..70 equal to HEAD |
| fixtures | `verify.rs` rebuilt in release against the four gp CSVs | isqrt 90, ispower_root 2486, xgcd_lehmer 24, binom_big 80 rows all pass; gp re-run of `binomial(396200,245522)` matches row 1 |
| 8 | §8.1 reasoning ((a+1)k > 2^32-1 with a >= 1 forces ak >= 2^31); gp `sigma(2^10*3^5,3)`, `sigma(7^40,5) % (2^127-1)`, `#binary` | 18285256324908244, 108698265444746923986228897305478117662, 562 |
| 10 | gp and GAP lists, Sage doc (doc.sagemath.org `sage.arith.misc.binomial`: "If m < 0, return 0", examples 0, -6, 0) | all as in §10 |
| 9 | grep of all `.to_i64()` calls | 166 calls, 138 lines without Option handling, as stated |

Digest sections (§18.4.5, §18.4.8, §18.6, §18.7, §18.8) and the GCD crossover note (digest :14,
:223) are cited correctly.

## Blocker

### B1. Item 9 (S7): the migration table misses 7 `Integer::to_i64` call sites, one a public trait impl in rustmath-rationals

The Appendix A grep excludes every line that contains `Some(`. That also drops lines where the
panicking inherent `Integer::to_i64` is wrapped in `Some(..)`. At HEAD these are:

| site | code | what it is |
|---|---|---|
| `rustmath-integers/src/quadratic_sieve.rs:93` | `return Some(n.modulo(&p_int).to_i64());` | residue mod p < 2^63 (G) |
| `rustmath-integers/src/quadratic_sieve.rs:112` | `return Some(root.to_i64());` | residue (G) |
| `rustmath-integers/src/quadratic_sieve.rs:145` | `return Some(r.to_i64());` | residue (G) |
| `rustmath-rationals/src/rational.rs:530` | `Some(self.numerator.to_i64())` | `impl NumericConversion for Rational::to_i64`, whose contract is `Option` |
| `rustmath-symbolic/src/substitute.rs:158` | `Expr::Integer(n) => Some(n.to_i64() as f64)` | `eval_float` (PROJECT_TRACKER row 2144 already flags it) |
| `rustmath-symbolic/src/integrate.rs:681` | `Some(n.to_i64())` | `polynomial_degree` exponent |
| `rustmath-symbolic/src/solve.rs:1749` | `Expr::Integer(n) => Some(n.to_i64())` | `expr_to_integer` (feeds `expr_to_rational` :1725) |

So the Integer sites are 137, not 130, in 24 crates, not 23. rustmath-rationals is not in the
step-10 crate order at all, so step 11 breaks its build with no planned replacement.

The rationals site is a live defect, reproduced (`review1/trat.rs`, HEAD debug rlibs):
`NumericConversion::to_i64(&Rational(2^70))` panics with "Integer value too large for i64"
(backtrace through rational.rs:530), while `to_i64(1/2)` returns `None`. The trait promises
`None`. Every generic `R: NumericConversion` caller with `R = Rational` inherits the panic.

The compiler will flag all seven after step 11 (each becomes `Some(Option<i64>)` or
`Option<i64> as f64`), so nothing changes silently. But the design gives no class and no
replacement for them, and two of them fall in the design's own R-keep / R-fix hazard: in
`polynomial_degree` and `expr_to_integer` a `None` already means "not a polynomial" / "not an
integer expression", the same situation the design keeps as an explicit panic at
`plane_curve_ops.rs:90` and `symbolic/polynomial.rs:147`. The obvious mechanical edit (drop the
`Some`) would push a huge exponent or integer into that channel without the per-site analysis the
design requires. Item S7 asks for the decision "with all callers in the workspace".

Required change:
- Add the 7 rows to Appendix A with a class and a replacement. Suggested: quadratic_sieve.rs
  :93/:112/:145 G (`expect` naming p < 2^63; migrate them in step 9 with the crate's other
  sites). rational.rs:530 R-fix `self.numerator.to_i64()` (exact, the trait contract).
  substitute.rs:158 R-fix `n.to_f64()` (an f64 value exists for every Integer in range; check what
  `BigInt::to_f64` returns above f64::MAX and keep `eval_float`'s `None` meaning "cannot
  evaluate"). integrate.rs:681 and solve.rs:1749: read the callers (partial_fractions :722/:899,
  expr_to_rational :1725) and decide R-fix (if `None` there only means "cannot handle", so the
  caller falls back to an unevaluated or "unknown" result) or R-keep (explicit `expect`), and state
  why.
- Add rustmath-rationals to the step-10 crate list. Its acceptance command is already in the plan.
- Add tests that panic on HEAD: `rustmath-rationals`: `NumericConversion::to_i64` of the integral
  rational 2^70 is `None` and of -2^63 is `Some(i64::MIN)`; `rustmath-symbolic`:
  `Expr::Integer(10^30).eval_float()` is `Some(1e30)` (no panic).
- Fix the counts in §9.1 / §9.3 (137 Integer sites, 24 crates) and the grep note in Appendix A
  (the `Some\(` exclusion must not drop `Some(x.to_i64())`).

## Minor

### m1. Item 8: the exponent check must come before any power is formed
HEAD computes `prime.pow(k)` (integer.rs:396) before `(multiplicity + 1) * k` (:397). §8.2 does
not fix the order. If the implementer only replaces line 397, `Integer::from(2).sigma(1 << 31)`
still builds 2^(2^31) (256 MiB; 0.56 s in debug, measured in `tsqrt.rs`) and the 1024 case about
500 MiB before returning `Err`. For an odd prime it never returns: `Integer::from(3).sigma(1 << 31)`
would compute 3^(2^31). Required: compute and check e for each factor before `prime.pow(k)` (and
before the k = 1 power). Add `Integer::from(3).sigma(1 << 31)` = `Err(NumericalError(_))` to
`test_sigma_exponent_overflow_is_an_error`, run under a time bound (it hangs on HEAD).

### m2. Item 9 test: `unramified_modulus(2^61 - 1, 2)` does not fail on HEAD
2^61 - 1 fits in i64, so HEAD's `prime.to_i64()` at unramified.rs:89 does not panic for it. Use a
prime above 2^63, for example 2^89 - 1 (gp: `isprime(2^89-1)` = 1, `(2^89-1) % 4` = 3, so the
fallback finds x^2 + 1 at counter 1; `is_irreducible_mod_p`, poly_arith.rs:308, works on
`Integer`). Check the test stays fast.

### m3. Plan steps 10 and 11: the Appendix replacements do not compile before step 11
Many R-fix replacements are written in the post-change form (`x.to_i64() == Some(c)`,
`if let Some(v) = x.to_i64()`, `.is_some_and`). In step 10 the inherent method still returns
`i64`, so these do not compile, while step 10 promises "each commit build-green". Either write the
step-10 forms as `i64::try_from(&x).ok()` (available from step 9), or do the signature change and
all downstream migrations in one commit. Say which.

### m4. Cost-item tests do not fail on HEAD, they only run long
`isqrt_of_a_100k_bit_number` (HEAD about 193 s in debug), `is_perfect_power_huge_inputs` (about
12 s), the big central rows of `binomial_big_fixture` (minutes) and the new sigma case of m1 pass
on HEAD given enough time. Wrap them in a time bound (the `within(secs, ..)` helper of
`rustmath-polynomials/tests/ch23_quotrem_gcd.rs:68`, copied into the integers test file) so that
they fail on HEAD, as the design claims.

### m5. Lehmer: use checked conversions for the matrix entries
The prototype returns `(a as i64, b as i64, c as i64, d as i64)` from i128. The bound holds
(proof: u_hat = abs(D)·u_j + abs(B)·v_j and v_hat = abs(C)·u_j + abs(A)·v_j; instrumented runs
found no violation), but §4.2 point 3 relies on "a broken invariant would panic". With `as` it
would truncate silently. Use `i64::try_from(..).expect(..)` (or a `debug_assert` plus
`try_from`).

### m6. Numberfields facade in the function the migration edits
`class_number_quadratic` (rustmath-numberfields/src/lib.rs:429-432) returns `Ok(1)` for every
polynomial discriminant with abs < 100. For x^2 + 5 (discriminant -20) it returns 1, but gp
`quadclassunit(-20).no` = 2. The design records it in §11. Since step 10 edits this function
(:419) and adds a test through it, replacing the branch with `Err(NotImplemented)` is a
three-line honest fix. Do it, or keep it as a HANDOFF follow-up with the gp evidence.

### m7. Stale comment after deleting `ArithLLong::miller_rabin`
`tests/ch18_signs_roots.rs:565-566` (`fast_arith_miller_rabin_after_pow_mod_change`, "ArithLLong::is_prime
runs Miller-Rabin through the internal pow_mod helper") becomes false. Update the comment and the
test name.

### m8. Citation nit in §4.1
The "TODO: consider Binary or Lehmer's GCD algorithms" at num-bigint `biguint.rs:939` is in
`BigUint::modinv`, not in the gcd. `BigUint::gcd` (:241) is Stein's algorithm. The conclusion (no
Lehmer in num-bigint 0.4.8) holds.

### m9. Half-gcd decision is an estimate
§4.4 argues from M(n) at 10^6 bits; no half-gcd was prototyped. That is acceptable for "decide with
numbers" given Lehmer's measured 10x to 60x, but the tracker note should say "estimated
crossover", not imply a measurement.

### m10. Binomial: very large K aborts instead of panicking
For 2^40 <= K < 2^64, `prime_range(2, K + 1)` asks for K bytes and the allocator aborts the process
(HEAD loops instead). Both are non-answers. Optional: panic with a documented message once K
exceeds a bound where the result could not be stored (C(N, K) >= 2^K bits).

## Summary for the orchestrator

Items 1 to 8 and 10 are correct, complete and reproduced. The cost claims hold within 10 percent,
the fixtures are genuine gp output, and the 2-adic root and Lehmer algorithms survived extra
adversarial stress tests. Item 9 (S7) is the right decision (checked API), but the caller table
misses 7 sites. One of them (`NumericConversion::to_i64` for `Rational`) is a reproduced panic
behind an `Option` contract, in a crate the plan never migrates. Fix B1 and the minors m1 to m4,
then this design can be approved.
