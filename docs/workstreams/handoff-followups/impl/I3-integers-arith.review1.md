# Review 1 of the implementation of Unit I3-integers-arith

Base: HEAD `c79a582` (I3 is the first unit of the round; `git log c79a582..HEAD` is empty).
Uncommitted tree: 75 modified files, 3 new modules (`binomial.rs`, `lehmer.rs`,
`perfect_power.rs`), `gen_i3.gp` and 4 new CSV fixtures. All read.
Scratch: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/I3-integers-arith.rev1/`
(test logs `t_*.log`, `check.log`, gp script `v.gp`, regenerated fixtures in `gen/`, programs in
`src/`: `stress.rs`, `smallcmp.rs`, `timing.rs`).

Verdict: **approved**. No blocker. Tests pass. Four minor notes below (docs only).

## Acceptance commands (run here)

| command | result |
|---|---|
| `cargo check --workspace --tests` | exit 0, no error. Warnings in touched files all pre-exist (e.g. the unused `NumericConversion` imports in chevalley_basis.rs, local_densities.rs, theta_series.rs were already unused at HEAD, since the inherent method won resolution) |
| `cargo test -p rustmath-core --no-fail-fast` | 49 passed, 0 failed, 1 ignored |
| `cargo test -p rustmath-integers --no-fail-fast` | 298 passed, 0 failed, 3 ignored |
| `cargo test -p rustmath-rationals --no-fail-fast` | 101 passed, 0 failed, 3 ignored |
| `cargo test -p rustmath-polynomials -p rustmath-matrix --no-fail-fast` | 672 passed, 0 failed, 5 ignored |
| `cargo test -p rustmath-symbolic ... -p rustmath-jupyter --no-fail-fast` (the 21 crates of the Plan) | 7973 passed, 0 failed, 153 ignored |
| `cargo clippy -p rustmath-core -p rustmath-integers -p rustmath-rationals` | no warning in new or rewritten code; the remaining ones are pre-existing (fast_arith.rs `prime_range`, integer.rs:578/:813, prime.rs:671, quadratic_sieve.rs, rationals sage_wrapper.rs:736) |

Counts equal the implementation report. No `rustmath-misc/output.txt` or `.tmp_dir_*` change.
The release build used for timing was deleted (`target/release` gone).

## What I checked

**Reference values.** Re-ran every gp value cited in the new tests and docs (`v.gp`):
the core gcd/lcm/gcdext lists and extremes, the fast_arith gcd/lcm/mod values,
`isprime(2147483647)`, `isprime(2147483629)`, `factor(2146654199)` = 46327·46337,
`sqrtint(3^63093+12345)` residue and 50001 bits, all `ispower(.., &r)` values,
`ispower(2^1000000+1)` etc., `binomial(2^33,3)`, `binomial(1000,500) % (2^127-1)`,
`sigma(2^10*3^5,3)`, `sigma(7^40,5)` residue and 562 bits, the negative-r binomial list,
`quadclassunit(-20|-15|-23|-3|-4|-8|-9995|-39980|40).no` = 2, 2, 3, 1, 1, 1, 40, 120, 2,
`coredisc(-39980)` = -9995, the `poldisc` values, `isprime(2^89-1)`, `polisirreducible(..)`,
`fibonacci(92)`, `fibonacci(91)`. All equal. Regenerated the four fixtures with
`tests/data/gen_i3.gp` in an empty directory: byte-identical to the committed CSVs.

**Item 1 (core i32/i64).** `gcd_u64`, `xgcd_i128`, the u64/u128 lcm and the checked `div_rem`
match the design. Panics only where MAGMA's value does not fit (gcd 2^31 / 2^63, lcm above MAX),
with the reason. Tests: gp grid through both types, 100,000 LCG pairs against Integer and
num-integer, extremes, five `should_panic`.

**Item 2 (fast_arith).** Unsigned gcd/lcm are total; `add_mod`/`mul_mod` use `rem_euclid(|m|)`
in i64/i128 (`|i32::MIN|` and `|i64::MIN|` handled), `Err(DivisionByZero)` for 0; `is_prime` is
`is_prime_u64` (Sorenson-Webster, a proof on u64); `miller_rabin` deleted; renamed test with the
new comment. Only crate-internal callers (grep over the workspace and rustmath-py).

**Item 3 (sqrt).** `BigUint::sqrt` of the magnitude; read num-bigint 0.4.8 `biguint.rs:453`:
Newton from a scaled float estimate (Brent-Zimmermann 1.13), exact floor. `prime.rs` isqrt gone.

**Item 4 (Lehmer).** Read `lehmer.rs` against Knuth Algorithm L: the L2-L3 test and update,
the B = 0 division step, the `combine` sign cases, the i64 bound of the entries, and the
cofactor update in `gcd_cofactor` (including the q = 0 swap when a < b and the unbalanced
cases). `xgcd_magnitudes` gives t by exact division. `Integer::extended_gcd` returns
`(g, sign(a)·s, sign(b)·t)`, which equals num-integer's truncated recurrence (the design's proof).
Independent stress (`src/stress.rs`, linked against the debug rlib so all debug_asserts run):
13,832 signed pairs (63..2049-bit boundary sizes, a ± 1, a >> 63/64/65, 2^k ± 1 pairs, Fibonacci
pairs up to F_6000 in both orders, q·b + 1, 3,000 random pairs with common factors), plus the
zero cases, against `num_integer` gcd and extended_gcd: all equal. Small sizes are not slower:
on the debug rlib, 200k word-size gcds 38 ms (HEAD's BigUint Euclid 45 ms), 100-bit gcd 64 ms vs
446 ms, word-size xgcd 110 ms vs 1.68 s (`src/smallcmp.rs`).

**Items 5, 6 (exact_root, is_perfect_power).** Re-derived the 2-adic argument in
`perfect_power.rs` (inverse of k from z = k correct mod 8 with five doublings, y from 1 with six,
BigUint lifting with masked products, r = a·y^(k-1), bits(c) = ceil(bits(m)/k)). Every `None` is
a proof. Stress: 16,800 `exact_root` checks (random b^k ± 1, + 2, even parts up to 2^4, k up to
120, negative input with odd k) and 600 `is_perfect_power` checks (positive and negative,
maximal exponent by brute force over all e) against `BigUint::nth_root`: all equal.

**Item 7 (binomial).** Window indices `i ≡ n (mod p)`, Legendre counts, the signed count, block
processing on u64 and BigUint windows, product trees. The caller passes K = min(k, N - k) <= N/2,
so no window term is 0 (no infinite division loop). The 2^32-bit refusal is a proven lower
bound and runs before any allocation. Stress: 374 values (random n < 3000, n at 2^32, 2^63,
2^64 - 1 and + 2, central rows at 2^12..2^17) against the product formula: all equal.

**Item 8 (sigma).** Pre-pass of every exponent in u64 before any power; k = 0 returns earlier
through `num_divisors`; the error-size claim (a·k >= 2^31) holds.

**Item 9 (to_i64).** Read every hunk of the migration. G sites use `try_from(..).expect(<invariant>)`
(where the target was `as usize/u8/u32/u64` the conversion now goes straight to that type, which
also removes silent `as` truncation); T sites `.unwrap()`; R-fix replacements are exact (Integer
comparisons, `signum`, `is_even`, `if let Ok(..)` fallthrough to the unevaluated form,
`to_f64().filter(is_finite)`, `checked_add`, `u32::try_from(..).map_err(..)?` in bsdratio and
lseries); R-keep sites keep HEAD's panic with an explicit message and a comment; D deletes the
two dead functions. `ideals.rs:535` is indeed non-test code (`rational_prime_factors`, test
modules at :166, :374, :564), so R-keep there is right. Remaining `.to_i64()` calls in the
workspace: only BigInt, rug, MPFR/BigFloat/mpc, generic `NumericConversion` receivers and the new
Option-returning inherent method in integer.rs tests; no path call (`Integer::to_i64`) and no doc
example uses it. rustmath-py: `numerator()` is `&Integer`, edition 2021, so `i64::try_from` compiles
by inspection (not built, as the design accepts).

**Class number deviation** (review 2 m3, optional part). `hcf::reduced_forms` is the standard
reduced-form count (|b| <= a <= c, b >= 0 on the boundary, primitive, a <= sqrt(|d|/3));
`fundamental_discriminant` is correct (squarefree part by trial division, then s or 4s). The
polynomial discriminant of a quadratic over Q gives the field Q(sqrt(d)) also for non-monic or
rational coefficients, and a non-integral d is refused. Cost bounded (|d| <= 10^7, about
1.3·10^7 loop steps at most). Exact values, gp-checked; `test_class_number_quadratic` (x^2 + 3)
now passes honestly (h(-3) = 1), so not moving it is fine.

**Costs (release, `src/timing.rs`, sample).** C(524288, 262144) 16 ms (doc: about 16 ms),
C(10^6, 500000) 36 ms, gcd / xgcd at 100,000 bits 15 ms / 32 ms, `is_perfect_power(2^10^6 + 1)`
21 ms, `sqrt` of the 100k-bit input 2.9 ms. Debug times of the bounded tests are well inside their
bounds (integers test binary 5 s in all).

**Other.** No unsafe; no em-dash in added prose; every new file cites MAGMA ch. 18 and the papers;
no assertion removed or weakened (the fast_arith asserts are kept in `Ok(..)` form, the sieve test
moved unchanged to `perfect_power.rs`).

## Blockers

None.

## Minor (docs only; none blocks)

1. `Integer::exact_root` doc (integer.rs, "for odd `k` at least `bits(self)` it is O(1) after
   the trailing-zero count"): `exact_root_biguint` always builds `odd_part`, which shifts the input
   and takes two word residues, so that case is O(bits), not O(1). Suggested: "it costs one O(bits)
   pass (the odd part and its residues)".
2. `rustmath-numberfields/src/ideals.rs:520` doc of `rational_prime_factors` says the leftover
   cofactor "is taken as one prime if it fits i64", which reads as if a larger one were dropped. It
   panics (as at HEAD, now with an explicit message). Suggested: "...; a cofactor beyond i64 panics
   (not supported)".
3. The `fast_arith` module header and `ArithInt::is_prime` / `ArithLLong::is_prime` docs link to
   the `pub(crate)` `is_prime_u64` (`[`is_prime_u64`]`) from public items; rustdoc reports
   `private_intra_doc_links` for that. Plain backticks avoid it.
4. `sage_wrapper.rs` binomial comment `// < log2(N/K)` on `per_factor`: with `.max(1)` the value
   can equal 1 = log2(2), so "<=" is exact.

## Review-note bookkeeping

The report folds in all six notes of design review 2 (m1 to m5 done, m6 accepted); I confirmed
each in the code: Stein outside the `within(5)` closure, the memory figure and the single signed
count per prime, the class-number delegation, the 128-bit threshold on the Lehmer debug_assert,
the direct `solve_constant_inequality` test. Handoffs to J1a/J1b (binomial work model, repl.rs
comment), S1 (series.rs fixed here, parser untouched) and C1 are recorded in the report.
