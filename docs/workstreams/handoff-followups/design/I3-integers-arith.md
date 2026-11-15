# Design I3-integers-arith (rustmath-core, rustmath-integers, rustmath-rationals)

Unit I3 of the HANDOFF "Open follow-ups" (docs/HANDOFF.md:251-333 at `9c24389`; HEAD `c79a582`
has the same code). Spec: MAGMA Handbook ch. 18, digest
`rustmath-lava/docs/Chapter_18_Ring_of_Integers.md` (§18.4.5 `IsPower`, §18.4.8 `Isqrt`, §18.6
`Gcd`/`Lcm`/`Xgcd`, §18.7 `DivisorSigma`, §18.8 `Binomial`).

Experiments, prototypes and gp scripts:
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/I3-integers-arith/`
(`rc.sh` builds a program against the debug rlibs, `rr.sh` against a release build of
rustmath-integers in `target/release`; `gen_i3.gp` makes the new fixtures in `gp/`; `verify.rs`
checks every prototype against them). All timings below were measured on this machine.
Revision 1 experiments are in the subdirectory `rev1/` of that directory.

## Revision 1 (after review 1)

Review: `I3-integers-arith.review1.md`. Every issue is accepted. Each one is listed below with
what changed in the body. The body has been updated in place, so it is one consistent design.

| issue | change |
|---|---|
| B1 (blocker), item 9: the Appendix A grep dropped every line with `Some(`, which hid 7 sites of the panicking `Integer::to_i64` | Confirmed: `grep -rn "\.to_i64()" --include=*.rs . \| grep "Some("` lists exactly these 7 Integer sites; the other `Some(` lines are Option-typed asserts or chains. Reproduced on HEAD rlibs: `NumericConversion::to_i64` of the integral Rationals 2^70, 2^63 and -2^63-1 panics, -2^63 gives `Some(i64::MIN)` (`rev1/trat2.rs`); `Expr::Integer(10^30).eval_float()` panics at substitute.rs:158 (`rev1/sym.rs`); `polynomial_degree(x^(10^30), x)` panics at integrate.rs:681 and `polynomial_degree(x^(2^62)·x^(2^62), x)` overflows at integrate.rs:697 (`rev1/pdeg.rs`). Appendix A gains the 7 rows: quadratic_sieve.rs:93/:112/:145 are G, migrated in step 9; rational.rs:530 is R-fix `i64::try_from(&self.numerator).ok()` (the trait contract); substitute.rs:158 is R-fix `n.to_f64().filter(\|v\| v.is_finite())` (num-bigint returns `Some(inf)` above f64::MAX, so the filter keeps `None` = "cannot evaluate"); integrate.rs:681 is R-fix (`None`, plus `checked_add` at :697), with the caller analysis in §9.3; solve.rs:1749 and :1722 get a new class D: `expr_to_integer` and `expr_to_rational` are dead code (rustc at HEAD: "function `expr_to_rational` is never used", "function `expr_to_integer` is never used"), so both are deleted. Re-reading solve.rs also changed rows :1133/:1137: the old `to_f64` replacement gives NaN for a rational with huge parts and so a false `Empty`; they are now exact sign tests. Counts in §9.1/§9.3: 145 rows, 137 Integer sites in 24 crates. rustmath-rationals is first in the step-10 crate list. New tests in §9.4 (rationals, eval_float, polynomial_degree, constant inequalities); each fails on HEAD except the marked boundary guards. |
| m1, item 8: the exponent check must come before any power | §8.2 now checks every exponent (a+1)·k in a pre-pass, before any power is formed, the k = 1 branch included. §8.4 adds `Integer::from(3).sigma(1 << 31)` = `Err(NumericalError)` under a 10 s bound (it does not return on HEAD). |
| m2: the rings test with 2^61-1 does not fail on HEAD | Confirmed (`rev1/unr.rs`): at HEAD `unramified_modulus(2^61-1, 2)` returns x^2+1 in 9 ms (debug), `unramified_modulus(2^89-1, 2)` panics at unramified.rs:89. The test now uses 2^89-1 (gp `isprime(2^89-1)` = 1, `(2^89-1)%4` = 3, `polisirreducible(Mod(1,2^89-1)*(x^2+1))` = 1). A second test reaches :212 through `UnramifiedExtension::with_modulus(2^89-1, x^2+1, 3)`, which panics on HEAD (`rev1/unr2.rs`) and must give degree 2, `is_conway() = false`. |
| m3: Option-form replacements do not compile in step 10 | Steps 10 and 11 stay separate commits. Every step-10 replacement avoids the inherent method: it uses `i64::try_from` / `u32::try_from` / `usize::try_from` (from step 9), Integer comparisons (`is_one()`, `== Integer::from(c)`), `signum()`, `is_even()` or `to_f64()`. These compile before and after step 11. Appendix A rows are rewritten in that form (§9.3 states the rule). |
| m4: cost tests only run long on HEAD | A copy of the `within(secs, ..)` helper (rustmath-polynomials/tests/ch23_quotrem_gcd.rs:66-84) goes into `rustmath-integers/tests/ch18_signs_roots.rs`. Bounds, with measured debug times after the fix: isqrt 10 s (48 ms, `rev1/tsq.rs`), is_perfect_power 5 s (about 0.2 s; HEAD 13.2 s in debug, `rev1/tsq.rs`), binomial central rows 10 s (0.7 s, prototype `binom.rs` built in debug), whole binomial fixture 60 s (5.1 s, `rev1/vbinom.rs`), sigma 10 s (instant). A Lehmer speed test is added too: 100,000-bit gcd and xgcd under 5 s (debug: HEAD 9.7 s + 11.3 s, Lehmer 0.14 s + 0.32 s, `rev1/tlehmer.rs`). §3.4, §4.5, §6.4, §7.4 and §8.4 say which fail on HEAD in debug and which also in release. |
| m5: `as i64` in lehmer_matrix | §4.2: the entries are converted with `i64::try_from(..).expect(..)`. |
| m6: `class_number_quadratic` placeholder returns 1 for every \|d\| < 100 | Fixed in the step-10 numberfields commit: the branch returns `Err(NotImplemented)`. The existing `test_class_number_quadratic` used x^2+3 (polynomial discriminant -12), which passed only through the placeholder; it moves to x^2+x+1 (discriminant -3, gp `quadclassunit(-3).no` = 1). New test: x^2+5 gives `Err(NotImplemented)` (HEAD `Ok(1)`, gp `quadclassunit(-20).no` = 2). x^2+3 (h = 1) now also gives `Err(NotImplemented)`: honest, since no algorithm covers it. §11 and the HANDOFF note record the remaining gap. |
| m7: stale comment after deleting `ArithLLong::miller_rabin` | §2.4 and Plan step 2: the test becomes `fast_arith_llong_is_prime_mersenne_61` with a comment naming `is_prime_u64`. |
| m8: citation | §4.1: `BigUint::gcd` (biguint.rs:241) is Stein's algorithm; the Lehmer TODO (biguint.rs:939) is in `BigUint::modinv`. |
| m9: half-gcd decision is an estimate | §4.4 and Plan step 12 say "estimated crossover"; no half-gcd was prototyped. |
| m10: binomial with an enormous K aborts the process | Adopted. §7.2: `SageInteger::binomial` panics with a documented message when a proven lower bound on the result size exceeds 2^32 bits. This replaces the earlier K >= 2^64 rule. The bound never refuses a result of at most 2^32 bits (proof in §7.2; checked against `math.comb` for all N < 400 and 2,000 random cases, `rev1/lb_check.py`). The only non-test caller, the REPL, caps results at 2^22 bits first. Test in §7.4. |

Items and verdicts:

| # | item | verdict |
|---|---|---|
| 1 | i32/i64 sign-arbitrary default gcd/lcm/xgcd | fix (override in rustmath-core) |
| 2 | ArithInt/ArithLLong gcd, lcm overflow on MIN; add_mod/mul_mod assume m > 0 | fix (plus a found overflow in ArithInt::is_prime) |
| 3 | Integer::sqrt starts Newton at x = self | fix (BigUint::sqrt; also the private prime.rs isqrt) |
| 4 | faster balanced gcd (Lehmer, half-gcd) | Lehmer gcd and xgcd: fix. Half-gcd: not now, decided with measured Lehmer numbers and an estimated crossover (§4.4) |
| 5 | IsPower(n, k) returns no root | fix (`Integer::exact_root`) |
| 6 | is_perfect_power costs about bits^2 on huge non-powers | fix (2-adic exact roots, essentially linear) |
| 7 | S1 faster SageInteger::binomial | fix (Goetgheluck prime splitting + product tree; documented panic beyond 2^32 result bits) |
| 8 | S5 Integer::sigma forms (multiplicity + 1) * k unchecked | fix (u64 exponents checked before any power; Err beyond u32) |
| 9 | S7 panicking inherent Integer::to_i64 | decision: checked API; migrate all 137 Integer call sites (24 crates) |
| 10 | Binomial(n, r) for r < 0 unverified | doc fix: PARI, GAP and Sage agree on 0; MAGMA stays unconfirmed |

Nothing is declared out of scope. Item 4's half-gcd is a "not worth it yet" (the item asks to
"decide with numbers"): Lehmer is measured, the half-gcd crossover is estimated. It gets a
tracker note.

Conventions: "gp" is PARI/GP 2.15.4 (`gp -q`), "GAP" is GAP 4.12.1. Each expected value below
names its command.

---

## 1. i32 / i64 gcd, lcm, xgcd (rustmath-core)

### 1.1 Defect at HEAD

`rustmath-core/src/traits.rs:439` (`impl EuclideanDomain for i32`) and `:483` (i64) override only
`norm`, `div_rem` and `normalization_unit`; gcd, lcm and extended_gcd are the trait defaults
(`:227`, `:241`, `:251`), which run Euclid over the truncating `div_rem` and keep the sign of the
last remainder. Reproduction (`t1_prim_gcd.rs`, debug):

| call | HEAD | gp |
|---|---|---|
| gcd(4, -6) | -2 | `gcd(4,-6)` -> 2 |
| gcd(0, -5) | -5 | 5 |
| lcm(-4, 6) | -12 | `lcm(-4,6)` -> 12 |
| extended_gcd(4, -6) | (-2, 1, 1) | `gcdext(4,-6)` -> [-1, -1, 2], so (2, -1, -1) |
| extended_gcd(-12, 18) | (6, 1, 1) | fine |
| lcm(2^40, 2^40) (i64) | panic "attempt to multiply with overflow" (the default forms a·b) | 2^40 |
| lcm(65536, 65536) (i32) | panic (overflow) | 65536 |
| gcd(i32::MIN, 0) | -2147483648 | 2147483648 (does not fit i32) |
| gcd(i32::MIN, -1) | panic "attempt to divide with overflow" (div_rem(MIN, -1)) | 1 |
| norm(i32::MIN) | panic "attempt to negate with overflow" (`self.abs()`) | 2^31 |
| div_rem(i64::MIN, -1) | panic | quotient 2^63 does not fit |

MAGMA §18.6: `Gcd` is non-negative, `Lcm` is non-negative, `Xgcd` gives g >= 0.

### 1.2 Fix

In `rustmath-core/src/traits.rs`, two private helpers and full overrides for i32 and i64 (the
same shape; the i32 version uses u64 / i64-wide intermediates, the i64 version u64 / u128 /
i128):

```rust
/// Euclid on magnitudes ([Knu97] §4.5.2 Algorithm A); gcd(0, 0) = 0.
fn gcd_u64(mut a: u64, mut b: u64) -> u64 { while b != 0 { let r = a % b; a = b; b = r; } a }

/// Extended Euclid with truncated quotients, the recurrence of num_integer::Integer::extended_gcd
/// and of Integer::extended_gcd, then (g, x, y) negated when g < 0; (0, 0, 0) for (0, 0).
fn xgcd_i128(a: i128, b: i128) -> (i128, i128, i128) {
    if a == 0 && b == 0 { return (0, 0, 0); }
    let (mut r0, mut r1, mut s0, mut s1, mut t0, mut t1) = (a, b, 1i128, 0i128, 0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (s0, s1) = (s1, s0 - q * s1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    if r0 < 0 { (-r0, -s0, -t0) } else { (r0, s0, t0) }
}

impl EuclideanDomain for i64 {
    fn norm(&self) -> u64 { self.unsigned_abs() }
    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if *other == 0 { return Err(MathError::DivisionByZero); }
        match (self.checked_div(*other), self.checked_rem(*other)) {
            (Some(q), Some(r)) => Ok((q, r)),
            _ => Err(MathError::NumericalError("i64::MIN / -1 overflows i64".to_string())),
        }
    }
    /// MAGMA Gcd (ch. 18 §18.6): non-negative. # Panics: gcd = 2^63 (both inputs in {0, MIN}).
    fn gcd(&self, other: &Self) -> Self {
        let g = gcd_u64(self.unsigned_abs(), other.unsigned_abs());
        i64::try_from(g).expect("gcd(i64::MIN, 0) = 2^63 does not fit in i64")
    }
    /// MAGMA Lcm: |a|/g·|b| formed in u128; 0 when an input is 0. # Panics: lcm > i64::MAX.
    fn lcm(&self, other: &Self) -> Self {
        if *self == 0 || *other == 0 { return 0; }
        let (a, b) = (self.unsigned_abs(), other.unsigned_abs());
        let l = u128::from(a / gcd_u64(a, b)) * u128::from(b);
        i64::try_from(l).expect("lcm does not fit in i64")
    }
    /// MAGMA Xgcd: g >= 0, same multipliers as Integer::extended_gcd (= PARI gcdext).
    /// # Panics: only g = 2^63 (inputs (MIN, 0), (0, MIN), (MIN, MIN)).
    fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        let (g, x, y) = xgcd_i128(i128::from(*self), i128::from(*other));
        let f = |v: i128| i64::try_from(v).expect("extended_gcd: gcd = 2^63 does not fit in i64");
        (f(g), f(x), f(y))
    }
    fn normalization_unit(&self) -> Option<Self> { /* unchanged */ }
}
```

Correctness. gcd: Euclid on the magnitudes returns gcd(|a|, |b|) >= 0, which fits in i64
except for 2^63, which needs both inputs in {0, i64::MIN}. lcm: |a|/g·|b| < 2^126 fits u128;
the conversion fails exactly when MAGMA's value does not fit. xgcd: the i128 recurrence is
num_integer's and Integer's; the cofactors of Euclid satisfy |x| <= |b|/g and |y| <= |a|/g
(Knuth §4.5.2), so only g itself can leave i64, again only at 2^63. Rust integer overflow
panics in debug builds only; here every non-representable result panics in every build, with
the reason. That is the honest choice for a `-> Self` trait method.

### 1.3 Callers and behaviour changes

Every generic `R: EuclideanDomain` call with R = i32 or i64. Found by grepping files that have
EuclideanDomain in scope, call gcd/lcm/extended_gcd, and mention i32/i64:
`rustmath-polynomials/src/univariate.rs` (gcd/lcm/content of `UnivariatePolynomial<i64>`; it
normalises with `normalization_unit`, so its results do not change;
`rustmath-polynomials/tests/ch23_quotrem_gcd.rs:712` `gcd_lcm_over_i64` keeps passing),
`rustmath-matrix/src/integer_forms.rs` (SNF/HNF; instantiated with `Integer` only in the
workspace). No other generic use with i32/i64 was found. `div_rem(MIN, -1)` changes from a
panic to `Err`. No test pins a negative gcd for i32/i64.

### 1.4 Tests (all fail on HEAD except the grid rows with positive gcd)

In `rustmath-core/src/traits.rs` `mod tests`:

- `test_i32_i64_gcd_lcm_xgcd_signs_match_pari`, for both types: gcd(4,-6)=2, gcd(0,-5)=5,
  gcd(-5,0)=5, gcd(0,0)=0 (gp `[gcd(4,-6),gcd(0,-5),gcd(-5,0),gcd(0,0)]` -> [2,5,5,0]);
  lcm(-4,6)=12, lcm(-4,-6)=12, lcm(0,-5)=0 (gp); extended_gcd(3,-7)=(1,-2,-1) (gp
  `gcdext(3,-7)` -> [-2,-1,1]), (4,-6)=(2,-1,-1) (gp -> [-1,-1,2]), (-4,-6)=(2,1,-1) (gp ->
  [1,-1,2]), (0,-5)=(5,0,-1), (5,-5)=(5,0,-1), (-5,5)=(5,0,1), (0,0)=(0,0,0).
- `test_i64_euclidean_extremes`: gcd(i64::MIN, 6) = 2 (gp `gcd(-2^63,6)`); gcd(i64::MIN, -1) =
  1; extended_gcd(i64::MIN, 3) = (1, 1, 3074457345618258603) (gp `gcdext(-2^63,3)` ->
  [1, 3074457345618258603, 1]); extended_gcd(i64::MIN, -1) = (1, 0, -1) (gp -> [0,-1,1]);
  extended_gcd(i64::MAX, i64::MIN) = (1, -1, -1) (gp `gcdext(2^63-1,-2^63)` -> [-1,-1,1]);
  lcm(-2^61, 3) = 6917529027641081856 (gp); lcm(2^40, 2^40) = 2^40; norm(i64::MIN) = 2^63;
  div_rem(i64::MIN, -1) is `Err(NumericalError(_))`; div_rem(i64::MIN, 2) = Ok((-2^62, 0)).
- `test_i32_euclidean_extremes`: gcd(i32::MIN, 6) = 2; extended_gcd(i32::MIN, 3) =
  (1, 1, 715827883) (gp `gcdext(-2^31,3)` -> [1, 715827883, 1]); extended_gcd(i32::MAX,
  i32::MIN) = (1, -1, -1) (gp `gcdext(2^31-1,-2^31)`); lcm(65536, 65536) = 65536;
  lcm(-2^29, 3) = 1610612736 (gp); norm(i32::MIN) = 2^31.
- `#[should_panic(expected = "does not fit in i64")] test_i64_gcd_min_zero_panics` (gp
  `gcd(-2^63,0)` -> 9223372036854775808 > i64::MAX); same for `lcm(-2^62, 3)` (gp ->
  13835058055282163712) and `extended_gcd(i64::MIN, 0)`; i32: `gcd(i32::MIN, 0)`,
  `lcm(-2^30, 3)` (gp -> 3221225472).

In `rustmath-integers/tests/ch18_signs_roots.rs` (it already depends on rustmath-core):

- `gcd_grid_fixture_through_i32_and_i64`: every row of `tests/data/gcd_grid.csv` (3,721 gp rows
  a, b in -30..30: gcd, lcm, gcdext) through `<i32 as EuclideanDomain>` and
  `<i64 as EuclideanDomain>`. The prototype `prim.rs` passes all rows.
- `machine_word_gcd_agrees_with_integer`: 100,000 LCG pairs (i64 values shifted right by a
  random amount, every 17th b = i64::MIN, every 19th b = 0) against `Integer` gcd / lcm /
  extended_gcd, whenever the Integer result fits. The prototype ran 200,000 pairs.

---

## 2. ArithInt / ArithLLong (rustmath-integers/src/fast_arith.rs)

### 2.1 Defects at HEAD (`t3_fast.rs`, debug)

| call | HEAD | gp |
|---|---|---|
| `ArithInt::gcd(i32::MIN, 0)` (:202) | panic "attempt to negate with overflow" (`a.abs()`) | 2147483648 |
| `ArithInt::gcd(i32::MIN, 6)` | panic | 2 |
| `ArithInt::lcm(i32::MIN, 1)` (:215) | panic | 2147483648 |
| `ArithInt::lcm(65536, 65537)` | panic "attempt to multiply with overflow" | 4295032832 |
| `ArithLLong::gcd(i64::MIN, 6)` (:370), `lcm(i64::MIN, 1)` (:383) | panic | 2, 9223372036854775808 |
| `ArithLLong::lcm(2^32, 2^32+1)` | panic | 18446744078004518912 |
| `add_mod(3, 4, 0)`, `mul_mod(3, 4, 0)` (:147, :158, :315, :326) | panic "remainder with a divisor of zero" | gp `Mod(7, 0)` refuses |
| `add_mod(-1, 0, -5)` (both) | -6 | gp `(-1+0)%5` -> 4 |
| `ArithInt::is_prime(2147483647)` (:223, loop at :246) | panic "attempt to multiply with overflow" (`i * i` passes i32::MAX) | gp `isprime(2^31-1)` -> 1 |

The last row is not in the HANDOFF list; it is the same struct and found while reproducing.

### 2.2 Fix (API changes; the only callers are tests in this crate)

- `ArithInt::gcd(&self, a: i32, b: i32) -> u32` (was i32): Euclid on `unsigned_abs`, total.
- `ArithInt::lcm(&self, a: i32, b: i32) -> u64` (was i32): `|a|/g·|b| <= 2^62`, total; 0 when an
  input is 0.
- `ArithLLong::gcd -> u64`, `ArithLLong::lcm -> u128`, same formulas.
- `add_mod`, `mul_mod` -> `Result<i32>` / `Result<i64>` with `pow_mod`'s conventions:
  `m = 0` is `Err(DivisionByZero)`, the result is the least non-negative residue modulo `|m|`
  (i64 / i128 arithmetic, `rem_euclid(|m|)`; `|i64::MIN| = 2^63` as i128, and the residue
  `< 2^63` fits i64). Docs lose the "Requires m > 0" sentences.
- `ArithInt::is_prime` and `ArithLLong::is_prime`: `n >= 2 && is_prime_u64(n as u64)` (the
  deterministic test at :70, proven for all u64 by Sorenson-Webster). Delete the private
  `ArithLLong::miller_rabin` (:417), which was the only internal user of `mul_mod`.
  `small_primes` and `get_small_primes` stay (public API).

MAGMA: §18.6 Gcd/Lcm non-negative; §18.12.1 `Modexp` conventions already used by `pow_mod`.
The unsigned return types encode non-negativity and make gcd/lcm total, so no panic remains.

### 2.3 Callers

`grep -rn "ArithInt\|ArithLLong"`: only `rustmath-integers/src/lib.rs:21` (re-export),
`fast_arith.rs` tests, `tests/ch18_signs_roots.rs` (pow_mod only) and `tests/ch18_residues.rs`
(:757-809; `add_mod` at :767 becomes `Ok(9_223_372_036_854_775_805)`). In `fast_arith.rs`
tests, `add_mod`/`mul_mod` expectations become `Ok(..)`; the gcd/lcm asserts keep compiling
(literals infer u32/u64). Nothing outside the crate (rustmath-py included) uses them.

### 2.4 Tests (in `tests/ch18_residues.rs`, section T8)

- `fast_arith_gcd_lcm_extremes` (fail on HEAD): `ArithInt`: gcd(i32::MIN, 0) = 2147483648 (gp
  `gcd(-2^31,0)`), gcd(i32::MIN, i32::MIN) = 2147483648, gcd(i32::MIN, 6) = 2, lcm(i32::MIN, 1)
  = 2147483648, lcm(65536, 65537) = 4295032832, lcm(i32::MIN, i32::MIN + 1) =
  4611686016279904256 (gp `lcm(-2^31,-2^31+1)`). `ArithLLong`: gcd(i64::MIN, 0) =
  9223372036854775808 (gp), gcd(i64::MIN, 6) = 2, lcm(i64::MIN, 1) = 9223372036854775808,
  lcm(2^32, 2^32+1) = 18446744078004518912 (gp).
- `fast_arith_add_mul_mod_any_modulus` (fail on HEAD): add_mod(-1, 0, -5) = Ok(4) (gp
  `(-1+0)%5`), mul_mod(3, 4, -5) = Ok(2), add_mod(3, 4, 0) = Err(DivisionByZero);
  `ArithInt::mul_mod(i32::MIN, i32::MIN, i32::MAX)` = Ok(1) (gp `(-2^31)^2 % (2^31-1)` -> 1),
  `ArithInt::add_mod(i32::MAX, i32::MAX, i32::MAX)` = Ok(0); `ArithLLong::mul_mod(i64::MIN,
  i64::MIN, i64::MAX)` = Ok(1) (gp `(-2^63)^2 % (2^63-1)` -> 1), `add_mod(i64::MIN, i64::MIN,
  7)` = Ok(5) (gp `(-2^64) % 7` -> 5), `mul_mod(3, 5, i64::MIN)` = Ok(15),
  `add_mod(i64::MIN, 0, i64::MIN)` = Ok(0).
- `fast_arith_add_mul_mod_agree_with_integer`: a, b in -20..=20, m in -15..=15 without 0, plus
  the extremes, against `Integer::modulo(|m|)` of the exact sum / product.
- `fast_arith_int_is_prime_near_i32_max` (panics on HEAD): is_prime(2147483647) and
  is_prime(2147483629) true (gp `isprime`), is_prime(2146654199) false (gp: 46337 ·
  `precprime(46336)`), and agreement with `Integer::is_prime` for every n in
  [2^31 - 20000, 2^31 - 1] (ArithInt) and [2^63 - 2000, 2^63 - 1] (ArithLLong).
- Existing `fast_arith_miller_rabin_after_pow_mod_change` (`tests/ch18_signs_roots.rs:565`):
  its comment "ArithLLong::is_prime runs Miller-Rabin through the internal pow_mod helper"
  becomes false once `miller_rabin` is deleted. Rename it `fast_arith_llong_is_prime_mersenne_61`
  and write the comment "ArithLLong::is_prime is the deterministic is_prime_u64
  (fast_arith.rs:70)". The asserts stay (gp `isprime(2^61-1)` -> 1, `isprime(2^61+1)` -> 0).

---

## 3. Integer::sqrt (rustmath-integers/src/integer.rs:216)

### 3.1 Defect

`Integer::sqrt` runs Newton's iteration from x = self (`:228`), so it needs about bits/2
iterations, each a full division before it gets near the root. The private
`prime.rs:180 isqrt` (used by `is_prime_trial_division`, `prime.rs:19`) has the same start.
Measured (`t2_sqrt.rs`, input 3^k + 12345; values identical to `BigUint::sqrt`):

| bits | HEAD debug | BigUint::sqrt debug | HEAD release | BigUint::sqrt release |
|---|---|---|---|---|
| 1,000 | 5.7 ms | 38 µs | 0.43 ms | 2.9 µs |
| 10,000 | 768 ms | 0.9 ms | 73 ms | 57 µs |
| 30,000 | 9.3 s | 3.7 ms | 825 ms | 0.35 ms |
| 100,000 | 193 s | 42 ms | 13.3 s | 3.0 ms |
| 300,000 | not run | 313 ms | 217 s | 22 ms |

### 3.2 Fix

```rust
/// MAGMA Isqrt (Handbook ch. 18 §18.4.8): floor(sqrt(self)) for self >= 0.
/// num-bigint's BigUint::sqrt: Newton's iteration from a floating-point estimate
/// (Brent-Zimmermann, Modern Computer Arithmetic, Algorithm 1.13 SqrtInt), exact floor.
/// # Errors: InvalidArgument for self < 0.
pub fn sqrt(&self) -> Result<Self> {
    if self.signum() < 0 { return Err(MathError::InvalidArgument("Square root of negative number".into())); }
    Ok(Integer::new(BigInt::from(self.value.magnitude().sqrt())))
}
```

`prime.rs`: delete `isqrt`; `is_prime_trial_division` uses `n.sqrt().expect("n > 3")`.
Header line of integer.rs gains "§18.4.8 Isqrt".

### 3.3 Callers

Values are identical (both are the exact floor), so every caller keeps its results and gets
faster: `is_perfect_square` (integer.rs:675), `crt.rs:111`, `factorint.rs:191`,
`quadratic_sieve.rs:228`, `:376`, and `Integer::sqrt` callers in other crates.

### 3.4 Tests

- `isqrt_big_fixture` (`tests/ch18_signs_roots.rs`): new `tests/data/isqrt_big.csv`, 90 gp rows
  `a, sqrtint(a)` (65 to 3000 bits; every third row a square b^2, b^2 + 1 or b^2 - 1), plus the
  floor property r^2 <= a < (r+1)^2. Generated by `gen_i3.gp` (35 KB). Passes on HEAD (value
  guard).
- Time bounds. The cost items (3, 4, 6, 7, 8) are tested under a time bound, so that HEAD
  fails instead of passing slowly. `tests/ch18_signs_roots.rs` gets one copy of
  `within(secs, f)` from `rustmath-polynomials/tests/ch23_quotrem_gcd.rs:66-84` (with its
  `use std::sync::mpsc; use std::time::Duration;`): it runs `f` on a thread, re-raises a panic
  of `f`, and panics "timed out after {secs} s" otherwise. Each bound is at least 10 times the
  measured debug time after the fix.
- `isqrt_of_a_100k_bit_number` (`tests/ch18_signs_roots.rs`, run as `within(10, ..)`):
  `(3^63093 + 12345).sqrt()` has 50001 bits and is
  130117212609866437730867252961009611028 mod 2^127 - 1 (gp `sqrtint(3^63093+12345) %
  (2^127-1)`, `#binary(..)`). After the fix the whole test (power, root, reduction) takes
  48 ms in debug (`rev1/tsq.rs`, through `BigUint::sqrt`). On HEAD the root alone takes 193 s
  in debug and 13.3 s in release, so the test fails on HEAD in both profiles.
- `integer.rs::test_sqrt` stays.

---

## 4. Faster balanced gcd: Lehmer gcd and xgcd

### 4.1 What Integer uses now, measured

`EuclideanDomain::gcd` for Integer (`integer.rs:1529`) is Euclid with `BigUint %`;
`extended_gcd` (`:1559`) is `num_integer::Integer::extended_gcd` on `BigInt` (truncated
extended Euclid, a BigInt division and two BigInt multiply-subtracts per step). num-bigint 0.4.8
has no Lehmer: `BigUint::gcd` (num_integer trait, `biguint.rs:241`) is Stein's binary gcd
(its comment at :247 says "Stein's algorithm"). The crate's only Lehmer mention is a TODO in
`BigUint::modinv` ("TODO: consider Binary or Lehmer's GCD algorithms", `biguint.rs:939`).

Prototype `lehmer.rs` / `lehmer_x.rs` (Knuth Algorithm L with 63-bit leading parts). Release
build, balanced random operands (b has 7 bits fewer):

| bits | Euclid (HEAD gcd) | Lehmer gcd | Stein (num-integer) | xgcd HEAD | Lehmer xgcd | a·b |
|---|---|---|---|---|---|---|
| 128 | 3.9 µs | 1.1 µs | 1.4 µs | 17.8 µs | | 0.5 µs |
| 1,000 | 151 µs | 15 µs | 17.7 µs | 212 µs (194 µs) | 33 µs | 1.1 µs |
| 3,000 | 516 µs | 98 µs | 120 µs | 1.41 ms | | 4.9 µs |
| 10,000 | 8.1 ms | 0.53 ms | 1.07 ms | 14.3 ms (7.8 ms) | 0.66 ms | 30 µs |
| 30,000 | 75.5 ms | 1.84 ms | 7.0 ms | 87 ms | | 94 µs |
| 100,000 | 994 ms | 16.2 ms | 65 ms | 1.12 s (1.17 s) | 32.9 ms | 0.55 ms |
| 300,000 | 10.4 s | 154 ms | 582 ms | 12.8 s | | 2.5 ms |
| 1,000,000 | 117.9 s | 1.92 s | 6.5 s | not run | | 15.7 ms |

(The 128-bit row is a second run; the bracketed xgcd times are from the `lehmer_x.rs` run that
also timed Lehmer xgcd, with b negative.)

Debug build: 128 bits 40 µs vs 5.1 µs; 1,000 bits 775 µs vs 67 µs; 10,000 bits 45 ms vs
1.7 ms. Unbalanced operands keep Euclid's single division (Lehmer's leading part of the small
operand is 0, so step L4 divides), e.g. `gcd(2^100000 + 1, 1)`.

### 4.2 Fix: new crate-private module `rustmath-integers/src/lehmer.rs`

Header: MAGMA ch. 18 §18.6 (`Gcd`, `Xgcd`; the digest names Weber's accelerated gcd [Web95] and
Lehmer's XGCD [Knu97, pp. 345-348] for the classical range, Schönhage's half-gcd [Sch71]
above it); D. E. Knuth, TAOCP vol. 2, 3rd ed., §4.5.2 Algorithm L (Lehmer 1938).

```rust
/// Leading 63 bits of u at shift s = bits(u) - 63, and v's bits at the same shift.
fn shr_u64(x: &BigUint, s: u64) -> u64          // reads two digits via iter_u64_digits()
/// Knuth L2-L3 on (uh, vh): the cofactor matrix [[A, B], [C, D]] of the Euclid quotients that
/// are certain (q = floor((uh+A)/(vh+C)) = floor((uh+B)/(vh+D))). i128 arithmetic; entries
/// fit i64 (|.| <= uh < 2^63). Each entry leaves through
/// `i64::try_from(x).expect("lehmer_matrix: |entry| <= uh < 2^63 (Knuth 4.5.2)")`, never `as`.
fn lehmer_matrix(uh: u64, vh: u64) -> (i64, i64, i64, i64)
/// p·x + q·y as a BigUint, for a pair (p, q) of opposite signs (or one zero) with a result >= 0.
fn combine(x: &BigUint, p: i64, y: &BigUint, q: i64) -> BigUint
pub(crate) fn gcd(a: &BigUint, b: &BigUint) -> BigUint
/// (g, s) with g = gcd(a, b) and s the Euclid cofactor of a: g = s·a + t·b for an integer t.
pub(crate) fn gcd_cofactor(a: &BigUint, b: &BigUint) -> (BigUint, BigInt)
```

`gcd`: order (u, v) with u >= v; loop: v = 0 returns u; u < 2^64 finishes with u64 Euclid;
otherwise take (uh, vh) = leading 63 bits at the same shift, run `lehmer_matrix`; if B = 0, one
multiprecision step (u, v) <- (v, u mod v); else (u, v) <- (A·u + B·v, C·u + D·v). This is the
prototype that agreed with Euclid on 3,000 random pairs (with common factors, b = a, b = 3a).

`gcd_cofactor`: the same loop on (a, b) without swapping; when u < v or u < 2^64 or B = 0 it
does a plain step (q, r) = divmod(u, v), (su, sv) <- (sv, su - q·sv); a matrix step also maps
(su, sv) <- (A·su + B·sv, C·su + D·sv) (BigInt). It agreed with `num_integer::extended_gcd` on
6,000 random signed pairs (zeros, a = b, b = 5a included) and on F_5001, F_5000.

`Integer` changes (`integer.rs`):
- `EuclideanDomain::gcd` (:1529): `Integer::new(BigInt::from(lehmer::gcd(|a|, |b|)))`.
- `lcm` (:1542): unchanged formula over the new gcd.
- `extended_gcd` (:1559): (0, 0) gives (0, 0, 0); else `(g, s) = gcd_cofactor(|a|, |b|)`,
  t = (g - s·|a|) / |b| (exact; t = 0 when b = 0), and the result is
  (g, sign(a)·s, sign(b)·t).
- Docs of the inherent gcd/extended_gcd (:100-:130) and of the overrides name Lehmer.

Correctness.
1. gcd: Knuth §4.5.2 proves that step L3 runs only when q is the true Euclid quotient of (u, v)
   (both bounds agree), so the matrix maps (u, v) to two consecutive remainders of Euclid's
   sequence, and a B = 0 step is a Euclid step. Each pass makes progress, so the loop ends at
   gcd(|a|, |b|) >= 0.
2. xgcd equals HEAD's (= PARI gcdext on the existing fixtures). Write Euclid on the signed (a, b)
   with truncated quotients as HEAD does: r_{i+1} = r_{i-1} - q_i r_i, and a truncated remainder
   has the sign of its dividend, so sign(r_{i+1}) = sign(r_{i-1}), |q_i| = floor(|r_{i-1}|/|r_i|)
   and |r_i| is Euclid's sequence on (|a|, |b|). By induction s_i = σ_i·sign(a)·s'_i and
   t_i = σ_i·sign(b)·t'_i, where σ_i = sign(r_i) and (s'_i, t'_i) are the cofactors on the
   magnitudes (base cases i = 0, 1; step uses σ_{i+1} = σ_{i-1}). HEAD negates the final triple
   when r_last < 0, which multiplies by σ_last, so HEAD returns (|g|, sign(a)·s', sign(b)·t').
   Lehmer applies exactly Euclid's quotients (point 1), so its (g, s') is Euclid's, and t' is
   determined by g = s'|a| + t'|b|. The case |a| < |b| starts with a quotient 0 (a swap) in
   both.
3. The `combine` subtraction cannot underflow: after at least one L3 step the pairs (A, B) and
   (C, D) have opposite signs or a zero entry (Knuth), and the results are remainders >= 0.
   The matrix entries satisfy |entry| <= uh < 2^63: with (u_j, v_j) the current pair,
   uh = |D|·u_j + |B|·v_j and vh = |C|·u_j + |A|·v_j (review 1 re-derived this, and an
   instrumented run of 40,413 structured pairs never saw an entry out of range). A broken
   invariant would panic, in the checked i64 conversion or in BigUint subtraction. It would
   never return a wrong value.

### 4.3 Callers

Every `Integer` gcd / lcm / extended_gcd in the workspace (Rational normalisation, polynomial
contents, HNF/SNF, CRT, the factorisation drivers). Values do not change (§4.2 point 2); they
only get faster. `Integer::mod_inverse` keeps `BigInt::modinv`.

### 4.4 Half-gcd: decided not now (Lehmer measured, crossover estimated)

At 10^6 bits Lehmer costs 1.92 s (measured), the time of about 120 full 10^6-bit
multiplications (15.7 ms each, measured; num-bigint Toom-3; num-bigint has no FFT
multiplication). A half-gcd [Sch71] (in the form of N. Möller, *On Schönhage's algorithm and
subquadratic integer gcd computation*, Math. Comp. 77 (2008) 589-607) does O(log n) levels of
such multiplications plus its own base case. So, by this estimate, below about 10^5 to 10^6
bits it cannot beat Lehmer by much with Toom-3 products. No half-gcd was prototyped, so the
crossover is an estimate, not a measurement. MAGMA's crossover is 32,768 bits with
Schönhage-Strassen multiplication (digest :14, :223); ours would be higher. Lehmer gives the
10x to 60x that matters for the sizes this workspace uses. Record "half-gcd absent, speed
only; Lehmer 1.9 s at 10^6 bits (measured); estimated crossover 10^5 to 10^6 bits" in the
tracker row; no code.

### 4.5 Tests

- `xgcd_lehmer_fixture` (`tests/ch18_signs_roots.rs`): new `tests/data/xgcd_lehmer.csv`, 24 gp
  rows `a, b, d, u, v` (gcdext(a, b) = [u, v, d]) of 65 to 3,000 bits with signs: Fibonacci
  pairs (F_1300/F_1299 up to F_6100/F_6099, negated b), b = -a, b = 7a, an unbalanced 70-bit b,
  a = 0, b = 3·2^(bits+5), common 200-bit factors. Checked through `check_gcd_row`-style
  identities. Both HEAD and Lehmer pass it (regression guard; HEAD's multipliers equal gp).
- `lehmer_gcd_agrees_with_stein_and_extended_euclid`: 400 LCG pairs up to 3,000 bits (a third
  with a common factor, some b = a, b = k·a, a = 0, random signs): Integer gcd equals
  `num_integer::Integer::gcd` on the magnitudes (Stein, an independent implementation), and
  Integer extended_gcd equals `num_integer::Integer::extended_gcd` on BigInt (HEAD's algorithm).
- `xgcd_of_large_operands`: a, b of 30,000 bits with a common 5,000-bit factor: g = x·a + y·b,
  g | a, g | b, Stein gcd of a/g and b/g is 1 (HEAD in debug: about 0.5 s; Lehmer much less).
  A correctness test, passes on HEAD.
- `gcd_and_xgcd_of_100k_bit_operands` (speed test, `within(5, ..)`): LCG operands of 100,000
  and 99,993 bits, b negated; `EuclideanDomain::gcd` and `extended_gcd` of Integer, checked by
  g | a, g | b, g = x·a + y·b and equality with Stein's gcd on the magnitudes. Measured in
  debug (`rev1/tlehmer.rs`): HEAD gcd 9.7 s and xgcd 11.3 s; Lehmer 0.14 s and 0.32 s. It
  fails on HEAD in debug (the default `cargo test` profile). In release HEAD takes about 2.1 s
  for both and passes; the bound cannot separate the two in release without making the debug
  margin thin, so the debug failure is the regression signal.
- Existing: `gcd_lcm_xgcd_inline`, `gcd_grid_fixture` (3,721 rows), `xgcd_big_fixture` (300),
  `gcd_through_euclidean_domain_trait`, `gcd_with_a_small_operand_is_one_division`.
- Unit tests in `lehmer.rs`: `lehmer_matrix` on (2^63 - 1, F_92) gives Fibonacci cofactors;
  `combine` sign cases.

---

## 5. IsPower(n, k) with the root: `Integer::exact_root`

### 5.1 Defect

`Integer::is_power(&self, k: u32) -> bool` (`integer.rs:416`) answers only the predicate;
MAGMA `IsPower(n, k)` (§18.4.5) returns `true, b` with n = b^k (digest :163). The tracker row
`ch018.IsPower.2` is "partial" for this. It is also slow for large k, because it calls
`BigUint::nth_root(k)`, whose Newton iteration starts at 2^(bits/k + 1) when the input exceeds
the f64 range: `is_power(3^100003 + 2, 9999)` takes 680 ms release, 11.35 s debug.

### 5.2 Fix

New public method (and `is_power` becomes `self.exact_root(k).is_some()`):

```rust
/// The integer b with b^k = self, when self is a perfect k-th power: the second return value
/// of MAGMA IsPower(n, k) (Handbook ch. 18 §18.4.5); PARI/GP ispower(n, k, &b).
///
/// - Some(b) with b^k = self. For self > 0 and even k, b > 0 (as PARI); for self < 0 (odd k)
///   b < 0; 0 = 0^k for k >= 1; k = 1 gives self.
/// - None when no integer b has b^k = self (self < 0 with even k included).
/// - k = 0: Some(1) for self = 1 (1 = 1^0), None otherwise. MAGMA requires k > 1 and PARI
///   refuses k <= 0; these values only extend the definition.
pub fn exact_root(&self, k: u32) -> Option<Self>
```

Algorithm, in a new crate-private module `rustmath-integers/src/perfect_power.rs` (shared with
§6): write k = 2^e·k' with k' odd. Take the k'-th root of |self| by §6.2 `exact_root_odd`
(odd part of |self| by 2-adic Newton, then fingerprints, then one exact power), then e square
roots, each preceded by the quadratic-residue sieve `may_be_qth_power(x, 2)` and checked by
s·s = x. Restore the sign. Cost: O(log k · M(bits/k')) plus the square roots; for odd k at
least bits(self) it is O(1) after the trailing-zero count.

`rustmath-rationals/src/sage_wrapper.rs:457` (`SageRational::nth_root`) switches its
`exact_root` closure to `Integer::exact_root` on |numerator| and |denominator| (same values,
the cost of §6).

### 5.3 Callers

`is_power` callers: `integer.rs` tests, `tests/ch18_signs_roots.rs` only (no other crate;
`rustmath-combinatorics/src/words.rs:195` is a different method). Values unchanged.
`exact_root` is new. The REPL has no `ispower` command.

### 5.4 Tests (new ones fail to compile on HEAD; the API is new)

- `exact_root_inline`: gp `ispower(16,4,&r)` -> r = 2; `ispower(-8,3,&r)` -> -2;
  `ispower(64,6,&r)` -> 2; `ispower(-64,6)` -> 0 (None); exact_root(0, 3) = Some(0);
  exact_root(1, 0) = Some(1); exact_root(2, 0) = None; exact_root(-1, 2) = None;
  exact_root(3^20011, 20011) = Some(3); exact_root(-3^100055, 100055) = Some(-3);
  exact_root((2^61-1)^331, 331) = Some(2^61-1); exact_root(2^(64·7)·5^7, 7) = Some(2^64·5)
  (gp `ispower(2^(64*7)*5^7, 7, &r)` -> 1, r = 2^64*5); exact_root(3^100003 + 2, k) = None for
  k in [3, 101, 9999, 100003] (gp `[ispower(n,3), ispower(n,101), ispower(n,9999),
  ispower(n,100003)]` -> [0,0,0,0]; 41 ms in a debug build for all four, HEAD's is_power(9999)
  alone takes 11 s).
- `exact_root_fixture`: new `tests/data/ispower_root.csv`, 2,486 gp rows `a, k, ispower(a,k),
  r`: a in -200..200 with k in 1..6, plus 80 rows of ±b^k (b < 2^40, k in {2,3,4,5,6,7,9,12,
  15,30,31,64}), b^k + 2, and 2^(k·j)·b^k. Checks exact_root and is_power. 40 KB.
- Existing `is_power_inline`, `is_power_grid_fixture` (6,408 rows), `integer.rs::test_is_power`.
- `rustmath-rationals` `sage_wrapper.rs::test_nth_root` stays.

---

## 6. is_perfect_power: essentially linear on huge non-powers

### 6.1 Defect

`is_perfect_power` (`integer.rs:604`) loops over primes q < bits(n). For each q it calls
`may_be_qth_power` (`:1264`), which reduces the whole of n modulo up to four primes p ≡ 1 mod q:
O(bits) word divisions per q, so about (bits / ln bits)·4·(bits/64) in total, and an exact
`BigUint::nth_root` for every q the sieve cannot refute. Measured on HEAD (release): 2^20000+1
9.0 ms, 2^100000+1 114 ms, 2^1000000+1 7.5 s (the HANDOFF quotes 12 s in debug).

### 6.2 Fix: exact q-th roots through the 2-adic root

Module `perfect_power.rs` (crate-private). Header: MAGMA ch. 18 §18.4.5 (`IsPower(n)`,
`IsPower(n, k)`); [Ber98] D. J. Bernstein, Math. Comp. 67 (1998) 1253-1283, §1 (k <= log2 n);
[BS93] Bach-Sorenson (the residue sieve); [vzGG99] von zur Gathen-Gerhard, *Modern Computer
Algebra*, ch. 9 (Newton iteration, p-adic lifting); Knuth §4.6.3 (powering).

Data and functions (from the prototype `ppow.rs`):

```rust
const FP1: u64 = (1 << 61) - 1;               // 2^61 - 1, prime
const FP2: u64 = 18_446_744_073_709_551_557;  // 2^64 - 59 = precprime(2^64)
/// n = 2^v · m with m odd, and m mod FP1, m mod FP2 (computed once per n: O(bits)).
pub(crate) struct OddPart { v: u64, m: BigUint, fp: [u64; 2] }
pub(crate) fn odd_part(n: &BigUint) -> OddPart                 // n >= 1
/// The unique odd r < 2^prec with r^k ≡ a (mod 2^prec), for odd a and odd k >= 3.
fn root_2adic(a: &BigUint, k: u64, prec: u64) -> BigUint
/// For n = 2^v·m >= 2 and odd k >= 3: Some(b) with b^k = n, or None.
pub(crate) fn exact_root_odd(o: &OddPart, k: u64) -> Option<BigUint>
/// For n >= 2 and any k >= 1 (§5.2).
pub(crate) fn exact_root_biguint(n: &BigUint, k: u32) -> Option<BigUint>
pub(crate) fn may_be_qth_power(n: &BigUint, q: u64) -> bool    // moved unchanged from integer.rs
```

`root_2adic`: Newton for y = a^(-1/k): y <- y + y·(1 - a·y^k)·k^(-1) mod 2^(2j), from y = 1
(correct mod 2). Six steps in wrapping u64 arithmetic give 64 bits (k^(-1) mod 2^64 by
z <- z(2 - kz) from z = k, five steps). Above 64 bits, k^(-1) is lifted the same way and y is
doubled in BigUint arithmetic with every product masked to the current precision
(`pow_trunc`). Then r = a·y^(k-1) mod 2^prec. A `debug_assert` checks r^k ≡ a (mod 2^prec).

`exact_root_odd(o, k)`: if k ∤ v return None; if m = 1 return 2^(v/k); K = ceil(bits(m)/k); if
K <= 1 return None; r = root_2adic(m, k, K); if bits(r) != K return None; if
(r mod FP_i)^k mod FP_i != m mod FP_i for i = 1 or 2 return None; if r^k != m return None;
return r·2^(v/k).

`Integer::is_perfect_power`: the loop of HEAD (primes q ascending while q < bits(base), skip
q = 2 for negative input, strip every exact q-th root, k *= q), with the root step replaced:
for q < 32 first `may_be_qth_power` (cheap there: a few O(bits) passes refute almost every
non-power), then q = 2 by `BigUint::sqrt` and s·s = base, odd q by `exact_root_odd` on the
cached `OddPart` of base (recomputed only when base changes, at most log2 bits times).

Correctness.
1. Uniqueness: (Z/2^K)^* has order 2^(K-1); k odd is prime to it, so x -> x^k is a bijection on
   odd residues and the odd r with r^k ≡ m is unique.
2. Size: if m = c^k with c a positive integer (c odd since m is odd), then c^k < 2^bits(m)
   gives c < 2^(bits(m)/k) <= 2^K, and 2^((bits(c)-1)k) <= m < 2^(bits(c)·k) gives
   bits(c) = ceil(bits(m)/k) = K. So c is the unique odd K-bit residue root: c = r. Hence m is a
   k-th power iff r^k = m, and every "None" above is a proof: bits(r) != K, or a fingerprint
   mismatch (r^k = m would make every residue agree), or the exact power differs.
3. Even part: n = 2^v·m with m odd is a k-th power iff k | v and m is a k-th power (unique
   factorisation), with root 2^(v/k)·c.
4. Newton: if a·y^k = 1 - e with e ≡ 0 mod 2^j, then for y' = y(1 + e/k),
   a·y'^k = (1 - e)(1 + e/k)^k = (1 - e)(1 + e + O(e^2)) = 1 + O(e^2), because the binomial
   terms C(k, i)(e/k)^i, i >= 2, have 2-adic valuation >= 2j (k is a 2-adic unit). So the
   precision doubles, also from j = 1. And r = a·y^(k-1) gives r^k = a^k·(a^(-1))^(k-1) = a.
5. The search over q < bits(base) is complete ([Ber98] §1) and stripping prime exponents finds
   the maximal k, as on HEAD; the sieve only rules exponents out (unchanged argument).

Cost: per odd prime q, O(log q · M(bits/q)) for the root (u64 arithmetic once K <= 64, which is
every q > bits/64) plus O(bits/q) for r mod FP_i; the sum over q is a small multiple of
M(bits). An input built so that r_q^q ≡ m modulo FP1·FP2 for many q pays one exact power per
such q, the same worst case as HEAD's sieve.

Measured (prototype, same machine; values equal to HEAD and gp):

| input | HEAD release | new release | HEAD debug | new debug |
|---|---|---|---|---|
| 2^20000+1 | 9.0 ms | 0.35 ms | 15 ms | 2.1 ms |
| 2^100000+1 | 114 ms | 2.4 ms | 193 ms | 11.6 ms |
| 2^1000000+1 | 7.5 s | 23 ms | about 12 s (HANDOFF) | 133 ms |
| 3^20011 | 11.4 ms | 0.87 ms | 21 ms | 6.6 ms |
| 3001^1999 | 20.6 ms | 0.36 ms | 367 ms | 3.5 ms |
| (2^61-1)^331 | 7.9 ms | 0.56 ms | 125 ms | 6.1 ms |
| 3^30030, 2^100000 | | | | 23 ms, 7.2 ms |

The integer.rs doc block of `is_perfect_power` (:563-:603) is rewritten (algorithm, proof
sketch, these costs); the module header line about the sieve is updated.

### 6.3 Callers

`prime.rs:872` (`split_odd_composite`, the factorisation drivers) and tests. Values unchanged,
faster. `may_be_qth_power` moves with its test (`power_residue_sieve_never_refutes_a_power`,
integer.rs:2095) to `perfect_power.rs`.

### 6.4 Tests

- `is_perfect_power_huge_inputs` (`tests/ch18_signs_roots.rs`, `within(5, ..)`):
  pp(2^1000000 + 1) = None (gp `ispower(2^1000000+1)` -> 0); pp(3^30030) = Some((3, 30030))
  (gp `ispower(3^30030)` -> 30030); pp(3^30030 + 2) = None; pp(2^100000) = Some((2, 100000))
  (gp). About 0.2 s in debug together after the fix. HEAD needs 13.2 s in debug
  (`rev1/tsq.rs`) and 7.5 s in release for the first input, so the test fails on HEAD in both
  profiles.
- `exact_root_odd_exhaustive_small` (unit test in perfect_power.rs): for every n in 2..=20000
  and odd k in 3..=15, `exact_root_odd` agrees with `BigUint::nth_root(k)^k == n`; `root_2adic`
  satisfies r^k ≡ a (mod 2^prec) for random odd a, k in {3, 5, 7, 63, 65, 1001} and prec in
  {1, 2, 63, 64, 65, 128, 1000}.
- Existing: `is_perfect_power_inline`, `is_perfect_power_sieve_inline`,
  `is_perfect_power_grid_fixture` (6,001 gp rows), `is_perfect_power_big_fixture` (120 rows).
  The prototype also matched HEAD on every n in -20000..=20000 and 400 random b^k(+1).

---

## 7. S1: SageInteger::binomial

### 7.1 Defect

`sage_wrapper.rs:423`: after the (5.14) transformation it runs K = min(k, N-k) steps, each a
bignum multiplication and a bignum division of the growing result: O(K·B) word operations for
a result of B bits, quadratic. Measured (release): C(10000, 5000) 3.9 ms, C(100000, 50000)
377 ms, C(524288, 262144) 10.7 s; C(10^30 + 7, 1000) 7.4 ms.

### 7.2 Fix: prime splitting of the window, product tree

New crate-private module `rustmath-integers/src/binomial.rs`. Header: MAGMA ch. 18 §18.8
`Binomial`; P. Goetgheluck, *Computing binomial coefficients*, Amer. Math. Monthly 94 (1987)
360-365 (prime-power factorisation of C(n, k)); Legendre's formula v_p(K!) = Σ_j floor(K/p^j);
balanced product trees ([vzGG99] §10.1).

```rust
/// C(n, kk) for 0 <= kk <= n (the caller passes kk = min(k, n - k)).
pub(crate) fn binomial_nonneg(n: &BigUint, kk: u64) -> BigUint
fn product_tree(v: Vec<BigUint>) -> BigUint
/// Packs word factors into products below 2^64, then product_tree.
fn product_u64(factors: impl Iterator<Item = u64>) -> BigUint
```

Algorithm: kk = 0 gives 1, kk = 1 gives n. Primes p <= kk by `fast_arith::prime_range(2,
kk + 1)`; e_p = v_p(kk!) by Legendre; n mod p for each p. The window W_i = n - i,
i = 0..kk-1, is processed in blocks of max(2^16, ceil(kk/64)) terms (u64 terms when n < 2^64,
BigUint terms otherwise): for each p, the indices with p | W_i are i ≡ n (mod p); divide p out
of those terms completely and count the removals c_p. Multiply each block (product_u64 or
product_tree) and keep the block products. Finally multiply the block products and
Π p^(c_p - e_p) with a product tree.

`SageInteger::binomial` keeps its sign/zero cases (k < 0, 0 <= n < k, n < 0 by (5.14)) and
then refuses results that are certainly beyond 2^32 bits, before any allocation:

```rust
/// Largest result size SageInteger::binomial computes, in bits (512 MiB). The same order as
/// the u32 exponent of Integer::pow and the 2^31-bit limit of Integer::sigma (§8.2).
const BINOMIAL_MAX_BITS: u128 = 1 << 32;
// n_big = N >= 0 after (5.14); kmin = K = min(k, N - k) >= 2 here (K = 0, 1 return early).
let kk = match u64::try_from(&kmin) {
    Ok(v) if u128::from(v) <= BINOMIAL_MAX_BITS => v,
    _ => panic!("binomial: C(n, k) has more than 2^32 bits; not computed"),
};
let kbits = u64::from(64 - kk.leading_zeros());                    // bits(K)
let per_factor = n_big.bit_length().saturating_sub(1 + kbits).max(1); // <= log2(N/K) (N/K >= 2)
if u128::from(kk) * u128::from(per_factor) > BINOMIAL_MAX_BITS {
    panic!("binomial: C(n, k) has more than 2^32 bits; not computed");
}
binomial_nonneg(n_big.as_bigint().magnitude(), kk)
```

Why the bound is a proof. K = min(k, N - k) <= N/2. C(N, K) = Π_{i<K} (N - i)/(K - i), and
each factor is >= N/K >= 2, so C(N, K) >= (N/K)^K >= 2^K. Also N >= 2^(bits(N)-1) and
K < 2^bits(K), so log2(N/K) > bits(N) - 1 - bits(K). Hence log2 C(N, K) >=
K·max(1, bits(N) - 1 - bits(K)) = LB, and the result has more than LB bits. The code refuses
only when LB > 2^32 (or K itself exceeds 2^32, or does not fit u64), so it never refuses a
result of at most 2^32 bits. It can still compute somewhat larger results: C(N, K) <=
(eN/K)^K and log2(N/K) < bits(N) - 1 - bits(K) + 2, so LB undercounts by less than 3.5·K
bits, and an admitted result has fewer than about 4.5·2^32 bits (worst case: N/K just
below 8 with K just above a power of 2; central coefficients have about 2K = 2·LB bits). Documented as `# Panics` on `SageInteger::binomial`. HEAD loops
instead, and the uncapped prime sieve of the first draft (K bytes) would have made the
allocator abort for 2^40 <= K < 2^64 (review 1, m10). With the cap the sieve takes at most
4 GiB.

Correctness. C(N, K) = Π W_i / K!. Split each W_i = s_i·t_i where s_i is its K-smooth part
(primes <= K) and t_i has no prime factor <= K. Then Π W_i = S·T with S = Π s_i, T = Π t_i.
K! is K-smooth and divides Π W_i (C(N, K) is an integer), and gcd(T, K!) = 1, so K! | S and
C(N, K) = (S / K!)·T with S/K! = Π_{p <= K} p^(v_p(S) - v_p(K!)). The algorithm computes
v_p(S) = c_p by counting the removed factors, so every exponent is >= 0 and the product is
exact. No division of large numbers happens.

Cost: about K·ln ln K word divisions for the removal, plus O(M(B) log K) for the product tree.
Memory: the primes below K (`prime_range` uses K bytes, at most 4 GiB under the 2^32 cap),
one block, and the block products (about the size of the result). Measured (prototype
`binom.rs`, release; the debug build of the same prototype: C(524288, 262144) 155 ms,
C(10^6, 500000) 386 ms, C(4194304, 2097152) 3.1 s):

| C(n, k) | HEAD | new | bits |
|---|---|---|---|
| C(10000, 5000) | 3.9 ms | 0.17 ms | 9,994 |
| C(100000, 50000) | 377 ms | 2.2 ms | 99,992 |
| C(524288, 262144) | 10.7 s | 16 ms | 524,279 |
| C(10^6, 500000) | not run | 79 ms | 999,990 |
| C(4194304, 2097152) | not run | 426 ms | 4,194,293 |
| C(10^8, 1000) | not run | 0.14 ms | 18,047 |
| C(10^30+7, 1000) | 7.4 ms | 0.98 ms | 91,129 |
| C(10^30+7, 100000) | not run | 629 ms | 8,449,081 |

### 7.3 Callers and behaviour changes

Values unchanged (prototype equal to HEAD on n in -60..60, k in -5..70, several large cases,
and to gp on 80 large rows). Callers: `rustmath-jupyter/src/repl.rs:10652` (`checked_binomial`)
and tests. Cross-unit note for the REPL units (J1a/J1b): `repl_limits::BINOMIAL_WORK`
(repl.rs:10480-10488) and the `checked_binomial` doc (repl.rs:10605-10626) model and measure
the old K-step loop ("C(524288, 262143) takes 10.5 s"); after I3 that call takes about 16 ms
release. The limits stay safe (they only over-estimate), but the measured-time sentences become
stale; recalibrating them is J1's. The REPL caps results at `BINOMIAL_BITS` = 2^22 bits
(repl.rs:10481) before it calls `SageInteger::binomial`, so it never reaches the new 2^32-bit
panic. No other caller exists (grep of `Integer::binomial`, `SageInteger::binomial` and
`.binomial(` over the workspace and rustmath-py). `rustmath-combinatorics` u32 `binomial`
(C1, HANDOFF S2/S4) could call `SageInteger::binomial`.

### 7.4 Tests

- `binomial_big_fixture`: new `tests/data/binom_big.csv`, 80 gp rows `n, k, binomial(n,k) %
  (2^127-1), #binary(|binomial(n,k)|)`: 40 random n < 10^6 with random k, 10 central ones,
  10 with n = 10^30 + 7 + i (k < 2002), 6 with n = 2^64 - 59 + i (k < 5002, the BigUint
  window), 8 with n < 0 (k < 3002), (524288, 262144), (524288, 262143), (10^6, 500000),
  (5, -3), (-5, -3), (100, 101). Run as `within(60, ..)`. After the fix the whole fixture
  takes 5.1 s in debug, worst row (10^6, 500000) 0.39 s (`rev1/vbinom.rs`, the prototype
  against `gp/binom_big.csv`). HEAD needs minutes in debug, so it fails there. In release HEAD
  needs about 60 s for the three central rows alone (10.7 s, 10.7 s and about 40 s by the
  quadratic cost), which is at the bound. The test that fails on HEAD in release is the next
  one.
- `binomial_central_rows_within_10s` (the decisive speed test, `within(10, ..)`):
  C(524288, 262144), C(524288, 262143) and C(10^6, 500000) against their fixture rows. After
  the fix 0.7 s in debug. HEAD takes 10.7 s in release for the first row alone, so the test
  fails on HEAD in both profiles.
- `binomial_beyond_2_pow_32_bits_is_refused` (`within(10, ..)`): for (n, k) in
  (2^64, 2^40) (LB = 2^40·23), (10^30, 10^9) (LB = 10^9·69 = 6.9·10^10) and (2^70, 2^68)
  (K = 2^68 does not fit the cap), `std::panic::catch_unwind(|| Integer::binomial(&n, &k))`
  is `Err` and the payload contains "more than 2^32 bits". HEAD loops on all three, so the
  test fails there by the time bound. Boundary guard: C(2^33, 3) (LB = 3·(34 - 1 - 2) = 93) is
  computed and equals 105637549982125628646836142080 (gp `binomial(2^33,3)`).
- `binomial_agrees_with_the_product_formula`: a test-local copy of HEAD's loop as the reference
  for n in -40..=300, k in -3..=n+2.
- `binomial_1000_500`: C(1000, 500) mod 2^127 - 1 = 62526922371930188635637843119963540756
  (gp `binomial(1000,500) % (2^127-1)`).
- Existing: `binomial_negative_upper_index_inline`, `binomial_fixture` (2,196 rows),
  `binomial_identities`, `sage_wrapper.rs::test_binomial_*` (incl. `test_binomial_performance`).
- Unit tests in `binomial.rs`: product_tree of [] is 1; `binomial_nonneg` on the u64 and BigUint
  window paths at the 2^64 boundary (n = 2^64 - 1 and 2^64, K = 1000) against the reference.

---

## 8. S5: Integer::sigma exponent overflow (integer.rs:398)

### 8.1 Defect

`prime.pow((multiplicity + 1) * k)` multiplies two u32. (a+1)·k > u32::MAX panics in debug and
wraps in release, which gives a wrong value (`t4_sigma.rs`):

| call | debug | release | truth |
|---|---|---|---|
| `Integer::from(2).sigma(1 << 31)` | panic | Ok(0) | 1 + 2^(2^31) |
| `Integer::from(8).sigma(1 << 30)` | panic | Ok(0) | |
| `Integer::from(1024).sigma(400_000_000)` | panic | Ok(0) | |

`num_divisors` (:341) and the k = 1 branch (:390) form `multiplicity + 1` in u32 too (overflow
needs a > 2^32 - 2, unreachable in memory, but free to make safe).

The order matters too (review 1, m1). In the general branch HEAD forms `prime.pow(k)` (:397)
before the overflowing product (:398). So replacing only :398 would still build p^k first:
`Integer::from(2).sigma(1 << 31)` would form 2^(2^31) (256 MiB, 0.56 s in debug) before the
`Err`, and `Integer::from(3).sigma(1 << 31)` would form 3^(2^31) (about 3.4·10^9 bits) and
not return in any reasonable time.

### 8.2 Fix

Check every exponent first, then form powers. After `factor(&n)` and before the product loop:

```rust
// Every exponent (a + 1)·k is checked before any power is formed: a power formed first can
// take minutes (3^(2^31)) and then the call would fail anyway.
let mut exps: Vec<u32> = Vec::with_capacity(factors.len());
for (_, a) in &factors {
    let e = (u64::from(*a) + 1) * u64::from(k);
    let e = u32::try_from(e).map_err(|_| MathError::NumericalError(format!(
        "sigma: the exponent (a + 1)·k = {e} of p^((a+1)k) exceeds u32::MAX; \
         sigma_k(n) would have more than 2^31 bits")))?;
    exps.push(e);
}
for ((prime, _), e) in factors.iter().zip(exps) {
    // numerator p^e - 1; denominator p - 1 for k = 1, else p^k - 1 (k < e, so p^k is
    // formed only after the check above)
}
```

The k = 1 branch uses the same checked exponent (e = a + 1). `num_divisors` uses
`Integer::from(u64::from(multiplicity) + 1)`. Doc: `# Errors` gains this case. No power is
formed before the last exponent has been checked, so every `Err` is returned after O(number
of prime factors) work beyond the factorisation.

When every exponent fits u32 the value is computed as on HEAD, even if it is huge (for
example sigma(6, 2^31 - 1), with e = 2^32 - 2): that is a legal input, as for `Integer::pow`.
The REPL bounds such calls first (`SIGMA_BITS`, §8.3).

Why Err and not a u64 power: (a+1)·k > 2^32 with a >= 1 forces a·k >= 2^31, so the value has
more than 2^31 bits (256 MiB); `Integer::pow` takes a u32 exponent everywhere in this crate.
MAGMA `DivisorSigma(i, n)` is specified for "small non-negative integer i" (§18.7).

### 8.3 Callers

`sum_divisors` (sigma(1)), `rustmath-jupyter/src/repl.rs:2629` (the REPL guards sizes first,
`SIGMA_BITS`, so it never reaches the new Err), `rustmath-py/src/integers.rs:173`.

### 8.4 Tests

- `sigma_exponent_overflow_is_an_error` (`tests/ch18_signs_roots.rs`, `within(10, ..)`, the
  helper of §3.4): `Integer::from(2).sigma(1 << 31)`, `Integer::from(8).sigma(1 << 30)`,
  `Integer::from(1024).sigma(400_000_000)` and `Integer::from(3).sigma(1 << 31)` all return
  `Err(NumericalError(_))`, instantly after the fix. On HEAD the first three panic in debug
  (and return `Ok(0)` in release), and the fourth does not return, so the test fails on HEAD
  in both profiles.
- `test_sigma_values_pari` (`integer.rs` tests): sigma(2^10·3^5, 3) = 18285256324908244 (gp `sigma(2^10*3^5,3)`);
  sigma(7^40, 5) mod 2^127 - 1 = 108698265444746923986228897305478117662 and 562 bits (gp
  `sigma(7^40,5) % (2^127-1)`, `#binary(..)`).

---

## 9. S7: the panicking inherent Integer::to_i64 (integer.rs:88)

### 9.1 Facts

`Integer::to_i64(&self) -> i64` panics when the value does not fit. Inherent methods win
method resolution, so `x.to_i64()` on an Integer calls it even where `NumericConversion` (whose
`to_i64` returns `Option<i64>`) is in scope, while `x.to_u64()`, `x.to_usize()` (trait) and the
inherent `x.to_f64()` return Options.

The workspace has 166 `.to_i64()` calls (`grep -rn "\.to_i64()" --include=*.rs .`). The first
draft listed 138 of them as not handling an Option, but its grep dropped every line containing
`Some(`, and that hid 7 Integer sites (review 1, B1): `quadratic_sieve.rs:93`, `:112`, `:145`,
`rational.rs:530`, `substitute.rs:158`, `integrate.rs:681`, `solve.rs:1749`. The other lines
with `Some(` are Option-typed (MPFR, BigFloat and ComplexMPFR asserts, `integer_range.rs:274`,
`ratfunc.rs:290`, `:293`). So 145 lines do not handle an Option. Eight of them are not
Integer's (`integer.rs:1590` and `bigfloat.rs:1099` are BigInt, `mpc.rs:699` is rug,
`bigfloat.rs:1208`, `:1216` and `mpc.rs:927` are BigFloat / ComplexMPFR test asserts on `None`,
`ratfunc.rs:280` is inside an Option chain, `s_unit_solver.rs:232` is an orphan file). That
leaves 137 call sites in 24 crates (23 in the workspace and the excluded rustmath-py) that use
the panicking method. Fully qualified calls (`<Integer as NumericConversion>::to_i64(p)` in
`ellipticcurves/src/curve.rs:357`, `:415`, `NumericConversion::to_i64(x)` in
`combinatorics/src/integer_vectors.rs:97`) already return Option and are unaffected. No doc
test or Markdown example calls the method. The compiler is the final check: after step 11 any
missed site fails to build.

Evidence that the panic is a footgun, not a contract: several callers were written for an
Option or for saturation and panic instead:
`rustmath-rationals/src/rational.rs:530` (`impl NumericConversion for Rational`: the trait
promises `None`, but the integral rationals 2^70, 2^63 and -2^63-1 panic at HEAD,
`rev1/trat2.rs`; every generic `R: NumericConversion` caller with R = Rational inherits it),
`rustmath-symbolic/src/substitute.rs:158` (`eval_float -> Option<f64>` panics for
`Expr::Integer(10^30)`, `rev1/sym.rs`),
`rustmath-polynomials/src/factorization.rs:702` (error text "requires a prime p >= 2 that fits
in i64", but `p.to_i64()` panics first), `rustmath-numbertheory/src/quadratic_forms.rs:479`
(comment "modulus too large or invalid"), `rustmath-rings/src/padics/unramified.rs:89`, `:212`
(re-check `Integer::from(v) == prime` after the panicking call; `unramified_modulus(2^89-1, 2)`
panics at HEAD, `rev1/unr.rs`), `rustmath-powerseries/src/algebraic.rs:243`
(`i128_of -> Option`), `rustmath-liealgebras/src/chevalley_basis.rs:276` ("otherwise use 0 as
fallback"), `rustmath-quadraticforms/src/local_densities.rs:231` (`.min(1000)`),
`rustmath-symbolic/src/series.rs:34` (HANDOFF S7 "the expansion point": `x^2` expanded at
10^30 panics). The REPL already had to wrap its own calls (repl.rs:11999 "Each of these
panicked before (inherent Integer::to_i64), which killed the kernel").

### 9.2 Decision: checked API

- `pub fn to_i64(&self) -> Option<i64>` (same as `NumericConversion::to_i64`, `BigInt::to_i64`,
  `Integer::to_f64`). Doc: "None when the value is outside i64".
- `impl TryFrom<&Integer>` and `impl TryFrom<Integer>` for i8, i16, i32, i64, i128, isize, u8,
  u16, u32, u64, u128, usize (one macro, delegating to num-bigint's `TryFrom<&BigInt>`), with
  `type Error = MathError` and `MathError::NumericalError(format!("an integer of {bits} bits{} does not fit in {ty}", if negative {" (negative)"} else {""}))`
  (no decimal formatting of a huge value). `?` then works in `Result<_, MathError>` code. No
  `From<Integer>` impl for a primitive exists, so the blanket `TryFrom` impl cannot conflict
  (grep `impl From<Integer> for`, `impl TryFrom<.*Integer`).
- A documented panic was rejected: it leaves the call sites above panicking on user input, and
  the per-site intent is invisible in `x.to_i64()`.

### 9.3 Migration (all 137 Integer sites; full table in Appendix A)

Rules, so that no site turns "does not fit" into a value-bearing answer:

- G (50 sites, guarded: a residue modulo a word modulus, a checked bit length, |k| <= 4096,
  etc.): `i64::try_from(&x).expect("<the invariant>")`.
- T (23 test sites): `i64::try_from(&x).unwrap()`.
- R-fix (50 sites): an exact replacement that cannot invent a value: comparisons with a small
  constant become Integer comparisons (`i.is_one()`, `*i == Integer::from(c)`), sign tests
  `x.signum()`, parity `x.is_even()`, the symbolic special-value tables
  `if let Ok(v) = i64::try_from(x)` so that large arguments fall through to the existing
  unevaluated form, existing error channels (`Err`, `?`) where the function already reports
  "too large", `u32::try_from` instead of `as u32` in `bsdratio.rs` (which truncated silently),
  and `.ok()` where the function already returns `Option` with the meaning "cannot handle".
- R-keep (12 sites): where `None` would flow into a channel that means something else (for
  example "not a polynomial" turned into degree 0 by a caller, "no roots", "count 0"), keep
  today's panic, made explicit with `i64::try_from(&x).expect("<function>: ...")`. These are
  flagged as follow-ups (§11).
- D (2 sites, new in revision 1): dead code, deleted (`solve.rs:1722`, `:1749`, below).
- N (8 sites): not Integer's method; unchanged.

Step-10 forms (review 1, m3). No replacement in step 10 calls the inherent `to_i64`. Each one
uses the TryFrom impls of step 9 (`.ok()` where an Option is wanted, `.expect(..)` for G and
R-keep, `.unwrap()` for T, `.map_err(..)?` for an existing error channel) or an Integer method
(`is_zero`, `is_one`, `== Integer::from(c)`, `signum`, `is_even`, `to_f64`). These compile
both before and after step 11. So steps 10 and 11 stay separate commits, each build-green.

The seven sites added in revision 1, with the reasoning:

- `quadratic_sieve.rs:93`, `:112`, `:145` (G, step 9): inside `tonelli_shanks(n, p: u64)`, which
  asserts p < 2^63 prime at entry (:86-89, `fn tonelli_shanks` at :85). The values are residues in [0, p), so they fit:
  `i64::try_from(&r).expect("tonelli_shanks: a residue mod p < 2^63 fits in i64")`.
- `rational.rs:530` (R-fix): `Some(self.numerator.to_i64())` becomes
  `i64::try_from(&self.numerator).ok()`. Exact, and it is the trait's own contract ("None when
  the value does not fit").
- `substitute.rs:158` (R-fix): `Expr::Integer(n) => Some(n.to_i64() as f64)` becomes
  `Expr::Integer(n) => n.to_f64().filter(|v| v.is_finite())`. `Integer::to_f64` (integer.rs:193)
  delegates to num-bigint, which returns `Some(f64::INFINITY)` above f64::MAX (num-bigint 0.4.8
  `biguint/convert.rs:375`). The filter keeps `eval_float`'s `None` = "cannot evaluate" for an
  integer outside the f64 range. Below 2^63 in absolute value the result equals HEAD's
  `n.to_i64() as f64` (both round to nearest; `rev1/sym.rs` checks 10^30 -> 1e30 and
  -2^63 -> -9.223372036854776e18). Callers: REPL plotting (`repl.rs:3031`, `:3093`, `:3123`,
  `:3154`, `unwrap_or(f64::NAN)`, a non-plotted point), `rustmath-manifolds/src/riemannian.rs:678`
  (`ok_or_else` into an error), and symbolic internals that propagate `None`. A huge integer
  now gives `None` instead of a panic.
- `integrate.rs:681` (R-fix, with `:697`): in `advanced::polynomial_degree`,
  `Some(n.to_i64())` becomes `i64::try_from(n).ok()`, and the Mul branch
  `Some(deg_left + deg_right)` becomes `deg_left.checked_add(deg_right)` (at HEAD
  `polynomial_degree(x^(2^62)·x^(2^62), x)` panics with "attempt to add with overflow" in
  debug and wraps in release, `rev1/pdeg.rs`). Why R-fix and not R-keep: every caller only
  propagates the `None` as "this tier declines". The recursion (:690-697) uses `?`;
  `partial_fractions` (:722-723) and `integrate_rational` (:899-900) use `?` and return `None`;
  `integrate_rational` is reached from `Expr::integrate` (:155), whose documented `None` is
  "Refused (`None`)" (integrate.rs:58-61; the budget text at :43-45 calls the final `None` the honest "not integrated"), an unresolved answer, never a decision. No
  caller turns the `None` into a value. That is the difference from `plane_curve_ops.rs:90`,
  which stays R-keep because its caller does `self.degree().unwrap_or(0)`
  (plane_curve_ops.rs:373). Precedent: rustmath-calculus's own `polynomial_degree` already
  returns `None` for an exponent outside i64 (`expr.rs:233-234`, `n_str.parse().ok()?`). The
  doc of `polynomial_degree` changes to "None: not recognised as a polynomial in var, or the
  degree does not fit in i64".
- `solve.rs:1749` and `solve.rs:1722` (D): `expr_to_integer` (:1747) and `expr_to_rational`
  (:1720) are never called; `expr_to_rational` only calls itself and `expr_to_integer`.
  `cargo check -p rustmath-symbolic` at HEAD warns "function `expr_to_rational` is never
  used" and "function `expr_to_integer` is never used". Both functions are deleted. No
  behaviour changes; two sites and two warnings go away. (The S1 symbolic design does not use
  them.)

One earlier R-fix row was corrected while re-reading solve.rs: `solve.rs:1133`, `:1137`
(`solve_constant_inequality`) use the value only for its sign against 0 (:1142-1148). The
first draft replaced them by `to_f64`, which gives NaN for a rational with huge numerator and
denominator (both convert to infinity) and so a false `Empty`. They now use the exact sign:
`Expr::Integer(n) => n.signum()`, `Expr::Rational(r) => r.numerator().signum()` (the
denominator is kept positive, rational.rs:507), then `LessThan => sign < 0`, and so on.

Order, each commit build-green: (1) rustmath-integers adds the TryFrom impls and migrates its
own sites to `try_from` (quadratic_sieve.rs:50, :60, :93, :96, :112, :145 and integer.rs:457,
G; tests/ch18_signs_roots.rs:550, T); (2) one commit per downstream crate, in this order:
rationals (first, since most crates depend on it), polynomials, numberfields (with the
class-number fix of §11), symbolic, ellipticcurves, modular, numbertheory, powerseries,
crypto, padics, curves, algebraic, liealgebras, galois, groups, combinatorics, matrix,
quadraticforms, rings, functionfields, geometry, finitefields tests, rustmath-py (22 workspace
crates plus rustmath-py); (3) rustmath-integers changes the inherent signature, when nothing
calls the old one. `rustmath-py` is outside the workspace (`Cargo.toml:5`), so its two sites
are edited without a compile check (state it in the commit).

Cross-unit note: the series.rs:34/:206/:233 edits fix the S7 sub-item "the expansion point in
series.rs:34" (owned by the symbolic unit); the parser sub-item (parser.rs:148-154) is not
touched here.

### 9.4 Tests

- `integer.rs::test_to_i64_is_checked`: to_i64 of i64::MIN, i64::MAX are Some; of
  i64::MAX + 1 and i64::MIN - 1 are None (gp `2^63` does not fit).
- `integer.rs::test_try_from_integer`: i64::try_from(&-2^63) = Ok(i64::MIN); u64::try_from(&-1)
  is Err(NumericalError); u128::try_from(&2^127) = Ok; usize, i8 boundaries; the error message
  names the bit length.
- `rustmath-rationals`, `rational.rs` tests, `numeric_conversion_to_i64_is_checked`:
  `NumericConversion::to_i64` of the integral rationals 2^70, 2^63 and -2^63-1 is `None` (each
  panics on HEAD, `rev1/trat2.rs`); of -2^63 it is `Some(i64::MIN)` and of -6/3 `Some(-2)`
  (boundary guards, they pass on HEAD); of 1/2 it is `None`.
- `rustmath-symbolic`, `substitute.rs` tests, `eval_float_of_huge_integers`:
  `Expr::Integer(10^30).eval_float()` = `Some(1e30)` (panics on HEAD);
  `Expr::Integer(10^400).eval_float()` = `None` (panics on HEAD); `Expr::Integer(-2^63)` gives
  `Some(-9223372036854775808.0)` (passes on HEAD).
- `rustmath-symbolic`, `integrate.rs` tests, `polynomial_degree_beyond_i64_is_none`:
  `polynomial_degree(x^(10^30), x)` = `None` (HEAD panics at :681);
  `polynomial_degree(x^(2^62)·x^(2^62), x)` = `None` (HEAD panics at :697 in debug);
  `polynomial_degree(x^(2^62)·x, x)` = `Some(2^62 + 1)`.
- `rustmath-symbolic`, `solve.rs` tests, `constant_inequality_with_huge_values`: 0 < 10^30 is
  `All` and 10^30 < 0 is `Empty` (HEAD panics at :1133); with r = -10^400/(10^400 + 1),
  `Inequality::new(Expr::Rational(r), LessThan, Expr::from(0))` is `All` (HEAD panics at
  :1137; the first draft's `to_f64` would have given `Empty`). The implementer checks that the
  standard form reaches `solve_constant_inequality` as a bare Integer / Rational, as the
  existing `test_solve_constant_inequality_true` does for 0 < 1.
- Further R-fix tests in `rustmath-symbolic`: `x^2` taylor at 10^30 does not panic (`point`
  is a non-zero Integer, the first term is 10^60); `chebyshev_t(10^30, x)` stays the
  unevaluated function; `prime_pi(-10^30)` = 0.
- `rustmath-polynomials`: `hensel_lift` with p = 2^89 - 1 returns `Err(InvalidArgument)`.
- `rustmath-numberfields`, `lib.rs` tests, `class_number_quadratic_is_honest`: x^2 + 5 gives
  `Err(NotImplemented)` (HEAD `Ok(1)`; gp `poldisc(x^2+5)` = -20, `quadclassunit(-20).no` =
  2); x^2 + (10^30 + 3) gives `Err(NotImplemented)` (HEAD panics at :419); x^2 + x + 1 gives
  `Ok(1)` (gp `poldisc(x^2+x+1)` = -3, `quadclassunit(-3).no` = 1); x^2 + 1 gives `Ok(1)` (gp
  `quadclassunit(-4).no` = 1). `test_class_number_quadratic` (lib.rs:787) moves from x^2 + 3
  (polynomial discriminant -12, which passed only through the placeholder) to x^2 + x + 1.
- `rustmath-rings`, `unramified.rs` tests, `unramified_modulus_above_2_pow_63`:
  `unramified_modulus(2^89 - 1, 2)` = `Ok(x^2 + 1)`, coefficients [1, 0, 1] (HEAD panics at
  :89, `rev1/unr.rs`; gp `isprime(2^89-1)` = 1, `(2^89-1) % 4` = 3,
  `polisirreducible(Mod(1,2^89-1)*(x^2+1))` = 1 and x^2 at counter 0 is reducible).
  `UnramifiedExtension::with_modulus(2^89 - 1, &(x^2 + 1), 3)` is `Ok` with `degree()` = 2
  and `is_conway()` = false (HEAD panics at :212 in `tag_conway`, `rev1/unr2.rs`). Cost: at
  HEAD the same fallback path for 2^61 - 1 takes 9 ms (`unramified_modulus`) and 7 ms
  (`with_modulus`) in debug; 2^89 - 1 runs the same quadratic irreducibility test and Frobenius
  image on a two-word prime, so the test stays in the tens of milliseconds.
- `rustmath-ellipticcurves`: no test can reach a Sha above 2^32, so the `bsdratio.rs` change
  is covered by review only.

Each listed test that is not marked as a guard fails (panics, or returns the wrong value) on
HEAD.

---

## 10. MAGMA convention: Binomial(n, r) for r < 0 (documentation)

What the code does: k < 0 gives 0 for every n (`sage_wrapper.rs:424`, `integer.rs:1012` for
`binomial_mod`). Cross-checks:
- Sage `binomial(x, m)` (doc.sagemath.org, sage.arith.misc.binomial): "If m < 0, return 0";
  examples `binomial(3, -1)` = 0, `binomial(-5, -2)` = 0, `binomial(-2, 5)` = -6.
- gp `[binomial(5,-1), binomial(-1,-1), binomial(-5,-2), binomial(-2,5), binomial(3,-1),
  binomial(-3,-5), binomial(0,-1)]` -> [0, 0, 0, -6, 0, 0, 0].
- GAP `[Binomial(5,-1), Binomial(-1,-1), Binomial(-5,-2), Binomial(-2,5), Binomial(3,-1),
  Binomial(-3,-5), Binomial(0,-1)]` -> the same list.
- Mathematica's symmetric convention differs (`Binomial[-1, -1]` = 1).

Why: the three systems that can be run agree, and the value is the coefficient of x^k in
(1 + x)^n, which has no negative powers. MAGMA's own value stays unconfirmed (no binary, the
handbook gives no range). Doc fix at `sage_wrapper.rs:244-249`: add the Sage sentence and
examples, keep "MAGMA unconfirmed". Test line in `binomial_negative_upper_index_inline`:
binomial(3, -1) = 0, binomial(-5, -2) = 0, binomial(-2, 5) = -6 (Sage doc, gp, GAP). HANDOFF:
move this item from "unverified" to "cross-checked with PARI, GAP and Sage; MAGMA unconfirmed".

---

## 11. Found while designing (tracker notes or other units, except where marked fixed)

- `Integer::nth_root(n)` (MAGMA `Iroot`) is slow for large n on large inputs: num-bigint's
  Newton starts at 2^(bits/n + 1) once the input leaves the f64 range (3^100003 + 2, n = 9999:
  680 ms release, 11.35 s debug). A start from the leading bits would fix it. New tracker
  note on the nth_root row (integer.rs:253).
- Facades reachable only through panics today (R-keep sites): `rustmath-powerseries/src/
  algebraic.rs:207` turns a coefficient beyond i128 into an empty root list;
  `rustmath-algebraic/src/radicals.rs:39` returns alpha itself as its n-th root;
  `rustmath-numbertheory/src/quadratic_forms.rs:480` counts 0 solutions for m <= 0;
  `rustmath-quadraticforms/src/local_densities.rs:231` counts only the first 1000 residues.
  The chevalley_basis.rs:275 comment promises a 0 fallback (tracker row 1625 already says so).
- Fixed here (review 1, m6), in the step-10 numberfields commit:
  `rustmath-numberfields/src/lib.rs:429-432` returned class number 1 for every quadratic
  polynomial discriminant with |d| < 100 (a placeholder: x^2 + 5 has discriminant -20 and
  h = 2, x^2 + 15 has -60 and h = 2; gp `quadclassunit(-20).no` -> 2,
  `quadclassunit(-15).no` -> 2). The branch now returns `Err(NotImplemented)`. The list
  check at :419 stays correct: the field is Q(sqrt(d)) for the polynomial discriminant d, and
  each of the nine values -3, -4, -7, -8, -11, -19, -43, -67, -163 is itself a fundamental
  discriminant, so membership means the field discriminant is d and h = 1. Remaining gap, for HANDOFF and tracker row 1799: every other quadratic
  field gets `Err(NotImplemented)`, x^2 + 3 (h = 1) included. An exact algorithm for imaginary
  quadratic fields is the count of reduced primitive forms of the fundamental discriminant
  (H. Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 5.3.5); real
  quadratic fields need the regulator (Cohen §5.7). `classgroup::class_group`
  (classgroup.rs:307) is a relation-based computation; whether its output is certified must be
  checked before `NumberField::class_number` delegates to it.
- `rustmath-symbolic/src/integrate.rs:674` `polynomial_degree` returns `Some(-1)` for x^(-1),
  a degree for a non-polynomial (pre-existing, heuristic tier; tracker note).
- REPL binomial work model (§7.3) and combinatorics binomial (C1).
- Half-gcd (§4.4).

---

## Plan

File-by-file, in this order (one commit per numbered step; workspace build-green after each).

1. `rustmath-core/src/traits.rs`: `gcd_u64`, `xgcd_i128`; i32 and i64 `norm`, `div_rem`, `gcd`,
   `lcm`, `extended_gcd` (§1.2); tests (§1.4). Run `cargo test -p rustmath-core`,
   `cargo test -p rustmath-polynomials --test ch23_quotrem_gcd`.
2. `rustmath-integers/src/fast_arith.rs`: §2.2; update its tests and
   `tests/ch18_residues.rs:767`; new tests (§2.4); rename
   `fast_arith_miller_rabin_after_pow_mod_change` to `fast_arith_llong_is_prime_mersenne_61`
   with the new comment (§2.4).
3. `rustmath-integers/src/integer.rs` sqrt, `prime.rs` isqrt removal (§3.2); fixture
   `tests/data/isqrt_big.csv` and tests (§3.4). Copy the `within` helper into
   `tests/ch18_signs_roots.rs` (§3.4). Add `tests/data/gen_i3.gp` (copy of the
   scratch `gen_i3.gp`, header in the style of `gen_signs_roots.gp`) and commit all four CSVs
   it produces (`isqrt_big.csv`, `ispower_root.csv`, `xgcd_lehmer.csv`, `binom_big.csv`; 109 KB
   together; run it in an empty directory, it needs `default(parisizemax, 2000000000)`).
4. `rustmath-integers/src/lehmer.rs` (new, checked i64 conversion of the matrix entries),
   `lib.rs` (`mod lehmer;`), integer.rs gcd / extended_gcd overrides and docs (§4.2); tests
   (§4.5, including the 5 s speed test).
5. `rustmath-integers/src/perfect_power.rs` (new, `mod perfect_power;` in lib.rs; moves
   `may_be_qth_power`, `POWER_SIEVE_*` and its test), integer.rs `exact_root`, `is_power`,
   `is_perfect_power`, their docs and the module header (§5.2, §6.2); tests (§5.4, §6.4).
6. `rustmath-rationals/src/sage_wrapper.rs:457`: `nth_root` via `Integer::exact_root`.
7. `rustmath-integers/src/binomial.rs` (new, `mod binomial;` in lib.rs), `sage_wrapper.rs`
   binomial body (with the 2^32-bit refusal of §7.2) and doc (cost, `# Panics`, the Sage
   sentence of §10); tests (§7.4, §10).
8. integer.rs sigma (exponent pre-pass) / num_divisors (§8.2); tests (§8.4).
9. integer.rs TryFrom impls + rustmath-integers internal to_i64 sites (§9.3 step 1:
   quadratic_sieve.rs:50, :60, :93, :96, :112, :145, integer.rs:457, and the test site
   tests/ch18_signs_roots.rs:550); tests (§9.4).
10. One commit per downstream crate for the to_i64 sites (Appendix A), in the §9.3 order:
    rustmath-rationals first (rational.rs:530 and its test), then polynomials, numberfields
    (also the class-number placeholder of §11, with `class_number_quadratic_is_honest` and the
    moved `test_class_number_quadratic`), symbolic (also the deletion of the dead
    `expr_to_rational` / `expr_to_integer`, the `polynomial_degree` doc and `checked_add`),
    ..., rings (tests with 2^89 - 1), ..., rustmath-py. Every replacement is in the step-10
    form of §9.3 (no call of the inherent `to_i64`), so each commit is build-green. Each
    commit carries its crate's R-fix tests (§9.4).
11. integer.rs: `to_i64 -> Option<i64>`; `cargo check --workspace --tests`.
12. Docs: PROJECT_TRACKER rows 355 (core i32/i64 note), 400 (sigma), 417 (gcd: Lehmer
    measured; half-gcd absent, "estimated crossover 10^5 to 10^6 bits", §4.4), 418 (IsPower:
    2-adic roots, exact_root), 421 (fast_arith; remove "Open:"), 424 (Isqrt), 427/428
    (binomial algorithm, 2^32-bit refusal, Sage cross-check), 1625 (comment fixed), 1799
    (class_number_quadratic: placeholder removed, the remaining gap of §11), 2144
    (`eval_float` of a huge Integer: fixed);
    `docs/magma/status_overrides.csv` rows ch018.IsPower.2 (partial -> verified with
    exact_root), ch018.IsPower, ch018.Isqrt, ch018.GreatestCommonDivisor,
    ch018.ExtendedGreatestCommonDivisor, ch018.Binomial, ch018.DivisorSigma; run
    `python3 tools/magma_tracker/build.py`; docs/HANDOFF.md: remove the fixed follow-ups
    (lines 252-254 Binomial, 296 S1, 300 S5, 302 S7 first part, 324-327) and add §11's notes
    (the quadratic class-number gap with the gp evidence and the Cohen 5.3.5 pointer).

Acceptance commands:

```bash
cargo test -p rustmath-core --no-fail-fast
cargo test -p rustmath-integers --no-fail-fast
cargo test -p rustmath-rationals --no-fail-fast
cargo test -p rustmath-polynomials -p rustmath-matrix --no-fail-fast        # i32/i64 and gcd consumers
cargo check --workspace --tests                                             # after steps 10 and 11
cargo test -p rustmath-symbolic -p rustmath-numberfields -p rustmath-ellipticcurves \
  -p rustmath-modular -p rustmath-numbertheory -p rustmath-powerseries -p rustmath-crypto \
  -p rustmath-padics -p rustmath-curves -p rustmath-algebraic -p rustmath-liealgebras \
  -p rustmath-galois -p rustmath-groups -p rustmath-combinatorics -p rustmath-quadraticforms \
  -p rustmath-rings -p rustmath-functionfields -p rustmath-geometry -p rustmath-finitefields \
  -p rustmath-manifolds -p rustmath-jupyter --no-fail-fast                  # to_i64 sites, eval_float, binomial, gcd users
cargo clippy -p rustmath-core -p rustmath-integers -p rustmath-rationals
nohup cargo test --workspace --no-fail-fast > <scratch>/ws.log 2>&1 &      # about an hour; baseline 14,173 passed
```

Risks:
- A Lehmer bug would corrupt every Rational. Mitigation: differential tests against Stein and
  HEAD's extended Euclid on hundreds of random pairs, the gp fixtures, and a
  `debug_assert!(a % g == 0 && b % g == 0)` in `lehmer::gcd` (debug builds only).
- A 2-adic root bug would turn a power into "not a power" (a false decision). Mitigation: the
  exhaustive small-range unit test, the debug_assert in `root_2adic`, gp fixtures with
  composite k and even parts.
- The to_i64 migration touches 24 crates (23 in the workspace plus rustmath-py). The first
  draft's grep missed 7 sites (review 1); the compiler finds every remaining one at step 11,
  Appendix A gives the replacement, and the full workspace test is the gate. Later units (S1
  symbolic, J1 REPL) must rebase over these one-line edits.
- The time-bounded tests (§3.4) can fail spuriously on a heavily loaded machine. Each bound is
  at least 10 times the measured debug time after the fix.
- Memory of `binomial`: `prime_range(2, K+1)` takes K bytes (C(2^26, 2^25) would need 64 MiB
  for the sieve and has a 8 MiB result). The 2^32-bit cap of §7.2 bounds the sieve at 4 GiB;
  an admitted call near the cap can still exhaust a small machine. A segmented sieve would
  remove this; it is not needed for any size the workspace uses.
- `ArithInt`/`ArithLLong` signatures change (public API, but no caller outside the crate).

---

## Appendix A: the 145 `.to_i64()` lines without Option handling (HEAD): 137 Integer sites

Found with two greps (166 calls in all). First, the lines without an Option handler on the same
line:
`grep -rn "\.to_i64()" --include=*.rs . | grep -vE '\.to_i64\(\)\s*(\?|\.unwrap|\.expect|\.ok_or|\.map|\.and_then|\.is_some|\.is_none|\.filter|\.unwrap_or|\.ok\(\)|\.zip)|Some\(|let Some|None =>|== Some|!= Some|fn to_i64'`
(138 lines). The `Some\(` exclusion in it also drops lines where the inherent method sits
inside `Some(..)` (review 1, B1). So second, every excluded line was read:
`grep -rn "\.to_i64()" --include=*.rs . | grep -E '<the same alternatives>'` gives 28 lines;
7 of them are Integer sites (`quadratic_sieve.rs:93`, `:112`, `:145`, `rational.rs:530`,
`substitute.rs:158`, `integrate.rs:681`, `solve.rs:1749`, marked "(rev. 1)" below) and the other
21 are Option-typed (an Option handler, or a non-Integer receiver: BigInt, rug, MPFR, BigFloat,
ComplexMPFR, `integer_range.rs`, `ratfunc.rs`). Test sites were found by a preceding
`#[cfg(test)] mod` or a `tests/` path. Classes as in §9.3. Every replacement is written in the
step-10 form (§9.3): it never calls the inherent `to_i64`, so it compiles before and after
step 11. "try_from" below means the TryFrom impls of step 9, for example `i64::try_from(&x)`.
G rows use `i64::try_from(&x).expect("<the invariant in the row>")`, T rows
`i64::try_from(&x).unwrap()`, R-keep rows `i64::try_from(&x).expect("<function>: ...")`.

| site (HEAD) | class | replacement |
|---|---|---|
| `rustmath-integers/src/quadratic_sieve.rs:50` | G | residue mod p (factor-base prime p < 2^63) |
| `rustmath-integers/src/quadratic_sieve.rs:60` | G | mod_pow result in [0, p) |
| `rustmath-integers/src/quadratic_sieve.rs:93` | G | (rev. 1) `tonelli_shanks`, p = 2: residue mod p, and p < 2^63 is asserted at :86-89; migrated in step 9 |
| `rustmath-integers/src/quadratic_sieve.rs:96` | G | residue mod p |
| `rustmath-integers/src/quadratic_sieve.rs:112` | G | (rev. 1) mod_pow result in [0, p), p < 2^63; step 9 |
| `rustmath-integers/src/quadratic_sieve.rs:145` | G | (rev. 1) Tonelli-Shanks root r in [0, p), p < 2^63; step 9 |
| `rustmath-integers/src/integer.rs:457` | G | digit < base <= 36 |
| `rustmath-integers/src/integer.rs:1590` | N | BigInt (ToPrimitive), already Option |
| `rustmath-integers/tests/ch18_signs_roots.rs:550` | T | residue mod \|m\| <= 15 |
| `rustmath-rationals/src/rational.rs:530` | R-fix | (rev. 1) `impl NumericConversion for Rational::to_i64`: `i64::try_from(&self.numerator).ok()` (the trait contract; HEAD panics for 2^70) |
| `rustmath-geometry/src/plane_curve_ops.rs:90` | R-keep | exponent of x^n; None means "not a polynomial", so a huge exponent must not map to None; explicit expect |
| `rustmath-py/src/rationals.rs:228` | R-keep | Python to_float (crate outside the workspace); explicit expect; S11 owns float conversion of large rationals |
| `rustmath-py/src/rationals.rs:229` | R-keep | same |
| `rustmath-modular/src/hecke.rs:613` | T | test value, known small |
| `rustmath-modular/src/modsym/gamma0.rs:106` | G | \|x % n\| < n <= i64::MAX |
| `rustmath-modular/src/modsym/lseries.rs:402` | R-fix | fn returns Result<_, String>: `i64::try_from(ap.numerator()).map_err(\|_\| format!("a_{p} does not fit in i64"))?` |
| `rustmath-modular/src/modsym/degeneracy.rs:1654` | T | test value, known small |
| `rustmath-modular/src/modsym/degeneracy.rs:2100` | T | test value, known small |
| `rustmath-modular/src/modsym/decomposition.rs:697` | T | test value, known small |
| `rustmath-modular/src/modsym/decomposition.rs:912` | T | test value, known small |
| `rustmath-modular/src/modsym/decomposition.rs:1006` | T | test value, known small |
| `rustmath-ellipticcurves/src/bsdratio.rs:345` | R-fix | `as u32` truncated silently; fn returns Result<_, String>: `u32::try_from(&sha_int).map_err(..)?` (TryFrom from step 9) |
| `rustmath-ellipticcurves/src/bsdratio.rs:346` | R-fix | same as :345 (no silent `as u32` truncation) |
| `rustmath-ellipticcurves/src/bsdratio.rs:484` | R-fix | same as :345 (no silent `as u32` truncation) |
| `rustmath-ellipticcurves/src/bsdratio.rs:485` | R-fix | same as :345 (no silent `as u32` truncation) |
| `rustmath-ellipticcurves/src/bsdratio.rs:689` | R-fix | same as :345 (no silent `as u32` truncation) |
| `rustmath-ellipticcurves/src/bsdratio.rs:690` | R-fix | same as :345 (no silent `as u32` truncation) |
| `rustmath-ellipticcurves/src/rizzo.rs:136` | G | residue mod small m |
| `rustmath-ellipticcurves/src/division.rs:440` | G | residue mod q (q small prime) |
| `rustmath-ellipticcurves/src/torsion.rs:261` | G | residue mod p (p small prime) |
| `rustmath-padics/src/montes.rs:94` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-curves/src/igp_closure.rs:69` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-curves/src/igp_closure.rs:179` | G | entries of PolySystem::jacobian_mod (reduced mod p) |
| `rustmath-numbertheory/src/bernoulli.rs:399` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-numbertheory/src/bernoulli.rs:400` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-numbertheory/src/bernoulli.rs:755` | T | test value, known small |
| `rustmath-numbertheory/src/quadratic_forms.rs:479` | R-keep | brute force over [0, m); explicit expect (a None must not join the existing `m <= 0 => return`, which would count 0 solutions) |
| `rustmath-powerseries/src/algebraic.rs:243` | R-keep | i128_of: callers turn None into an empty root list (a false "no roots"), so keep the panic explicit: `i128::from(i64::try_from(n).expect("i128_of: ..."))`; flag the callers |
| `rustmath-powerseries/src/algebraic.rs:655` | T | test value, known small |
| `rustmath-crypto/src/classical.rs:190` | G | residue in [0, 26) (comment already says so) |
| `rustmath-numberfields/src/round2.rs:441` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-numberfields/src/panayi.rs:360` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-numberfields/src/classgroup.rs:412` | G | invariant factor of a class group computed from i64 relations; `as usize` becomes usize::try_from(..).expect(..) |
| `rustmath-numberfields/src/lib.rs:419` | R-fix | membership in a list of 9 small discriminants: `i64::try_from(d).is_ok_and(\|v\| class_one_discs.contains(&v))` (exact); same commit: the \|d\| < 100 placeholder at :429-432 becomes `Err(NotImplemented)` (§11) |
| `rustmath-numberfields/src/ideals.rs:53` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-numberfields/src/ideals.rs:535` | T | test value, known small |
| `rustmath-numberfields/src/ramification.rs:139` | G | integer slopes of a ramification polygon (bounded by degree times valuations) |
| `rustmath-numberfields/src/ramification.rs:263` | G | floor of a bounded rational; `.max(0) as usize` keeps |
| `rustmath-numberfields/src/ramification.rs:433` | T | test value, known small |
| `rustmath-numberfields/src/local_field.rs:312` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-numberfields/src/local_field.rs:386` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-numberfields/src/local_field.rs:587` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-numberfields/src/local_field.rs:955` | T | test value, known small |
| `rustmath-polynomials/src/factorization.rs:702` | R-fix | the error text already says "a prime p >= 2 that fits in i64": `let p_i64 = match i64::try_from(p) { Ok(v) if v > 1 => v, _ => return Err(..) };` |
| `rustmath-polynomials/src/factorization.rs:717` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-polynomials/src/root_label.rs:755` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-polynomials/src/root_label.rs:1032` | T | test value, known small |
| `rustmath-polynomials/src/root_label.rs:1050` | T | test value, known small |
| `rustmath-polynomials/src/zp_hensel.rs:126` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-polynomials/src/zp_hensel.rs:234` | T | test value, known small |
| `rustmath-polynomials/src/disc.rs:65` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-polynomials/src/disc.rs:182` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-polynomials/src/disc.rs:184` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-polynomials/src/zassenhaus.rs:48` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-polynomials/src/berlekamp.rs:468` | G | p <= deg f (checked just above) |
| `rustmath-polynomials/src/berlekamp.rs:518` | G | split_small runs only for p < SMALL_P (:505) |
| `rustmath-polynomials/src/berlekamp.rs:565` | G | residue mod 2^32 |
| `rustmath-polynomials/src/padic_factor.rs:54` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-polynomials/src/padic_factor.rs:103` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-polynomials/tests/ch23_berlekamp.rs:659` | T | test value, known small |
| `rustmath-reals/src/bigfloat.rs:1099` | N | BigInt (ToPrimitive), already Option |
| `rustmath-reals/src/bigfloat.rs:1208` | N | BigFloat::to_i64 (Option), test |
| `rustmath-reals/src/bigfloat.rs:1216` | N | BigFloat::to_i64 (Option), test |
| `rustmath-algebraic/src/radicals.rs:34` | R-keep | explicit expect; the fallback at :39 returns alpha itself as its n-th root (a facade, flag it), so None must not reach it |
| `rustmath-algebraic/src/descriptor.rs:555` | R-keep | explicit expect (None of eval_interval means "cannot evaluate"; check before turning a huge exponent into it) |
| `rustmath-liealgebras/src/chevalley_basis.rs:276` | R-keep | structure constants are small; explicit expect, and fix the comment that promises "0 as fallback" (tracker row 1625) |
| `rustmath-galois/src/galois_ctx.rs:447` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-galois/src/galois_ctx.rs:714` | T | test value, known small |
| `rustmath-galois/src/galois_ctx.rs:780` | T | test value, known small |
| `rustmath-galois/src/sieve.rs:59` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-complex/src/mpc.rs:699` | N | rug::Integer::to_i64, already Option |
| `rustmath-complex/src/mpc.rs:927` | N | ComplexMPFR::to_i64 (Option), test |
| `rustmath-groups/src/pgalois.rs:131` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-finitefields/src/poly_factor.rs:1107` | T | test value, known small |
| `rustmath-finitefields/src/extension_field.rs:337` | T | test value, known small |
| `rustmath-finitefields/tests/square_roots.rs:237` | T | test value, known small |
| `rustmath-combinatorics/src/symmetric_group_representations.rs:598` | G | multiplicity of an irreducible (bounded by the dimension) |
| `rustmath-matrix/src/lattice.rs:48` | G | bit_length <= 52 checked at :46 |
| `rustmath-matrix/src/lattice.rs:55` | G | remainder mod 2^32 |
| `rustmath-symbolic/src/series.rs:34` | R-fix | `i.is_zero()` (exact; this is HANDOFF S7 "expansion point in series.rs:34") |
| `rustmath-symbolic/src/series.rs:206` | R-fix | `i.is_zero()` (exact; this is HANDOFF S7 "expansion point in series.rs:34") |
| `rustmath-symbolic/src/series.rs:233` | R-fix | `i.is_zero()` (exact; this is HANDOFF S7 "expansion point in series.rs:34") |
| `rustmath-symbolic/src/integrate_exact.rs:666` | G | \|k\| <= 4096 checked at :662 |
| `rustmath-symbolic/src/polynomial.rs:147` | R-keep | degree_recursive: the fallthrough says "Constant power, Some(0)", so a huge exponent must not reach it; explicit expect |
| `rustmath-symbolic/src/polynomial.rs:253` | R-keep | same shape as :147 |
| `rustmath-symbolic/src/specialfunctions/orthogonal_polys.rs:73` | R-fix | `if let Ok(n_val) = i64::try_from(n_int)` around the small-n table; a large n falls through to the unevaluated function (exact) |
| `rustmath-symbolic/src/specialfunctions/orthogonal_polys.rs:132` | R-fix | `if let Ok(n_val) = i64::try_from(n_int)` around the small-n table; a large n falls through to the unevaluated function (exact) |
| `rustmath-symbolic/src/specialfunctions/orthogonal_polys.rs:178` | R-fix | `if let Ok(n_val) = i64::try_from(n_int)` around the small-n table; a large n falls through to the unevaluated function (exact) |
| `rustmath-symbolic/src/specialfunctions/orthogonal_polys.rs:245` | R-fix | `if let Ok(n_val) = i64::try_from(n_int)` around the small-n table; a large n falls through to the unevaluated function (exact) |
| `rustmath-symbolic/src/specialfunctions/orthogonal_polys.rs:290` | R-fix | `if let Ok(n_val) = i64::try_from(n_int)` around the small-n table; a large n falls through to the unevaluated function (exact) |
| `rustmath-symbolic/src/specialfunctions/transcendental.rs:90` | R-fix | `if let Ok(n_val) = i64::try_from(n)`: a large s falls through to unevaluated zeta(s) |
| `rustmath-symbolic/src/specialfunctions/prime_pi.rs:67` | R-fix | test `n.signum() < 0` / `n < 2` on the Integer first (returns 0 exactly), then `if let Ok(n_val) = i64::try_from(n)` for the table; else unevaluated |
| `rustmath-symbolic/src/specialfunctions/prime_pi.rs:142` | R-fix | `if let (Ok(x), Ok(a)) = (i64::try_from(x_int), i64::try_from(a_int))` around the small table; else unevaluated |
| `rustmath-symbolic/src/specialfunctions/other.rs:200` | R-fix | `if let Ok(n_val) = i64::try_from(i)` around the n <= 20 table; else unevaluated factorial |
| `rustmath-symbolic/src/specialfunctions/other.rs:240` | R-fix | `if let (Ok(n), Ok(k)) = (i64::try_from(n_int), i64::try_from(k_int))` around the n <= 30 table; else unevaluated binomial |
| `rustmath-symbolic/src/risch.rs:380` | G | \|k\| <= 4096 checked at :377 |
| `rustmath-symbolic/src/risch.rs:484` | G | \|k\| <= 4096 checked at :480 |
| `rustmath-symbolic/src/substitute.rs:124` | R-fix | `match u32::try_from(exp) { Ok(e) => Some(l.pow(e)), Err(_) => None }` (same range 0..=u32::MAX as HEAD; None = not evaluated, as the existing else branch) |
| `rustmath-symbolic/src/substitute.rs:158` | R-fix | (rev. 1) `eval_float`: `Expr::Integer(n) => n.to_f64().filter(\|v\| v.is_finite())` (None = cannot evaluate; num-bigint gives `Some(inf)` above f64::MAX; HEAD panics for 10^30) |
| `rustmath-symbolic/src/integrate.rs:251` | R-fix | `matches!(self, Expr::Integer(i) if i.is_one())` (exact) |
| `rustmath-symbolic/src/integrate.rs:256` | R-fix | `matches!(self, Expr::Integer(i) if *i == Integer::from(-1))` (exact) |
| `rustmath-symbolic/src/integrate.rs:516` | R-fix | `*i == Integer::from(2)` (exact comparison with a small constant) |
| `rustmath-symbolic/src/integrate.rs:681` | R-fix | (rev. 1) `polynomial_degree`: `i64::try_from(n).ok()`, and `deg_left.checked_add(deg_right)` at :697; every caller propagates None as "this tier declines" (§9.3); doc: None also when the degree does not fit i64 |
| `rustmath-symbolic/src/integrate.rs:972` | R-fix | `*i == Integer::from(2)` (exact comparison with a small constant) |
| `rustmath-symbolic/src/radical.rs:124` | R-fix | `n.is_even()` |
| `rustmath-symbolic/src/radical.rs:129` | R-fix | `*n > Integer::from(2)` and `k = n / 2` as an Integer |
| `rustmath-symbolic/src/radical.rs:259` | R-fix | `Rational::new(Integer::one(), r.denominator().clone())` |
| `rustmath-symbolic/src/expand.rs:49` | R-fix | `if let Ok(e) = u32::try_from(exp) { if e <= 10 { return expand_power(&left_exp, e); } }`; a large exponent stays unexpanded |
| `rustmath-symbolic/src/limits.rs:323` | R-fix | `i.signum() > 0` |
| `rustmath-symbolic/src/limits.rs:332` | R-fix | `i.signum() < 0` |
| `rustmath-symbolic/src/solve.rs:907` | R-fix | `n.is_one()` (exact) |
| `rustmath-symbolic/src/solve.rs:915` | R-fix | `*n == Integer::from(-1)` (exact) |
| `rustmath-symbolic/src/solve.rs:946` | R-fix | `n.is_one()` (exact) |
| `rustmath-symbolic/src/solve.rs:950` | R-fix | `*n == Integer::from(-1)` (exact) |
| `rustmath-symbolic/src/solve.rs:973` | R-fix | `n.is_one()` (exact) |
| `rustmath-symbolic/src/solve.rs:981` | R-fix | `*n == Integer::from(-1)` (exact) |
| `rustmath-symbolic/src/solve.rs:1133` | R-fix | (corrected in rev. 1) `solve_constant_inequality` needs only the sign: `n.signum()` compared with 0 (exact; the first draft's `to_f64` is dropped) |
| `rustmath-symbolic/src/solve.rs:1137` | R-fix | (corrected in rev. 1) sign of the rational: `r.numerator().signum()` (denominator > 0, exact; `to_f64` would give NaN for huge parts and a false `Empty`) |
| `rustmath-symbolic/src/solve.rs:1190` | R-fix | sign test: `n.signum() > 0` / `r.numerator().signum() > 0` |
| `rustmath-symbolic/src/solve.rs:1192` | R-fix | sign test: `n.signum() > 0` / `r.numerator().signum() > 0` |
| `rustmath-symbolic/src/solve.rs:1266` | R-fix | sign test: `n.signum() > 0` / `r.numerator().signum() > 0` |
| `rustmath-symbolic/src/solve.rs:1268` | R-fix | sign test: `n.signum() > 0` / `r.numerator().signum() > 0` |
| `rustmath-symbolic/src/solve.rs:1722` | D | (rev. 1) inside `expr_to_rational`, dead code (rustc: "never used"); the function is deleted |
| `rustmath-symbolic/src/solve.rs:1749` | D | (rev. 1) inside `expr_to_integer`, dead code (rustc: "never used"); the function is deleted |
| `rustmath-symbolic/src/simplify.rs:596` | G | \|n\| <= MAX_FOLD_EXPONENT checked at :593 |
| `rustmath-quadraticforms/src/theta_series.rs:136` | R-keep | explicit expect (a float bound; keep today's behaviour) |
| `rustmath-quadraticforms/src/local_densities.rs:231` | R-keep | explicit expect; flag the pre-existing cap of 1000 residues (a facade for p^k > 1000) |
| `rustmath-rings/src/number_field/s_unit_solver.rs:232` | N | orphan file (not compiled); already written for an Option |
| `rustmath-rings/src/valuation/maclane.rs:135` | G | bit_length <= 62 checked at :130 |
| `rustmath-rings/src/padics/newton_lift.rs:643` | T | test value, known small |
| `rustmath-rings/src/padics/newton_lift.rs:770` | T | test value, known small |
| `rustmath-rings/src/padics/unramified.rs:89` | R-fix | `let p_small: Option<u32> = u32::try_from(prime).ok();` (prime >= 2 is checked at :82; exact, replaces the panicking call and the round-trip re-check) |
| `rustmath-rings/src/padics/unramified.rs:212` | R-fix | same as :89 in `tag_conway`: `u32::try_from(&self.prime).ok()` |
| `rustmath-functionfields/src/branch_cycles.rs:40` | G | residue of a reduction modulo a word-size modulus |
| `rustmath-functionfields/src/ratfunc.rs:280` | N | coefficient type has an Option to_i64 (inside `and_then`) |

## Appendix B: prototypes

All in the scratch directory named at the top; each was run against the debug and release
rlibs and checked by `verify.rs` / `verify2.rs` against the gp fixtures in `gp/`:
`prim.rs` (§1), `t3_fast.rs` (§2 reproduction), `t2_sqrt.rs` (§3), `lehmer.rs`,
`lehmer_small.rs`, `lehmer_x.rs` (§4), `ppow.rs` (§5, §6), `binom.rs` (§7), `t4_sigma.rs`
(§8), `to_i64_sites.txt`, `to_i64_ctx.txt`, `to_i64_rows.tsv`, `classify.py` (§9),
`gen_i3.gp` (fixtures). Revision 1, in `rev1/` (build commands in `rev1/BUILD.txt`):
`trat2.rs` (Rational to_i64), `sym.rs` (eval_float, Integer::to_f64), `pdeg.rs`
(polynomial_degree), `unr.rs`, `unr2.rs` (unramified.rs:89, :212), `tsq.rs` (isqrt and HEAD
is_perfect_power timings), `tlehmer.rs` (gcd / xgcd timings at 30k and 100k bits),
`vbinom.rs` (binomial fixture in debug), `lb_check.py` (the §7.2 size bound). Note: the 7
sites of revision 1 are not in `to_i64_rows.tsv` (it was built from the first draft's grep);
the design's Appendix A is the complete list.
