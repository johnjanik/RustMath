//! Arbitrary precision integers (MAGMA Handbook ch. 18, "Ring of Integers";
//! digest `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`).
//!
//! Residue arithmetic follows the handbook's conventions:
//! - §18.7 `LegendreSymbol`, `JacobiSymbol`, `KroneckerSymbol`: [`Integer::kronecker_symbol`]
//!   implements [Coh93] Algorithm 1.4.10 (binary Jacobi with quadratic reciprocity, extended to
//!   all moduli by the rules at 2 and -1); the Legendre and Jacobi symbols are restrictions of it.
//! - §18.12.1 `Modexp`, `Modinv`, `Modsqrt`: results are least non-negative residues in
//!   `[0, |m|)` ([`Integer::mod_pow`], [`Integer::mod_inverse`], [`Integer::sqrt_mod_prime`]).
//!   Square roots modulo a prime use [Coh93] Algorithm 1.5.1 (Tonelli-Shanks) and are verified
//!   before they are returned.
//! - `binomial_mod` reduces the generalised binomial C(n, k) (any n, k) modulo a prime p with
//!   Lucas' theorem and C(n, k) = (-1)^k C(k-n-1, k).
//! - §18.6 `Gcd`, `Lcm`, `Xgcd`: [`Integer::gcd`] and [`Integer::lcm`] are non-negative, and
//!   [`Integer::extended_gcd`] returns `g >= 0` (`g = 0` only for `(0, 0)`), by Lehmer's
//!   algorithm on the magnitudes ([Knu97] §4.5.2 Algorithm L; it applies exactly Euclid's
//!   quotients, so the multipliers are Euclid's).
//! - §18.4.8 `Isqrt`: [`Integer::sqrt`] is num-bigint's Newton iteration from a floating-point
//!   start (exact floor).
//! - §18.4.8 `Iroot` and §18.4.5 `IsPower`: [`Integer::nth_root`] is the floor of the real
//!   n-th root (also for negative `self` and odd n); [`Integer::is_power`] and
//!   [`Integer::is_perfect_power`] follow `IsPower(n, k)` / `IsPower(n)` and PARI/GP
//!   `ispower` for negative inputs; [`Integer::exact_root`] returns the root of
//!   `IsPower(n, k)`. `is_perfect_power` strips prime exponents q <= log2 |n| ([Ber98] §1), so
//!   it decides every input with no cap on the exponent. Odd exact roots are 2-adic roots
//!   (Newton's iteration, [Ber98]) checked by word fingerprints and one exact power; a
//!   power-residue sieve modulo primes p ≡ 1 (mod q) ([BS93]) runs first for q < 32.
//!
//! References:
//! - [Coh93] H. Cohen, *A Course in Computational Algebraic Number Theory*, GTM 138,
//!   Springer 1993, Algorithms 1.4.10 and 1.5.1.
//! - [Knu97] D. E. Knuth, *The Art of Computer Programming* vol. 2, 3rd ed., §4.5.2
//!   (Euclid's algorithm, extended) and §4.6.3 (binary exponentiation).
//! - [Ber98] D. J. Bernstein, *Detecting perfect powers in essentially linear time*, Math.
//!   Comp. 67 (1998) 1253-1283, §1 (n = b^k with b >= 2 forces k <= log2 n).
//! - [BS93] E. Bach, J. Sorenson, *Sieve algorithms for perfect power testing*, Algorithmica 9
//!   (1993) 313-328 (a q-th power is 0 or a q-th power residue modulo every prime p ≡ 1 mod q).
//! - E. Lucas, *Théorie des fonctions numériques simplement périodiques*, Amer. J. Math. 1
//!   (1878) 184-240, 289-321 (Lucas' theorem); Graham-Knuth-Patashnik, *Concrete
//!   Mathematics*, eq. (5.14) (negating the upper index).
//! - Primality: [`crate::prime::is_prime`] is a proof for every size (the twelve strong tests
//!   2..37 below psi_12 = 318665857834031151167461, J. Sorenson, J. Webster, *Strong
//!   pseudoprimes to twelve prime bases*, Math. Comp. 86 (2017) 985-1003; Baillie-PSW and the
//!   Atkin-Morain ECPP above it, [`crate::ecpp`]). The residue functions (§18.7 "a fast
//!   probabilistic primality test is applied") use [`crate::prime::is_probable_prime`]
//!   (Baillie-PSW, exact below psi_12), because their answers check themselves; `binomial_mod`
//!   needs a proven prime modulus and uses [`crate::prime::try_is_prime`].
//!
//! Note on `%`: the `Rem` impls (and `EuclideanDomain::div_rem`) truncate toward zero, so the
//! remainder has the sign of the dividend. Every residue computation in this file reduces with
//! `mod_floor` on the absolute modulus instead.

use num_bigint::BigInt;
use num_integer::Integer as _;
use num_traits::{One, Signed, Zero};
use rustmath_core::{
    CommutativeRing, EuclideanDomain, IntegralDomain, MathError, NumericConversion, Result, Ring,
};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Arbitrary precision integer
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Integer {
    value: BigInt,
}

impl Integer {
    /// Create a new integer from a BigInt
    pub fn new(value: BigInt) -> Self {
        Integer { value }
    }

    /// Create an integer with value 0
    pub fn zero() -> Self {
        Integer::new(BigInt::zero())
    }

    /// Create an integer with value 1
    pub fn one() -> Self {
        Integer::new(BigInt::one())
    }

    /// Parse an integer from its base-10 string representation (exact, bignum).
    /// Returns `None` if the string is not a valid decimal integer.
    pub fn from_decimal_str(s: &str) -> Option<Self> {
        use std::str::FromStr;
        BigInt::from_str(s).ok().map(Integer::new)
    }

    /// Get the absolute value
    pub fn abs(&self) -> Self {
        Integer::new(self.value.abs())
    }

    /// The value as an `i64`; `None` when the value is outside i64 (the same contract as
    /// `NumericConversion::to_i64`, `BigInt::to_i64` and [`Integer::to_f64`]). For a typed
    /// error use `i64::try_from(&n)` (`Err(NumericalError)`), and the same for the other
    /// primitive integer types.
    pub fn to_i64(&self) -> Option<i64> {
        use num_traits::ToPrimitive;
        self.value.to_i64()
    }

    /// Get the sign: -1 for negative, 0 for zero, 1 for positive
    pub fn signum(&self) -> i8 {
        if self.value.is_zero() {
            0
        } else if self.value.is_positive() {
            1
        } else {
            -1
        }
    }

    /// Greatest common divisor (MAGMA `Gcd(m, n)`, Handbook ch. 18 §18.6): always
    /// non-negative; `|m|` when `n = 0` (and vice versa); `0` for `(0, 0)`. PARI/GP `gcd`
    /// gives the same values: `gcd(4, -6) = 2`, `gcd(0, -5) = 5`.
    ///
    /// Lehmer's algorithm on `|m|` and `|n|` ([Knu97] §4.5.2 Algorithm L).
    pub fn gcd(&self, other: &Self) -> Self {
        EuclideanDomain::gcd(self, other)
    }

    /// Least common multiple (MAGMA `Lcm(m, n)`, Handbook ch. 18 §18.6): the smallest
    /// non-negative common multiple, so `lcm(m, n) · gcd(m, n) = |m · n|`; `0` when `m` or `n`
    /// is `0`. PARI/GP: `lcm(-4, -6) = 12`.
    pub fn lcm(&self, other: &Self) -> Self {
        EuclideanDomain::lcm(self, other)
    }

    /// Extended gcd (MAGMA `Xgcd(m, n)`, Handbook ch. 18 §18.6): `(g, x, y)` with
    /// `g = gcd(m, n) = x·m + y·n` and `g >= 0`; `g = 0` only for `(0, 0)`, which gives
    /// `(0, 0, 0)`.
    ///
    /// The multipliers are those of the extended Euclidean algorithm ([Knu97] §4.5.2) with
    /// truncated quotients, negated together with `g` when `g` comes out negative. They equal
    /// PARI/GP `gcdext(m, n)` (which returns `[x, y, g]`): `xgcd(3, -7) = (1, -2, -1)`,
    /// `xgcd(0, -5) = (5, 0, -1)`. Computed by Lehmer's algorithm (Algorithm L), which
    /// applies exactly Euclid's quotients.
    pub fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        EuclideanDomain::extended_gcd(self, other)
    }

    /// Compute self raised to the power exp
    pub fn pow(&self, exp: u32) -> Self {
        Ring::pow(self, exp)
    }

    /// Modular exponentiation `self^exp mod |modulus|` (MAGMA `Modexp`, Handbook ch. 18
    /// §18.12.1; binary exponentiation [Knu97] §4.6.3). Exponents longer than 32 bits go to
    /// `BigInt::modpow`, which uses Montgomery multiplication only when `|modulus|` is odd and
    /// plain square-and-multiply with division otherwise.
    ///
    /// The result is the least non-negative residue, in `[0, |modulus|)`, for every sign of
    /// `self`, `exp` and `modulus` (PARI/GP `lift(Mod(self, modulus)^exp)`):
    ///
    /// | input | result |
    /// |---|---|
    /// | `modulus = 0` | `Err(DivisionByZero)` |
    /// | `\|modulus\| = 1` | `Ok(0)` for every `exp` (Z/1Z = {0}) |
    /// | `modulus < 0` | computed modulo `\|modulus\|` |
    /// | `exp >= 0` | `self^exp` reduced; `0^0 = 1` |
    /// | `exp < 0`, `gcd(self, modulus) = 1` | `(self^-1)^\|exp\|` reduced |
    /// | `exp < 0`, not invertible | `Err(NotInvertible)` |
    pub fn mod_pow(&self, exp: &Self, modulus: &Self) -> Result<Self> {
        if modulus.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let m = modulus.value.abs();
        if m.is_one() {
            return Ok(Integer::zero());
        }
        if exp.value.is_negative() {
            let inv = self
                .value
                .mod_floor(&m)
                .modinv(&m)
                .ok_or(MathError::NotInvertible)?;
            return Ok(Integer::new(mod_pow_nonneg(&inv, &(-&exp.value), &m)));
        }
        Ok(Integer::new(mod_pow_nonneg(&self.value, &exp.value, &m)))
    }

    /// Check if the number is even
    pub fn is_even(&self) -> bool {
        (&self.value % 2) == BigInt::zero()
    }

    /// Check if the number is odd
    pub fn is_odd(&self) -> bool {
        !self.is_even()
    }

    /// Check if the number is zero
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// Check if the number is one
    pub fn is_one(&self) -> bool {
        self.value == BigInt::one()
    }

    /// Convert to f64 if within range
    pub fn to_f64(&self) -> Option<f64> {
        use num_traits::ToPrimitive;
        self.value.to_f64()
    }

    /// Get the bit length
    pub fn bit_length(&self) -> u64 {
        self.value.bits()
    }

    /// Convert to BigInt reference
    pub fn as_bigint(&self) -> &BigInt {
        &self.value
    }

    /// Convert into BigInt
    pub fn into_bigint(self) -> BigInt {
        self.value
    }

    /// MAGMA `Isqrt(n)` (Handbook ch. 18 §18.4.8): `floor(sqrt(self))` for `self >= 0`.
    ///
    /// num-bigint's `BigUint::sqrt`: Newton's iteration started from a floating-point
    /// estimate of the root (R. P. Brent, P. Zimmermann, *Modern Computer Arithmetic*,
    /// Algorithm 1.13 SqrtInt), which returns the exact floor. The old start at `x = self`
    /// needed about bits/2 full divisions before it neared the root.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` for `self < 0`.
    pub fn sqrt(&self) -> Result<Self> {
        if self.signum() < 0 {
            return Err(MathError::InvalidArgument(
                "Square root of negative number".to_string(),
            ));
        }
        Ok(Integer::new(BigInt::from(self.value.magnitude().sqrt())))
    }

    /// The floor of the real n-th root: the largest integer `r` with `r^n <= self`.
    ///
    /// For `self >= 0` this is MAGMA `Iroot(a, n)` (Handbook ch. 18 §18.4.8) and PARI/GP
    /// `sqrtnint(a, n)`. MAGMA and PARI define it only for `a >= 0`; for `self < 0` and odd
    /// `n` the real root exists and this returns its floor, `-ceil(|self|^(1/n))`:
    /// `nth_root(-8, 3) = -2`, `nth_root(-9, 3) = -3`. That is not the truncation toward zero
    /// of GMP `mpz_root` or Sage `nth_root(truncate_mode=1)`, which give `-2` for `-9`.
    ///
    /// The root of `|self|` is `num_bigint::BigUint::nth_root` (exact floor root).
    ///
    /// # Errors
    ///
    /// `InvalidArgument` when `n = 0`, or when `self < 0` and `n` is even.
    pub fn nth_root(&self, n: u32) -> Result<Self> {
        if n == 0 {
            return Err(MathError::InvalidArgument(
                "Root degree cannot be 0".to_string(),
            ));
        }

        if self.signum() < 0 && n.is_multiple_of(2) {
            return Err(MathError::InvalidArgument(
                "Even root of negative number".to_string(),
            ));
        }

        let mag = self.value.magnitude();
        let t = mag.nth_root(n);
        if !self.value.is_negative() {
            return Ok(Integer::new(BigInt::from(t)));
        }
        // self < 0 and n odd: floor(self^(1/n)) = -ceil(|self|^(1/n)). t^n <= |self|, so the
        // check below never builds a number larger than |self|.
        let r = if &t.pow(n) == mag { t } else { t + 1u32 };
        Ok(Integer::new(-BigInt::from(r)))
    }

    /// Get all positive divisors of this number
    ///
    /// Returns divisors in ascending order
    pub fn divisors(&self) -> Result<Vec<Self>> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Zero has infinitely many divisors".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(vec![Integer::one()]);
        }

        // Get prime factorization
        let factors = factor(&n);

        // Generate divisors from prime factorization
        // If n = p1^a1 * p2^a2 * ... * pk^ak
        // then divisors are all products p1^b1 * p2^b2 * ... * pk^bk where 0 <= bi <= ai
        let mut divisors = vec![Integer::one()];

        for (prime, multiplicity) in factors {
            let mut new_divisors = Vec::new();
            let mut prime_power = Integer::one();

            for _ in 0..=multiplicity {
                for div in &divisors {
                    new_divisors.push(div.clone() * prime_power.clone());
                }
                prime_power = prime_power * prime.clone();
            }

            divisors = new_divisors;
        }

        divisors.sort();
        Ok(divisors)
    }

    /// Count the number of divisors (tau function, also written σ₀(n))
    pub fn num_divisors(&self) -> Result<Self> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Zero has infinitely many divisors".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(Integer::one());
        }

        // If n = p1^a1 * p2^a2 * ... * pk^ak
        // then tau(n) = (a1+1)(a2+1)...(ak+1)
        let factors = factor(&n);
        let mut result = Integer::one();

        for (_, multiplicity) in factors {
            result = result * Integer::from(u64::from(multiplicity) + 1);
        }

        Ok(result)
    }

    /// Sum of divisors (sigma function, also written σ₁(n))
    pub fn sum_divisors(&self) -> Result<Self> {
        self.sigma(1)
    }

    /// Generalized divisor function σₖ(n) - sum of k-th powers of divisors
    ///
    /// σₖ(n) = Σ d^k for all divisors d of n
    ///
    /// Special cases:
    /// - σ₀(n) = number of divisors (tau function)
    /// - σ₁(n) = sum of divisors
    ///
    /// Formula: If n = p₁^a₁ * p₂^a₂ * ... * pₖ^aₖ, then
    /// σₖ(n) = ∏ [(pᵢ^((aᵢ+1)*k) - 1) / (pᵢ^k - 1)]
    /// (MAGMA `DivisorSigma(k, n)`, Handbook ch. 18 §18.7, for "small non-negative k").
    ///
    /// # Errors
    ///
    /// - `InvalidArgument` for `self = 0`.
    /// - `NumericalError` when an exponent `(a + 1)·k` exceeds `u32::MAX` (the exponent type
    ///   of [`Integer::pow`]). With `a >= 1` that forces `a·k >= 2^31`, so σₖ(n) would have
    ///   more than 2^31 bits (256 MiB). Every exponent is checked before any power is formed,
    ///   so the error costs no more than the factorisation.
    pub fn sigma(&self, k: u32) -> Result<Self> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Zero has infinitely many divisors".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(Integer::one());
        }

        // Special case: σ₀(n) is the count of divisors
        if k == 0 {
            return self.num_divisors();
        }

        // Get prime factorization
        let factors = factor(&n);
        let mut result = Integer::one();

        // Every exponent (a + 1)·k is checked before any power is formed: a power formed
        // first can take minutes (3^(2^31)), and then the call would fail anyway.
        let mut exps: Vec<u32> = Vec::with_capacity(factors.len());
        for (_, a) in &factors {
            let e = (u64::from(*a) + 1) * u64::from(k);
            let e = u32::try_from(e).map_err(|_| {
                MathError::NumericalError(format!(
                    "sigma: the exponent (a + 1)·k = {e} of p^((a+1)k) exceeds u32::MAX; \
                     sigma_k(n) would have more than 2^31 bits"
                ))
            })?;
            exps.push(e);
        }

        // If n = p1^a1 * p2^a2 * ... * pk^ak
        // then σₖ(n) = Π[(p_i^((a_i+1)*k) - 1) / (p_i^k - 1)]
        for ((prime, _), e) in factors.iter().zip(exps) {
            let numerator = prime.pow(e) - Integer::one();
            // p - 1 for k = 1, else p^k - 1 (k < e, and p^k is formed only after the check).
            let denominator = if k == 1 {
                prime.clone() - Integer::one()
            } else {
                prime.pow(k) - Integer::one()
            };
            let (quotient, _) = numerator.div_rem(&denominator)?;
            result = result * quotient;
        }

        Ok(result)
    }

    /// Whether `self = b^k` for some integer `b` (MAGMA `IsPower(n, k)`, Handbook ch. 18
    /// §18.4.5; PARI/GP `ispower(n, k)`): `self.exact_root(k).is_some()`.
    ///
    /// MAGMA defines it for `n > 1` and `k > 1`; outside that range this answers the same
    /// question: every integer is a first power, `0 = 0^k` for `k >= 1`, a negative number is
    /// a k-th power only for odd `k` (`-8 = (-2)^3`, but `-4` is not a square), and `b^0 = 1`
    /// for every `b`, so `k = 0` gives `true` only for `1`.
    pub fn is_power(&self, k: u32) -> bool {
        self.exact_root(k).is_some()
    }

    /// The integer `b` with `b^k = self`, when `self` is a perfect k-th power: the second
    /// return value of MAGMA `IsPower(n, k)` (Handbook ch. 18 §18.4.5); PARI/GP
    /// `ispower(n, k, &b)`.
    ///
    /// - `Some(b)` with `b^k = self`. For `self > 0` and even `k`, `b > 0` (as PARI); for
    ///   `self < 0` (odd `k`) `b < 0`; `0 = 0^k` for `k >= 1`; `k = 1` gives `self`.
    /// - `None` when no integer `b` has `b^k = self` (`self < 0` with even `k` included).
    /// - `k = 0`: `Some(1)` for `self = 1` (`1 = 1^0`), `None` otherwise. MAGMA requires
    ///   `k > 1` and PARI refuses `k <= 0`; these values only extend the definition.
    ///
    /// Algorithm (module `perfect_power`): `k = 2^e·k'` with `k'` odd. The k'-th root of
    /// `|self|` comes from the 2-adic root of its odd part (Newton's iteration doubles the
    /// precision; [Ber98]), checked by two word-size fingerprints and one exact power; then
    /// `e` square roots, each after the quadratic-residue sieve ([BS93]). Every `None` is a
    /// proof. Cost: `O(log k · M(bits/k'))` plus the square roots; for odd `k` at least
    /// `bits(self)` it is O(1) after the trailing-zero count. (`exact_root(3^100003 + 2, 9999)`
    /// takes milliseconds; the floor root `BigUint::nth_root` took 11 s in a debug build.)
    pub fn exact_root(&self, k: u32) -> Option<Self> {
        if k == 0 {
            return self.is_one().then(Integer::one);
        }
        let negative = self.value.is_negative();
        if negative && k.is_multiple_of(2) {
            return None;
        }
        let mag = self.value.magnitude();
        if mag.bits() <= 1 {
            return Some(self.clone()); // 0, 1, -1 (odd k): their own k-th roots
        }
        // For odd k, (-t)^k = -t^k, so the root of |self| decides negative self too.
        let r = BigInt::from(crate::perfect_power::exact_root_biguint(mag, k)?);
        Some(Integer::new(if negative { -r } else { r }))
    }

    /// Get digits of the number in the given base
    ///
    /// Returns digits from least significant to most significant
    /// (reverse of normal writing order)
    pub fn digits(&self, base: u32) -> Result<Vec<u8>> {
        if base < 2 {
            return Err(MathError::InvalidArgument(
                "Base must be at least 2".to_string(),
            ));
        }

        if base > 36 {
            return Err(MathError::InvalidArgument(
                "Base must be at most 36".to_string(),
            ));
        }

        if self.is_zero() {
            return Ok(vec![0]);
        }

        let mut n = self.abs();
        let mut digits = Vec::new();
        let base_int = Integer::from(base);

        while !n.is_zero() {
            let (quotient, remainder) = n.div_rem(&base_int)?;
            let digit = u8::try_from(&remainder).expect("digits: 0 <= remainder < base <= 36");
            digits.push(digit);
            n = quotient;
        }

        Ok(digits)
    }

    /// Compute Euler's totient function φ(n)
    ///
    /// φ(n) counts the positive integers up to n that are relatively prime to n
    ///
    /// Formula: If n = p₁^a₁ * p₂^a₂ * ... * pₖ^aₖ, then
    /// φ(n) = n * ∏(1 - 1/pᵢ) = ∏(pᵢ^(aᵢ-1) * (pᵢ - 1))
    pub fn euler_phi(&self) -> Result<Self> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Euler's totient is not defined for 0".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(Integer::one());
        }

        // Get prime factorization
        let factors = factor(&n);
        let mut result = Integer::one();

        // φ(n) = ∏ p^(a-1) * (p - 1) for each prime power p^a
        for (prime, exponent) in factors {
            // p^(a-1)
            let prime_power = if exponent > 1 {
                prime.pow(exponent - 1)
            } else {
                Integer::one()
            };

            // (p - 1)
            let p_minus_1 = prime - Integer::one();

            result = result * prime_power * p_minus_1;
        }

        Ok(result)
    }

    /// Compute the Möbius function μ(n)
    ///
    /// μ(n) = 1 if n is a square-free positive integer with an even number of prime factors
    /// μ(n) = -1 if n is a square-free positive integer with an odd number of prime factors
    /// μ(n) = 0 if n has a squared prime factor
    pub fn moebius(&self) -> Result<i8> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Möbius function is not defined for 0".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(1);
        }

        // Get prime factorization
        let factors = factor(&n);

        // Check if any prime has exponent > 1 (not square-free)
        for (_, exponent) in &factors {
            if *exponent > 1 {
                return Ok(0);
            }
        }

        // n is square-free, return (-1)^k where k is the number of prime factors
        if factors.len() % 2 == 0 {
            Ok(1)
        } else {
            Ok(-1)
        }
    }

    /// Check if the number is square-free (not divisible by any perfect square > 1)
    pub fn is_square_free(&self) -> bool {
        use crate::prime::factor;

        if self.is_zero() {
            return false;
        }

        let n = self.abs();
        if n.is_one() {
            return true;
        }

        // Check if any prime factor has exponent > 1
        let factors = factor(&n);
        factors.iter().all(|(_, exp)| *exp == 1)
    }

    /// Perfect-power decomposition (MAGMA `IsPower(n)`, Handbook ch. 18 §18.4.5; PARI/GP
    /// `ispower(n, , &b)`).
    ///
    /// - `self > 1`: `Some((b, k))` with `self = b^k`, `k >= 2` maximal and so `b >= 2`
    ///   minimal (MAGMA's "minimal positive b and its associated k"): `64 -> (2, 6)`,
    ///   `2^67 -> (2, 67)`. `None` when `self` is not a perfect power.
    /// - `self < -1`: `Some((b, k))` with `b < 0` and `k` the largest **odd** exponent with
    ///   `self = b^k`, as PARI/GP returns: `-64 -> (-4, 3)`, `-32 -> (-2, 5)`. `None` when
    ///   there is no odd exponent `k >= 3` (`-4` is not a perfect power).
    /// - `|self| <= 1`: `None`. This is PARI's convention (`ispower(0)` = `ispower(±1)` = 0)
    ///   and lies outside MAGMA's domain `n > 1`; it does not claim that 0 and ±1 are not
    ///   powers (`1 = 1^k`).
    ///
    /// Algorithm: write `|self| = c^m` with `c` not a perfect power. `|self|` is a j-th power
    /// iff `j | m`, and `c >= 2` gives `m <= log2 |self|` ([Ber98] §1). So stripping exact
    /// q-th roots for each prime `q` (ascending, while `q` is below the bit length of what
    /// remains) finds `m` exactly: the search covers every possible exponent and decides
    /// every input. For `self < 0` only odd primes are stripped.
    ///
    /// The root step (module `perfect_power`). For `q < 32` a power-residue sieve ([BS93])
    /// first tries up to four primes `p ≡ 1 (mod q)`: if `p ∤ base` and
    /// `base^((p-1)/q) ≢ 1 (mod p)`, then `base` is not a q-th power (`base = d^q` would give
    /// `d^(p-1) ≡ 1`, Fermat). Then `q = 2` takes `BigUint::sqrt` and checks `s·s = base`.
    /// An odd `q` takes the unique odd q-th root `r` of the odd part `m` of `base` modulo
    /// `2^K`, `K = ceil(bits(m)/q)` (2-adic Newton iteration, [Ber98]): `m` is a q-th power iff
    /// `bits(r) = K` and `r^q = m`, which is checked first modulo two word primes and then
    /// exactly. The odd part and its residues are computed once per base. Every exponent in
    /// the answer comes from an exact root raised back to `base`, and every rejection is a
    /// proof.
    ///
    /// Cost: per odd prime `q`, `O(log q · M(bits/q))` for the 2-adic root (u64 arithmetic
    /// once `K <= 64`, i.e. for every `q > bits/64`) and `O(bits/q)` for the residues, so the
    /// sum over `q` is a small multiple of `M(bits)`. An input built to agree with a q-th power
    /// modulo both word primes for many `q` pays one exact power per such `q`. Measured in a
    /// debug build: `2^1000000 + 1` 0.13 s (the floor-root loop before took about 12 s),
    /// `3001^1999` 3.5 ms (0.37 s before), `(2^61 - 1)^331` 6 ms (0.13 s before).
    ///
    /// # Panics
    ///
    /// Only on inputs of more than `u32::MAX` bits (512 MiB), where the exponent would not
    /// fit the `u32` result.
    pub fn is_perfect_power(&self) -> Option<(Self, u32)> {
        use crate::perfect_power::{exact_root_odd, may_be_qth_power, odd_part};
        /// Below this exponent the residue sieve runs before the root: a few O(bits) passes
        /// refute almost every non-power, and the root costs more there.
        const SIEVE_BELOW: u64 = 32;
        if self.abs() <= Integer::one() {
            return None;
        }
        let negative = self.value.is_negative();
        let mut base = self.value.magnitude().clone();
        let bits = u32::try_from(base.bits())
            .expect("is_perfect_power: input has more than u32::MAX bits");
        let mut odd = odd_part(&base);
        let mut k: u32 = 1;
        for q in crate::fast_arith::prime_range(2, u64::from(bits) + 1) {
            // base = d^q with d >= 2 forces 2^q <= base, i.e. q < bits(base).
            if q >= base.bits() {
                break;
            }
            if negative && q == 2 {
                continue; // a negative number is only an odd power
            }
            let q32 = u32::try_from(q).expect("q < bits(base) <= u32::MAX");
            loop {
                if q < SIEVE_BELOW && !may_be_qth_power(&base, q) {
                    break; // proof that base is not a q-th power
                }
                let root = if q == 2 {
                    let s = base.sqrt();
                    (&s * &s == base).then_some(s)
                } else {
                    exact_root_odd(&odd, q)
                };
                let Some(r) = root else { break };
                base = r;
                odd = odd_part(&base);
                k *= q32; // base >= 2 and base^k = |self|, so k < bits(|self|) <= u32::MAX
                if q >= base.bits() {
                    break;
                }
            }
        }
        if k == 1 {
            return None;
        }
        let b = BigInt::from(base);
        Some((Integer::new(if negative { -b } else { b }), k))
    }

    /// Compute the radical of n (product of distinct prime factors)
    ///
    /// rad(n) = ∏ p for each prime p dividing n
    pub fn radical(&self) -> Result<Self> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Radical is not defined for 0".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(Integer::one());
        }

        // Get prime factorization and multiply distinct primes
        let factors = factor(&n);
        let mut result = Integer::one();

        for (prime, _) in factors {
            result = result * prime;
        }

        Ok(result)
    }

    /// Check if the number is a perfect square
    pub fn is_perfect_square(&self) -> bool {
        if self.signum() < 0 {
            return false;
        }

        if let Ok(sqrt) = self.sqrt() {
            sqrt.clone() * sqrt == *self
        } else {
            false
        }
    }

    /// Legendre symbol (self/p) for an odd prime p (MAGMA `LegendreSymbol`, Handbook ch. 18
    /// §18.7), for every integer `self` (negative included).
    ///
    /// Returns:
    /// - 0 if self ≡ 0 (mod p)
    /// - 1 if self is a non-zero quadratic residue mod p
    /// - -1 if self is a quadratic non-residue mod p
    ///
    /// Computed as [`Integer::kronecker_symbol`] ([Coh93] Alg. 1.4.10, O(log^2 p)).
    ///
    /// Errors:
    /// - p <= 2 or p not accepted by [`crate::prime::is_probable_prime`]: `InvalidArgument`.
    /// - The symbol is 0 although p does not divide self: this proves p composite
    ///   (gcd(self, p) is a proper factor), so `InvalidArgument("... modulus is not prime ...")`
    ///   is returned instead of a 0 that would claim p | self.
    ///
    /// Primality, as in MAGMA ("a fast probabilistic primality test is applied"), is the
    /// Baillie-PSW probable-prime test: exact below psi_12 = 318665857834031151167461, and no
    /// BPSW pseudoprime is known above it. For a BPSW pseudoprime p (if one exists) the value
    /// returned is the Jacobi symbol (self/p), which is not the Legendre symbol of a prime. A -1
    /// is a proof of non-residuosity for any odd p; a 1 is not.
    pub fn legendre_symbol(&self, p: &Self) -> Result<i8> {
        use crate::prime::is_probable_prime;

        if *p <= Integer::from(2) || !is_probable_prime(p) {
            return Err(MathError::InvalidArgument(
                "Second argument must be an odd prime".to_string(),
            ));
        }

        let k = self.kronecker_symbol(p);
        if k == 0 && !self.value.mod_floor(&p.value).is_zero() {
            return Err(MathError::InvalidArgument(
                "legendre_symbol: modulus is not prime (gcd(a, p) > 1)".to_string(),
            ));
        }
        Ok(k)
    }

    /// Kronecker symbol (self/n) for all integers self and n (MAGMA `KroneckerSymbol`,
    /// Handbook ch. 18 §18.7). The symbol is defined everywhere, so no error is possible.
    ///
    /// - (a/0) = 1 if |a| = 1, else 0.
    /// - (a/2) = 0 for a even, (-1)^((a^2-1)/8) for a odd.
    /// - (a/-1) = -1 if a < 0, else 1 (so (0/-1) = 1).
    /// - Completely multiplicative in n. It equals the Jacobi symbol for odd n > 0 and the
    ///   Legendre symbol for odd prime n.
    ///
    /// Algorithm: [Coh93] Algorithm 1.4.10 (binary Jacobi algorithm with quadratic
    /// reciprocity), O(log^2 max(|a|, |n|)) bit operations; n is never factored.
    pub fn kronecker_symbol(&self, n: &Self) -> i8 {
        kronecker_bigint(&self.value, &n.value)
    }

    /// Compute the p-adic valuation: largest k such that p^k divides self
    ///
    /// Returns the exponent k in the prime factorization where this prime appears.
    /// Returns 0 if p does not divide self.
    ///
    /// # Examples
    /// ```ignore
    /// // 12 = 2^2 * 3, so valuation(12, 2) = 2
    /// assert_eq!(Integer::from(12).valuation(&Integer::from(2)), 2);
    /// // valuation(12, 3) = 1
    /// assert_eq!(Integer::from(12).valuation(&Integer::from(3)), 1);
    /// // valuation(12, 5) = 0 (5 doesn't divide 12)
    /// assert_eq!(Integer::from(12).valuation(&Integer::from(5)), 0);
    /// ```
    pub fn valuation(&self, p: &Self) -> u32 {
        if self.is_zero() {
            return u32::MAX; // Infinite valuation for 0
        }

        if p.abs() <= Integer::one() {
            return 0; // Invalid prime
        }

        let mut n = self.abs();
        let p_abs = p.abs();
        let mut count = 0u32;

        loop {
            let (quotient, remainder) = match n.div_rem(&p_abs) {
                Ok((q, r)) => (q, r),
                Err(_) => break,
            };

            if !remainder.is_zero() {
                break;
            }

            count += 1;
            n = quotient;

            if n.is_zero() {
                break;
            }
        }

        count
    }

    /// Get the number of bits required to represent this integer
    ///
    /// This is an alias for `bit_length()` for SageMath compatibility.
    /// Returns the minimum number of bits needed to represent this number.
    pub fn bits(&self) -> u64 {
        self.bit_length()
    }

    /// Jacobi symbol (self/n) for odd n > 0 (MAGMA `JacobiSymbol`, Handbook ch. 18 §18.7), for
    /// every integer `self` (negative included). (self/1) = 1.
    ///
    /// Computed by quadratic reciprocity as [`Integer::kronecker_symbol`] ([Coh93]
    /// Alg. 1.4.10); n is not factored.
    ///
    /// Errors: n <= 0 or n even gives `InvalidArgument` (use [`Integer::kronecker_symbol`] for
    /// those moduli).
    pub fn jacobi_symbol(&self, n: &Self) -> Result<i8> {
        if n.is_even() || n.signum() <= 0 {
            return Err(MathError::InvalidArgument(
                "Second argument must be an odd positive integer".to_string(),
            ));
        }
        Ok(self.kronecker_symbol(n))
    }

    /// Decide whether `self` is a square modulo a prime p, i.e. whether some x has
    /// x^2 ≡ self (mod p). 0 counts as a square, and every integer is a square mod 2.
    ///
    /// Every `Ok` is a decision:
    /// - `Ok(false)` is returned only when the Jacobi symbol (self/p) is -1, which proves that
    ///   self is a non-square modulo any odd p, prime or not.
    /// - `Ok(true)` is returned only after [`Integer::sqrt_mod_prime`] produced a verified root.
    ///
    /// Errors (`InvalidArgument`):
    /// - p not accepted by [`crate::prime::is_probable_prime`] ("Modulus must be prime";
    ///   Baillie-PSW, exact below psi_12 = 318665857834031151167461, no BPSW pseudoprime is
    ///   known).
    /// - (self/p) = 0 with p not dividing self, which proves p composite.
    /// - p is a BPSW pseudoprime and the root computation failed (a bounded search ran out or
    ///   the root check failed), which proves p composite.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// // 4 is a QR mod 7: 2² = 4
    /// assert!(Integer::from(4).is_quadratic_residue_prime(&Integer::from(7)).unwrap());
    /// // 3 is not a QR mod 7
    /// assert!(!Integer::from(3).is_quadratic_residue_prime(&Integer::from(7)).unwrap());
    /// // -1 is not a QR mod 7 (7 = 3 mod 4)
    /// assert!(!Integer::from(-1).is_quadratic_residue_prime(&Integer::from(7)).unwrap());
    /// ```
    pub fn is_quadratic_residue_prime(&self, p: &Self) -> Result<bool> {
        use crate::prime::is_probable_prime;

        if !is_probable_prime(p) {
            return Err(MathError::InvalidArgument(
                "Modulus must be prime".to_string(),
            ));
        }
        match sqrt_mod_prime_checked(&self.value, &p.value)? {
            Some(_) => Ok(true),
            None => Ok(false),
        }
    }

    /// Square root modulo a prime (MAGMA `Modsqrt`, Handbook ch. 18 §18.12.1): returns r with
    /// 0 <= r < p and r^2 ≡ self (mod p), for every integer `self` (negative included).
    /// Which of the two roots is returned is unspecified.
    ///
    /// # Algorithm
    ///
    /// `a = self mod p` (least non-negative residue). For p = 2 or a = 0 the root is a.
    /// Otherwise the Kronecker symbol (a/p) is computed ([Coh93] Alg. 1.4.10); for
    /// p ≡ 3 (mod 4) the root is a^((p+1)/4), and for p ≡ 1 (mod 4) it is found by
    /// Tonelli-Shanks ([Coh93] Alg. 1.5.1). Every root is checked (r^2 ≡ a) before it is
    /// returned, so a non-root is never returned, even for a pseudoprime modulus.
    ///
    /// # Errors (`InvalidArgument`)
    ///
    /// - p not accepted by [`crate::prime::is_probable_prime`]: "Modulus must be prime"
    ///   (Baillie-PSW, exact below psi_12 = 318665857834031151167461, no BPSW pseudoprime is
    ///   known). A caller that has already proved p prime uses
    ///   [`Integer::try_sqrt_mod_prime`], which runs no primality test.
    /// - (a/p) = -1: "Not a quadratic residue".
    /// - (a/p) = 0 with a ≠ 0, a bounded Tonelli-Shanks search runs out, or the final check
    ///   fails: "sqrt_mod_prime: modulus is not prime (...)". Each proves p composite.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// // sqrt(4) mod 7 is 2 or 5
    /// let root = Integer::from(4).sqrt_mod_prime(&Integer::from(7)).unwrap();
    /// assert!(root == Integer::from(2) || root == Integer::from(5));
    /// // sqrt(-1) mod 5 is 2 or 3
    /// let root = Integer::from(-1).sqrt_mod_prime(&Integer::from(5)).unwrap();
    /// assert!(root == Integer::from(2) || root == Integer::from(3));
    /// // -1 is not a square mod 7
    /// assert!(Integer::from(-1).sqrt_mod_prime(&Integer::from(7)).is_err());
    /// ```
    pub fn sqrt_mod_prime(&self, p: &Self) -> Result<Self> {
        use crate::prime::is_probable_prime;

        if !is_probable_prime(p) {
            return Err(MathError::InvalidArgument(
                "Modulus must be prime".to_string(),
            ));
        }
        match sqrt_mod_prime_checked(&self.value, &p.value)? {
            Some(r) => Ok(Integer::new(r)),
            None => Err(MathError::InvalidArgument(
                "Not a quadratic residue".to_string(),
            )),
        }
    }

    /// Square root modulo p without testing p for primality, for callers that already proved
    /// p prime (MAGMA `Modsqrt`, Handbook ch. 18 §18.12.1).
    ///
    /// Precondition: p is prime. `Ok(Some(r))`: 0 <= r < p and r^2 ≡ self (mod p), checked
    /// before it is returned. `Ok(None)`: the Jacobi symbol (self/p) is -1, a proof that self is
    /// not a square mod p (for any odd p). `Err(InvalidArgument)`: p < 2, p even and > 2, p a
    /// perfect square (checked first), or the computation proved p composite (gcd(self, p) > 1
    /// with p not dividing self, the Tonelli-Shanks order bound ran out, or the root did not
    /// square back). For a composite p that is not a square the call still ends: a quadratic
    /// non-residue modulo p (Jacobi symbol -1) exists below p, and the search stops at the first
    /// one; there is no time bound beyond that for composite p. (For a square p every Jacobi
    /// symbol is 0 or 1, so that search would run to p; hence the square check.)
    ///
    /// Algorithm: [Coh93] Algorithm 1.5.1 (Tonelli-Shanks), a^((p+1)/4) for p ≡ 3 (mod 4).
    pub fn try_sqrt_mod_prime(&self, p: &Self) -> Result<Option<Self>> {
        let pv = &p.value;
        let two = BigInt::from(2u32);
        if *pv < two {
            return Err(MathError::InvalidArgument(format!(
                "try_sqrt_mod_prime: p = {p} is below 2"
            )));
        }
        if pv.is_even() && *pv != two {
            return Err(MathError::InvalidArgument(format!(
                "try_sqrt_mod_prime: p = {p} is even and not 2"
            )));
        }
        let r = pv.sqrt();
        if &r * &r == *pv {
            return Err(MathError::InvalidArgument(format!(
                "try_sqrt_mod_prime: p = {p} is a perfect square"
            )));
        }
        Ok(sqrt_mod_prime_checked(&self.value, pv)?.map(Integer::new))
    }

    /// Get all quadratic residues modulo a prime p
    ///
    /// Returns a sorted vector of all quadratic residues in the range [0, p).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let qrs = Integer::quadratic_residues(&Integer::from(7));
    /// // Returns [0, 1, 2, 4] since 0²=0, 1²=1, 2²=4, 3²=2 (mod 7)
    /// ```
    pub fn quadratic_residues(p: &Self) -> Vec<Self> {
        use crate::prime::is_probable_prime;

        if !is_probable_prime(p) || *p <= Integer::one() {
            return vec![];
        }

        let mut residues = std::collections::HashSet::new();
        residues.insert(Integer::zero()); // 0 is always a QR

        // For prime p, there are (p-1)/2 non-zero quadratic residues
        let limit = (p.clone() + Integer::one()) / Integer::from(2);

        for i in 1..limit.to_usize().unwrap_or(1000).min(1000) {
            let x = Integer::from(i as i64);
            let sq = (x.clone() * x.clone()) % p.clone();
            residues.insert(sq);
        }

        let mut result: Vec<Self> = residues.into_iter().collect();
        result.sort();
        result
    }

    /// Modular inverse (MAGMA `Modinv`, Handbook ch. 18 §18.12.1), by the extended Euclidean
    /// algorithm (`BigInt::modinv`).
    ///
    /// For m = |modulus| >= 2 returns `Some(u)` with 1 <= u < m and self*u ≡ 1 (mod m), or
    /// `None` when gcd(self, m) > 1 (no inverse exists). Any sign of `self` and `modulus` is
    /// accepted (PARI/GP `lift(Mod(self, modulus)^-1)`). For m = 1 the result is `Some(0)`
    /// (Z/1Z = {0}).
    ///
    /// `modulus = 0` is not a valid modulus (MAGMA requires m > 1, and PARI/GP raises an
    /// error): the result is `None` for every `self`. This `None` does not say that `self` has
    /// no inverse in Z/0Z = Z (where ±1 are units).
    pub fn mod_inverse(&self, modulus: &Self) -> Option<Self> {
        if modulus.is_zero() {
            return None;
        }
        let m = modulus.value.abs();
        if m.is_one() {
            return Some(Integer::zero());
        }
        self.value.mod_floor(&m).modinv(&m).map(Integer::new)
    }

    /// Least non-negative residue of `self` modulo `|modulus|`, in `[0, |modulus|)`.
    ///
    /// Unlike the `%` operator, which truncates and gives the remainder the sign of `self`,
    /// this always returns a value in `[0, |modulus|)`. (MAGMA `mod` instead gives the result
    /// the sign of the modulus; for `modulus > 0` the two agree.)
    ///
    /// # Panics
    ///
    /// Panics if `modulus` is zero (division by zero).
    pub fn modulo(&self, modulus: &Self) -> Self {
        let result = self.clone() % modulus.clone();
        if result.signum() < 0 {
            result + modulus.abs()
        } else {
            result
        }
    }

    /// Alias for mod_pow for compatibility
    pub fn modpow(&self, exp: &Self, modulus: &Self) -> Result<Self> {
        self.mod_pow(exp, modulus)
    }

    /// MAGMA `IsPrime(n)` (`Proof := true`): [`crate::prime::is_prime`], a proof for every size
    /// (the deterministic word test below 2^64, the twelve strong tests below psi_12, then
    /// Baillie-PSW and ECPP). It panics when the ECPP proof of a probable prime is unresolved
    /// (see [`crate::prime::try_is_prime`] for the rates and the `Result` form).
    pub fn is_prime(&self) -> bool {
        use crate::prime::is_prime;
        is_prime(self)
    }

    /// Generalised binomial coefficient C(self, k) reduced modulo a prime p, in `[0, p)`
    /// (MAGMA `Binomial(n, r)` mod p; PARI/GP `binomial(n, k) % p`).
    ///
    /// - k < 0: C(n, k) = 0.
    /// - n < 0: C(n, k) = (-1)^k C(k-n-1, k) (Graham-Knuth-Patashnik eq. (5.14)).
    /// - 0 <= n < k: 0.
    /// - Otherwise Lucas' theorem on the base-p digits (n_i, k_i): C(n, k) ≡ prod C(n_i, k_i),
    ///   each factor computed as prod_{i<k'} (n_i - i) / k'! mod p with k' = min(k_i, n_i - k_i).
    ///
    /// Errors: p proven not prime by [`crate::prime::try_is_prime`] (`InvalidArgument`); p a
    /// probable prime whose ECPP proof is unresolved (`NotSupported`, from `try_is_prime`); a
    /// Lucas digit k' that does not fit in a `u64` (only possible for p > 2^64) gives
    /// `NotSupported`.
    ///
    /// Lucas' theorem does not hold for a composite modulus, so a probable prime is not enough:
    /// p is proven prime first (a proof for every size; above psi_12 one ECPP proof per process
    /// and prime, then cached).
    pub fn binomial_mod(&self, k: &Self, p: &Self) -> Result<Self> {
        if !crate::prime::try_is_prime(p)? {
            return Err(MathError::InvalidArgument(
                "Modulus must be prime".to_string(),
            ));
        }
        let p = &p.value;

        if k.value.is_negative() {
            return Ok(Integer::zero());
        }
        // C(n, k) = (-1)^k C(k - n - 1, k) moves a negative upper index to k - n - 1 >= k.
        let (mut n, negate) = if self.value.is_negative() {
            (&k.value - &self.value - 1u32, k.value.is_odd())
        } else {
            (self.value.clone(), false)
        };
        if k.value > n {
            return Ok(Integer::zero());
        }

        let mut k = k.value.clone();
        let mut result = BigInt::one();
        while !k.is_zero() {
            let (n_next, n_i) = n.div_mod_floor(p);
            let (k_next, k_i) = k.div_mod_floor(p);
            if k_i > n_i {
                return Ok(Integer::zero());
            }
            // C(n_i, k_i) mod p with 0 <= k_i <= n_i < p, by the product formula.
            let k_small = std::cmp::min(k_i.clone(), &n_i - &k_i);
            let k_small = u64::try_from(&k_small).map_err(|_| {
                MathError::NotSupported(
                    "binomial_mod: Lucas digit does not fit in a u64".to_string(),
                )
            })?;
            let mut num = BigInt::one();
            let mut den = BigInt::one();
            for i in 0..k_small {
                num = (num * (&n_i - i)).mod_floor(p);
                den = (den * (i + 1)).mod_floor(p);
            }
            // den is a product of 1..=k_small with k_small < p, hence a unit mod p.
            let den_inv = den.modinv(p).ok_or(MathError::NotInvertible)?;
            result = (result * num * den_inv).mod_floor(p);
            n = n_next;
            k = k_next;
        }
        if negate {
            result = (-result).mod_floor(p);
        }
        Ok(Integer::new(result))
    }
}

/// `base^exp mod m` in `[0, m)` for `exp >= 0` and `m >= 2`.
///
/// `BigInt::modpow` (Montgomery multiplication for odd m; it reduces like `mod_floor`) has a
/// fixed set-up cost that dominates for short exponents, such as the squarings in
/// Miller-Rabin, so exponents of at most 32 bits use right-to-left binary exponentiation
/// ([Knu97] §4.6.3) with `mod_floor` reductions instead. Both give the same residue.
fn mod_pow_nonneg(base: &BigInt, exp: &BigInt, m: &BigInt) -> BigInt {
    debug_assert!(!exp.is_negative() && *m > BigInt::one());
    if exp.bits() > 32 {
        return base.modpow(exp, m);
    }
    let mut e = u32::try_from(exp).expect("exponent has at most 32 bits");
    let mut b = base.mod_floor(m);
    let mut result = BigInt::one();
    while e > 0 {
        if e & 1 == 1 {
            result = (&result * &b).mod_floor(m);
        }
        e >>= 1;
        if e > 0 {
            b = (&b * &b).mod_floor(m);
        }
    }
    result
}

/// Kronecker symbol (a/b) on `BigInt`s: [Coh93] Algorithm 1.4.10.
///
/// Steps: (a/0); strip the power of 2 from b with the rule (a/2); make b positive with the rule
/// (a/-1); then the binary Jacobi loop with b odd and positive. The reciprocity step is also
/// valid on the first pass, where a may be negative and odd:
/// (a/b) = (-1)^(((a-1)/2)((b-1)/2)) (b/|a|), where a ≡ 3 (mod 4) is tested with `mod_floor`.
pub(crate) fn kronecker_bigint(a: &BigInt, b: &BigInt) -> i8 {
    if b.is_zero() {
        return if a.abs().is_one() { 1 } else { 0 };
    }
    if a.is_even() && b.is_even() {
        return 0;
    }
    let mut a = a.clone();
    let mut b = b.clone();
    let mut k: i8 = 1;
    let (three, four, five, eight) = (
        BigInt::from(3u32),
        BigInt::from(4u32),
        BigInt::from(5u32),
        BigInt::from(8u32),
    );
    // (a/2)^v with a odd (a even and b even was handled above).
    let v = b.trailing_zeros().expect("b is non-zero");
    if v > 0 {
        b >>= v; // exact: b has exactly v trailing zero bits
        if v % 2 == 1 {
            let a8 = a.mod_floor(&eight);
            if a8 == three || a8 == five {
                k = -k;
            }
        }
    }
    // (a/-1) = -1 iff a < 0.
    if b.is_negative() {
        b = -b;
        if a.is_negative() {
            k = -k;
        }
    }
    // Invariant: b odd and positive.
    loop {
        if a.is_zero() {
            return if b.is_one() { k } else { 0 };
        }
        let v = a.trailing_zeros().expect("a is non-zero");
        if v > 0 {
            a >>= v; // exact
            if v % 2 == 1 {
                let b8 = b.mod_floor(&eight);
                if b8 == three || b8 == five {
                    k = -k;
                }
            }
        }
        if a.mod_floor(&four) == three && b.mod_floor(&four) == three {
            k = -k;
        }
        let r = a.abs();
        a = b.mod_floor(&r);
        b = r;
    }
}

/// Square root of `a` modulo `p`, where `p` is 2 or an odd non-square (a probable prime in
/// every caller: [`crate::prime::is_probable_prime`], a proven prime, or a BPSW probable prime
/// inside the ECPP prover). Returns `Ok(Some(r))` with 0 <= r < p and r^2 ≡ a (mod p),
/// `Ok(None)` when (a/p) = -1 (a proven non-square), and `Err` when the computation proves p
/// composite.
///
/// [Coh93] Algorithm 1.5.1 (Tonelli-Shanks) for p ≡ 1 (mod 4), a^((p+1)/4) for p ≡ 3 (mod 4).
/// Both bounded searches (least non-residue n, least m with b^(2^m) = 1) end in `Err` when they
/// run out, which can happen only for composite p, never in a decision.
pub(crate) fn sqrt_mod_prime_checked(a: &BigInt, p: &BigInt) -> Result<Option<BigInt>> {
    let not_prime = |why: &str| -> Result<Option<BigInt>> {
        Err(MathError::InvalidArgument(format!(
            "sqrt_mod_prime: modulus is not prime ({why})"
        )))
    };
    let a = a.mod_floor(p);
    let two = BigInt::from(2u32);
    if *p == two || a.is_zero() {
        return Ok(Some(a));
    }
    match kronecker_bigint(&a, p) {
        -1 => return Ok(None),
        0 => return not_prime("gcd(a, p) > 1"),
        _ => {}
    }

    let x = if p.mod_floor(&BigInt::from(4u32)) == BigInt::from(3u32) {
        mod_pow_nonneg(&a, &((p + 1u32) >> 2), p)
    } else {
        // p - 1 = 2^e q with q odd.
        let p_minus_1 = p - 1u32;
        let e = p_minus_1.trailing_zeros().expect("p > 2");
        let q = &p_minus_1 >> e;
        // Least n >= 2 with (n/p) = -1; for a prime p it is below p.
        let mut n = two.clone();
        while kronecker_bigint(&n, p) != -1 {
            n += 1u32;
            if n >= *p {
                return not_prime("no quadratic non-residue below p");
            }
        }
        let mut y = mod_pow_nonneg(&n, &q, p);
        let mut r = e;
        let mut x = mod_pow_nonneg(&a, &((&q - 1u32) >> 1), p);
        let mut b = (&a * &x * &x).mod_floor(p);
        x = (&a * &x).mod_floor(p);
        loop {
            if b.is_one() {
                break x;
            }
            // Least m with b^(2^m) = 1; for a prime p and a square a it is below r.
            let mut m = 0u64;
            let mut t = b.clone();
            while !t.is_one() {
                t = (&t * &t).mod_floor(p);
                m += 1;
                if m >= r {
                    return not_prime("Tonelli-Shanks order bound exceeded");
                }
            }
            let mut t = y.clone();
            for _ in 0..(r - m - 1) {
                t = (&t * &t).mod_floor(p);
            }
            y = (&t * &t).mod_floor(p);
            r = m;
            x = (&x * &t).mod_floor(p);
            b = (&b * &y).mod_floor(p);
        }
    };
    if (&x * &x).mod_floor(p) != a {
        return not_prime("verification failed");
    }
    Ok(Some(x))
}

// Implement From traits for convenient construction
impl From<i64> for Integer {
    fn from(n: i64) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<u64> for Integer {
    fn from(n: u64) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<i32> for Integer {
    fn from(n: i32) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<u32> for Integer {
    fn from(n: u32) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<BigInt> for Integer {
    fn from(n: BigInt) -> Self {
        Integer::new(n)
    }
}

/// Checked conversion to a primitive integer: `Err(NumericalError)` when the value is outside
/// the target range. The error names the bit length and the sign, never the decimal value
/// (which can be huge). Delegates to num-bigint's `TryFrom<&BigInt>`.
macro_rules! impl_try_from_integer {
    ($($t:ty),*) => {$(
        impl TryFrom<&Integer> for $t {
            type Error = MathError;
            fn try_from(n: &Integer) -> Result<$t> {
                <$t>::try_from(&n.value).map_err(|_| {
                    MathError::NumericalError(format!(
                        "an integer of {} bits{} does not fit in {}",
                        n.value.bits(),
                        if n.value.is_negative() { " (negative)" } else { "" },
                        stringify!($t)
                    ))
                })
            }
        }

        impl TryFrom<Integer> for $t {
            type Error = MathError;
            fn try_from(n: Integer) -> Result<$t> {
                <$t>::try_from(&n)
            }
        }
    )*};
}

impl_try_from_integer!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

// Display implementation
impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Debug for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Integer({})", self.value)
    }
}

// Typesetting implementation
impl rustmath_typesetting::MathDisplay for Integer {
    fn math_format(&self, options: &rustmath_typesetting::FormatOptions) -> String {
        use rustmath_typesetting::OutputFormat;

        let value_str = self.value.to_string();

        match options.format {
            OutputFormat::LaTeX => {
                // For large numbers, might want to add formatting
                if let Some(max_width) = options.max_width {
                    if value_str.len() > max_width {
                        return rustmath_typesetting::utils::truncate(&value_str, max_width);
                    }
                }
                value_str
            }
            OutputFormat::Html => {
                // Wrap in MathML number element
                rustmath_typesetting::html::number(&value_str)
            }
            OutputFormat::Unicode | OutputFormat::Ascii | OutputFormat::Plain => value_str,
        }
    }

    fn precedence(&self) -> i32 {
        rustmath_typesetting::utils::precedence::ATOMIC
    }
}

// Arithmetic operations
impl Add for Integer {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Integer::new(self.value + other.value)
    }
}

impl<'b> Add<&'b Integer> for &Integer {
    type Output = Integer;

    fn add(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value + &other.value)
    }
}

impl Sub for Integer {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Integer::new(self.value - other.value)
    }
}

impl<'b> Sub<&'b Integer> for &Integer {
    type Output = Integer;

    fn sub(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value - &other.value)
    }
}

impl Mul for Integer {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Integer::new(self.value * other.value)
    }
}

impl<'b> Mul<&'b Integer> for &Integer {
    type Output = Integer;

    fn mul(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value * &other.value)
    }
}

impl Div for Integer {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Integer::new(self.value / other.value)
    }
}

impl<'b> Div<&'b Integer> for &Integer {
    type Output = Integer;

    fn div(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value / &other.value)
    }
}

/// Truncated remainder (num-bigint `%`): `a % m` has the sign of `a`, so `-1 % 7 == -1`.
/// MAGMA `mod` is floored instead (Handbook ch. 18 §18.4.1, §18.12.1). For a least
/// non-negative residue use [`Integer::modulo`].
impl Rem for Integer {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        Integer::new(self.value % other.value)
    }
}

impl<'b> Rem<&'b Integer> for &Integer {
    type Output = Integer;

    fn rem(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value % &other.value)
    }
}

impl Neg for Integer {
    type Output = Self;

    fn neg(self) -> Self {
        Integer::new(-self.value)
    }
}

impl Neg for &Integer {
    type Output = Integer;

    fn neg(self) -> Integer {
        Integer::new(-&self.value)
    }
}

// Ring trait implementation
impl Ring for Integer {
    fn zero() -> Self {
        Integer::new(BigInt::zero())
    }

    fn one() -> Self {
        Integer::new(BigInt::one())
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value == BigInt::one()
    }

    fn pow(&self, n: u32) -> Self {
        Integer::new(num_traits::pow(self.value.clone(), n as usize))
    }

    // Element predicates (MAGMA Ch. 17 §17.5.5 `IsUnit`/`IsNilpotent`, Ch. 19 `Order`):
    // Z is a domain whose unit group is {±1} (GAP: IsUnit(Integers, 2) = false,
    // Order(-1) = 2).

    fn unit_status(&self) -> Option<bool> {
        Some(self.value.abs().is_one())
    }

    fn nilpotent_status(&self) -> Option<bool> {
        Some(self.value.is_zero())
    }

    fn unit_inverse(&self) -> Result<Self> {
        if self.value.abs().is_one() {
            Ok(self.clone())
        } else {
            Err(MathError::NotInvertible)
        }
    }

    fn multiplicative_order_status(&self) -> rustmath_core::MultiplicativeOrder {
        use rustmath_core::MultiplicativeOrder;
        if self.value.is_one() {
            MultiplicativeOrder::Finite(1)
        } else if self.value.abs().is_one() {
            MultiplicativeOrder::Finite(2)
        } else {
            MultiplicativeOrder::NotAUnit
        }
    }

    fn static_characteristic() -> Option<u64> {
        Some(0)
    }

    fn static_is_commutative() -> Option<bool> {
        Some(true)
    }
}

impl CommutativeRing for Integer {}
impl IntegralDomain for Integer {}

// Z with its usual total order: translation-invariant, and products of
// non-negative integers are non-negative, as `OrderedRing` requires.
// (Path-qualified per the core naming policy: `ordering` is not root-re-exported.)
impl rustmath_core::ordering::OrderedRing for Integer {
    fn sign(&self) -> i32 {
        self.signum() as i32
    }

    fn abs(&self) -> Self {
        Integer::abs(self)
    }
}

impl EuclideanDomain for Integer {
    fn norm(&self) -> u64 {
        // For integers, use absolute value as norm
        // This is a simplification; true norm could be arbitrarily large
        self.abs().to_u64().unwrap_or(u64::MAX)
    }

    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        let q = &self.value / &other.value;
        let r = &self.value % &other.value;

        Ok((Integer::new(q), Integer::new(r)))
    }

    /// MAGMA `Gcd(m, n)` (Handbook ch. 18 §18.6): non-negative; `|m|` when `n = 0`; `0` for
    /// `(0, 0)`. Lehmer's algorithm on `|m|`, `|n|` ([Knu97] §4.5.2 Algorithm L, module
    /// `lehmer`), so the result is non-negative by construction. (The trait default runs
    /// Euclid over the truncating `div_rem`, whose last remainder is `±gcd`.) Lehmer keeps
    /// Euclid's single division when one operand is small, e.g. `gcd(num, 1)` in
    /// `Rational::new`, where binary gcd (Stein) would be far slower; on balanced operands it
    /// is about 10x (1,000 bits) to 60x (10^6 bits) faster than Euclid.
    fn gcd(&self, other: &Self) -> Self {
        Integer::new(BigInt::from(crate::lehmer::gcd(
            self.value.magnitude(),
            other.value.magnitude(),
        )))
    }

    /// MAGMA `Lcm(m, n)` (Handbook ch. 18 §18.6): `|m| / gcd(m, n) · |n|`, so
    /// `lcm · gcd = |m · n|`; `0` when `m` or `n` is `0`.
    fn lcm(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Integer::zero();
        }
        let g = EuclideanDomain::gcd(self, other);
        Integer::new(BigInt::from(
            self.value.magnitude() / g.value.magnitude() * other.value.magnitude(),
        ))
    }

    /// MAGMA `Xgcd(m, n)` (Handbook ch. 18 §18.6): `(g, x, y)` with `g = x·m + y·n`,
    /// `g >= 0`, and `g = 0` only for `(0, 0)`, which gives `(0, 0, 0)` as PARI/GP
    /// `gcdext(0, 0)` and GMP `mpz_gcdext` do.
    ///
    /// The multipliers are those of the truncated-quotient extended Euclid on the signed
    /// `(m, n)` ([Knu97] §4.5.2, as `num_integer::Integer::extended_gcd`), with `(g, x, y)`
    /// negated when `g < 0`. A truncated remainder has the sign of its dividend, so that
    /// triple is `(gcd, sign(m)·s, sign(n)·t)` with `(s, t)` the Euclid cofactors of the
    /// magnitudes. Those come from Lehmer's algorithm (module `lehmer`), which applies
    /// exactly Euclid's quotients. The triples equal PARI/GP `gcdext` (see
    /// `tests/ch18_signs_roots.rs`).
    fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        if self.is_zero() && other.is_zero() {
            return (Integer::zero(), Integer::zero(), Integer::zero());
        }
        let (g, s, t) =
            crate::lehmer::xgcd_magnitudes(self.value.magnitude(), other.value.magnitude());
        let s = if self.value.is_negative() { -s } else { s };
        let t = if other.value.is_negative() { -t } else { t };
        (Integer::new(BigInt::from(g)), Integer::new(s), Integer::new(t))
    }

    /// The sign: MAGMA `Normalize` over Z (Handbook ch. 23 §23.5.1) takes the associate with
    /// positive leading coefficient, so the canonical associate of `n` is `|n|` and
    /// `n = sign(n) · |n|`; `0` gives `1`.
    fn normalization_unit(&self) -> Option<Self> {
        Some(if self.signum() < 0 {
            -Integer::one()
        } else {
            Integer::one()
        })
    }
}

impl NumericConversion for Integer {
    fn from_i64(n: i64) -> Self {
        Integer::from(n)
    }

    fn from_u64(n: u64) -> Self {
        Integer::from(n)
    }

    fn to_i64(&self) -> Option<i64> {
        use num_traits::ToPrimitive;
        self.value.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        use num_traits::ToPrimitive;
        self.value.to_u64()
    }

    fn to_usize(&self) -> Option<usize> {
        use num_traits::ToPrimitive;
        self.value.to_usize()
    }

    fn to_f64(&self) -> Option<f64> {
        use num_traits::ToPrimitive;
        self.value.to_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = Integer::from(42);
        let b = Integer::from(17);

        assert_eq!(a.clone() + b.clone(), Integer::from(59));
        assert_eq!(a.clone() - b.clone(), Integer::from(25));
        assert_eq!(a.clone() * b.clone(), Integer::from(714));
        assert_eq!(a.clone() / b.clone(), Integer::from(2));
        assert_eq!(a.clone() % b.clone(), Integer::from(8));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(Integer::from(48).gcd(&Integer::from(18)), Integer::from(6));
        assert_eq!(Integer::from(100).gcd(&Integer::from(35)), Integer::from(5));
    }

    #[test]
    fn test_lcm() {
        assert_eq!(Integer::from(12).lcm(&Integer::from(18)), Integer::from(36));
        assert_eq!(Integer::from(21).lcm(&Integer::from(6)), Integer::from(42));
    }

    #[test]
    fn test_extended_gcd() {
        let a = Integer::from(240);
        let b = Integer::from(46);
        let (gcd, s, t) = a.extended_gcd(&b);

        assert_eq!(gcd, Integer::from(2));
        assert_eq!(&a * &s + &b * &t, gcd);
    }

    #[test]
    fn test_mod_pow() {
        let base = Integer::from(2);
        let exp = Integer::from(10);
        let modulus = Integer::from(1000);

        let result = base.mod_pow(&exp, &modulus).unwrap();
        assert_eq!(result, Integer::from(24));
    }

    #[test]
    fn test_pow() {
        assert_eq!(Integer::from(2).pow(10), Integer::from(1024));
        assert_eq!(Integer::from(3).pow(5), Integer::from(243));
    }

    #[test]
    fn test_large_numbers() {
        // Test with numbers larger than 64-bit
        let a = Integer::from(1u64 << 63);
        let b = Integer::from(1u64 << 62);
        let c = a.clone() + b.clone();

        assert!(c > a);
        assert!(c > b);
    }

    #[test]
    fn test_to_i64_is_checked() {
        // gp: 2^63 = 9223372036854775808 > i64::MAX; -2^63-1 < i64::MIN. Both panicked before.
        let two = Integer::from(2);
        assert_eq!(Integer::from(i64::MIN).to_i64(), Some(i64::MIN));
        assert_eq!(Integer::from(i64::MAX).to_i64(), Some(i64::MAX));
        assert_eq!(two.pow(63).to_i64(), None);
        assert_eq!((-two.pow(63) - Integer::one()).to_i64(), None);
        assert_eq!(Integer::from(-7).to_i64(), Some(-7));
    }

    #[test]
    fn test_try_from_integer() {
        // Ranges: i64 is [-2^63, 2^63 - 1] (gp: -2^63 -> -9223372036854775808), u64 is
        // [0, 2^64 - 1], u128 is [0, 2^128 - 1], i8 is [-128, 127].
        let two = Integer::from(2);
        let min64 = -two.pow(63);
        assert_eq!(i64::try_from(&min64), Ok(i64::MIN));
        assert_eq!(i64::try_from(min64.clone()), Ok(i64::MIN));
        assert!(matches!(
            i64::try_from(&(min64 - Integer::one())),
            Err(MathError::NumericalError(_))
        ));
        assert!(matches!(
            i64::try_from(&two.pow(63)),
            Err(MathError::NumericalError(_))
        ));
        match u64::try_from(&Integer::from(-1)) {
            Err(MathError::NumericalError(msg)) => {
                assert_eq!(msg, "an integer of 1 bits (negative) does not fit in u64");
            }
            other => panic!("u64::try_from(-1) = {other:?}"),
        }
        match u32::try_from(&two.pow(100)) {
            Err(MathError::NumericalError(msg)) => {
                assert_eq!(msg, "an integer of 101 bits does not fit in u32");
            }
            other => panic!("u32::try_from(2^100) = {other:?}"),
        }
        assert_eq!(u128::try_from(&two.pow(127)), Ok(1u128 << 127));
        assert!(u128::try_from(&two.pow(128)).is_err());
        assert_eq!(i128::try_from(&-two.pow(127)), Ok(i128::MIN));
        assert_eq!(usize::try_from(&Integer::from(usize::MAX as u64)), Ok(usize::MAX));
        assert!(usize::try_from(&Integer::from(-1)).is_err());
        assert_eq!(i8::try_from(&Integer::from(-128)), Ok(i8::MIN));
        assert!(i8::try_from(&Integer::from(128)).is_err());
        assert_eq!(u8::try_from(&Integer::from(255)), Ok(255u8));
        assert!(u8::try_from(&Integer::from(256)).is_err());
        assert_eq!(i16::try_from(&Integer::from(-32768)), Ok(i16::MIN));
        assert_eq!(isize::try_from(&Integer::from(-5)), Ok(-5isize));
        assert_eq!(u16::try_from(Integer::from(65535)), Ok(u16::MAX));
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(Integer::from(0).sqrt().unwrap(), Integer::from(0));
        assert_eq!(Integer::from(1).sqrt().unwrap(), Integer::from(1));
        assert_eq!(Integer::from(4).sqrt().unwrap(), Integer::from(2));
        assert_eq!(Integer::from(9).sqrt().unwrap(), Integer::from(3));
        assert_eq!(Integer::from(16).sqrt().unwrap(), Integer::from(4));
        assert_eq!(Integer::from(100).sqrt().unwrap(), Integer::from(10));

        // Non-perfect squares (floor)
        assert_eq!(Integer::from(8).sqrt().unwrap(), Integer::from(2));
        assert_eq!(Integer::from(15).sqrt().unwrap(), Integer::from(3));
        assert_eq!(Integer::from(99).sqrt().unwrap(), Integer::from(9));

        // Negative numbers should error
        assert!(Integer::from(-4).sqrt().is_err());
    }

    #[test]
    fn test_nth_root() {
        // Square roots
        assert_eq!(Integer::from(16).nth_root(2).unwrap(), Integer::from(4));
        assert_eq!(Integer::from(25).nth_root(2).unwrap(), Integer::from(5));

        // Cube roots
        assert_eq!(Integer::from(8).nth_root(3).unwrap(), Integer::from(2));
        assert_eq!(Integer::from(27).nth_root(3).unwrap(), Integer::from(3));
        assert_eq!(Integer::from(64).nth_root(3).unwrap(), Integer::from(4));

        // Fourth root
        assert_eq!(Integer::from(16).nth_root(4).unwrap(), Integer::from(2));
        assert_eq!(Integer::from(81).nth_root(4).unwrap(), Integer::from(3));

        // Edge cases
        assert_eq!(Integer::from(1).nth_root(5).unwrap(), Integer::from(1));
        assert!(Integer::from(0).nth_root(0).is_err());
    }

    #[test]
    fn test_divisors() {
        // 12 = 2^2 * 3, divisors: 1, 2, 3, 4, 6, 12
        let divs = Integer::from(12).divisors().unwrap();
        assert_eq!(divs, vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
            Integer::from(6),
            Integer::from(12)
        ]);

        // Prime number has only 1 and itself
        let divs = Integer::from(7).divisors().unwrap();
        assert_eq!(divs, vec![Integer::from(1), Integer::from(7)]);

        // 1 has only itself
        let divs = Integer::from(1).divisors().unwrap();
        assert_eq!(divs, vec![Integer::from(1)]);

        // 0 should error
        assert!(Integer::from(0).divisors().is_err());
    }

    #[test]
    fn test_num_divisors() {
        assert_eq!(Integer::from(1).num_divisors().unwrap(), Integer::from(1));
        assert_eq!(Integer::from(12).num_divisors().unwrap(), Integer::from(6)); // 1,2,3,4,6,12
        assert_eq!(Integer::from(7).num_divisors().unwrap(), Integer::from(2)); // 1,7
        assert_eq!(Integer::from(28).num_divisors().unwrap(), Integer::from(6)); // 1,2,4,7,14,28
    }

    #[test]
    fn test_sum_divisors() {
        // σ(1) = 1
        assert_eq!(Integer::from(1).sum_divisors().unwrap(), Integer::from(1));

        // σ(12) = 1 + 2 + 3 + 4 + 6 + 12 = 28
        assert_eq!(Integer::from(12).sum_divisors().unwrap(), Integer::from(28));

        // σ(7) = 1 + 7 = 8 (prime)
        assert_eq!(Integer::from(7).sum_divisors().unwrap(), Integer::from(8));
    }

    #[test]
    fn test_sigma_function() {
        // σ₀(12) = number of divisors = 6
        assert_eq!(Integer::from(12).sigma(0).unwrap(), Integer::from(6));

        // σ₁(12) = sum of divisors = 28
        assert_eq!(Integer::from(12).sigma(1).unwrap(), Integer::from(28));

        // σ₂(12) = 1² + 2² + 3² + 4² + 6² + 12² = 1 + 4 + 9 + 16 + 36 + 144 = 210
        assert_eq!(Integer::from(12).sigma(2).unwrap(), Integer::from(210));

        // σ₂(6) = 1² + 2² + 3² + 6² = 1 + 4 + 9 + 36 = 50
        assert_eq!(Integer::from(6).sigma(2).unwrap(), Integer::from(50));
    }

    /// `try_sqrt_mod_prime` runs no primality test but refuses squares first (review 1 m5: for
    /// a square p every Jacobi symbol is 0 or 1, so the non-residue search would run to p).
    /// gp: isprime(1000003) = 1, kronecker(2, 15) = 1, Mod(2, 15)^4 = 1 (15 = 3 mod 4, so the
    /// candidate root 2^((15+1)/4) = 1 does not square back to 2).
    #[test]
    fn try_sqrt_mod_prime_refuses_squares() {
        let start = std::time::Instant::now();
        let sq = Integer::from(1_000_003u32).pow(2);
        assert!(Integer::from(2).try_sqrt_mod_prime(&sq).is_err());
        let m61 = Integer::from((1u64 << 61) - 1);
        assert!(Integer::from(3).try_sqrt_mod_prime(&(m61.clone() * m61)).is_err());
        assert!(start.elapsed().as_secs() < 5, "the square check comes first");
        assert_eq!(
            Integer::from(4).try_sqrt_mod_prime(&Integer::from(7)),
            Ok(Some(Integer::from(2)))
        );
        assert_eq!(Integer::from(3).try_sqrt_mod_prime(&Integer::from(7)), Ok(None));
        assert!(Integer::from(2).try_sqrt_mod_prime(&Integer::from(15)).is_err());
        assert_eq!(
            Integer::from(5).try_sqrt_mod_prime(&Integer::from(2)),
            Ok(Some(Integer::one()))
        );
        assert!(Integer::from(5).try_sqrt_mod_prime(&Integer::from(1)).is_err());
        assert!(Integer::from(5).try_sqrt_mod_prime(&Integer::from(10)).is_err());
    }

    #[test]
    fn test_sigma_values_pari() {
        // gp: sigma(2^10*3^5,3) -> 18285256324908244
        let n = Integer::from(2).pow(10) * Integer::from(3).pow(5);
        assert_eq!(n.sigma(3).unwrap(), Integer::from(18_285_256_324_908_244u64));
        // gp: s=sigma(7^40,5); [s % (2^127-1), #binary(s)]
        //     -> [108698265444746923986228897305478117662, 562]
        let s = Integer::from(7).pow(40).sigma(5).unwrap();
        let m127 = Integer::from(2).pow(127) - Integer::one();
        assert_eq!(
            s.modulo(&m127),
            Integer::from_decimal_str("108698265444746923986228897305478117662").unwrap()
        );
        assert_eq!(s.bit_length(), 562);
    }

    #[test]
    fn test_is_power() {
        // Perfect squares
        assert!(Integer::from(4).is_power(2));
        assert!(Integer::from(9).is_power(2));
        assert!(Integer::from(16).is_power(2));

        // Perfect cubes
        assert!(Integer::from(8).is_power(3));
        assert!(Integer::from(27).is_power(3));

        // Not perfect powers
        assert!(!Integer::from(5).is_power(2));
        assert!(!Integer::from(10).is_power(3));

        // Edge cases
        assert!(Integer::from(1).is_power(5));
        assert!(!Integer::from(2).is_power(0));
        assert!(Integer::from(100).is_power(1));
    }

    #[test]
    fn test_digits() {
        // Binary
        assert_eq!(Integer::from(5).digits(2).unwrap(), vec![1, 0, 1]); // 101
        assert_eq!(Integer::from(8).digits(2).unwrap(), vec![0, 0, 0, 1]); // 1000

        // Decimal
        assert_eq!(Integer::from(123).digits(10).unwrap(), vec![3, 2, 1]);
        assert_eq!(Integer::from(0).digits(10).unwrap(), vec![0]);

        // Hexadecimal
        assert_eq!(Integer::from(255).digits(16).unwrap(), vec![15, 15]); // FF
    }

    #[test]
    fn test_euler_phi() {
        // φ(1) = 1
        assert_eq!(Integer::from(1).euler_phi().unwrap(), Integer::from(1));

        // φ(p) = p - 1 for prime p
        assert_eq!(Integer::from(7).euler_phi().unwrap(), Integer::from(6));
        assert_eq!(Integer::from(11).euler_phi().unwrap(), Integer::from(10));

        // φ(12) = φ(2^2 * 3) = 12 * (1 - 1/2) * (1 - 1/3) = 12 * 1/2 * 2/3 = 4
        assert_eq!(Integer::from(12).euler_phi().unwrap(), Integer::from(4));

        // φ(9) = φ(3^2) = 9 * (1 - 1/3) = 6
        assert_eq!(Integer::from(9).euler_phi().unwrap(), Integer::from(6));
    }

    #[test]
    fn test_moebius() {
        // μ(1) = 1
        assert_eq!(Integer::from(1).moebius().unwrap(), 1);

        // μ(p) = -1 for prime p
        assert_eq!(Integer::from(2).moebius().unwrap(), -1);
        assert_eq!(Integer::from(3).moebius().unwrap(), -1);
        assert_eq!(Integer::from(5).moebius().unwrap(), -1);

        // μ(6) = μ(2*3) = 1 (two distinct primes)
        assert_eq!(Integer::from(6).moebius().unwrap(), 1);

        // μ(30) = μ(2*3*5) = -1 (three distinct primes)
        assert_eq!(Integer::from(30).moebius().unwrap(), -1);

        // μ(4) = μ(2^2) = 0 (has squared factor)
        assert_eq!(Integer::from(4).moebius().unwrap(), 0);
        assert_eq!(Integer::from(8).moebius().unwrap(), 0);
        assert_eq!(Integer::from(12).moebius().unwrap(), 0);
    }

    #[test]
    fn test_is_square_free() {
        assert!(Integer::from(1).is_square_free());
        assert!(Integer::from(2).is_square_free());
        assert!(Integer::from(3).is_square_free());
        assert!(Integer::from(6).is_square_free()); // 2*3
        assert!(Integer::from(30).is_square_free()); // 2*3*5

        assert!(!Integer::from(4).is_square_free()); // 2^2
        assert!(!Integer::from(8).is_square_free()); // 2^3
        assert!(!Integer::from(12).is_square_free()); // 2^2 * 3
        assert!(!Integer::from(0).is_square_free());
    }

    #[test]
    fn test_radical() {
        // rad(1) = 1
        assert_eq!(Integer::from(1).radical().unwrap(), Integer::from(1));

        // rad(p) = p for prime p
        assert_eq!(Integer::from(7).radical().unwrap(), Integer::from(7));

        // rad(12) = rad(2^2 * 3) = 2 * 3 = 6
        assert_eq!(Integer::from(12).radical().unwrap(), Integer::from(6));

        // rad(30) = rad(2 * 3 * 5) = 30
        assert_eq!(Integer::from(30).radical().unwrap(), Integer::from(30));

        // rad(16) = rad(2^4) = 2
        assert_eq!(Integer::from(16).radical().unwrap(), Integer::from(2));
    }

    #[test]
    fn test_is_perfect_square() {
        assert!(Integer::from(0).is_perfect_square());
        assert!(Integer::from(1).is_perfect_square());
        assert!(Integer::from(4).is_perfect_square());
        assert!(Integer::from(9).is_perfect_square());
        assert!(Integer::from(16).is_perfect_square());
        assert!(Integer::from(100).is_perfect_square());

        assert!(!Integer::from(2).is_perfect_square());
        assert!(!Integer::from(3).is_perfect_square());
        assert!(!Integer::from(5).is_perfect_square());
        assert!(!Integer::from(99).is_perfect_square());
        assert!(!Integer::from(-4).is_perfect_square());
    }

    #[test]
    fn test_legendre_symbol() {
        // (2/7) = 1 (2 is a QR mod 7, since 3² = 9 ≡ 2 (mod 7))
        assert_eq!(Integer::from(2).legendre_symbol(&Integer::from(7)).unwrap(), 1);

        // (3/7) = -1 (3 is not a QR mod 7)
        assert_eq!(Integer::from(3).legendre_symbol(&Integer::from(7)).unwrap(), -1);

        // (0/7) = 0
        assert_eq!(Integer::from(0).legendre_symbol(&Integer::from(7)).unwrap(), 0);

        // (1/p) = 1 for any prime p
        assert_eq!(Integer::from(1).legendre_symbol(&Integer::from(11)).unwrap(), 1);
    }

    #[test]
    fn test_jacobi_symbol() {
        // (2/15) where 15 = 3 * 5
        let result = Integer::from(2).jacobi_symbol(&Integer::from(15)).unwrap();
        // (2/3) = -1, (2/5) = -1, so (2/15) = 1
        assert_eq!(result, 1);

        // (1/n) = 1 for any odd n
        assert_eq!(Integer::from(1).jacobi_symbol(&Integer::from(21)).unwrap(), 1);

        // (0/n) = 0 if n > 1
        assert_eq!(Integer::from(0).jacobi_symbol(&Integer::from(15)).unwrap(), 0);
    }

    #[test]
    fn test_binomial_mod() {
        // C(5, 2) mod 3 = 10 mod 3 = 1
        assert_eq!(
            Integer::from(5).binomial_mod(&Integer::from(2), &Integer::from(3)).unwrap(),
            Integer::from(1)
        );

        // C(10, 5) mod 7 = 252 mod 7 = 0
        assert_eq!(
            Integer::from(10).binomial_mod(&Integer::from(5), &Integer::from(7)).unwrap(),
            Integer::from(0)
        );

        // C(p, k) mod p = 0 for 0 < k < p
        assert_eq!(
            Integer::from(7).binomial_mod(&Integer::from(3), &Integer::from(7)).unwrap(),
            Integer::from(0)
        );
    }

    #[test]
    fn test_is_quadratic_residue_prime() {
        let p = Integer::from(7);

        // Quadratic residues mod 7: 0, 1, 2, 4
        assert!(Integer::from(0).is_quadratic_residue_prime(&p).unwrap());
        assert!(Integer::from(1).is_quadratic_residue_prime(&p).unwrap()); // 1² = 1
        assert!(Integer::from(2).is_quadratic_residue_prime(&p).unwrap()); // 3² = 9 ≡ 2
        assert!(Integer::from(4).is_quadratic_residue_prime(&p).unwrap()); // 2² = 4

        // Non-residues mod 7: 3, 5, 6
        assert!(!Integer::from(3).is_quadratic_residue_prime(&p).unwrap());
        assert!(!Integer::from(5).is_quadratic_residue_prime(&p).unwrap());
        assert!(!Integer::from(6).is_quadratic_residue_prime(&p).unwrap());

        // Test with p = 11
        let p11 = Integer::from(11);
        assert!(Integer::from(4).is_quadratic_residue_prime(&p11).unwrap()); // 2² = 4
        assert!(Integer::from(9).is_quadratic_residue_prime(&p11).unwrap()); // 3² = 9
        assert!(!Integer::from(2).is_quadratic_residue_prime(&p11).unwrap());
    }

    #[test]
    fn test_sqrt_mod_prime() {
        // Test p ≡ 3 (mod 4) case
        let p7 = Integer::from(7);

        // sqrt(4) mod 7 should be 2 or 5
        let root = Integer::from(4).sqrt_mod_prime(&p7).unwrap();
        assert!(root == Integer::from(2) || root == Integer::from(5));
        assert_eq!((root.clone() * root) % p7.clone(), Integer::from(4));

        // sqrt(2) mod 7 should be 3 or 4
        let root = Integer::from(2).sqrt_mod_prime(&p7).unwrap();
        assert!(root == Integer::from(3) || root == Integer::from(4));
        assert_eq!((root.clone() * root) % p7.clone(), Integer::from(2));

        // Test general case with p ≡ 1 (mod 4)
        let p13 = Integer::from(13);

        // sqrt(3) mod 13 should be 4 or 9
        let root = Integer::from(3).sqrt_mod_prime(&p13).unwrap();
        assert_eq!((root.clone() * root) % p13.clone(), Integer::from(3));

        // sqrt(1) mod any prime should be 1
        assert_eq!(Integer::from(1).sqrt_mod_prime(&p7).unwrap(), Integer::from(1));
        assert_eq!(Integer::from(1).sqrt_mod_prime(&p13).unwrap(), Integer::from(1));

        // sqrt(0) mod any prime should be 0
        assert_eq!(Integer::from(0).sqrt_mod_prime(&p7).unwrap(), Integer::from(0));

        // Non-residue should error
        assert!(Integer::from(3).sqrt_mod_prime(&p7).is_err());
    }

    #[test]
    fn test_quadratic_residues() {
        // QRs mod 7: {0, 1, 2, 4}
        let qrs = Integer::quadratic_residues(&Integer::from(7));
        assert_eq!(qrs.len(), 4);
        assert_eq!(qrs[0], Integer::from(0));
        assert_eq!(qrs[1], Integer::from(1));
        assert_eq!(qrs[2], Integer::from(2));
        assert_eq!(qrs[3], Integer::from(4));

        // QRs mod 11: {0, 1, 3, 4, 5, 9}
        let qrs = Integer::quadratic_residues(&Integer::from(11));
        assert_eq!(qrs.len(), 6); // (11-1)/2 + 1 = 6
        assert!(qrs.contains(&Integer::from(0)));
        assert!(qrs.contains(&Integer::from(1)));
        assert!(qrs.contains(&Integer::from(3)));
        assert!(qrs.contains(&Integer::from(4)));
        assert!(qrs.contains(&Integer::from(5)));
        assert!(qrs.contains(&Integer::from(9)));

        // Verify they're actually quadratic residues
        let p = Integer::from(11);
        for qr in &qrs {
            assert!(qr.is_quadratic_residue_prime(&p).unwrap());
        }
    }

    /// Headline regressions for negative inputs (MAGMA ch. 18 §18.7, §18.12.1); the full
    /// PARI/GP tables are in `tests/ch18_residues.rs`.
    #[test]
    fn test_residues_negative_inputs() {
        let i = Integer::from;
        // gp: lift(Mod(-2,7)^3) -> 6 (was -1); lift(Mod(5,1)^0) -> 0; lift(Mod(2,7)^-1) -> 4
        assert_eq!(i(-2).mod_pow(&i(3), &i(7)), Ok(i(6)));
        assert_eq!(i(5).mod_pow(&i(0), &i(1)), Ok(i(0)));
        assert_eq!(i(2).mod_pow(&i(-1), &i(7)), Ok(i(4)));
        // gp: kronecker(-1,7) -> -1, kronecker(-3,7) -> 1 (both were 0)
        assert_eq!(i(-1).legendre_symbol(&i(7)), Ok(-1));
        assert_eq!(i(-3).legendre_symbol(&i(7)), Ok(1));
        // gp: kronecker(-1,3) -> -1 (was 0), kronecker(-2,15) -> -1
        assert_eq!(i(-1).jacobi_symbol(&i(3)), Ok(-1));
        assert_eq!(i(-2).jacobi_symbol(&i(15)), Ok(-1));
        // gp: kronecker(3,2) -> -1, kronecker(0,-1) -> 1
        assert_eq!(i(3).kronecker_symbol(&i(2)), -1);
        assert_eq!(i(0).kronecker_symbol(&i(-1)), 1);
        // gp: issquare(Mod(-1,7)) -> 0 (was Ok(1)); lift(sqrt(Mod(-1,5))) in {2,3} (was Err)
        assert!(i(-1).sqrt_mod_prime(&i(7)).is_err());
        let r = i(-1).sqrt_mod_prime(&i(5)).unwrap();
        assert!(r == i(2) || r == i(3));
        // gp: issquare(Mod(1,2)) -> 1 (was false)
        assert_eq!(i(1).is_quadratic_residue_prime(&i(2)), Ok(true));
        // gp: lift(Mod(-1,7)^-1) -> 6 (was None)
        assert_eq!(i(-1).mod_inverse(&i(7)), Some(i(6)));
        // gp: binomial(-1,2) % 3 -> 1 (was 0); binomial(5,-1) % 3 -> 0 (was 1)
        assert_eq!(i(-1).binomial_mod(&i(2), &i(3)), Ok(i(1)));
        assert_eq!(i(5).binomial_mod(&i(-1), &i(3)), Ok(i(0)));
    }

    #[test]
    fn test_ordered_ring() {
        use rustmath_core::ordering::OrderedRing;

        assert_eq!(OrderedRing::sign(&Integer::from(42)), 1);
        assert_eq!(OrderedRing::sign(&Integer::zero()), 0);
        assert_eq!(OrderedRing::sign(&Integer::from(-7)), -1);

        assert_eq!(OrderedRing::abs(&Integer::from(-7)), Integer::from(7));
        assert_eq!(OrderedRing::abs(&Integer::from(7)), Integer::from(7));

        assert!(Integer::from(5).is_positive());
        assert!(Integer::from(-5).is_negative());
        assert!(!Integer::zero().is_positive());
        assert_eq!(Integer::from(3).max_with(&Integer::from(8)), Integer::from(8));
        assert_eq!(Integer::from(3).min_with(&Integer::from(8)), Integer::from(3));

        // Translation invariance on a sample.
        let (a, b, c) = (Integer::from(3), Integer::from(8), Integer::from(100));
        assert!(a < b);
        assert!(a + c.clone() < b + c);
    }

    // MAGMA `Normalize` over Z (Handbook ch. 23 §23.5.1): the canonical associate of n is
    // |n|, so the normalization unit is the sign. PARI/GP: `sign(-6)` -> -1, `sign(6)` -> 1,
    // `sign(-10^30)` -> -1; the unit of 0 is 1 (0 is its own canonical associate).
    #[test]
    fn test_normalization_unit_is_the_sign() {
        let unit = |n: i64| Integer::from(n).normalization_unit();
        assert_eq!(unit(-6), Some(Integer::from(-1)));
        assert_eq!(unit(6), Some(Integer::from(1)));
        assert_eq!(unit(0), Some(Integer::from(1)));
        assert_eq!(unit(-1), Some(Integer::from(-1)));
        let big = Integer::from(10).pow(30);
        assert_eq!((-big.clone()).normalization_unit(), Some(Integer::from(-1)));
        assert_eq!(big.normalization_unit(), Some(Integer::from(1)));
    }
}
