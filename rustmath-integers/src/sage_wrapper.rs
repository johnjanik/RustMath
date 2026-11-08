//! SageMath-compatible API wrapper for Integer type
//!
//! This module provides a SageMath-compatible interface for the Integer type,
//! making it easier to port SageMath code to RustMath. It includes additional
//! methods and behaviors that match SageMath's `sage.rings.integer.Integer` class.
//!
//! The primary specification is MAGMA Handbook ch. 18, "Ring of Integers" (digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`): `binomial` is MAGMA `Binomial(n, r)`
//! (§18.8) generalised to negative n by Graham-Knuth-Patashnik, *Concrete Mathematics*,
//! eq. (5.14), and `nth_root` is `Iroot` (§18.4.8).
//!
//! # Overview
//!
//! SageMath provides a comprehensive Integer class (`sage.rings.integer.Integer`)
//! with extensive functionality for integer arithmetic, number theory, and
//! combinatorics. This module wraps RustMath's `Integer` type to provide a
//! similar API surface.
//!
//! # Key Features
//!
//! - **Number Theoretic Functions**: `is_prime()`, `is_prime_power()`, `factor()`, etc.
//! - **Combinatorial Functions**: `factorial()`, `binomial()`, `nth_root()`
//! - **Divisor Functions**: `divisors()`, `euler_phi()`, `sigma()`
//! - **Modular Arithmetic**: `mod_pow()`, `sqrt_mod_prime()`, `legendre_symbol()`
//!
//! # API Differences from SageMath
//!
//! While this wrapper aims to provide a similar API to SageMath, there are some
//! differences due to Rust's type system and design philosophy:
//!
//! ## Return Types
//!
//! - **SageMath**: Often returns `Integer`, `None`, or raises exceptions
//! - **RustMath**: Uses `Result<Integer, MathError>` for fallible operations
//!
//! ```rust
//! # use rustmath_integers::Integer;
//! # use rustmath_integers::sage_wrapper::SageInteger;
//! // SageMath: sqrt(-1) raises ValueError
//! // RustMath: sqrt() returns Result
//! let n = Integer::from(-1);
//! assert!(n.sqrt().is_err());
//! ```
//!
//! ## Method Names
//!
//! - Most method names match SageMath exactly
//! - Some Rust-specific methods use snake_case (e.g., `is_prime` instead of `isPrime`)
//! - SageMath's `__` methods are not directly exposed (use regular methods instead)
//!
//! ## Factorization Format
//!
//! - **SageMath**: Returns `Factorization` object with pretty printing
//! - **RustMath**: Returns `Vec<(Integer, u32)>` of `(prime, exponent)` pairs
//!
//! ```ignore
//! # SageMath
//! sage: factor(60)
//! 2^2 * 3 * 5
//!
//! # RustMath
//! use rustmath_integers::Integer;
//! use rustmath_integers::sage_wrapper::SageInteger;
//! let n = Integer::from(60);
//! let factors = n.factor();
//! // Returns: [(2, 2), (3, 1), (5, 1)]
//! ```
//!
//! ## Random Number Generation
//!
//! - SageMath integrates with its own random number framework
//! - RustMath uses Rust's `rand` crate (not yet fully implemented)
//!
//! ## BigInt Limits
//!
//! - Both use arbitrary precision integers
//! - RustMath uses `num_bigint::BigInt` internally
//! - SageMath uses GMP via Cython
//!
//! # Usage Example
//!
//! ```rust
//! use rustmath_integers::Integer;
//! use rustmath_integers::sage_wrapper::SageInteger;
//!
//! // Create an integer
//! let n = Integer::from(120);
//!
//! // Number theoretic functions
//! assert!(n.is_prime_power() == false);
//! assert_eq!(n.euler_phi().unwrap(), Integer::from(32));
//!
//! // Combinatorial functions
//! let fact5 = Integer::factorial(5);
//! assert_eq!(fact5, Integer::from(120));
//!
//! let binom = Integer::binomial(&Integer::from(10), &Integer::from(3));
//! assert_eq!(binom, Integer::from(120));
//!
//! // Divisor functions
//! let divs = n.divisors().unwrap();
//! assert_eq!(divs.len(), 16); // 120 has 16 divisors
//! ```
//!
//! # Comparison with SageMath
//!
//! | Feature | SageMath | RustMath | Notes |
//! |---------|----------|----------|-------|
//! | `factorial(n)` | ✓ | ✓ | Both support arbitrary precision |
//! | `binomial(n, k)` | ✓ | ✓ | Exact, generalised to negative n (MAGMA `Binomial`) |
//! | `is_prime()` | ✓ | ✓ | Uses Miller-Rabin |
//! | `is_prime_power()` | ✓ | ✓ | Checks if n = p^k |
//! | `nth_root(k)` | ✓ | ✓ | Returns floor(k-th root), also for negative self and odd k |
//! | `factor()` | ✓ | ✓ | Different output format |
//! | `divisors()` | ✓ | ✓ | Returns sorted list |
//! | `euler_phi()` | ✓ | ✓ | Euler's totient function |
//! | `sigma(k)` | ✓ | ✓ | Sum of k-th powers of divisors |
//! | `mod_pow(e, m)` | ✓ | ✓ | Modular exponentiation |
//! | `sqrt_mod(p)` | ✓ | ✓ | Tonelli-Shanks algorithm |
//! | `gcd()` | ✓ | ✓ | Non-negative (MAGMA `Gcd`); Euclid on the magnitudes |
//! | `lcm()` | ✓ | ✓ | Non-negative (MAGMA `Lcm`), via gcd |
//! | `valuation(p)` | ✓ | ✓ | p-adic valuation |
//! | `is_square()` | ✓ | ✓ | Perfect square test |
//! | `isqrt()` | ✓ | ✓ | Integer square root |
//! | `digits(base)` | ✓ | ✓ | Base representation |
//! | `bits()` | ✓ | ✓ | Bit length |
//!
//! # Missing Features
//!
//! Some SageMath features are not yet implemented:
//!
//! - `random_element()` - Random integer generation with specific properties
//! - `nth_root(k, truncate_mode)` - Different rounding modes
//! - `powermod()` - Alias for `mod_pow()`
//! - Integration with symbolic ring
//! - Coercion to other rings (automatic conversion)
//!
//! # See Also
//!
//! - [`crate::integer::Integer`] - The underlying Integer type
//! - [`crate::prime`] - Prime number algorithms
//! - [`crate::factorint`] - Integer factorization

use crate::prime::{factor, is_prime, is_prime_power};
use crate::Integer;
use rustmath_core::Result;

/// Extension trait providing SageMath-compatible methods for Integer
///
/// This trait extends the `Integer` type with additional methods that match
/// SageMath's API, making it easier to port SageMath code to RustMath.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::sage_wrapper::SageInteger;
///
/// let n = Integer::from(100);
///
/// // Check if it's a prime power (100 = 2^2 × 5^2, not a prime power)
/// assert!(!n.is_prime_power());
///
/// // Compute factorial
/// let fact = Integer::factorial(5);
/// assert_eq!(fact, Integer::from(120));
///
/// // Compute binomial coefficient
/// let binom = Integer::binomial(&Integer::from(5), &Integer::from(2));
/// assert_eq!(binom, Integer::from(10));
/// ```
pub trait SageInteger {
    /// Check if this number is a prime power (n = p^k for prime p and k ≥ 1)
    ///
    /// Returns `true` if this number can be expressed as p^k where p is prime
    /// and k ≥ 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_integers::sage_wrapper::SageInteger;
    ///
    /// assert!(Integer::from(8).is_prime_power());    // 2^3
    /// assert!(Integer::from(27).is_prime_power());   // 3^3
    /// assert!(Integer::from(7).is_prime_power());    // 7^1 (prime)
    /// assert!(!Integer::from(12).is_prime_power());  // 2^2 * 3
    /// assert!(!Integer::from(1).is_prime_power());   // Not a prime power
    /// ```
    ///
    /// # SageMath Equivalent
    ///
    /// ```python
    /// sage: Integer(8).is_prime_power()
    /// True
    /// sage: Integer(12).is_prime_power()
    /// False
    /// ```
    fn is_prime_power(&self) -> bool;

    /// Compute the factorial of n (n!)
    ///
    /// Returns n! = n × (n-1) × (n-2) × ... × 2 × 1
    ///
    /// # Errors
    ///
    /// Returns an error if n is negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_integers::sage_wrapper::SageInteger;
    ///
    /// assert_eq!(Integer::factorial(0), Integer::from(1));
    /// assert_eq!(Integer::factorial(5), Integer::from(120));
    /// assert_eq!(Integer::factorial(10), Integer::from(3628800));
    /// ```
    ///
    /// # SageMath Equivalent
    ///
    /// ```python
    /// sage: factorial(5)
    /// 120
    /// sage: Integer(5).factorial()
    /// 120
    /// ```
    fn factorial(n: u32) -> Integer;

    /// The generalised binomial coefficient C(n, k) (MAGMA `Binomial(n, r)`, Handbook ch. 18
    /// §18.8; PARI/GP `binomial`, GAP `Binomial`), for every integer n and k:
    ///
    /// | input | value |
    /// |---|---|
    /// | k < 0 (any n) | 0 |
    /// | 0 <= n < k | 0 |
    /// | 0 <= k <= n | n! / (k! (n-k)!) |
    /// | n < 0, k >= 0 | (-1)^k C(k-n-1, k) (negating the upper index) |
    ///
    /// For n >= 0 this counts the k-subsets of an n-set. For n < 0 it is the polynomial
    /// n(n-1)...(n-k+1)/k! evaluated at n, as in Graham-Knuth-Patashnik, *Concrete
    /// Mathematics*, eq. (5.14), and in GMP `mpz_bin_ui` (the arithmetic MAGMA's integers use);
    /// so C(-1, 2) = 1 and C(-3, 2) = 6 (gp `binomial(-1,2)`, `binomial(-3,2)`).
    ///
    /// For k < 0, MAGMA's own value is **unconfirmed**. The handbook states no range for n and
    /// r, and no MAGMA output for a negative r was available to check against. The 0 above
    /// follows PARI/GP and GAP (`binomial(5,-1)` = `binomial(-1,-1)` = 0) and
    /// [`Integer::binomial_mod`]. Other systems differ: Mathematica's symmetric convention gives
    /// `Binomial[-1, -1]` = 1. So MAGMA may differ here too.
    ///
    /// Cost: O(min(k, n'-k)) bignum multiplications and exact divisions, where n' is n after
    /// the upper index is negated.
    ///
    /// # Arguments
    ///
    /// * `n` - The upper index (any integer)
    /// * `k` - The lower index (any integer)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_integers::sage_wrapper::SageInteger;
    ///
    /// // C(5, 2) = 10
    /// assert_eq!(
    ///     Integer::binomial(&Integer::from(5), &Integer::from(2)),
    ///     Integer::from(10)
    /// );
    ///
    /// // C(10, 3) = 120
    /// assert_eq!(
    ///     Integer::binomial(&Integer::from(10), &Integer::from(3)),
    ///     Integer::from(120)
    /// );
    ///
    /// // C(n, 0) = 1
    /// assert_eq!(
    ///     Integer::binomial(&Integer::from(5), &Integer::from(0)),
    ///     Integer::from(1)
    /// );
    ///
    /// // C(n, k) = 0 when 0 <= n < k
    /// assert_eq!(
    ///     Integer::binomial(&Integer::from(5), &Integer::from(10)),
    ///     Integer::from(0)
    /// );
    ///
    /// // Negative upper index: C(-1, 2) = (-1)(-2)/2 = 1 (gp: binomial(-1,2) -> 1)
    /// assert_eq!(
    ///     Integer::binomial(&Integer::from(-1), &Integer::from(2)),
    ///     Integer::from(1)
    /// );
    ///
    /// // Negative lower index: 0 (gp: binomial(-1,-1) -> 0)
    /// assert_eq!(
    ///     Integer::binomial(&Integer::from(-1), &Integer::from(-1)),
    ///     Integer::from(0)
    /// );
    /// ```
    ///
    /// # SageMath Equivalent
    ///
    /// ```python
    /// sage: binomial(5, 2)
    /// 10
    /// sage: Integer(5).binomial(2)
    /// 10
    /// ```
    fn binomial(n: &Integer, k: &Integer) -> Integer;

    /// Compute the prime factorization of this number
    ///
    /// Returns a vector of (prime, exponent) pairs representing the complete
    /// prime factorization.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_integers::sage_wrapper::SageInteger;
    ///
    /// let n = Integer::from(60);
    /// let factors = n.factor();
    ///
    /// // 60 = 2^2 × 3 × 5
    /// assert_eq!(factors, vec![
    ///     (Integer::from(2), 2),
    ///     (Integer::from(3), 1),
    ///     (Integer::from(5), 1),
    /// ]);
    /// ```
    ///
    /// # SageMath Equivalent
    ///
    /// ```python
    /// sage: factor(60)
    /// 2^2 * 3 * 5
    /// ```
    ///
    /// Note: SageMath returns a `Factorization` object with pretty printing,
    /// while RustMath returns a vector of tuples.
    fn factor(&self) -> Vec<(Integer, u32)>;

    /// Check if this number is prime
    ///
    /// Uses a combination of trial division (for small numbers) and
    /// Miller-Rabin primality test (for larger numbers).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_integers::sage_wrapper::SageInteger;
    ///
    /// assert!(Integer::from(17).is_prime());
    /// assert!(!Integer::from(18).is_prime());
    /// assert!(Integer::from(97).is_prime());
    /// ```
    ///
    /// # SageMath Equivalent
    ///
    /// ```python
    /// sage: Integer(17).is_prime()
    /// True
    /// ```
    fn is_prime(&self) -> bool;

    /// Get the nth root of this number
    ///
    /// Returns the largest integer r such that r^n ≤ self, i.e. floor(self^(1/n)) (MAGMA
    /// `Iroot`, Handbook ch. 18 §18.4.8, for self >= 0). For self < 0 and odd n this is the
    /// floor of the real root: `nth_root(-9, 3) = -3`, whereas Sage's
    /// `nth_root(3, truncate_mode=1)` truncates toward zero and gives -2. See
    /// [`Integer::nth_root`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - n is 0
    /// - n is even and self is negative
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::Integer;
    /// use rustmath_integers::sage_wrapper::SageInteger;
    ///
    /// assert_eq!(Integer::from(27).nth_root(3).unwrap(), Integer::from(3));
    /// assert_eq!(Integer::from(100).nth_root(2).unwrap(), Integer::from(10));
    /// assert_eq!(Integer::from(15).nth_root(2).unwrap(), Integer::from(3)); // floor
    /// assert_eq!(Integer::from(-8).nth_root(3).unwrap(), Integer::from(-2));
    /// assert_eq!(Integer::from(-9).nth_root(3).unwrap(), Integer::from(-3)); // floor
    /// assert!(Integer::from(-4).nth_root(2).is_err());
    /// ```
    ///
    /// # SageMath Equivalent
    ///
    /// ```python
    /// sage: Integer(27).nth_root(3)
    /// 3
    /// ```
    fn nth_root(&self, n: u32) -> Result<Integer>;
}

impl SageInteger for Integer {
    fn is_prime_power(&self) -> bool {
        is_prime_power(self)
    }

    fn factorial(n: u32) -> Integer {
        if n == 0 || n == 1 {
            return Integer::one();
        }

        let mut result = Integer::one();
        for i in 2..=n {
            result = result * Integer::from(i);
        }
        result
    }

    fn binomial(n: &Integer, k: &Integer) -> Integer {
        if k.signum() < 0 {
            return Integer::zero();
        }
        // Negating the upper index (Concrete Mathematics (5.14)): C(n, k) = (-1)^k C(k-n-1, k),
        // and k - n - 1 >= k when n < 0, so the transformed coefficient is an ordinary one.
        let (n, negate) = if n.signum() < 0 {
            (k.clone() - n.clone() - Integer::one(), k.is_odd())
        } else {
            (n.clone(), false)
        };
        if *k > n {
            return Integer::zero();
        }
        // C(n, k) = C(n, n-k): run over the smaller lower index.
        let n_minus_k = n.clone() - k.clone();
        let kk = if *k > n_minus_k { n_minus_k } else { k.clone() };
        let mut result = Integer::one();
        let mut i = Integer::zero();
        while i < kk {
            // result = C(n, i), so result * (n - i) = C(n, i + 1) * (i + 1): the division is
            // exact.
            result = result * (n.clone() - i.clone());
            i = i + Integer::one();
            result = result / i.clone();
        }
        if negate {
            -result
        } else {
            result
        }
    }

    fn factor(&self) -> Vec<(Integer, u32)> {
        factor(self)
    }

    fn is_prime(&self) -> bool {
        is_prime(self)
    }

    fn nth_root(&self, n: u32) -> Result<Integer> {
        self.nth_root(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== is_prime_power Tests =====

    #[test]
    fn test_is_prime_power_basic() {
        // Prime powers
        assert!(Integer::from(2).is_prime_power()); // 2^1
        assert!(Integer::from(4).is_prime_power()); // 2^2
        assert!(Integer::from(8).is_prime_power()); // 2^3
        assert!(Integer::from(16).is_prime_power()); // 2^4
        assert!(Integer::from(32).is_prime_power()); // 2^5

        assert!(Integer::from(3).is_prime_power()); // 3^1
        assert!(Integer::from(9).is_prime_power()); // 3^2
        assert!(Integer::from(27).is_prime_power()); // 3^3
        assert!(Integer::from(81).is_prime_power()); // 3^4

        assert!(Integer::from(5).is_prime_power()); // 5^1
        assert!(Integer::from(25).is_prime_power()); // 5^2
        assert!(Integer::from(125).is_prime_power()); // 5^3
    }

    #[test]
    fn test_is_prime_power_not_prime_powers() {
        // Not prime powers
        assert!(!Integer::from(1).is_prime_power()); // 1 is not a prime power
        assert!(!Integer::from(6).is_prime_power()); // 2 × 3
        assert!(!Integer::from(12).is_prime_power()); // 2^2 × 3
        assert!(!Integer::from(15).is_prime_power()); // 3 × 5
        assert!(!Integer::from(30).is_prime_power()); // 2 × 3 × 5
        assert!(!Integer::from(100).is_prime_power()); // 2^2 × 5^2
    }

    #[test]
    fn test_is_prime_power_large() {
        // Large prime powers
        assert!(Integer::from(128).is_prime_power()); // 2^7
        assert!(Integer::from(243).is_prime_power()); // 3^5
        assert!(Integer::from(343).is_prime_power()); // 7^3
        assert!(Integer::from(1024).is_prime_power()); // 2^10

        // Large non-prime powers
        assert!(!Integer::from(1000).is_prime_power()); // 2^3 × 5^3
        assert!(!Integer::from(1001).is_prime_power()); // 7 × 11 × 13
    }

    // ===== factorial Tests =====

    #[test]
    fn test_factorial_small() {
        assert_eq!(Integer::factorial(0), Integer::from(1));
        assert_eq!(Integer::factorial(1), Integer::from(1));
        assert_eq!(Integer::factorial(2), Integer::from(2));
        assert_eq!(Integer::factorial(3), Integer::from(6));
        assert_eq!(Integer::factorial(4), Integer::from(24));
        assert_eq!(Integer::factorial(5), Integer::from(120));
        assert_eq!(Integer::factorial(6), Integer::from(720));
        assert_eq!(Integer::factorial(7), Integer::from(5040));
    }

    #[test]
    fn test_factorial_medium() {
        // 10! = 3,628,800
        assert_eq!(Integer::factorial(10), Integer::from(3628800));

        // 12! = 479,001,600
        assert_eq!(Integer::factorial(12), Integer::from(479001600i64));
    }

    #[test]
    fn test_factorial_large() {
        // 20! = 2432902008176640000
        let fact20 = Integer::factorial(20);
        assert_eq!(fact20, Integer::from(2432902008176640000i64));

        // Verify 21! = 21 × 20!
        let fact21 = Integer::factorial(21);
        assert_eq!(fact21, fact20 * Integer::from(21));
    }

    #[test]
    fn test_factorial_very_large() {
        // Test that very large factorials work (arbitrary precision)
        let fact50 = Integer::factorial(50);

        // 50! has 65 digits, verify it's greater than 10^64
        let ten_pow_64 = Integer::from(10).pow(64);
        assert!(fact50 > ten_pow_64);

        // Verify n! is divisible by n
        assert_eq!(&fact50 % &Integer::from(50), Integer::zero());
    }

    // ===== binomial Tests =====

    #[test]
    fn test_binomial_small() {
        // C(n, 0) = 1
        assert_eq!(
            Integer::binomial(&Integer::from(5), &Integer::from(0)),
            Integer::from(1)
        );

        // C(n, n) = 1
        assert_eq!(
            Integer::binomial(&Integer::from(5), &Integer::from(5)),
            Integer::from(1)
        );

        // C(n, 1) = n
        assert_eq!(
            Integer::binomial(&Integer::from(5), &Integer::from(1)),
            Integer::from(5)
        );

        // C(5, 2) = 10
        assert_eq!(
            Integer::binomial(&Integer::from(5), &Integer::from(2)),
            Integer::from(10)
        );

        // C(5, 3) = 10
        assert_eq!(
            Integer::binomial(&Integer::from(5), &Integer::from(3)),
            Integer::from(10)
        );
    }

    #[test]
    fn test_binomial_pascal_triangle() {
        // Test Pascal's triangle identity: C(n, k) = C(n-1, k-1) + C(n-1, k)
        for n in 2..10 {
            for k in 1..n {
                let n_int = Integer::from(n);
                let k_int = Integer::from(k);

                let left = Integer::binomial(&n_int, &k_int);
                let right1 = Integer::binomial(&(n_int.clone() - Integer::one()), &(k_int.clone() - Integer::one()));
                let right2 = Integer::binomial(&(n_int.clone() - Integer::one()), &k_int);

                assert_eq!(left, right1 + right2, "Pascal's triangle failed for C({}, {})", n, k);
            }
        }
        // The generalised binomial (C(n, k) = 0 for k < 0, negative n by GKP (5.14)) satisfies
        // Pascal's rule for every integer n and k.
        for n in -10..10 {
            for k in -3..12 {
                let n_int = Integer::from(n);
                let k_int = Integer::from(k);
                let left = Integer::binomial(&n_int, &k_int);
                let right1 = Integer::binomial(
                    &(n_int.clone() - Integer::one()),
                    &(k_int.clone() - Integer::one()),
                );
                let right2 = Integer::binomial(&(n_int.clone() - Integer::one()), &k_int);
                assert_eq!(
                    left,
                    right1 + right2,
                    "Pascal's rule failed for C({n}, {k})"
                );
            }
        }
    }

    #[test]
    fn test_binomial_negative_upper_index() {
        // MAGMA Binomial(n, r), Handbook ch. 18 §18.8, generalised by C(n, k) = (-1)^k C(k-n-1, k).
        // Reference: PARI/GP 2.15.4, `gp -q`, command in each comment.
        let b = |n: i64, k: i64| Integer::binomial(&Integer::from(n), &Integer::from(k));
        assert_eq!(b(-1, 2), Integer::from(1)); // gp: binomial(-1,2) -> 1
        assert_eq!(b(-1, 1), Integer::from(-1)); // gp: binomial(-1,1) -> -1
        assert_eq!(b(-1, 0), Integer::from(1)); // gp: binomial(-1,0) -> 1
        assert_eq!(b(-3, 2), Integer::from(6)); // gp: binomial(-3,2) -> 6
        assert_eq!(b(-2, 3), Integer::from(-4)); // gp: binomial(-2,3) -> -4
        assert_eq!(b(-10, 3), Integer::from(-220)); // gp: binomial(-10,3) -> -220
        assert_eq!(b(-150, 3), Integer::from(-573800)); // gp: binomial(-150,3) -> -573800
        assert_eq!(b(-7, 30), Integer::from(1947792)); // gp: binomial(-7,30) -> 1947792
        assert_eq!(b(-2, 1000), Integer::from(1001)); // gp: binomial(-2,1000) -> 1001

        // Negative lower index gives 0 (gp: binomial(-1,-1) -> 0, binomial(-4,-4) -> 0,
        // binomial(5,-1) -> 0; GAP 4.12.1 Binomial(-1,-1) -> 0).
        assert_eq!(b(-1, -1), Integer::zero());
        assert_eq!(b(-4, -4), Integer::zero());
        assert_eq!(b(5, -1), Integer::zero());
        assert_eq!(b(0, -1), Integer::zero());
    }

    #[test]
    fn test_binomial_edge_cases() {
        // C(n, k) = 0 when k > n
        assert_eq!(
            Integer::binomial(&Integer::from(5), &Integer::from(10)),
            Integer::from(0)
        );

        // C(0, 0) = 1
        assert_eq!(
            Integer::binomial(&Integer::from(0), &Integer::from(0)),
            Integer::from(1)
        );

        // C(n, n-1) = n
        assert_eq!(
            Integer::binomial(&Integer::from(10), &Integer::from(9)),
            Integer::from(10)
        );
    }

    #[test]
    fn test_binomial_medium() {
        // C(10, 3) = 120
        assert_eq!(
            Integer::binomial(&Integer::from(10), &Integer::from(3)),
            Integer::from(120)
        );

        // C(10, 5) = 252
        assert_eq!(
            Integer::binomial(&Integer::from(10), &Integer::from(5)),
            Integer::from(252)
        );

        // C(20, 10) = 184,756
        assert_eq!(
            Integer::binomial(&Integer::from(20), &Integer::from(10)),
            Integer::from(184756)
        );
    }

    #[test]
    fn test_binomial_large() {
        // C(100, 50) is a very large number
        let binom = Integer::binomial(&Integer::from(100), &Integer::from(50));

        // C(100, 50) ≈ 1.009 × 10^29
        // Verify it's in the right ballpark
        let lower_bound = Integer::from(10).pow(29);
        let upper_bound = Integer::from(10).pow(30);

        assert!(binom > lower_bound);
        assert!(binom < upper_bound);

        // Verify symmetry: C(n, k) = C(n, n-k)
        let binom_sym = Integer::binomial(&Integer::from(100), &Integer::from(50));
        assert_eq!(binom, binom_sym);
    }

    #[test]
    fn test_binomial_symmetry() {
        // C(n, k) = C(n, n-k)
        for n in 1..20 {
            for k in 0..=n {
                let n_int = Integer::from(n);
                let k_int = Integer::from(k);
                let nk_int = Integer::from(n - k);

                let left = Integer::binomial(&n_int, &k_int);
                let right = Integer::binomial(&n_int, &nk_int);

                assert_eq!(left, right, "Symmetry failed for C({}, {})", n, k);
            }
        }
    }

    // ===== Integration Tests with Existing Methods =====

    #[test]
    fn test_factor_integration() {
        let n = Integer::from(60);
        let factors = n.factor();

        // 60 = 2^2 × 3 × 5
        assert_eq!(factors.len(), 3);
        assert_eq!(factors[0], (Integer::from(2), 2));
        assert_eq!(factors[1], (Integer::from(3), 1));
        assert_eq!(factors[2], (Integer::from(5), 1));

        // Verify factorization is correct
        let mut product = Integer::one();
        for (prime, exp) in factors {
            product = product * prime.pow(exp);
        }
        assert_eq!(product, n);
    }

    #[test]
    fn test_is_prime_integration() {
        assert!(Integer::from(2).is_prime());
        assert!(Integer::from(3).is_prime());
        assert!(Integer::from(5).is_prime());
        assert!(Integer::from(7).is_prime());
        assert!(Integer::from(11).is_prime());
        assert!(Integer::from(97).is_prime());

        assert!(!Integer::from(1).is_prime());
        assert!(!Integer::from(4).is_prime());
        assert!(!Integer::from(6).is_prime());
        assert!(!Integer::from(100).is_prime());
    }

    #[test]
    fn test_nth_root_integration() {
        // Perfect powers
        assert_eq!(
            Integer::from(27).nth_root(3).unwrap(),
            Integer::from(3)
        );
        assert_eq!(
            Integer::from(16).nth_root(4).unwrap(),
            Integer::from(2)
        );
        assert_eq!(
            Integer::from(100).nth_root(2).unwrap(),
            Integer::from(10)
        );

        // Non-perfect powers (floor)
        assert_eq!(
            Integer::from(10).nth_root(2).unwrap(),
            Integer::from(3)
        );
        assert_eq!(
            Integer::from(100).nth_root(3).unwrap(),
            Integer::from(4)
        );
    }

    // ===== Comparison with SageMath Behavior =====

    #[test]
    fn test_sagemath_factorial_equivalence() {
        // These values are verified against SageMath
        // sage: [factorial(n) for n in range(10)]
        let expected = vec![
            1,          // 0!
            1,          // 1!
            2,          // 2!
            6,          // 3!
            24,         // 4!
            120,        // 5!
            720,        // 6!
            5040,       // 7!
            40320,      // 8!
            362880,     // 9!
        ];

        for (n, &expected_val) in expected.iter().enumerate() {
            let result = Integer::factorial(n as u32);
            assert_eq!(result, Integer::from(expected_val), "factorial({}) mismatch", n);
        }
    }

    #[test]
    fn test_sagemath_binomial_equivalence() {
        // These values are verified against SageMath
        // sage: binomial(10, 5)
        // 252
        assert_eq!(
            Integer::binomial(&Integer::from(10), &Integer::from(5)),
            Integer::from(252)
        );

        // sage: binomial(20, 10)
        // 184756
        assert_eq!(
            Integer::binomial(&Integer::from(20), &Integer::from(10)),
            Integer::from(184756)
        );

        // sage: binomial(7, 3)
        // 35
        assert_eq!(
            Integer::binomial(&Integer::from(7), &Integer::from(3)),
            Integer::from(35)
        );
    }

    #[test]
    fn test_sagemath_is_prime_power_equivalence() {
        // sage: [n.is_prime_power() for n in range(1, 30)]
        // Verified with SageMath:
        // 1: False, 2: True (2^1), 3: True (3^1), 4: True (2^2), 5: True (5^1)
        // 6: False (2*3), 7: True (7^1), 8: True (2^3), 9: True (3^2), 10: False (2*5)
        // 11: True (11^1), 12: False (2^2*3), 13: True (13^1), 14: False (2*7), 15: False (3*5)
        // 16: True (2^4), 17: True (17^1), 18: False (2*3^2), 19: True (19^1), 20: False (2^2*5)
        // 21: False (3*7), 22: False (2*11), 23: True (23^1), 24: False (2^3*3), 25: True (5^2)
        // 26: False (2*13), 27: True (3^3), 28: False (2^2*7), 29: True (29^1)
        let sage_results = vec![
            false, true, true, true, true, false, true, true, true, // 1-9
            false, true, false, true, false, false, true, true, false, // 10-18
            true, false, false, false, true, false, true, false, true, // 19-27
            false, true, // 28-29
        ];

        for (i, &expected) in sage_results.iter().enumerate() {
            let n = i + 1;
            let result = Integer::from(n as i64).is_prime_power();
            assert_eq!(
                result, expected,
                "is_prime_power({}) returned {} but SageMath returns {}",
                n, result, expected
            );
        }
    }

    #[test]
    fn test_sagemath_nth_root_equivalence() {
        // sage: Integer(27).nth_root(3)
        // 3
        assert_eq!(
            Integer::from(27).nth_root(3).unwrap(),
            Integer::from(3)
        );

        // sage: Integer(100).nth_root(2)
        // 10
        assert_eq!(
            Integer::from(100).nth_root(2).unwrap(),
            Integer::from(10)
        );

        // sage: Integer(1000).nth_root(3)
        // 10 (floor of cube root)
        assert_eq!(
            Integer::from(1000).nth_root(3).unwrap(),
            Integer::from(10)
        );

        // sage: Integer(1024).nth_root(10)
        // 2
        assert_eq!(
            Integer::from(1024).nth_root(10).unwrap(),
            Integer::from(2)
        );
    }

    // ===== Performance and Stress Tests =====

    #[test]
    fn test_factorial_performance() {
        // Test that we can compute moderately large factorials
        let fact30 = Integer::factorial(30);

        // 30! = 265252859812191058636308480000000
        // Verify it has the right magnitude
        assert!(fact30 > Integer::from(10).pow(32));
    }

    #[test]
    fn test_binomial_performance() {
        // Test moderately large binomial coefficients
        let binom = Integer::binomial(&Integer::from(50), &Integer::from(25));

        // C(50, 25) ≈ 1.26 × 10^14
        assert!(binom > Integer::from(10).pow(14));
        assert!(binom < Integer::from(10).pow(15));
    }

    // ===== Documentation Example Tests =====

    #[test]
    fn test_readme_example() {
        // Example from module documentation
        let n = Integer::from(120);

        // Number theoretic functions
        assert_eq!(n.is_prime_power(), false);
        assert_eq!(n.euler_phi().unwrap(), Integer::from(32));

        // Combinatorial functions
        let fact5 = Integer::factorial(5);
        assert_eq!(fact5, Integer::from(120));

        let binom = Integer::binomial(&Integer::from(10), &Integer::from(3));
        assert_eq!(binom, Integer::from(120));

        // Divisor functions
        let divs = n.divisors().unwrap();
        assert_eq!(divs.len(), 16); // 120 has 16 divisors
    }

    #[test]
    fn test_trait_example() {
        // Example from trait documentation
        let n = Integer::from(100);

        // Check if it's a prime power
        assert!(!n.is_prime_power());

        // Compute factorial
        let fact = Integer::factorial(5);
        assert_eq!(fact, Integer::from(120));

        // Compute binomial coefficient
        let binom = Integer::binomial(&Integer::from(5), &Integer::from(2));
        assert_eq!(binom, Integer::from(10));
    }
}
