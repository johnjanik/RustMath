//! Fast arithmetic operations using native types
//!
//! This module provides optimized arithmetic operations for common cases
//! where native integer types (i32, i64, u64) can be used instead of
//! arbitrary precision integers.
//!
//! Key optimizations:
//! - Using native CPU arithmetic instructions
//! - Avoiding heap allocations
//! - SIMD operations where applicable
//! - Efficient prime generation and testing

/// Fast arithmetic operations using i32
///
/// Provides optimized operations when working with 32-bit integers
#[derive(Clone, Debug)]
pub struct ArithInt {
    /// Cached values for common operations
    small_primes: Vec<i32>,
}

impl ArithInt {
    /// Create a new ArithInt instance
    pub fn new() -> Self {
        ArithInt {
            small_primes: Self::compute_small_primes(1000),
        }
    }

    /// Compute small primes up to n using sieve of Eratosthenes
    fn compute_small_primes(n: usize) -> Vec<i32> {
        if n < 2 {
            return Vec::new();
        }

        let mut is_prime = vec![true; n + 1];
        is_prime[0] = false;
        is_prime[1] = false;

        for i in 2..=((n as f64).sqrt() as usize) {
            if is_prime[i] {
                let mut j = i * i;
                while j <= n {
                    is_prime[j] = false;
                    j += i;
                }
            }
        }

        is_prime
            .iter()
            .enumerate()
            .filter_map(|(i, &prime)| if prime { Some(i as i32) } else { None })
            .collect()
    }

    /// Fast modular addition: (a + b) mod m
    pub fn add_mod(&self, a: i32, b: i32, m: i32) -> i32 {
        let sum = (a as i64 + b as i64) % m as i64;
        if sum < 0 {
            (sum + m as i64) as i32
        } else {
            sum as i32
        }
    }

    /// Fast modular multiplication: (a * b) mod m
    pub fn mul_mod(&self, a: i32, b: i32, m: i32) -> i32 {
        let product = (a as i64 * b as i64) % m as i64;
        if product < 0 {
            (product + m as i64) as i32
        } else {
            product as i32
        }
    }

    /// Fast modular exponentiation: a^b mod m
    pub fn pow_mod(&self, a: i32, mut b: i32, m: i32) -> i32 {
        let mut result = 1i64;
        let mut base = (a as i64) % (m as i64);

        while b > 0 {
            if b & 1 == 1 {
                result = (result * base) % (m as i64);
            }
            base = (base * base) % (m as i64);
            b >>= 1;
        }

        if result < 0 {
            (result + m as i64) as i32
        } else {
            result as i32
        }
    }

    /// Fast GCD using Euclidean algorithm
    pub fn gcd(&self, mut a: i32, mut b: i32) -> i32 {
        a = a.abs();
        b = b.abs();

        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    /// Fast LCM computation
    pub fn lcm(&self, a: i32, b: i32) -> i32 {
        if a == 0 || b == 0 {
            return 0;
        }
        (a.abs() / self.gcd(a, b)) * b.abs()
    }

    /// Check if a number is prime (using cached small primes)
    pub fn is_prime(&self, n: i32) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 || n % 3 == 0 {
            return false;
        }

        // Check against cached small primes
        for &p in &self.small_primes {
            if p * p > n {
                return true;
            }
            if n % p == 0 {
                return false;
            }
        }

        // If n is larger than our cached primes, use trial division
        let mut i = 5;
        while i * i <= n {
            if n % i == 0 || n % (i + 2) == 0 {
                return false;
            }
            i += 6;
        }
        true
    }

    /// Get the cached small primes
    pub fn get_small_primes(&self) -> &[i32] {
        &self.small_primes
    }
}

impl Default for ArithInt {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast arithmetic operations using i64
///
/// Provides optimized operations when working with 64-bit integers
#[derive(Clone, Debug)]
pub struct ArithLLong {
    /// Cached values for common operations
    small_primes: Vec<i64>,
}

impl ArithLLong {
    /// Create a new ArithLLong instance
    pub fn new() -> Self {
        ArithLLong {
            small_primes: Self::compute_small_primes(10000),
        }
    }

    /// Compute small primes up to n using sieve of Eratosthenes
    fn compute_small_primes(n: usize) -> Vec<i64> {
        if n < 2 {
            return Vec::new();
        }

        let mut is_prime = vec![true; n + 1];
        is_prime[0] = false;
        is_prime[1] = false;

        for i in 2..=((n as f64).sqrt() as usize) {
            if is_prime[i] {
                let mut j = i * i;
                while j <= n {
                    is_prime[j] = false;
                    j += i;
                }
            }
        }

        is_prime
            .iter()
            .enumerate()
            .filter_map(|(i, &prime)| if prime { Some(i as i64) } else { None })
            .collect()
    }

    /// Fast modular addition: (a + b) mod m
    pub fn add_mod(&self, a: i64, b: i64, m: i64) -> i64 {
        let sum = ((a % m) + (b % m)) % m;
        if sum < 0 {
            sum + m
        } else {
            sum
        }
    }

    /// Fast modular multiplication: (a * b) mod m using 128-bit arithmetic
    pub fn mul_mod(&self, a: i64, b: i64, m: i64) -> i64 {
        let product = (a as i128 * b as i128) % m as i128;
        if product < 0 {
            (product + m as i128) as i64
        } else {
            product as i64
        }
    }

    /// Fast modular exponentiation: a^b mod m
    pub fn pow_mod(&self, a: i64, mut b: i64, m: i64) -> i64 {
        let mut result = 1i128;
        let mut base = (a as i128) % (m as i128);

        while b > 0 {
            if b & 1 == 1 {
                result = (result * base) % (m as i128);
            }
            base = (base * base) % (m as i128);
            b >>= 1;
        }

        if result < 0 {
            (result + m as i128) as i64
        } else {
            result as i64
        }
    }

    /// Fast GCD using Euclidean algorithm
    pub fn gcd(&self, mut a: i64, mut b: i64) -> i64 {
        a = a.abs();
        b = b.abs();

        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    /// Fast LCM computation
    pub fn lcm(&self, a: i64, b: i64) -> i64 {
        if a == 0 || b == 0 {
            return 0;
        }
        (a.abs() / self.gcd(a, b)) * b.abs()
    }

    /// Check if a number is prime (using cached small primes and Miller-Rabin)
    pub fn is_prime(&self, n: i64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 || n % 3 == 0 {
            return false;
        }

        // Check against cached small primes
        for &p in &self.small_primes {
            if p * p > n {
                return true;
            }
            if n % p == 0 {
                return false;
            }
        }

        // For larger numbers, use Miller-Rabin
        self.miller_rabin(n, 20)
    }

    /// Miller-Rabin primality test
    fn miller_rabin(&self, n: i64, rounds: usize) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        // Write n-1 as 2^r * d
        let mut d = n - 1;
        let mut r = 0;
        while d % 2 == 0 {
            d /= 2;
            r += 1;
        }

        // Witnesses to test
        let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

        'witness_loop: for &a in witnesses.iter().take(rounds.min(witnesses.len())) {
            if a >= n {
                continue;
            }

            let mut x = self.pow_mod(a, d, n);

            if x == 1 || x == n - 1 {
                continue 'witness_loop;
            }

            for _ in 0..r - 1 {
                x = self.mul_mod(x, x, n);
                if x == n - 1 {
                    continue 'witness_loop;
                }
            }

            return false;
        }

        true
    }

    /// Get the cached small primes
    pub fn get_small_primes(&self) -> &[i64] {
        &self.small_primes
    }
}

impl Default for ArithLLong {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate all primes in a range [start, end) using sieve
///
/// This is an optimized function for generating many primes at once.
///
/// # Examples
/// ```
/// use rustmath_integers::fast_arith::prime_range;
///
/// let primes = prime_range(10, 30);
/// // Returns [11, 13, 17, 19, 23, 29]
/// ```
pub fn prime_range(start: u64, end: u64) -> Vec<u64> {
    if start >= end || end <= 2 {
        return Vec::new();
    }

    let actual_start = start.max(2);
    let size = (end - actual_start) as usize;

    if size == 0 {
        return Vec::new();
    }

    // Sieve of Eratosthenes for the range
    let mut is_prime = vec![true; size];

    // Mark multiples of 2
    let first_even = if actual_start % 2 == 0 {
        0
    } else {
        1
    };
    for i in (first_even..size).step_by(2) {
        is_prime[i] = false;
    }

    // Handle 2 specially if in range
    let mut primes = Vec::new();
    if actual_start <= 2 && 2 < end {
        primes.push(2);
    }

    // Sieve for odd numbers
    let sqrt_end = (end as f64).sqrt() as u64 + 1;
    for p in (3..=sqrt_end).step_by(2) {
        let first_multiple = if actual_start <= p {
            p * p
        } else {
            let rem = actual_start % p;
            if rem == 0 {
                actual_start
            } else {
                actual_start + (p - rem)
            }
        };

        if first_multiple >= end {
            continue;
        }

        let start_idx = (first_multiple - actual_start) as usize;
        for idx in (start_idx..size).step_by(p as usize) {
            is_prime[idx] = false;
        }
    }

    // Collect primes
    for (i, &prime) in is_prime.iter().enumerate() {
        if prime {
            let n = actual_start + i as u64;
            if n >= 2 && n != 2 && n % 2 != 0 {
                primes.push(n);
            }
        }
    }

    primes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arith_int_modular_ops() {
        let arith = ArithInt::new();

        // Test modular addition
        assert_eq!(arith.add_mod(5, 7, 10), 2);
        assert_eq!(arith.add_mod(8, 9, 10), 7);

        // Test modular multiplication
        assert_eq!(arith.mul_mod(3, 4, 10), 2);
        assert_eq!(arith.mul_mod(7, 8, 10), 6);

        // Test modular exponentiation
        assert_eq!(arith.pow_mod(2, 10, 1000), 24);
        assert_eq!(arith.pow_mod(3, 5, 13), 9);
    }

    #[test]
    fn test_arith_int_gcd_lcm() {
        let arith = ArithInt::new();

        assert_eq!(arith.gcd(48, 18), 6);
        assert_eq!(arith.gcd(100, 50), 50);
        assert_eq!(arith.gcd(17, 19), 1);

        assert_eq!(arith.lcm(4, 6), 12);
        assert_eq!(arith.lcm(21, 6), 42);
    }

    #[test]
    fn test_arith_int_is_prime() {
        let arith = ArithInt::new();

        assert!(arith.is_prime(2));
        assert!(arith.is_prime(3));
        assert!(arith.is_prime(17));
        assert!(arith.is_prime(97));

        assert!(!arith.is_prime(1));
        assert!(!arith.is_prime(4));
        assert!(!arith.is_prime(100));
    }

    #[test]
    fn test_arith_llong_modular_ops() {
        let arith = ArithLLong::new();

        // Test modular addition
        assert_eq!(arith.add_mod(5, 7, 10), 2);
        assert_eq!(arith.add_mod(8, 9, 10), 7);

        // Test modular multiplication (using 128-bit)
        assert_eq!(arith.mul_mod(3, 4, 10), 2);
        assert_eq!(arith.mul_mod(7, 8, 10), 6);

        // Test with larger numbers
        let large = 1_000_000_007i64;
        assert_eq!(arith.mul_mod(large - 1, large - 1, large), 1);
    }

    #[test]
    fn test_arith_llong_pow_mod() {
        let arith = ArithLLong::new();

        assert_eq!(arith.pow_mod(2, 10, 1000), 24);
        assert_eq!(arith.pow_mod(3, 5, 13), 9);

        // Test with larger numbers
        assert_eq!(arith.pow_mod(123456, 789, 1000000007), 885357843);
    }

    #[test]
    fn test_arith_llong_gcd_lcm() {
        let arith = ArithLLong::new();

        assert_eq!(arith.gcd(48, 18), 6);
        assert_eq!(arith.gcd(100, 50), 50);
        assert_eq!(arith.gcd(17, 19), 1);

        assert_eq!(arith.lcm(4, 6), 12);
        assert_eq!(arith.lcm(21, 6), 42);
    }

    #[test]
    fn test_arith_llong_is_prime() {
        let arith = ArithLLong::new();

        assert!(arith.is_prime(2));
        assert!(arith.is_prime(3));
        assert!(arith.is_prime(17));
        assert!(arith.is_prime(97));
        assert!(arith.is_prime(1000000007));

        assert!(!arith.is_prime(1));
        assert!(!arith.is_prime(4));
        assert!(!arith.is_prime(100));
        assert!(!arith.is_prime(1000000008));
    }

    #[test]
    fn test_prime_range_small() {
        let primes = prime_range(10, 30);
        assert_eq!(primes, vec![11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_prime_range_with_start() {
        let primes = prime_range(0, 20);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_prime_range_large() {
        let primes = prime_range(100, 130);
        assert_eq!(primes, vec![101, 103, 107, 109, 113, 127]);
    }

    #[test]
    fn test_prime_range_empty() {
        let primes = prime_range(50, 50);
        assert_eq!(primes, Vec::<u64>::new());

        let primes = prime_range(100, 50);
        assert_eq!(primes, Vec::<u64>::new());
    }

    #[test]
    fn test_prime_range_single_prime() {
        let primes = prime_range(17, 18);
        assert_eq!(primes, vec![17]);
    }
}
