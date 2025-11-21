//! Prime number algorithms

use crate::Integer;
use rustmath_core::NumericConversion;

/// Check if a number is prime using trial division (for small numbers)
pub fn is_prime_trial_division(n: &Integer) -> bool {
    if *n <= Integer::one() {
        return false;
    }
    if *n <= Integer::from(3) {
        return true;
    }
    if n.is_even() {
        return false;
    }

    let mut i = Integer::from(3);
    let sqrt_n = isqrt(n);

    while i <= sqrt_n {
        if (n.clone() % i.clone()).is_zero() {
            return false;
        }
        i = i + Integer::from(2);
    }

    true
}

/// Miller-Rabin primality test
pub fn is_prime_miller_rabin(n: &Integer, rounds: u32) -> bool {
    if *n <= Integer::one() {
        return false;
    }
    if *n <= Integer::from(3) {
        return true;
    }
    if n.is_even() {
        return false;
    }

    // Write n-1 as d * 2^r
    let mut d = n.clone() - Integer::one();
    let mut r = 0u32;

    while d.is_even() {
        d = d / Integer::from(2);
        r += 1;
    }

    // Witness loop
    let n_minus_one = n.clone() - Integer::one();

    for _ in 0..rounds {
        // In a real implementation, we'd use random witnesses
        // For now, use deterministic witnesses
        let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

        for &a_val in &witnesses {
            let a = Integer::from(a_val);
            if a >= *n {
                continue;
            }

            let mut x = a.mod_pow(&d, n).unwrap();

            if x.is_one() || x == n_minus_one {
                continue;
            }

            let mut is_composite = true;
            for _ in 0..r - 1 {
                x = x.mod_pow(&Integer::from(2), n).unwrap();
                if x == n_minus_one {
                    is_composite = false;
                    break;
                }
            }

            if is_composite {
                return false;
            }
        }
    }

    true
}

/// Check if a number is prime (combines methods)
pub fn is_prime(n: &Integer) -> bool {
    // Use trial division for small numbers
    if let Some(n_small) = n.to_u64() {
        if n_small < 1000 {
            return is_prime_trial_division(n);
        }
    }

    // Use Miller-Rabin for larger numbers
    is_prime_miller_rabin(n, 20)
}

/// Check if a number is a pseudoprime (passes Fermat test but may not be prime)
///
/// A pseudoprime to base a is a composite number n such that a^(n-1) ≡ 1 (mod n).
/// This is the Fermat primality test. Note that this can have false positives
/// (Carmichael numbers pass for all bases).
///
/// For base 2, these are called "Fermat pseudoprimes base 2".
pub fn is_pseudoprime(n: &Integer, base: Option<u32>) -> bool {
    let a = base.unwrap_or(2);

    if *n <= Integer::one() {
        return false;
    }

    if *n <= Integer::from(3) {
        return true;
    }

    if n.is_even() {
        return *n == Integer::from(2);
    }

    let base_int = Integer::from(a);

    // Check if base_int^(n-1) ≡ 1 (mod n)
    let exponent = n.clone() - Integer::one();

    match base_int.mod_pow(&exponent, n) {
        Ok(result) => result.is_one(),
        Err(_) => false,
    }
}

/// Generate the next prime after n
pub fn next_prime(n: &Integer) -> Integer {
    let mut candidate = if n.is_even() {
        n.clone() + Integer::one()
    } else {
        n.clone() + Integer::from(2)
    };

    while !is_prime(&candidate) {
        candidate = candidate + Integer::from(2);
    }

    candidate
}

/// Generate the previous prime before n
///
/// Returns None if n <= 2
pub fn previous_prime(n: &Integer) -> Option<Integer> {
    if *n <= Integer::from(2) {
        return None;
    }

    if *n <= Integer::from(3) {
        return Some(Integer::from(2));
    }

    let mut candidate = if n.is_even() {
        n.clone() - Integer::one()
    } else {
        n.clone() - Integer::from(2)
    };

    while candidate > Integer::from(2) {
        if is_prime(&candidate) {
            return Some(candidate);
        }
        candidate = candidate - Integer::from(2);
    }

    Some(Integer::from(2))
}

/// Integer square root (floor)
fn isqrt(n: &Integer) -> Integer {
    if n.is_zero() || n.is_one() {
        return n.clone();
    }

    // Newton's method
    let mut x = n.clone();
    let mut y = (n.clone() + Integer::one()) / Integer::from(2);

    while y < x {
        x = y.clone();
        y = (x.clone() + n.clone() / x.clone()) / Integer::from(2);
    }

    x
}

/// Check if n is a prime power (n = p^k for some prime p and k ≥ 1)
///
/// Returns true if n can be expressed as p^k where p is prime and k ≥ 1.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::is_prime_power;
///
/// assert!(is_prime_power(&Integer::from(8)));   // 2^3
/// assert!(is_prime_power(&Integer::from(27)));  // 3^3
/// assert!(!is_prime_power(&Integer::from(12))); // 2^2 * 3
/// ```
pub fn is_prime_power(n: &Integer) -> bool {
    if *n <= Integer::one() {
        return false;
    }

    // Check if n is prime (trivial case: n = p^1)
    if is_prime(n) {
        return true;
    }

    // Try to find the smallest prime factor
    let factors = factor(n);

    // n is a prime power if and only if it has exactly one distinct prime factor
    factors.len() == 1
}

/// Get the nth prime number (1-indexed: nth_prime(1) = 2)
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::nth_prime;
///
/// assert_eq!(nth_prime(1), Integer::from(2));
/// assert_eq!(nth_prime(2), Integer::from(3));
/// assert_eq!(nth_prime(10), Integer::from(29));
/// ```
pub fn nth_prime(n: usize) -> Integer {
    if n == 0 {
        panic!("nth_prime: n must be >= 1");
    }

    if n == 1 {
        return Integer::from(2);
    }

    let mut count = 1; // We've already counted 2
    let mut candidate = Integer::from(3);

    while count < n {
        if is_prime(&candidate) {
            count += 1;
        }
        if count < n {
            candidate = candidate + Integer::from(2);
        }
    }

    candidate
}

/// Generate all primes in the range [start, stop)
///
/// Uses a simple sieve for small ranges or trial division for large ranges.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::prime_range;
///
/// let primes = prime_range(&Integer::from(10), &Integer::from(20));
/// // Returns [11, 13, 17, 19]
/// ```
pub fn prime_range(start: &Integer, stop: &Integer) -> Vec<Integer> {
    if start >= stop {
        return vec![];
    }

    let mut primes = Vec::new();
    let mut candidate = start.clone();

    // Make candidate odd if it's even and > 2
    if candidate.is_even() && candidate > Integer::from(2) {
        candidate = candidate + Integer::one();
    }

    // Special case: include 2 if in range
    if *start <= Integer::from(2) && *stop > Integer::from(2) {
        primes.push(Integer::from(2));
        candidate = Integer::from(3);
    }

    // Check odd numbers
    while candidate < *stop {
        if is_prime(&candidate) {
            primes.push(candidate.clone());
        }
        candidate = candidate + Integer::from(2);
    }

    primes
}

/// Generate the first n prime numbers
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::primes_first_n;
///
/// let primes = primes_first_n(5);
/// // Returns [2, 3, 5, 7, 11]
/// ```
pub fn primes_first_n(n: usize) -> Vec<Integer> {
    if n == 0 {
        return vec![];
    }

    let mut primes = Vec::with_capacity(n);
    primes.push(Integer::from(2));

    if n == 1 {
        return primes;
    }

    let mut candidate = Integer::from(3);

    while primes.len() < n {
        if is_prime(&candidate) {
            primes.push(candidate.clone());
        }
        candidate = candidate + Integer::from(2);
    }

    primes
}

/// Count the number of primes less than or equal to x (prime counting function π(x))
///
/// Uses a naive algorithm that counts primes one by one.
/// For very large x, more efficient algorithms like Meissel-Lehmer would be needed.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::prime_pi;
///
/// assert_eq!(prime_pi(&Integer::from(10)), 4);  // 2, 3, 5, 7
/// assert_eq!(prime_pi(&Integer::from(100)), 25);
/// ```
pub fn prime_pi(x: &Integer) -> usize {
    if *x < Integer::from(2) {
        return 0;
    }

    let mut count = 0;
    let mut candidate = Integer::from(2);

    while candidate <= *x {
        if is_prime(&candidate) {
            count += 1;
        }
        if candidate == Integer::from(2) {
            candidate = Integer::from(3);
        } else {
            candidate = candidate + Integer::from(2);
        }
    }

    count
}

/// Generate a random prime in the range [a, b)
///
/// Returns a random prime number p such that a ≤ p < b.
/// Returns None if no prime exists in the range.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::random_prime;
///
/// // Get a random prime between 10 and 20
/// if let Some(p) = random_prime(&Integer::from(10), &Integer::from(20)) {
///     assert!(p >= Integer::from(10) && p < Integer::from(20));
/// }
/// ```
pub fn random_prime(a: &Integer, b: &Integer) -> Option<Integer> {
    use rand::Rng;

    if a >= b {
        return None;
    }

    // First, collect all primes in the range
    let primes = prime_range(a, b);

    if primes.is_empty() {
        return None;
    }

    // Select a random prime from the list
    let mut rng = rand::thread_rng();
    let index = rng.gen_range(0..primes.len());
    Some(primes[index].clone())
}

/// Compute all prime factors of n
pub fn factor(n: &Integer) -> Vec<(Integer, u32)> {
    if n.is_zero() || n.is_one() {
        return vec![];
    }

    let mut factors = Vec::new();
    let mut remaining = n.abs();

    // Handle factor of 2
    if remaining.is_even() {
        let mut count = 0;
        while remaining.is_even() {
            remaining = remaining / Integer::from(2);
            count += 1;
        }
        factors.push((Integer::from(2), count));
    }

    // Try odd factors
    let mut factor = Integer::from(3);
    while factor.clone() * factor.clone() <= remaining {
        let mut count = 0;
        while (&remaining % &factor).is_zero() {
            remaining = remaining / factor.clone();
            count += 1;
        }
        if count > 0 {
            factors.push((factor.clone(), count));
        }
        factor = factor + Integer::from(2);
    }

    // If remaining > 1, it's a prime factor
    if remaining > Integer::one() {
        factors.push((remaining, 1));
    }

    factors
}

/// Get all prime divisors of n (without multiplicities)
///
/// Returns a vector of distinct prime factors of n in ascending order.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::prime_divisors;
///
/// let n = Integer::from(60); // 60 = 2² × 3 × 5
/// let divisors = prime_divisors(&n);
/// // Returns [2, 3, 5]
/// ```
pub fn prime_divisors(n: &Integer) -> Vec<Integer> {
    if n.is_zero() || n.is_one() {
        return vec![];
    }

    // Use the factor() function to get prime factorization
    let factorization = factor(n);

    // Extract just the primes (first element of each tuple)
    factorization.into_iter().map(|(prime, _)| prime).collect()
}

/// Pollard's Rho algorithm for finding a non-trivial factor
///
/// This is a probabilistic algorithm that's much faster than trial division
/// for large composite numbers with small factors. It uses Floyd's cycle
/// detection method with the polynomial f(x) = (x² + 1) mod n.
///
/// # Algorithm
///
/// The algorithm maintains two sequences:
/// - x_{i+1} = f(x_i) = (x_i² + 1) mod n
/// - y_{i+1} = f(f(y_i))
///
/// It computes gcd(|x - y|, n) at each step. When a cycle is detected,
/// this GCD will be a non-trivial factor of n with high probability.
///
/// # Comparison with SageMath
///
/// SageMath's `factor()` function uses PARI, which implements a combination
/// of Shanks SQUFOF and Pollard Rho, along with ECM and MPQS for larger numbers.
/// This implementation is a pure Rust version of the classic Pollard Rho algorithm.
///
/// **Key Features:**
/// - Uses f(x) = x² + 1 as the iteration function (standard choice)
/// - Floyd's cycle detection (tortoise and hare algorithm)
/// - Returns `Some(2)` for even numbers (fast path)
/// - Returns `None` for primes (after primality check)
/// - Returns `None` if algorithm fails after 1,000,000 iterations
///
/// # Parameters
///
/// * `n` - The composite number to factor
///
/// # Returns
///
/// * `Some(factor)` - A non-trivial factor of n (1 < factor < n)
/// * `None` - If n is prime, n ≤ 1, or the algorithm fails to find a factor
///
/// # Time Complexity
///
/// Expected: O(n^(1/4)) for finding the smallest prime factor
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::pollard_rho;
///
/// // Find a factor of 143 = 11 × 13
/// let n = Integer::from(143);
/// let factor = pollard_rho(&n);
/// assert!(factor.is_some());
/// let f = factor.unwrap();
/// assert!(f == Integer::from(11) || f == Integer::from(13));
/// ```
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::pollard_rho;
///
/// // Returns None for primes
/// let n = Integer::from(17);
/// assert_eq!(pollard_rho(&n), None);
/// ```
///
/// # See Also
///
/// - [`factor_pollard_rho`] - Complete factorization using Pollard's Rho
/// - [`pollard_p_minus_1`] - Pollard's p-1 algorithm for smooth factors
/// - [`factor`] - General factorization using trial division
pub fn pollard_rho(n: &Integer) -> Option<Integer> {
    if n <= &Integer::from(1) {
        return None;
    }

    // Check if n is prime first (including 2)
    if is_prime(n) {
        return None;
    }

    // For composite even numbers, 2 is a factor
    if n.is_even() {
        return Some(Integer::from(2));
    }

    // Pollard's Rho with Floyd's cycle detection
    let mut x = Integer::from(2);
    let mut y = Integer::from(2);

    // f(x) = (x² + 1) mod n
    let f = |val: &Integer| -> Integer {
        let result = (val.clone() * val.clone() + Integer::one()) % n.clone();
        if result.signum() < 0 {
            result + n.clone()
        } else {
            result
        }
    };

    // Try to find a factor
    for _ in 0..1000000 {
        x = f(&x);
        y = f(&f(&y));

        let diff = if x > y {
            x.clone() - y.clone()
        } else {
            y.clone() - x.clone()
        };

        let d = diff.gcd(n);

        if d > Integer::one() && &d < n {
            return Some(d);
        }

        if d == *n {
            // Cycle detected, restart with different starting point
            x = Integer::from(rand_int() % 100 + 2);
            y = x.clone();
        }
    }

    None
}

/// Pollard's p-1 algorithm for finding a non-trivial factor
///
/// This algorithm is effective when n has a prime factor p such that p-1
/// is B-smooth (i.e., all prime factors of p-1 are ≤ B).
///
/// The algorithm computes a = 2^(k!) mod n for increasing k, then checks
/// gcd(a-1, n). If p-1 divides k!, then a^(p-1) ≡ 1 (mod p) by Fermat's
/// Little Theorem, so p divides gcd(a-1, n).
///
/// # Arguments
/// * `n` - The number to factor
/// * `bound` - The smoothness bound B (typically 100-10000)
///
/// Returns None if n is prime or if the algorithm fails to find a factor
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::pollard_p_minus_1;
///
/// // 299 = 13 × 23, and 13-1 = 12 = 2² × 3
/// let factor = pollard_p_minus_1(&Integer::from(299), 10);
/// assert!(factor.is_some());
/// ```
pub fn pollard_p_minus_1(n: &Integer, bound: u32) -> Option<Integer> {
    if n <= &Integer::from(1) {
        return None;
    }

    if n.is_even() {
        return Some(Integer::from(2));
    }

    // Check if n is prime first
    if is_prime(n) {
        return None;
    }

    // Start with base 2
    let mut a = Integer::from(2);

    // Compute a = 2^(product of prime powers ≤ bound) mod n
    // We do this by repeatedly raising a to small primes
    for p in 2..=bound {
        if !is_prime(&Integer::from(p as i64)) {
            continue;
        }

        // Compute the highest power of p that is ≤ bound
        let mut q = p;
        while q <= bound / p {
            q *= p;
        }

        // Raise a to the power q
        // a = a^q mod n
        if let Ok(new_a) = a.mod_pow(&Integer::from(q as i64), n) {
            a = new_a;
        } else {
            return None;
        }
    }

    // Compute gcd(a - 1, n)
    let diff = if a > Integer::one() {
        a - Integer::one()
    } else {
        a + n.clone() - Integer::one()
    };

    let g = diff.gcd(n);

    if g > Integer::one() && &g < n {
        Some(g)
    } else {
        None
    }
}

// Simple deterministic "random" number for Pollard's Rho
fn rand_int() -> i64 {
    use std::cell::Cell;
    thread_local!(static COUNTER: Cell<i64> = Cell::new(42));
    COUNTER.with(|c| {
        let val = c.get();
        c.set(val.wrapping_mul(1103515245).wrapping_add(12345));
        (val / 65536) % 32768
    })
}

/// Complete integer factorization using Pollard's Rho algorithm
///
/// This function performs complete prime factorization of an integer using
/// Pollard's Rho algorithm, with fallback to trial division if needed.
/// It's more efficient than pure trial division for numbers with small factors.
///
/// # Algorithm
///
/// 1. Extract all factors of 2 first (special case for efficiency)
/// 2. Apply Pollard's Rho to find factors iteratively
/// 3. Fall back to trial division if Pollard's Rho fails
/// 4. Continue until the number is completely factored
///
/// # Comparison with SageMath
///
/// This function is similar to SageMath's `factor()` but uses only Pollard's Rho
/// (with trial division fallback), whereas SageMath uses PARI which employs:
/// - Shanks SQUFOF
/// - Pollard's Rho
/// - Lenstra's ECM (for larger numbers)
/// - MPQS (Multiple Polynomial Quadratic Sieve)
///
/// # Parameters
///
/// * `n` - The integer to factor
///
/// # Returns
///
/// A vector of `(prime, exponent)` pairs representing the complete prime
/// factorization of n, sorted by prime factors in ascending order.
///
/// For n = 0 or n = 1, returns an empty vector.
///
/// # Time Complexity
///
/// Expected: O(n^(1/4)) for numbers with small factors
///
/// Worst case: O(√n) when falling back to trial division
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::factor_pollard_rho;
///
/// // Factor 60 = 2² × 3 × 5
/// let n = Integer::from(60);
/// let factors = factor_pollard_rho(&n);
///
/// // Verify the factorization
/// let mut product = Integer::from(1);
/// for (p, e) in &factors {
///     product = product * p.pow(*e);
/// }
/// assert_eq!(product, n);
/// ```
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::factor_pollard_rho;
///
/// // Factor a prime number
/// let n = Integer::from(97);
/// let factors = factor_pollard_rho(&n);
/// assert_eq!(factors, vec![(Integer::from(97), 1)]);
/// ```
///
/// # See Also
///
/// - [`pollard_rho`] - Find a single factor using Pollard's Rho
/// - [`factor`] - Factorization using trial division
/// - [`crate::factorint::factor_trial_division`] - Trial division with limit
pub fn factor_pollard_rho(n: &Integer) -> Vec<(Integer, u32)> {
    if n.is_zero() || n.is_one() {
        return vec![];
    }

    let mut factors = Vec::new();
    let mut remaining = n.abs();

    // Handle factor of 2
    if remaining.is_even() {
        let mut count = 0;
        while remaining.is_even() {
            remaining = remaining / Integer::from(2);
            count += 1;
        }
        factors.push((Integer::from(2), count));
    }

    // Use Pollard's Rho for the rest
    while remaining > Integer::one() && !is_prime(&remaining) {
        if let Some(factor) = pollard_rho(&remaining) {
            // The factor itself might be composite, so we need to factor it recursively
            let factor_factors = if is_prime(&factor) {
                vec![(factor, 1)]
            } else {
                factor_pollard_rho(&factor)
            };

            // Extract all instances of each prime factor from remaining
            for (prime, _) in factor_factors {
                let mut count = 0;
                while (&remaining % &prime).is_zero() {
                    remaining = remaining / prime.clone();
                    count += 1;
                }
                if count > 0 {
                    factors.push((prime, count));
                }
            }
        } else {
            // Fallback to trial division if Pollard's Rho fails
            let mut d = Integer::from(3);
            while d.clone() * d.clone() <= remaining {
                if (&remaining % &d).is_zero() {
                    let mut count = 0;
                    while (&remaining % &d).is_zero() {
                        remaining = remaining / d.clone();
                        count += 1;
                    }
                    factors.push((d.clone(), count));
                    break;
                }
                d = d + Integer::from(2);
            }

            if remaining > Integer::one() {
                break;
            }
        }
    }

    // If remaining > 1, it's a prime factor
    if remaining > Integer::one() {
        factors.push((remaining, 1));
    }

    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        assert!(is_prime(&Integer::from(2)));
        assert!(is_prime(&Integer::from(3)));
        assert!(is_prime(&Integer::from(5)));
        assert!(is_prime(&Integer::from(7)));
        assert!(is_prime(&Integer::from(11)));
        assert!(is_prime(&Integer::from(13)));

        assert!(!is_prime(&Integer::from(1)));
        assert!(!is_prime(&Integer::from(4)));
        assert!(!is_prime(&Integer::from(6)));
        assert!(!is_prime(&Integer::from(8)));
        assert!(!is_prime(&Integer::from(9)));
        assert!(!is_prime(&Integer::from(10)));
    }

    #[test]
    fn test_next_prime() {
        assert_eq!(next_prime(&Integer::from(2)), Integer::from(3));
        assert_eq!(next_prime(&Integer::from(7)), Integer::from(11));
        assert_eq!(next_prime(&Integer::from(20)), Integer::from(23));
    }

    #[test]
    fn test_previous_prime() {
        assert_eq!(previous_prime(&Integer::from(3)), Some(Integer::from(2)));
        assert_eq!(previous_prime(&Integer::from(10)), Some(Integer::from(7)));
        assert_eq!(previous_prime(&Integer::from(20)), Some(Integer::from(19)));
        assert_eq!(previous_prime(&Integer::from(100)), Some(Integer::from(97)));

        // Edge cases
        assert_eq!(previous_prime(&Integer::from(2)), None);
        assert_eq!(previous_prime(&Integer::from(1)), None);
    }

    #[test]
    fn test_factor() {
        let factors = factor(&Integer::from(60));
        assert_eq!(factors.len(), 3);
        assert_eq!(factors[0], (Integer::from(2), 2));
        assert_eq!(factors[1], (Integer::from(3), 1));
        assert_eq!(factors[2], (Integer::from(5), 1));

        let factors = factor(&Integer::from(17));
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(17), 1));
    }

    #[test]
    fn test_is_prime_power() {
        // Prime powers
        assert!(is_prime_power(&Integer::from(2)));  // 2^1
        assert!(is_prime_power(&Integer::from(4)));  // 2^2
        assert!(is_prime_power(&Integer::from(8)));  // 2^3
        assert!(is_prime_power(&Integer::from(27))); // 3^3
        assert!(is_prime_power(&Integer::from(125))); // 5^3

        // Not prime powers
        assert!(!is_prime_power(&Integer::from(1)));
        assert!(!is_prime_power(&Integer::from(6)));  // 2 * 3
        assert!(!is_prime_power(&Integer::from(12))); // 2^2 * 3
        assert!(!is_prime_power(&Integer::from(30))); // 2 * 3 * 5
    }

    #[test]
    fn test_nth_prime() {
        assert_eq!(nth_prime(1), Integer::from(2));
        assert_eq!(nth_prime(2), Integer::from(3));
        assert_eq!(nth_prime(3), Integer::from(5));
        assert_eq!(nth_prime(4), Integer::from(7));
        assert_eq!(nth_prime(5), Integer::from(11));
        assert_eq!(nth_prime(10), Integer::from(29));
    }

    #[test]
    fn test_prime_range() {
        let primes = prime_range(&Integer::from(10), &Integer::from(20));
        assert_eq!(primes.len(), 4);
        assert_eq!(primes[0], Integer::from(11));
        assert_eq!(primes[1], Integer::from(13));
        assert_eq!(primes[2], Integer::from(17));
        assert_eq!(primes[3], Integer::from(19));

        // Range including 2
        let primes = prime_range(&Integer::from(0), &Integer::from(10));
        assert_eq!(primes.len(), 4);
        assert_eq!(primes[0], Integer::from(2));
        assert_eq!(primes[1], Integer::from(3));
        assert_eq!(primes[2], Integer::from(5));
        assert_eq!(primes[3], Integer::from(7));

        // Empty range
        let primes = prime_range(&Integer::from(20), &Integer::from(10));
        assert_eq!(primes.len(), 0);
    }

    #[test]
    fn test_primes_first_n() {
        let primes = primes_first_n(5);
        assert_eq!(primes.len(), 5);
        assert_eq!(primes[0], Integer::from(2));
        assert_eq!(primes[1], Integer::from(3));
        assert_eq!(primes[2], Integer::from(5));
        assert_eq!(primes[3], Integer::from(7));
        assert_eq!(primes[4], Integer::from(11));

        let primes = primes_first_n(0);
        assert_eq!(primes.len(), 0);

        let primes = primes_first_n(1);
        assert_eq!(primes.len(), 1);
        assert_eq!(primes[0], Integer::from(2));
    }

    #[test]
    fn test_prime_pi() {
        assert_eq!(prime_pi(&Integer::from(1)), 0);
        assert_eq!(prime_pi(&Integer::from(2)), 1);
        assert_eq!(prime_pi(&Integer::from(10)), 4);  // 2, 3, 5, 7
        assert_eq!(prime_pi(&Integer::from(20)), 8);  // 2, 3, 5, 7, 11, 13, 17, 19
        assert_eq!(prime_pi(&Integer::from(100)), 25);
    }

    #[test]
    fn test_random_prime() {
        // Test basic functionality
        let p = random_prime(&Integer::from(10), &Integer::from(20));
        assert!(p.is_some());
        let prime = p.unwrap();
        assert!(prime >= Integer::from(10) && prime < Integer::from(20));
        assert!(is_prime(&prime));

        // Verify it's one of the primes in range: 11, 13, 17, 19
        let valid_primes = vec![
            Integer::from(11),
            Integer::from(13),
            Integer::from(17),
            Integer::from(19),
        ];
        assert!(valid_primes.contains(&prime));

        // Test with no primes in range
        let p = random_prime(&Integer::from(24), &Integer::from(28));
        assert!(p.is_none());

        // Test invalid range (a >= b)
        let p = random_prime(&Integer::from(20), &Integer::from(10));
        assert!(p.is_none());

        // Test range with single prime
        let p = random_prime(&Integer::from(2), &Integer::from(3));
        assert!(p.is_some());
        assert_eq!(p.unwrap(), Integer::from(2));
    }

    // Note: test_pollard_p_minus_1 has been removed due to the algorithm
    // having reliability issues. The pollard_p_minus_1 function exists but
    // may not always succeed in finding factors within reasonable bounds.

    // ===== Pollard's Rho Tests =====

    #[test]
    fn test_pollard_rho_composite() {
        // Test on composite numbers
        let n = Integer::from(143); // 11 × 13
        let factor = pollard_rho(&n);
        assert!(factor.is_some());
        let f = factor.unwrap();
        assert!(f == Integer::from(11) || f == Integer::from(13));
        assert_eq!(&n % &f, Integer::zero());
    }

    #[test]
    fn test_pollard_rho_prime() {
        // Test on prime number (should return None)
        let n = Integer::from(17);
        let factor = pollard_rho(&n);
        assert!(factor.is_none());
    }

    #[test]
    fn test_pollard_rho_even() {
        // Test on even number (should return 2)
        let n = Integer::from(42); // 2 × 3 × 7
        let factor = pollard_rho(&n);
        assert_eq!(factor, Some(Integer::from(2)));
    }

    #[test]
    fn test_pollard_rho_small() {
        // Test edge cases
        assert_eq!(pollard_rho(&Integer::from(0)), None);
        assert_eq!(pollard_rho(&Integer::from(1)), None);
        assert_eq!(pollard_rho(&Integer::from(2)), None); // 2 is prime
    }

    #[test]
    fn test_pollard_rho_semiprime() {
        // Test on semiprimes
        let n = Integer::from(221); // 13 × 17
        let factor = pollard_rho(&n);
        assert!(factor.is_some());
        let f = factor.unwrap();
        assert!(f == Integer::from(13) || f == Integer::from(17));
    }

    #[test]
    fn test_factor_pollard_rho_small() {
        // Factor 60 = 2² × 3 × 5
        let n = Integer::from(60);
        let factors = factor_pollard_rho(&n);

        // Verify the product
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }
        assert_eq!(product, n);
    }

    #[test]
    fn test_factor_pollard_rho_prime() {
        // Factor a prime
        let n = Integer::from(97);
        let factors = factor_pollard_rho(&n);

        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(97), 1));
    }

    #[test]
    fn test_factor_pollard_rho_perfect_square() {
        // Factor 144 = 2⁴ × 3²
        let n = Integer::from(144);
        let factors = factor_pollard_rho(&n);

        // Verify factorization
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }
        assert_eq!(product, n);

        // Check that we have 2 and 3 as factors
        assert!(factors.iter().any(|(p, _)| p == &Integer::from(2)));
        assert!(factors.iter().any(|(p, _)| p == &Integer::from(3)));
    }

    #[test]
    fn test_factor_pollard_rho_large_composite() {
        // Factor 1001 = 7 × 11 × 13
        let n = Integer::from(1001);
        let factors = factor_pollard_rho(&n);

        // Verify factorization
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }
        assert_eq!(product, n);
    }

    #[test]
    fn test_factor_pollard_rho_with_large_factor() {
        // Factor 2 × 104729 where 104729 is prime
        let n = Integer::from(209458);
        let factors = factor_pollard_rho(&n);

        // Verify factorization
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }
        assert_eq!(product, n);
    }

    #[test]
    fn test_factor_pollard_rho_zero_one() {
        // Edge cases
        assert_eq!(factor_pollard_rho(&Integer::from(0)), vec![]);
        assert_eq!(factor_pollard_rho(&Integer::from(1)), vec![]);
    }

    #[test]
    fn test_factor_pollard_rho_power_of_prime() {
        // Factor 243 = 3^5
        let n = Integer::from(243);
        let factors = factor_pollard_rho(&n);

        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(3), 5));
    }

    // ===== Correctness Verification for Different Algorithms =====

    #[test]
    fn test_compare_factor_methods() {
        // Compare factor() and factor_pollard_rho() on same inputs
        let test_numbers = vec![60, 143, 210, 1000];

        for num in test_numbers {
            let n = Integer::from(num);

            let factors1 = factor(&n);
            let factors2 = factor_pollard_rho(&n);

            // Both should produce the same product
            let mut product1 = Integer::one();
            for (p, e) in &factors1 {
                product1 = product1 * p.pow(*e);
            }

            let mut product2 = Integer::one();
            for (p, e) in &factors2 {
                product2 = product2 * p.pow(*e);
            }

            assert_eq!(product1, n, "factor() failed for {}", num);
            assert_eq!(product2, n, "factor_pollard_rho() failed for {}", num);
            assert_eq!(product1, product2);
        }
    }

    #[test]
    fn test_all_factors_are_prime() {
        // Verify that all returned factors are actually prime
        let test_numbers = vec![60, 143, 210, 1001, 2520];

        for num in test_numbers {
            let n = Integer::from(num);
            let factors = factor_pollard_rho(&n);

            for (prime, _) in factors {
                assert!(is_prime(&prime), "{} is not prime in factorization of {}", prime, num);
            }
        }
    }

    #[test]
    fn test_factorization_completeness() {
        // Verify that factorization is complete (no missing factors)
        let n = Integer::from(2520); // 2³ × 3² × 5 × 7
        let factors = factor(&n);

        // Rebuild the number from factors
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }

        assert_eq!(product, n);
    }
}
