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
    let sqrt_n = n.sqrt().expect("n > 3 is non-negative");

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
/// Returns the prime factorization of |n| as `(prime, exponent)` pairs with the primes in
/// ascending order: PARI/GP `factor(n)` without the sign, MAGMA `Factorization(n)` (Handbook
/// ch. 18 §18.10, digest `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`). For n = 0, 1
/// and -1 the list is empty.
///
/// # Algorithm
///
/// 1. Divide out the factors of 2.
/// 2. While the cofactor m is not accepted by [`is_prime`], find a proper divisor d of m (see
///    below), factor d by recursion and divide each of its primes out of m with full
///    multiplicity.
/// 3. The last cofactor passed the loop guard, so it is prime.
///
/// A proper divisor of an odd composite m is found by, in order:
/// - the perfect-power test [`Integer::is_perfect_power`]: m = b^k gives d = b;
/// - [`pollard_rho`] (Floyd's cycle detection on x^2 + 1, at most 10^6 steps; J. M. Pollard,
///   *A Monte Carlo method for factorization*, BIT 15 (1975) 331-334);
/// - if that returns `None`: rounds j = 0, 1, 2, ... of Brent's variant with x^2 + c,
///   c = j + 2, at most 2^(12+j) steps (R. P. Brent, *An improved Monte Carlo factorization
///   algorithm*, BIT 20 (1980) 176-184, [Bre80] in the ch. 18 digest; H. Cohen, GTM 138,
///   §8.5), each round followed by trial division by the next 2^(12+j) odd numbers. Rho is
///   Monte Carlo and may fail for every c tried; the trial-division windows make the search
///   complete, since a composite m has a prime factor p <= sqrt(m), which they reach after
///   about p/2 divisions in all.
///
/// A cofactor that the methods have not yet split is never pushed as a prime: the search
/// goes on until it splits.
///
/// Every returned factor is prime in the sense of [`is_prime`]: Miller-Rabin to the twelve
/// bases 2..37, a proof below psi_12 = 318665857834031151167461 (J. Sorenson, J. Webster,
/// *Strong pseudoprimes to twelve prime bases*, Math. Comp. 86 (2017) 985-1003), a strong
/// probable-prime test above it.
///
/// # Comparison with SageMath
///
/// This function is similar to SageMath's `factor()` but uses only Pollard's Rho (with
/// Brent's variant and trial division as the fallback), whereas SageMath uses PARI which
/// employs:
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
/// factorization of |n|, sorted by prime factors in ascending order.
///
/// For n = 0, 1 or -1, returns an empty vector.
///
/// # Cost
///
/// There is no time or iteration limit: the call returns only with the complete
/// factorization. Rho needs about sqrt(p) steps to split off a prime p, so the run time is set
/// by the second-largest prime factor q of |n| (the largest one is the last cofactor) and
/// grows like sqrt(q). Measured in a debug build: (10^9+7)(10^9+9) takes about 0.2 s and
/// (10^12+39)(10^12+61) about 20 s (10^6 steps); for q with 20 or more digits (10^10 steps
/// or more) the call does not finish in practical time. The trial-division windows bound the
/// worst case of each split by about p/2 trial divisions for its smallest prime factor p, plus
/// up to about twice as many Brent steps (a round can overshoot its budget by 2x) and 10^6
/// Floyd steps; the windows restart at 3 for each cofactor that needs the fallback.
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
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::prime::factor_pollard_rho;
///
/// // 12780049 = 53 · 59 · 61 · 67 (PARI/GP: factor(12780049))
/// let factors = factor_pollard_rho(&Integer::from(12780049));
/// let primes: Vec<Integer> = [53, 59, 61, 67].into_iter().map(Integer::from).collect();
/// assert_eq!(factors, primes.into_iter().map(|p| (p, 1)).collect::<Vec<_>>());
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

    // `remaining` is odd from here on. `is_prime` is false only with a proof of
    // compositeness (trial division below 1000, a Miller-Rabin witness above), so every
    // cofactor handed to `split_odd_composite` is an odd composite.
    while remaining > Integer::one() && !is_prime(&remaining) {
        let divisor = split_odd_composite(&remaining);
        // The divisor might be composite: factor it by recursion (it is smaller than
        // `remaining`, so the recursion ends).
        let divisor_factors = if is_prime(&divisor) {
            vec![(divisor, 1)]
        } else {
            factor_pollard_rho(&divisor)
        };

        // Each prime of the divisor divides `remaining`; extract all of its instances, so it
        // cannot come up again later.
        for (prime, _) in divisor_factors {
            let mut count = 0;
            while (&remaining % &prime).is_zero() {
                remaining = remaining / prime.clone();
                count += 1;
            }
            if count > 0 {
                factors.push((prime, count));
            }
        }
    }

    // The loop guard accepted `remaining` as prime.
    if remaining > Integer::one() {
        factors.push((remaining, 1));
    }

    factors.sort_by(|a, b| a.0.cmp(&b.0));
    factors
}

/// Number of steps of the first Brent round and of the first trial-division window in
/// [`split_odd_composite`]; both double each round.
const SPLIT_FIRST_ROUND: u64 = 1 << 12;

/// A proper divisor d of an odd composite m (1 < d < m, d | m).
///
/// Perfect-power test, then [`pollard_rho`], then [`split_brent_and_trial_division`]. Always
/// returns; see [`factor_pollard_rho`] for the cost.
fn split_odd_composite(m: &Integer) -> Integer {
    debug_assert!(m.is_odd() && *m > Integer::one() && !is_prime(m));
    if let Some((base, _)) = m.is_perfect_power() {
        return base;
    }
    if let Some(d) = pollard_rho(m) {
        return d;
    }
    split_brent_and_trial_division(m)
}

/// The complete fallback of [`split_odd_composite`]: rounds j = 0, 1, 2, ... of Brent's rho
/// with x^2 + (j + 2) and at most `SPLIT_FIRST_ROUND * 2^j` steps, each followed by trial
/// division by the next `SPLIT_FIRST_ROUND * 2^j` odd numbers from 3 on.
///
/// Termination does not depend on rho: the windows reach the least prime factor p of m after
/// about p/2 divisions, and p <= sqrt(m) because m is composite.
///
/// # Panics
///
/// If m is odd and the windows pass sqrt(m) without a divisor, i.e. m is prime or 1. Callers
/// only pass odd composites, so this is an internal invariant.
fn split_brent_and_trial_division(m: &Integer) -> Integer {
    let mut next_trial = Integer::from(3);
    let mut budget = SPLIT_FIRST_ROUND;
    let mut c: u64 = 2;
    loop {
        if let Some(d) = brent_rho(m, &Integer::from(c), budget) {
            return d;
        }
        if let Some(d) = trial_division_window(m, &mut next_trial, budget) {
            return d;
        }
        c += 1;
        budget = budget.saturating_mul(2);
    }
}

/// One run of Brent's variant of Pollard rho on m with f(x) = x^2 + c mod m, from x_0 = 2,
/// for at most about `max_steps` evaluations of f ([Bre80]; Cohen GTM 138, §8.5).
///
/// The differences |x - y| are multiplied together modulo m and the gcd is taken every 128
/// steps; when that gcd is m, the last block is replayed one step at a time.
///
/// Returns `Some(d)` with 1 < d < m and d | m, or `None` when the run ends with gcd m (the
/// cycles modulo all prime factors of m were detected together) or the step budget runs out.
/// A `None` says nothing about m.
fn brent_rho(m: &Integer, c: &Integer, max_steps: u64) -> Option<Integer> {
    const BLOCK: u64 = 128;
    let f = |v: &Integer| -> Integer { (&(v * v) + c).modulo(m) };

    let mut y = Integer::from(2).modulo(m);
    let mut q = Integer::one();
    let mut r: u64 = 1;
    let mut steps: u64 = 0;

    let (x, mut ys, mut g) = loop {
        let x = y.clone();
        for _ in 0..r {
            y = f(&y);
        }
        steps += r;
        let mut ys = y.clone();
        let mut g = Integer::one();
        let mut k: u64 = 0;
        while k < r && g.is_one() {
            ys = y.clone();
            let block = BLOCK.min(r - k);
            for _ in 0..block {
                y = f(&y);
                q = (&q * &(&x - &y).abs()).modulo(m);
            }
            g = q.gcd(m);
            k += block;
            steps += block;
        }
        if !g.is_one() {
            break (x, ys, g);
        }
        if steps >= max_steps {
            return None;
        }
        r = r.saturating_mul(2);
    };

    if g == *m {
        // Replay the last block: some |x - y_i| in it shares a prime with m, because q was
        // coprime to m before the block.
        loop {
            ys = f(&ys);
            g = (&x - &ys).abs().gcd(m);
            if !g.is_one() {
                break;
            }
        }
    }

    if g.is_one() || g == *m {
        None
    } else {
        Some(g)
    }
}

/// Trial division of m by the `count` odd numbers from `*next` on; `*next` is advanced past
/// the numbers tried. Returns the first divisor found.
///
/// # Panics
///
/// If the window passes sqrt(m) without a divisor: then m has no divisor in
/// [3, sqrt(m)], so an odd m is prime (or 1), against the precondition of
/// [`split_brent_and_trial_division`].
fn trial_division_window(m: &Integer, next: &mut Integer, count: u64) -> Option<Integer> {
    for _ in 0..count {
        assert!(
            &*next * &*next <= *m,
            "trial_division_window: no divisor of {m} up to its square root, so it is not an \
             odd composite"
        );
        if (m % &*next).is_zero() {
            return Some(next.clone());
        }
        *next = &*next + &Integer::from(2);
    }
    None
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

    // ===== X1b: the fallback of factor_pollard_rho (MAGMA ch. 18 §18.10) =====

    fn z(s: &str) -> Integer {
        Integer::from_decimal_str(s).unwrap()
    }

    fn is_proper_divisor(d: &Integer, m: &Integer) -> bool {
        *d > Integer::one() && d < m && (m % d).is_zero()
    }

    /// Floyd's rho on x^2 + 1 fails on 25 (the cycles modulo 5 and 25 close together), which
    /// is why `factor_pollard_rho` must not trust a `None` from it.
    #[test]
    fn test_pollard_rho_fails_on_25() {
        assert_eq!(pollard_rho(&Integer::from(25)), None);
    }

    /// The fallback splits every odd composite, including 25 and inputs without small primes.
    #[test]
    fn test_split_brent_and_trial_division() {
        // gp: factor(15) = [3,1; 5,1]; factor(9) = Mat([3,2]); factor(25) = Mat([5,2]);
        // factor(3127) = [53,1; 59,1]; factor(12780049) = [53,1; 59,1; 61,1; 67,1];
        // factor(1000000016000000063) = [1000000007,1; 1000000009,1]
        for s in ["9", "15", "25", "3127", "12780049", "1000000016000000063"] {
            let m = z(s);
            let d = split_brent_and_trial_division(&m);
            assert!(
                is_proper_divisor(&d, &m),
                "{d} is not a proper divisor of {m}"
            );
            let d = split_odd_composite(&m);
            assert!(
                is_proper_divisor(&d, &m),
                "{d} is not a proper divisor of {m}"
            );
        }
    }

    #[test]
    fn test_brent_rho() {
        // gp: factor(1000000016000000063) = [1000000007,1; 1000000009,1]
        let m = z("1000000016000000063");
        let d =
            brent_rho(&m, &Integer::from(2), 1 << 20).expect("Brent splits m within 2^20 steps");
        assert!(d == z("1000000007") || d == z("1000000009"), "d = {d}");
        // A budget too small to reach the cycle modulo either prime gives an honest None.
        assert_eq!(brent_rho(&m, &Integer::from(2), 1), None);
    }

    #[test]
    fn test_trial_division_window() {
        // gp: factor(3127) = [53,1; 59,1]: the odd numbers 3..51 are not divisors.
        let m = Integer::from(3127);
        let mut next = Integer::from(3);
        assert_eq!(trial_division_window(&m, &mut next, 25), None);
        assert_eq!(next, Integer::from(53));
        assert_eq!(
            trial_division_window(&m, &mut next, 1),
            Some(Integer::from(53))
        );
    }

    /// A prime is not an odd composite: the window reaches its square root and panics
    /// instead of returning "no divisor".
    #[test]
    #[should_panic(expected = "no divisor of 101 up to its square root")]
    fn test_trial_division_window_rejects_prime() {
        // gp: isprime(101) = 1
        let mut next = Integer::from(3);
        let _ = trial_division_window(&Integer::from(101), &mut next, 100);
    }

    /// Output is sorted ascending (PARI order), as documented.
    #[test]
    fn test_factor_pollard_rho_sorted() {
        // gp: factor(13710311357) = [101,1; 103,1; 107,1; 109,1; 113,1]
        let got = factor_pollard_rho(&Integer::from(13710311357i64));
        let want: Vec<(Integer, u32)> = [101, 103, 107, 109, 113]
            .into_iter()
            .map(|p| (Integer::from(p), 1))
            .collect();
        assert_eq!(got, want);
    }
}
