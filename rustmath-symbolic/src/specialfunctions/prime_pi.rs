//! Prime Counting Functions
//!
//! This module implements prime counting and related functions from number theory.
//!
//! Corresponds to sage.functions.prime_pi
//!
//! # Functions
//!
//! - `prime_pi(x)`: Prime counting function π(x) - counts primes ≤ x
//! - `legendre_phi(x, a)`: Legendre's phi function φ(x, a)
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::prime_pi::prime_pi;
//! use rustmath_symbolic::Expr;
//!
//! // π(10) = 4 since there are 4 primes ≤ 10: {2, 3, 5, 7}
//! assert_eq!(prime_pi(&Expr::from(10)), Expr::from(4));
//! ```
//!
//! # Mathematical Background
//!
//! The prime counting function π(x) is one of the most important functions
//! in number theory. It's closely related to the distribution of primes
//! via the Prime Number Theorem: π(x) ~ x / ln(x) as x → ∞.

use std::sync::Arc;
use crate::expression::Expr;

/// Prime counting function π(x)
///
/// Returns the number of prime numbers less than or equal to x.
///
/// # Arguments
///
/// * `x` - The upper limit (should be a non-negative number)
///
/// # Returns
///
/// The count of primes ≤ x
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::prime_pi::prime_pi;
/// use rustmath_symbolic::Expr;
///
/// assert_eq!(prime_pi(&Expr::from(2)), Expr::from(1));  // {2}
/// assert_eq!(prime_pi(&Expr::from(10)), Expr::from(4)); // {2, 3, 5, 7}
/// assert_eq!(prime_pi(&Expr::from(20)), Expr::from(8)); // {2,3,5,7,11,13,17,19}
/// ```
///
/// # Performance
///
/// For small values (< 10,000), this function computes the exact count.
/// For larger values, it returns a symbolic representation.
///
/// # Properties
///
/// - π(x) = 0 for x < 2
/// - π(x) is a step function, increasing by 1 at each prime
/// - π(x) ~ x / ln(x) (Prime Number Theorem)
pub fn prime_pi(x: &Expr) -> Expr {
    // Try to compute for integer values
    if let Expr::Integer(n) = x {
        let n_val = n.to_i64();
        {
            if n_val < 2 {
                return Expr::from(0);
            }

            // For reasonable sizes, compute exact count
            if n_val <= 10_000 {
                let count = count_primes_up_to(n_val as u64);
                return Expr::from(count as i64);
            }
        }
    }

    // Return symbolic form for large values or non-integers
    Expr::Function("prime_pi".to_string(), vec![Arc::new(x.clone())])
}

/// Count primes up to n using sieve of Eratosthenes
fn count_primes_up_to(n: u64) -> usize {
    if n < 2 {
        return 0;
    }

    let limit = n as usize;
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    is_prime[1] = false;

    let sqrt_n = (n as f64).sqrt() as usize;
    for i in 2..=sqrt_n {
        if is_prime[i] {
            let mut j = i * i;
            while j <= limit {
                is_prime[j] = false;
                j += i;
            }
        }
    }

    is_prime.iter().filter(|&&x| x).count()
}

/// Legendre's phi function φ(x, a)
///
/// Counts the positive integers ≤ x that are not divisible by any of
/// the first a primes.
///
/// # Arguments
///
/// * `x` - The upper limit
/// * `a` - The number of primes to exclude
///
/// # Returns
///
/// φ(x, a) = |{n : n ≤ x, gcd(n, P_a) = 1}| where P_a = ∏_{i=1}^a p_i
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::prime_pi::legendre_phi;
/// use rustmath_symbolic::Expr;
///
/// // φ(10, 2) counts integers ≤ 10 not divisible by 2 or 3
/// // That's {1, 5, 7} so the answer is 3
/// assert_eq!(legendre_phi(&Expr::from(10), &Expr::from(2)), Expr::from(3));
/// ```
///
/// # Mathematical Background
///
/// Legendre's formula uses φ to compute π(x):
/// π(x) = φ(x, a) + a - 1 for a = π(√x)
pub fn legendre_phi(x: &Expr, a: &Expr) -> Expr {
    // Try to compute for small integer values
    if let (Expr::Integer(x_int), Expr::Integer(a_int)) = (x, a) {
        let (x_val, a_val) = (x_int.to_i64(), a_int.to_i64());
        {
            if x_val >= 0 && a_val >= 0 && x_val <= 1000 && a_val <= 20 {
                let result = compute_legendre_phi(x_val as u64, a_val as usize);
                return Expr::from(result as i64);
            }
        }
    }

    // Return symbolic form
    Expr::Function(
        "legendre_phi".to_string(),
        vec![Arc::new(x.clone()), Arc::new(a.clone())],
    )
}

/// Compute Legendre phi function
fn compute_legendre_phi(x: u64, a: usize) -> u64 {
    if a == 0 {
        return x; // No primes to exclude
    }
    if x == 0 {
        return 0;
    }

    // Generate first a primes
    let primes = generate_first_primes(a);
    if primes.is_empty() {
        return x;
    }

    // Use inclusion-exclusion principle or direct counting for small values
    // For simplicity, we use direct counting
    let mut count = 0u64;
    for n in 1..=x {
        let mut coprime = true;
        for &p in &primes {
            if n % p == 0 {
                coprime = false;
                break;
            }
        }
        if coprime {
            count += 1;
        }
    }
    count
}

/// Generate first n prime numbers
fn generate_first_primes(n: usize) -> Vec<u64> {
    if n == 0 {
        return vec![];
    }

    let mut primes = Vec::new();
    let mut candidate = 2u64;

    while primes.len() < n {
        if is_prime_simple(candidate) {
            primes.push(candidate);
        }
        candidate += if candidate == 2 { 1 } else { 2 };
    }

    primes
}

/// Simple primality test for small numbers
fn is_prime_simple(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    let sqrt_n = (n as f64).sqrt() as u64;
    for i in (3..=sqrt_n).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_prime_pi_small() {
        assert_eq!(prime_pi(&Expr::from(0)), Expr::from(0));
        assert_eq!(prime_pi(&Expr::from(1)), Expr::from(0));
        assert_eq!(prime_pi(&Expr::from(2)), Expr::from(1)); // {2}
        assert_eq!(prime_pi(&Expr::from(3)), Expr::from(2)); // {2, 3}
        assert_eq!(prime_pi(&Expr::from(10)), Expr::from(4)); // {2, 3, 5, 7}
    }

    #[test]
    fn test_prime_pi_medium() {
        assert_eq!(prime_pi(&Expr::from(20)), Expr::from(8)); // {2,3,5,7,11,13,17,19}
        assert_eq!(prime_pi(&Expr::from(100)), Expr::from(25));
    }

    #[test]
    fn test_prime_pi_symbolic() {
        let x = Symbol::new("x");
        let pi_x = prime_pi(&Expr::Symbol(x));
        assert!(matches!(pi_x, Expr::Function(name, _) if name == "prime_pi"));
    }

    #[test]
    fn test_count_primes_helper() {
        assert_eq!(count_primes_up_to(10), 4);
        assert_eq!(count_primes_up_to(20), 8);
        assert_eq!(count_primes_up_to(100), 25);
        assert_eq!(count_primes_up_to(1000), 168);
    }

    #[test]
    fn test_legendre_phi_basic() {
        // φ(10, 0) = 10 (no primes to exclude)
        assert_eq!(legendre_phi(&Expr::from(10), &Expr::from(0)), Expr::from(10));

        // φ(10, 1) excludes multiples of 2: {1,3,5,7,9} = 5
        assert_eq!(legendre_phi(&Expr::from(10), &Expr::from(1)), Expr::from(5));

        // φ(10, 2) excludes multiples of 2 and 3: {1,5,7} = 3
        assert_eq!(legendre_phi(&Expr::from(10), &Expr::from(2)), Expr::from(3));
    }

    #[test]
    fn test_legendre_phi_symbolic() {
        let x = Symbol::new("x");
        let a = Symbol::new("a");
        let phi = legendre_phi(&Expr::Symbol(x), &Expr::Symbol(a));
        assert!(matches!(phi, Expr::Function(name, _) if name == "legendre_phi"));
    }

    #[test]
    fn test_generate_first_primes() {
        let primes = generate_first_primes(5);
        assert_eq!(primes, vec![2, 3, 5, 7, 11]);

        let primes_10 = generate_first_primes(10);
        assert_eq!(primes_10, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_is_prime_simple() {
        assert!(!is_prime_simple(0));
        assert!(!is_prime_simple(1));
        assert!(is_prime_simple(2));
        assert!(is_prime_simple(3));
        assert!(!is_prime_simple(4));
        assert!(is_prime_simple(5));
        assert!(is_prime_simple(7));
        assert!(!is_prime_simple(9));
        assert!(is_prime_simple(11));
        assert!(is_prime_simple(13));
    }

    #[test]
    fn test_prime_pi_properties() {
        // π(x) is monotone increasing
        let pi_10 = prime_pi(&Expr::from(10));
        let pi_20 = prime_pi(&Expr::from(20));

        if let (Expr::Integer(a), Expr::Integer(b)) = (&pi_10, &pi_20) {
            assert!(a < b);
        }
    }
}
