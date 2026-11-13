//! Specialized integer factorization algorithms
//!
//! This module provides various specialized factorization methods that complement
//! the general-purpose algorithms in the [`prime`](crate::prime) module. The API
//! is designed to be compatible with SageMath's `sage.rings.factorint` module while
//! providing pure Rust implementations without unsafe code.
//!
//! # Overview
//!
//! This module implements several factorization strategies:
//!
//! - **Trial division**: Factoring by dividing by small primes up to a bound
//! - **Aurifeuillian factorization**: Special factorizations of numbers of the form a^n ± 1
//! - **Cunningham factorization**: Pattern-based factorization for numbers like b^n ± 1
//!
//! MAGMA `TrialDivision(n, B)` (Handbook ch. 18 §18.10.3, digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`) returns the primes found and the
//! unfactored part. [`trial_division_with_primes`] does this for an explicit list of primes.
//! [`crate::factorization::factorization_with_limits`] with `trial_division_limit: B`,
//! `pollard_rho_limit: 0` and `mpqs_limit: Some(0)` is close to `TrialDivision(n, B)`: perfect
//! powers are split too, and the cofactor is classified (use `proof: false` to skip ECPP).
//!
//! For general-purpose factorization including Pollard's Rho algorithm, see
//! [`prime::factor`](crate::prime::factor) and [`prime::pollard_rho`](crate::prime::pollard_rho).
//!
//! # Comparison with SageMath
//!
//! This implementation provides similar functionality to SageMath's `sage.rings.factorint`:
//!
//! | SageMath Function | RustMath Equivalent | Notes |
//! |-------------------|---------------------|-------|
//! | `factor_trial_division(m, limit)` | [`factor_trial_division(&n, limit)`] | Returns `Vec<(Integer, u32)>` instead of `IntegerFactorization` |
//! | `aurifeuillian(n, m, F, check)` | [`aurifeuillian(&n)`] | Simplified implementation for detection only |
//! | `factor_aurifeuillian(n, check)` | [`factor_aurifeuillian(&n)`] | Compatible behavior with basic patterns |
//! | `factor_cunningham(m, proof)` | [`factor_cunningham(&n)`] | Pattern detection without pre-computed tables |
//!
//! **Key Differences:**
//! - SageMath uses PARI/FLINT for performance; RustMath uses pure Rust
//! - SageMath's Aurifeuillian uses cyclotomic polynomials; RustMath uses simplified detection
//! - SageMath's Cunningham uses optional pre-computed tables; RustMath uses pattern matching
//! - Return type is `Vec<(Integer, u32)>` (prime-exponent pairs) vs `IntegerFactorization` object
//!
//! # Examples
//!
//! ## Basic Trial Division
//!
//! ```
//! use rustmath_integers::{Integer, factorint};
//!
//! // Factor 60 = 2² × 3 × 5
//! let n = Integer::from(60);
//! let factors = factorint::factor_trial_division(&n, None);
//!
//! assert_eq!(factors, vec![
//!     (Integer::from(2), 2),
//!     (Integer::from(3), 1),
//!     (Integer::from(5), 1),
//! ]);
//! ```
//!
//! ## Trial Division with Limit
//!
//! ```
//! use rustmath_integers::{Integer, factorint};
//!
//! // Partial factorization with small limit
//! let n = Integer::from(221); // 13 × 17
//! let factors = factorint::factor_trial_division(&n, Some(10));
//!
//! // Since 10 < 13, the number remains unfactored
//! assert_eq!(factors, vec![(Integer::from(221), 1)]);
//! ```
//!
//! ## Aurifeuillian Factorization
//!
//! ```
//! use rustmath_integers::{Integer, factorint};
//!
//! // Factor 2⁴ - 1 = 15 = 3 × 5
//! let n = Integer::from(15);
//! let factors = factorint::factor_aurifeuillian(&n);
//!
//! // Verify the factorization
//! let mut product = Integer::from(1);
//! for (p, e) in &factors {
//!     product = product * p.pow(*e);
//! }
//! assert_eq!(product, n);
//! ```
//!
//! ## Cunningham Factorization
//!
//! ```
//! use rustmath_integers::{Integer, factorint};
//!
//! // Factor 2⁵ - 1 = 31 (Mersenne prime)
//! let n = Integer::from(31);
//! let factors = factorint::factor_cunningham(&n);
//!
//! assert_eq!(factors, vec![(Integer::from(31), 1)]);
//! ```
//!
//! # Algorithm Details
//!
//! ## Trial Division
//!
//! The trial division algorithm factors an integer by testing divisibility by all
//! integers up to a limit (default: √n). The implementation:
//! - Handles 2 separately for efficiency
//! - Only tests odd divisors after 2
//! - Returns partial factorization if limit is too small
//! - Properly handles negative integers (includes -1 as a factor)
//!
//! **Time Complexity:** O(√n) for full factorization, O(limit) for bounded
//!
//! **Space Complexity:** O(k) where k is the number of distinct prime factors
//!
//! ## Pollard's Rho (in prime module)
//!
//! Pollard's Rho algorithm uses Floyd's cycle detection with the polynomial
//! f(x) = (x² + 1) mod n to find non-trivial factors. See [`prime::pollard_rho`](crate::prime::pollard_rho).
//!
//! **Time Complexity:** O(n^(1/4)) expected
//!
//! **Space Complexity:** O(1)
//!
//! ## Aurifeuillian Factorization
//!
//! Detects numbers of the form a^n ± 1 and applies algebraic factorizations.
//! For example, a^n - 1 = (a - 1)(a^(n-1) + a^(n-2) + ... + a + 1).
//!
//! ## Cunningham Factorization
//!
//! Recognizes patterns like b^n ± 1 for small bases b ∈ {2, 3, 5, 7, 10, 11, 12}
//! and applies known factorization formulas.
//!
//! # See Also
//!
//! - [`prime::factor`](crate::prime::factor) - General-purpose factorization using trial division
//! - [`prime::pollard_rho`](crate::prime::pollard_rho) - Pollard's Rho algorithm
//! - [`prime::factor_pollard_rho`](crate::prime::factor_pollard_rho) - Complete factorization using Pollard's Rho
//! - [`prime::pollard_p_minus_1`](crate::prime::pollard_p_minus_1) - Pollard's p-1 algorithm

use crate::factorization::{classify, Class, LimitedFactorization};
use crate::prime::factor;
use crate::Integer;
use rustmath_core::NumericConversion;

/// Perform trial division factorization up to a given bound
///
/// This is the simplest factorization method: try dividing by all primes
/// up to the square root of n (or up to the given limit).
///
/// # Arguments
/// * `n` - The number to factor
/// * `limit` - Maximum trial divisor (if None, uses sqrt(n))
///
/// # Returns
/// A vector of (prime, exponent) pairs representing the factorization,
/// potentially with a composite cofactor if limit is too small
///
/// # Examples
/// ```
/// use rustmath_integers::{Integer, factorint};
///
/// let n = Integer::from(60);
/// let factors = factorint::factor_trial_division(&n, Some(10));
/// // Returns [(2, 2), (3, 1), (5, 1)] since 60 = 2^2 * 3 * 5
/// ```
pub fn factor_trial_division(n: &Integer, limit: Option<u64>) -> Vec<(Integer, u32)> {
    if n.is_zero() || n.is_one() {
        return Vec::new();
    }

    let mut factors = Vec::new();
    let mut remaining = n.abs();

    // Handle negative numbers
    if n < &Integer::zero() {
        factors.push((Integer::from(-1), 1));
    }

    // Handle 2 separately for efficiency
    let two = Integer::from(2);
    let mut exp = 0u32;
    while &remaining % &two == Integer::zero() {
        exp += 1;
        remaining = &remaining / &two;
    }
    if exp > 0 {
        factors.push((two, exp));
    }

    // Determine the limit
    let max_divisor = if let Some(lim) = limit {
        lim
    } else {
        // Use sqrt(n) as default limit
        remaining.sqrt().unwrap_or(Integer::from(1000000))
            .to_u64().unwrap_or(1000000)
    };

    // Try odd divisors
    let mut d = 3u64;
    while d <= max_divisor && remaining > Integer::one() {
        let divisor = Integer::from(d);
        let mut exp = 0u32;

        while &remaining % &divisor == Integer::zero() {
            exp += 1;
            remaining = remaining / divisor.clone();
        }

        if exp > 0 {
            factors.push((divisor, exp));
        }

        d += 2; // Only try odd numbers
    }

    // If there's a remaining part, it's either prime or composite
    if remaining > Integer::one() {
        factors.push((remaining, 1));
    }

    factors
}

/// Check if a number is an Aurifeuillian number
///
/// Aurifeuillian numbers are numbers of the form (a^n ± b^n) that have
/// special algebraic factorizations for certain values of n and a, b.
///
/// Common cases:
/// - 2^(2n+1) + 1 has an Aurifeuillian factorization when n ≡ 1, 2 (mod 3)
/// - Numbers of the form (a^p ± 1)/gcd where p is prime
///
/// # Returns
/// `Some((a, n, sign))` if the number is of the form a^n ± 1, else None
pub fn aurifeuillian(n: &Integer) -> Option<(Integer, u32, i8)> {
    if n.is_zero() || n.is_one() {
        return None;
    }

    // Try to detect if n is of the form a^k ± 1
    // This is a simplified version - a full implementation would be more sophisticated
    // We prefer representations with larger exponents (smaller bases)

    let mut best: Option<(Integer, u32, i8)> = None;

    // Check for n = a^k + 1 (which means a^k = n - 1)
    if *n > Integer::one() {
        let n_minus_1 = n.clone() - Integer::one();
        for exp in 2..=64 {
            let root = n_minus_1.nth_root(exp);
            if let Ok(a) = root {
                if a > Integer::one() && a.pow(exp) == n_minus_1 {
                    // Prefer larger exponents (smaller bases)
                    if best.is_none() || exp > best.as_ref().unwrap().1 {
                        best = Some((a, exp, 1));
                    }
                }
            }
        }
    }

    // Check for n = a^k - 1 (which means a^k = n + 1)
    let n_plus_1 = n.clone() + Integer::one();
    for exp in 2..=64 {
        let root = n_plus_1.nth_root(exp);
        if let Ok(a) = root {
            if a > Integer::one() && a.pow(exp) == n_plus_1 {
                // Prefer larger exponents (smaller bases)
                if best.is_none() || exp > best.as_ref().unwrap().1 {
                    best = Some((a, exp, -1));
                }
            }
        }
    }

    best
}

/// Factor using Aurifeuillian factorization
///
/// This attempts to factor numbers of the form (a^n ± b^n) using
/// algebraic identities.
///
/// For example:
/// - x^4 + 4y^4 = (x^2 + 2xy + 2y^2)(x^2 - 2xy + 2y^2)
/// - 2^(4k+2) + 1 = L_{2k+1} * M_{2k+1} where L and M are Aurifeuillian factors
///
/// # Returns
/// Factorization if Aurifeuillian pattern is detected, otherwise uses standard factorization
pub fn factor_aurifeuillian(n: &Integer) -> Vec<(Integer, u32)> {
    // Check if this is an Aurifeuillian-type number
    if let Some((base, exp, sign)) = aurifeuillian(n) {
        // For numbers of the form a^n + 1 or a^n - 1, we can use algebraic factorizations

        if sign == 1 && exp % 2 == 0 {
            // a^(2k) + 1 case
            // We can use: x^2 + 1 = (x + 1)^2 - 2x for some algebraic tricks
            // For now, fall back to standard factorization
            return factor(n);
        } else if sign == -1 {
            // a^n - 1 = (a - 1)(a^(n-1) + a^(n-2) + ... + a + 1)
            let a_minus_1 = base.clone() - Integer::one();

            if !a_minus_1.is_zero() && n % &a_minus_1 == Integer::zero() {
                let quotient = n.clone() / a_minus_1.clone();

                // Recursively factor both parts
                let mut factors = factor(&a_minus_1);
                let cofactor_factors = factor(&quotient);
                factors.extend(cofactor_factors);

                // Combine duplicate factors
                return combine_factors(factors);
            }
        }
    }

    // No Aurifeuillian pattern detected, use standard factorization
    factor(n)
}

/// Factor using Cunningham tables
///
/// The Cunningham Project maintains tables of factorizations for numbers
/// of the form b^n ± 1 for small bases b and various n.
///
/// This is a simplified implementation that handles common small cases.
/// A full implementation would use precomputed tables.
///
/// # Arguments
/// * `n` - The number to factor
///
/// # Returns
/// Factorization using known patterns, or standard factorization
pub fn factor_cunningham(n: &Integer) -> Vec<(Integer, u32)> {
    // Check if n matches known Cunningham patterns

    // Common patterns from Cunningham tables:
    // 2^n - 1 (Mersenne numbers)
    // 2^n + 1 (Fermat numbers)
    // 3^n ± 1
    // 5^n ± 1
    // etc.

    // Try to detect pattern
    for base in [2, 3, 5, 7, 10, 11, 12].iter() {
        let b = Integer::from(*base);

        // Try different exponents
        for exp in 2..=100 {
            let power = b.pow(exp);

            // Check b^exp - 1
            if power.clone() - Integer::one() == *n {
                // Found pattern: n = b^exp - 1
                // Use algebraic factorization: b^n - 1 = (b-1)(b^(n-1) + ... + 1)
                let b_minus_1 = b.clone() - Integer::one();
                let quotient = n.clone() / b_minus_1.clone();

                let mut factors = factor(&b_minus_1);
                let cofactor_factors = factor(&quotient);
                factors.extend(cofactor_factors);

                return combine_factors(factors);
            }

            // Check b^exp + 1
            if power.clone() + Integer::one() == *n {
                // Found pattern: n = b^exp + 1
                // These are harder to factor in general
                return factor(n);
            }

            // Stop if power exceeds n
            if power > *n {
                break;
            }
        }
    }

    // No Cunningham pattern found, use standard factorization
    factor(n)
}

/// Helper function to combine duplicate factors in a factorization
fn combine_factors(factors: Vec<(Integer, u32)>) -> Vec<(Integer, u32)> {
    use std::collections::HashMap;

    let mut factor_map: HashMap<Integer, u32> = HashMap::new();

    for (prime, exp) in factors {
        *factor_map.entry(prime).or_insert(0) += exp;
    }

    let mut result: Vec<_> = factor_map.into_iter().collect();
    result.sort_by(|a, b| a.0.cmp(&b.0));
    result
}

/// Trial division of n by the given list (any order, duplicates allowed): MAGMA
/// `TrialDivision(n, B)` with the primes given explicitly (ch. 18 §18.10.3), which returns the
/// primes found and the unfactored part.
///
/// The result is a [`LimitedFactorization`]: the listed primes that divide n with their exact
/// exponents in `primes`; the cofactor m = |n| / prod p^e, when m > 1, classified without a
/// proof attempt: in `primes` if m < psi_12 = 318665857834031151167461 and
/// [`is_probable_prime`](crate::prime::is_probable_prime)`(m)` (exact there), in `composites`
/// if `!is_probable_prime(m)` (a proof), else in `probable_primes`. The sign of n is `sign`;
/// 0 gives sign 0 and empty lists.
///
/// Every list entry that divides n is checked to be prime (and proven when it is >= psi_12; an
/// entry whose ECPP proof is unresolved goes to `probable_primes`). Entries that do not divide n
/// are not tested. Cost: one remainder per entry, one classification per dividing entry, one
/// Baillie-PSW test for the cofactor.
///
/// # Panics
///
/// If an entry is < 2, or an entry that divides n is proven composite (it would otherwise be
/// reported as a prime).
///
/// # Examples
///
/// ```
/// use rustmath_integers::{factorint::trial_division_with_primes, Integer};
///
/// // gp: factor(1225) = [5,2; 7,2]: 2 and 3 do not divide it, the cofactor is composite.
/// let f = trial_division_with_primes(&Integer::from(1225), &[Integer::from(2), Integer::from(3)]);
/// assert!(f.primes.is_empty());
/// assert_eq!(f.composites, vec![(Integer::from(1225), 1)]);
/// ```
pub fn trial_division_with_primes(n: &Integer, primes: &[Integer]) -> LimitedFactorization {
    let mut out = LimitedFactorization {
        sign: n.signum(),
        primes: Vec::new(),
        probable_primes: Vec::new(),
        composites: Vec::new(),
    };
    let two = Integer::from(2);
    for p in primes {
        assert!(
            *p >= two,
            "trial_division_with_primes: the list entry {p} is below 2"
        );
    }
    if n.is_zero() {
        return out;
    }
    let mut remaining = n.abs();
    for p in primes {
        if !(&remaining % p).is_zero() {
            continue;
        }
        let mut exp = 0u32;
        while (&remaining % p).is_zero() {
            exp += 1;
            remaining = &remaining / p;
        }
        match classify(p, true) {
            Class::Prime => out.primes.push((p.clone(), exp)),
            Class::ProbablePrime => out.probable_primes.push((p.clone(), exp)),
            Class::Composite => {
                panic!("trial_division_with_primes: the list entry {p} is not prime")
            }
        }
    }
    if remaining > Integer::one() {
        match classify(&remaining, false) {
            Class::Prime => out.primes.push((remaining, 1)),
            Class::ProbablePrime => out.probable_primes.push((remaining, 1)),
            Class::Composite => out.composites.push((remaining, 1)),
        }
    }
    out.primes.sort();
    out.probable_primes.sort();
    out.composites.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Basic Trial Division Tests =====

    #[test]
    fn test_trial_division_small() {
        // Factor 60 = 2^2 * 3 * 5
        let n = Integer::from(60);
        let factors = factor_trial_division(&n, None);

        assert_eq!(factors.len(), 3);
        assert_eq!(factors[0], (Integer::from(2), 2));
        assert_eq!(factors[1], (Integer::from(3), 1));
        assert_eq!(factors[2], (Integer::from(5), 1));
    }

    #[test]
    fn test_trial_division_prime() {
        // Factor a prime number
        let n = Integer::from(17);
        let factors = factor_trial_division(&n, None);

        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(17), 1));
    }

    #[test]
    fn test_trial_division_large() {
        // Factor 1000 = 2^3 * 5^3
        let n = Integer::from(1000);
        let factors = factor_trial_division(&n, None);

        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0], (Integer::from(2), 3));
        assert_eq!(factors[1], (Integer::from(5), 3));
    }

    #[test]
    fn test_trial_division_with_limit() {
        // Factor 60 with a small limit
        let n = Integer::from(60);
        let factors = factor_trial_division(&n, Some(3));

        // Should find 2^2 and 3, leaving 5 as cofactor
        assert!(factors.iter().any(|(p, _)| p == &Integer::from(2)));
        assert!(factors.iter().any(|(p, _)| p == &Integer::from(3)));
    }

    #[test]
    fn test_trial_division_negative() {
        // Factor -60 = -1 * 2^2 * 3 * 5
        let n = Integer::from(-60);
        let factors = factor_trial_division(&n, None);

        assert!(factors[0].0 == Integer::from(-1));
        assert!(factors.iter().any(|(p, e)| p == &Integer::from(2) && *e == 2));
        assert!(factors.iter().any(|(p, e)| p == &Integer::from(3) && *e == 1));
        assert!(factors.iter().any(|(p, e)| p == &Integer::from(5) && *e == 1));
    }

    #[test]
    fn test_trial_division_power_of_two() {
        // Factor 128 = 2^7
        let n = Integer::from(128);
        let factors = factor_trial_division(&n, None);

        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(2), 7));
    }

    // ===== Edge Cases =====

    #[test]
    fn test_trial_division_zero() {
        // 0 should return empty factorization
        let n = Integer::from(0);
        let factors = factor_trial_division(&n, None);
        assert_eq!(factors.len(), 0);
    }

    #[test]
    fn test_trial_division_one() {
        // 1 should return empty factorization
        let n = Integer::from(1);
        let factors = factor_trial_division(&n, None);
        assert_eq!(factors.len(), 0);
    }

    #[test]
    fn test_trial_division_two() {
        // 2 is the smallest prime
        let n = Integer::from(2);
        let factors = factor_trial_division(&n, None);
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(2), 1));
    }

    #[test]
    fn test_trial_division_large_prime() {
        // Factor a larger prime: 104729
        let n = Integer::from(104729);
        let factors = factor_trial_division(&n, None);
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(104729), 1));
    }

    #[test]
    fn test_trial_division_semiprime() {
        // Factor 143 = 11 * 13 (semiprime)
        let n = Integer::from(143);
        let factors = factor_trial_division(&n, None);
        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0], (Integer::from(11), 1));
        assert_eq!(factors[1], (Integer::from(13), 1));
    }

    #[test]
    fn test_trial_division_many_small_factors() {
        // Factor 2520 = 2^3 * 3^2 * 5 * 7
        let n = Integer::from(2520);
        let factors = factor_trial_division(&n, None);
        assert_eq!(factors.len(), 4);
        assert_eq!(factors[0], (Integer::from(2), 3));
        assert_eq!(factors[1], (Integer::from(3), 2));
        assert_eq!(factors[2], (Integer::from(5), 1));
        assert_eq!(factors[3], (Integer::from(7), 1));
    }

    #[test]
    fn test_trial_division_perfect_square() {
        // Factor 144 = 2^4 * 3^2
        let n = Integer::from(144);
        let factors = factor_trial_division(&n, None);
        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0], (Integer::from(2), 4));
        assert_eq!(factors[1], (Integer::from(3), 2));
    }

    #[test]
    fn test_trial_division_perfect_cube() {
        // Factor 216 = 2^3 * 3^3
        let n = Integer::from(216);
        let factors = factor_trial_division(&n, None);
        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0], (Integer::from(2), 3));
        assert_eq!(factors[1], (Integer::from(3), 3));
    }

    #[test]
    fn test_trial_division_with_large_cofactor() {
        // Factor 2 * 104729 = 209458 where 104729 is prime
        let n = Integer::from(209458);
        let factors = factor_trial_division(&n, None);
        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0], (Integer::from(2), 1));
        assert_eq!(factors[1], (Integer::from(104729), 1));
    }

    #[test]
    fn test_trial_division_limit_stops_early() {
        // Factor 221 = 13 * 17 with limit 10
        // Should leave 221 unfactored
        let n = Integer::from(221);
        let factors = factor_trial_division(&n, Some(10));
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(221), 1));
    }

    // ===== Aurifeuillian Tests =====

    #[test]
    fn test_aurifeuillian_detection() {
        // Test detection of a^n ± 1 patterns

        // 2^4 - 1 = 15
        let n = Integer::from(15);
        let result = aurifeuillian(&n);
        assert!(result.is_some());
        if let Some((a, exp, sign)) = result {
            assert_eq!(a, Integer::from(2));
            assert_eq!(exp, 4);
            assert_eq!(sign, -1);
        }

        // 2^3 + 1 = 9
        let n = Integer::from(9);
        let result = aurifeuillian(&n);
        assert!(result.is_some());
        if let Some((a, exp, sign)) = result {
            assert_eq!(a, Integer::from(2));
            assert_eq!(exp, 3);
            assert_eq!(sign, 1);
        }
    }

    #[test]
    fn test_aurifeuillian_zero_one() {
        // Edge cases: 0 and 1
        assert!(aurifeuillian(&Integer::from(0)).is_none());
        assert!(aurifeuillian(&Integer::from(1)).is_none());
    }

    #[test]
    fn test_aurifeuillian_no_pattern() {
        // Number that doesn't match a^n ± 1
        let n = Integer::from(100);
        let result = aurifeuillian(&n);
        // 100 is not of the form a^n ± 1 for small n
        assert!(result.is_none());
    }

    #[test]
    fn test_factor_aurifeuillian() {
        // Factor 2^4 - 1 = 15 = 3 * 5
        let n = Integer::from(15);
        let factors = factor_aurifeuillian(&n);

        assert_eq!(factors.len(), 2);
        assert!(factors.iter().any(|(p, e)| p == &Integer::from(3) && *e == 1));
        assert!(factors.iter().any(|(p, e)| p == &Integer::from(5) && *e == 1));
    }

    #[test]
    fn test_factor_aurifeuillian_mersenne() {
        // Factor 2^5 - 1 = 31 (prime Mersenne)
        let n = Integer::from(31);
        let factors = factor_aurifeuillian(&n);
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(31), 1));
    }

    #[test]
    fn test_factor_aurifeuillian_larger() {
        // Factor 2^6 - 1 = 63 = 3^2 * 7
        let n = Integer::from(63);
        let factors = factor_aurifeuillian(&n);

        // Verify factorization
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }
        assert_eq!(product, n);
    }

    // ===== Cunningham Tests =====

    #[test]
    fn test_factor_cunningham_mersenne() {
        // Factor 2^4 - 1 = 15 (Mersenne-like)
        let n = Integer::from(15);
        let factors = factor_cunningham(&n);

        assert_eq!(factors.len(), 2);
        assert!(factors.iter().any(|(p, _)| p == &Integer::from(3)));
        assert!(factors.iter().any(|(p, _)| p == &Integer::from(5)));
    }

    #[test]
    fn test_factor_cunningham_small() {
        // Factor 2^5 - 1 = 31 (prime)
        let n = Integer::from(31);
        let factors = factor_cunningham(&n);

        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].0, Integer::from(31));
    }

    #[test]
    fn test_factor_cunningham_fermat_like() {
        // Factor 2^4 + 1 = 17 (Fermat-like, prime)
        let n = Integer::from(17);
        let factors = factor_cunningham(&n);

        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(17), 1));
    }

    #[test]
    fn test_factor_cunningham_base_3() {
        // Factor 3^3 - 1 = 26 = 2 * 13
        let n = Integer::from(26);
        let factors = factor_cunningham(&n);

        // Verify factorization
        let mut product = Integer::one();
        for (p, e) in &factors {
            product = product * p.pow(*e);
        }
        assert_eq!(product, n);
    }

    #[test]
    fn test_factor_cunningham_no_pattern() {
        // Number that doesn't match Cunningham pattern
        let n = Integer::from(97); // prime, not b^n ± 1 for small b, n
        let factors = factor_cunningham(&n);

        // Should still factor correctly
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(97), 1));
    }

    // ===== Helper Function Tests =====

    #[test]
    fn test_combine_factors() {
        // Test combining duplicate factors
        let factors = vec![
            (Integer::from(2), 1),
            (Integer::from(3), 2),
            (Integer::from(2), 1),
            (Integer::from(3), 1),
        ];

        let combined = combine_factors(factors);

        assert_eq!(combined.len(), 2);
        assert_eq!(combined[0], (Integer::from(2), 2));
        assert_eq!(combined[1], (Integer::from(3), 3));
    }

    #[test]
    fn test_combine_factors_single() {
        let factors = vec![(Integer::from(7), 3)];
        let combined = combine_factors(factors);

        assert_eq!(combined.len(), 1);
        assert_eq!(combined[0], (Integer::from(7), 3));
    }

    #[test]
    fn test_combine_factors_empty() {
        let factors = vec![];
        let combined = combine_factors(factors);
        assert_eq!(combined.len(), 0);
    }

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&p| Integer::from(p)).collect()
    }

    fn pairs(v: &[(i64, u32)]) -> Vec<(Integer, u32)> {
        v.iter().map(|&(p, e)| (Integer::from(p), e)).collect()
    }

    fn z(s: &str) -> Integer {
        Integer::from_decimal_str(s).unwrap()
    }

    #[test]
    fn trial_division_with_primes_full() {
        // gp: factor(210) = [2,1; 3,1; 5,1; 7,1]
        let f = trial_division_with_primes(&Integer::from(210), &ints(&[2, 3, 5, 7]));
        assert_eq!(f.primes, pairs(&[(2, 1), (3, 1), (5, 1), (7, 1)]));
        assert!(f.is_complete());
        assert_eq!(f.sign, 1);
    }

    #[test]
    fn trial_division_with_primes_prime_cofactor() {
        // gp: factor(30) = [2,1; 3,1; 5,1]: the cofactor 5 < psi_12 is decided.
        let f = trial_division_with_primes(&Integer::from(30), &ints(&[2, 3]));
        assert_eq!(f.primes, pairs(&[(2, 1), (3, 1), (5, 1)]));
        assert!(f.is_complete());
        // gp: factor(-12) = [-1,1; 2,2; 3,1]: the sign is a field, not a factor.
        let f = trial_division_with_primes(&Integer::from(-12), &ints(&[2, 3]));
        assert_eq!(f.sign, -1);
        assert_eq!(f.primes, pairs(&[(2, 2), (3, 1)]));
        assert_eq!(f.value(), Integer::from(-12));
    }

    /// HEAD reported the composite cofactor as a prime: `[(1225, 1)]`, `[(1001, 1)]`.
    #[test]
    fn trial_division_with_primes_composite_cofactor() {
        // gp: factor(1225) = [5,2; 7,2]; factor(1001) = [7,1; 11,1; 13,1]
        let list = ints(&[2, 3]);
        let f = trial_division_with_primes(&Integer::from(1225), &list);
        assert!(f.primes.is_empty() && f.probable_primes.is_empty());
        assert_eq!(f.composites, pairs(&[(1225, 1)]));
        let f = trial_division_with_primes(&Integer::from(-1225), &list);
        assert_eq!(f.sign, -1);
        assert_eq!(f.composites, pairs(&[(1225, 1)]));
        assert_eq!(f.value(), Integer::from(-1225));
        let f = trial_division_with_primes(&Integer::from(1001), &list);
        assert_eq!(f.composites, pairs(&[(1001, 1)]));
    }

    /// HEAD stopped at the first entry p with p^2 > cofactor, which is wrong for an unsorted
    /// list: (8, [5, 2]) gave `[(8, 1)]`.
    #[test]
    fn trial_division_with_primes_unsorted_list() {
        // gp: factor(8) = Mat([2,3])
        let f = trial_division_with_primes(&Integer::from(8), &ints(&[5, 2]));
        assert_eq!(f.primes, pairs(&[(2, 3)]));
        assert!(f.is_complete());
        // Duplicates are harmless.
        let f = trial_division_with_primes(&Integer::from(8), &ints(&[2, 2]));
        assert_eq!(f.primes, pairs(&[(2, 3)]));
    }

    #[test]
    fn trial_division_with_primes_probable_prime_cofactor() {
        // gp: isprime(nextprime(10^29)) = 1; 2 * nextprime(10^29) = 200000000000000000000000000638
        let f = trial_division_with_primes(&z("200000000000000000000000000638"), &ints(&[2]));
        assert_eq!(f.primes, pairs(&[(2, 1)]));
        assert_eq!(
            f.probable_primes,
            vec![(z("100000000000000000000000000319"), 1)]
        );
        assert!(f.composites.is_empty());
    }

    #[test]
    fn trial_division_with_primes_large_composite_cofactor() {
        // gp: 6 * nextprime(10^14) * nextprime(2*10^14) = 120000000000053400000000005022;
        //     factor(%) = [2,1; 3,1; 100000000000031,1; 200000000000027,1]
        let f = trial_division_with_primes(&z("120000000000053400000000005022"), &ints(&[2, 3]));
        assert_eq!(f.primes, pairs(&[(2, 1), (3, 1)]));
        assert_eq!(f.composites, vec![(z("20000000000008900000000000837"), 1)]);
    }

    #[test]
    fn trial_division_with_primes_zero_and_units() {
        for (n, sign) in [(0i64, 0i8), (1, 1), (-1, -1)] {
            let f = trial_division_with_primes(&Integer::from(n), &ints(&[2, 3]));
            assert_eq!(f.sign, sign, "n = {n}");
            assert!(f.primes.is_empty() && f.is_complete(), "n = {n}");
            assert_eq!(f.value(), Integer::from(n));
        }
    }

    #[test]
    #[should_panic(expected = "trial_division_with_primes: the list entry 4 is not prime")]
    fn trial_division_with_primes_rejects_composite_entry() {
        let _ = trial_division_with_primes(&Integer::from(12), &ints(&[4, 3]));
    }

    #[test]
    #[should_panic(expected = "trial_division_with_primes: the list entry 1 is below 2")]
    fn trial_division_with_primes_rejects_one() {
        let _ = trial_division_with_primes(&Integer::from(12), &ints(&[1, 3]));
    }

    // ===== Correctness Verification Tests =====

    #[test]
    fn test_verify_factorization_product() {
        // Verify that the product of factors equals the original number
        let test_numbers = vec![60, 100, 144, 1000, 2520, 9999];

        for num in test_numbers {
            let n = Integer::from(num);
            let factors = factor_trial_division(&n, None);

            let mut product = Integer::one();
            for (prime, exp) in factors {
                product = product * prime.pow(exp);
            }

            assert_eq!(product, n, "Factorization of {} is incorrect", num);
        }
    }

    #[test]
    fn test_verify_all_factors_prime() {
        // Verify that all factors are actually prime
        let n = Integer::from(2520); // 2^3 * 3^2 * 5 * 7
        let factors = factor_trial_division(&n, None);

        for (prime, _) in factors {
            if prime > Integer::one() {
                assert!(crate::prime::is_prime(&prime),
                    "{} is not prime", prime);
            }
        }
    }
}
