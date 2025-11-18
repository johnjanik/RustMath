//! # Sum of Squares
//!
//! This module provides algorithms for expressing integers as sums of squares.
//!
//! ## Overview
//!
//! Classical results from number theory:
//! - **Lagrange's Four Square Theorem**: Every non-negative integer can be expressed
//!   as the sum of four integer squares
//! - **Fermat's Two Square Theorem**: A prime p can be expressed as sum of two squares
//!   iff p = 2 or p ≡ 1 (mod 4)
//! - **Legendre's Three Square Theorem**: A positive integer can be expressed as sum
//!   of three squares iff it's not of the form 4^a(8b+7)
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::sum_of_squares;
//!
//! // Express 25 as sum of two squares: 25 = 3² + 4²
//! let (a, b) = sum_of_squares::two_squares(25).unwrap();
//! assert_eq!(a*a + b*b, 25);
//!
//! // Express 7 as sum of four squares: 7 = 2² + 1² + 1² + 1²
//! let (a, b, c, d) = sum_of_squares::four_squares(7);
//! assert_eq!(a*a + b*b + c*c + d*d, 7);
//! ```

use rustmath_integers::Integer;

/// Check if a positive integer can be expressed as sum of two squares
///
/// Uses Fermat's criterion: n is sum of two squares iff in its prime factorization,
/// every prime of the form 4k+3 occurs to an even power.
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::is_sum_of_two_squares;
///
/// assert!(is_sum_of_two_squares(25)); // 25 = 3² + 4²
/// assert!(is_sum_of_two_squares(2));  // 2 = 1² + 1²
/// assert!(!is_sum_of_two_squares(3)); // 3 cannot be expressed
/// ```
pub fn is_sum_of_two_squares(n: u64) -> bool {
    if n == 0 {
        return true;
    }

    // Factor out all 2s
    let mut n = n;
    while n % 2 == 0 {
        n /= 2;
    }

    // Check odd prime factors
    let mut p = 3;
    while p * p <= n {
        if n % p == 0 {
            let mut count = 0;
            while n % p == 0 {
                n /= p;
                count += 1;
            }

            // If p ≡ 3 (mod 4) and appears odd number of times, return false
            if p % 4 == 3 && count % 2 == 1 {
                return false;
            }
        }
        p += 2;
    }

    // Check remaining factor
    if n > 1 && n % 4 == 3 {
        return false;
    }

    true
}

/// Express n as sum of two squares: n = a² + b² with a ≥ b ≥ 0
///
/// Returns None if n cannot be expressed as sum of two squares.
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::two_squares;
///
/// let (a, b) = two_squares(25).unwrap();
/// assert_eq!(a*a + b*b, 25);
/// assert!(a >= b);
/// ```
pub fn two_squares(n: u64) -> Option<(u64, u64)> {
    if !is_sum_of_two_squares(n) {
        return None;
    }

    // Simple exhaustive search for small n
    let limit = (n as f64).sqrt() as u64 + 1;
    for a in 0..=limit {
        let a_sq = a * a;
        if a_sq > n {
            break;
        }

        let remainder = n - a_sq;
        let b = (remainder as f64).sqrt() as u64;

        if b * b == remainder {
            return Some((a.max(b), a.min(b)));
        }
    }

    None
}

/// Internal implementation for two_squares (mirrors SageMath's interface)
pub fn two_squares_pyx(n: u64) -> Option<(u64, u64)> {
    two_squares(n)
}

/// Check if n can be expressed as sum of three squares
///
/// By Legendre's three-square theorem, n is sum of three squares iff
/// n is not of the form 4^a(8b+7).
pub fn is_sum_of_three_squares(n: u64) -> bool {
    if n == 0 {
        return true;
    }

    let mut n = n;

    // Remove all factors of 4
    while n % 4 == 0 {
        n /= 4;
    }

    // Check if it's of form 8k+7
    n % 8 != 7
}

/// Express n as sum of three squares: n = a² + b² + c²
///
/// Returns None if n cannot be expressed as sum of three squares.
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::three_squares;
///
/// let (a, b, c) = three_squares(14).unwrap();
/// assert_eq!(a*a + b*b + c*c, 14);
/// ```
pub fn three_squares(n: u64) -> Option<(u64, u64, u64)> {
    if !is_sum_of_three_squares(n) {
        return None;
    }

    // Try all combinations a² + two_squares(n - a²)
    let limit = (n as f64).sqrt() as u64 + 1;
    for a in 0..=limit {
        let a_sq = a * a;
        if a_sq > n {
            break;
        }

        let remainder = n - a_sq;
        if let Some((b, c)) = two_squares(remainder) {
            return Some((a, b, c));
        }
    }

    None
}

/// Internal implementation for three_squares (mirrors SageMath's interface)
pub fn three_squares_pyx(n: u64) -> Option<(u64, u64, u64)> {
    three_squares(n)
}

/// Express n as sum of four squares: n = a² + b² + c² + d²
///
/// By Lagrange's four-square theorem, this always succeeds for non-negative n.
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::four_squares;
///
/// let (a, b, c, d) = four_squares(7);
/// assert_eq!(a*a + b*b + c*c + d*d, 7);
/// ```
pub fn four_squares(n: u64) -> (u64, u64, u64, u64) {
    if n == 0 {
        return (0, 0, 0, 0);
    }

    // Try to express as sum of fewer squares first
    if let Some((a, b)) = two_squares(n) {
        return (a, b, 0, 0);
    }

    if let Some((a, b, c)) = three_squares(n) {
        return (a, b, c, 0);
    }

    // Otherwise, use exhaustive search
    let limit = (n as f64).sqrt() as u64 + 1;
    for a in 0..=limit {
        let a_sq = a * a;
        if a_sq > n {
            break;
        }

        if let Some((b, c, d)) = three_squares(n - a_sq) {
            return (a, b, c, d);
        }
    }

    // This should never happen by Lagrange's theorem
    panic!("Four square theorem failed - this should be impossible!");
}

/// Internal implementation for four_squares (mirrors SageMath's interface)
pub fn four_squares_pyx(n: u64) -> (u64, u64, u64, u64) {
    four_squares(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_sum_of_two_squares() {
        assert!(is_sum_of_two_squares(0));
        assert!(is_sum_of_two_squares(1));
        assert!(is_sum_of_two_squares(2));
        assert!(!is_sum_of_two_squares(3));
        assert!(is_sum_of_two_squares(4));
        assert!(is_sum_of_two_squares(5)); // 1² + 2²
        assert!(!is_sum_of_two_squares(6));
        assert!(!is_sum_of_two_squares(7));
        assert!(is_sum_of_two_squares(8));
        assert!(is_sum_of_two_squares(25)); // 3² + 4²
        assert!(is_sum_of_two_squares(50)); // 1² + 7²
    }

    #[test]
    fn test_two_squares() {
        assert_eq!(two_squares(0), Some((0, 0)));
        assert_eq!(two_squares(1), Some((1, 0)));
        assert_eq!(two_squares(2), Some((1, 1)));
        assert_eq!(two_squares(3), None);
        assert_eq!(two_squares(4), Some((2, 0)));
        assert_eq!(two_squares(5), Some((2, 1)));

        let (a, b) = two_squares(25).unwrap();
        assert_eq!(a * a + b * b, 25);
        assert!(a >= b);

        let (a, b) = two_squares(50).unwrap();
        assert_eq!(a * a + b * b, 50);
    }

    #[test]
    fn test_two_squares_pyx() {
        assert_eq!(two_squares_pyx(25), Some((4, 3)));
    }

    #[test]
    fn test_is_sum_of_three_squares() {
        assert!(is_sum_of_three_squares(0));
        assert!(is_sum_of_three_squares(1));
        assert!(is_sum_of_three_squares(2));
        assert!(is_sum_of_three_squares(3));
        assert!(!is_sum_of_three_squares(7)); // 7 = 4^0(8*0+7)
        assert!(is_sum_of_three_squares(14));
        assert!(!is_sum_of_three_squares(15)); // 15 = 4^0(8*1+7)
    }

    #[test]
    fn test_three_squares() {
        assert_eq!(three_squares(0), Some((0, 0, 0)));
        assert_eq!(three_squares(7), None);

        let (a, b, c) = three_squares(14).unwrap();
        assert_eq!(a * a + b * b + c * c, 14);

        let (a, b, c) = three_squares(3).unwrap();
        assert_eq!(a * a + b * b + c * c, 3);
    }

    #[test]
    fn test_three_squares_pyx() {
        let result = three_squares_pyx(14);
        assert!(result.is_some());
        if let Some((a, b, c)) = result {
            assert_eq!(a * a + b * b + c * c, 14);
        }
    }

    #[test]
    fn test_four_squares() {
        // Test that every number can be expressed
        for n in 0..100 {
            let (a, b, c, d) = four_squares(n);
            assert_eq!(a * a + b * b + c * c + d * d, n);
        }

        // Specific cases
        let (a, b, c, d) = four_squares(7);
        assert_eq!(a * a + b * b + c * c + d * d, 7);

        let (a, b, c, d) = four_squares(15);
        assert_eq!(a * a + b * b + c * c + d * d, 15);
    }

    #[test]
    fn test_four_squares_pyx() {
        let (a, b, c, d) = four_squares_pyx(7);
        assert_eq!(a * a + b * b + c * c + d * d, 7);
    }

    #[test]
    fn test_lagrange_theorem() {
        // Verify Lagrange's four-square theorem for first 200 positive integers
        for n in 1..=200 {
            let (a, b, c, d) = four_squares(n);
            assert_eq!(
                a * a + b * b + c * c + d * d,
                n,
                "Failed for n = {}",
                n
            );
        }
    }
}
