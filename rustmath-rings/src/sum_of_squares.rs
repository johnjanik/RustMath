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
/// # Input Constraints
/// For inputs ≥ 2^32, behavior is platform-dependent (u64 on 64-bit systems).
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::is_sum_of_two_squares;
///
/// assert!(is_sum_of_two_squares(0));
/// assert!(is_sum_of_two_squares(1));
/// assert!(is_sum_of_two_squares(2));
/// assert!(!is_sum_of_two_squares(3));
/// assert!(is_sum_of_two_squares(25)); // 25 = 3² + 4²
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

/// Express n as sum of two squares: n = a² + b² with a ≤ b
///
/// Returns the lexicographically smallest representation where a ≤ b.
/// Returns None if n cannot be expressed as sum of two squares.
///
/// # Input Constraints
/// For inputs ≥ 2^32, behavior is platform-dependent (u64 on 64-bit systems).
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::two_squares;
///
/// assert_eq!(two_squares(0), Some((0, 0)));
/// assert_eq!(two_squares(2), Some((1, 1)));
/// let (a, b) = two_squares(25).unwrap();
/// assert_eq!(a*a + b*b, 25);
/// assert!(a <= b);
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
            // Return in ascending order: a ≤ b
            return Some((a.min(b), a.max(b)));
        }
    }

    None
}

/// Internal implementation for two_squares (mirrors SageMath's interface)
///
/// Returns a 2-tuple `(i, j)` where i² + j² = n and i ≤ j, representing the
/// lexicographically smallest solution. Returns None if n is not a sum of two squares.
///
/// This function mirrors SageMath's `two_squares_pyx` interface.
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::two_squares_pyx;
///
/// assert_eq!(two_squares_pyx(0), Some((0, 0)));
/// assert_eq!(two_squares_pyx(2), Some((1, 1)));
/// assert_eq!(two_squares_pyx(106), Some((5, 9)));
/// assert_eq!(two_squares_pyx(3), None);
/// ```
pub fn two_squares_pyx(n: u64) -> Option<(u64, u64)> {
    two_squares(n)
}

/// Check if n is a sum of two squares (mirrors SageMath's interface)
///
/// This function mirrors SageMath's `is_sum_of_two_squares_pyx` interface.
/// Returns true if n can be expressed as i² + j² for some nonnegative integers i, j.
///
/// # Input Constraints
/// For inputs ≥ 2^32, behavior is platform-dependent (u64 on 64-bit systems).
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::is_sum_of_two_squares_pyx;
///
/// assert!(is_sum_of_two_squares_pyx(0));
/// assert!(is_sum_of_two_squares_pyx(1));
/// assert!(is_sum_of_two_squares_pyx(2));
/// assert!(!is_sum_of_two_squares_pyx(3));
/// assert!(is_sum_of_two_squares_pyx(5));
/// ```
pub fn is_sum_of_two_squares_pyx(n: u64) -> bool {
    is_sum_of_two_squares(n)
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

/// Express n as sum of three squares: n = a² + b² + c² with a ≤ b ≤ c
///
/// Returns a 3-tuple `(a, b, c)` where a² + b² + c² = n and a ≤ b ≤ c.
/// Returns None if n cannot be expressed as sum of three squares.
///
/// By Legendre's three-square theorem, n is a sum of three squares if and only if
/// it is not of the form 4^a(8b+7).
///
/// # Input Constraints
/// For inputs ≥ 2^32, behavior is platform-dependent (u64 on 64-bit systems).
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::three_squares;
///
/// assert_eq!(three_squares(5), Some((0, 1, 2)));
/// assert_eq!(three_squares(107), Some((1, 5, 9)));
/// let (a, b, c) = three_squares(14).unwrap();
/// assert_eq!(a*a + b*b + c*c, 14);
/// assert!(a <= b && b <= c);
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
            // Sort the three values to ensure a ≤ b ≤ c
            let mut values = [a, b, c];
            values.sort_unstable();
            return Some((values[0], values[1], values[2]));
        }
    }

    None
}

/// Internal implementation for three_squares (mirrors SageMath's interface)
///
/// Returns a 3-tuple `(i, j, k)` where i² + j² + k² = n and i ≤ j ≤ k.
/// Returns None if n cannot be expressed as sum of three squares.
///
/// This function mirrors SageMath's `three_squares_pyx` interface.
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::three_squares_pyx;
///
/// assert_eq!(three_squares_pyx(5), Some((0, 1, 2)));
/// assert_eq!(three_squares_pyx(107), Some((1, 5, 9)));
/// assert_eq!(three_squares_pyx(7), None);
/// ```
pub fn three_squares_pyx(n: u64) -> Option<(u64, u64, u64)> {
    three_squares(n)
}

/// Express n as sum of four squares: n = a² + b² + c² + d² with a ≤ b ≤ c ≤ d
///
/// Returns a 4-tuple `(a, b, c, d)` where a² + b² + c² + d² = n and a ≤ b ≤ c ≤ d.
///
/// By Lagrange's four-square theorem, this always succeeds for non-negative n.
///
/// # Input Constraints
/// For inputs ≥ 2^32, behavior is platform-dependent (u64 on 64-bit systems).
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::four_squares;
///
/// assert_eq!(four_squares(15447), (2, 5, 17, 123));
/// let (a, b, c, d) = four_squares(7);
/// assert_eq!(a*a + b*b + c*c + d*d, 7);
/// assert!(a <= b && b <= c && c <= d);
/// ```
pub fn four_squares(n: u64) -> (u64, u64, u64, u64) {
    if n == 0 {
        return (0, 0, 0, 0);
    }

    // Try to express as sum of fewer squares first
    if let Some((a, b)) = two_squares(n) {
        // Already in ascending order
        return (0, 0, a, b);
    }

    if let Some((a, b, c)) = three_squares(n) {
        // Already in ascending order
        return (0, a, b, c);
    }

    // Otherwise, use exhaustive search
    let limit = (n as f64).sqrt() as u64 + 1;
    for a in 0..=limit {
        let a_sq = a * a;
        if a_sq > n {
            break;
        }

        if let Some((b, c, d)) = three_squares(n - a_sq) {
            // Sort all four values to ensure a ≤ b ≤ c ≤ d
            let mut values = [a, b, c, d];
            values.sort_unstable();
            return (values[0], values[1], values[2], values[3]);
        }
    }

    // This should never happen by Lagrange's theorem
    panic!("Four square theorem failed - this should be impossible!");
}

/// Internal implementation for four_squares (mirrors SageMath's interface)
///
/// Returns a 4-tuple `(i, j, k, l)` where i² + j² + k² + l² = n and i ≤ j ≤ k ≤ l.
///
/// This function mirrors SageMath's `four_squares_pyx` interface.
///
/// # Examples
/// ```
/// use rustmath_rings::sum_of_squares::four_squares_pyx;
///
/// assert_eq!(four_squares_pyx(15447), (2, 5, 17, 123));
/// assert_eq!(four_squares_pyx(523439), (3, 5, 26, 723));
/// let (a, b, c, d) = four_squares_pyx(7);
/// assert_eq!(a*a + b*b + c*c + d*d, 7);
/// ```
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
        // Test basic cases with ascending order (a ≤ b)
        assert_eq!(two_squares(0), Some((0, 0)));
        assert_eq!(two_squares(1), Some((0, 1)));
        assert_eq!(two_squares(2), Some((1, 1)));
        assert_eq!(two_squares(3), None);
        assert_eq!(two_squares(4), Some((0, 2)));
        assert_eq!(two_squares(5), Some((1, 2)));
        assert_eq!(two_squares(106), Some((5, 9)));

        // Verify ordering and sum
        let (a, b) = two_squares(25).unwrap();
        assert_eq!(a * a + b * b, 25);
        assert!(a <= b, "Expected a <= b, got {} > {}", a, b);

        let (a, b) = two_squares(50).unwrap();
        assert_eq!(a * a + b * b, 50);
        assert!(a <= b);
    }

    #[test]
    fn test_two_squares_pyx() {
        // Test SageMath examples with ascending order
        assert_eq!(two_squares_pyx(0), Some((0, 0)));
        assert_eq!(two_squares_pyx(2), Some((1, 1)));
        assert_eq!(two_squares_pyx(106), Some((5, 9)));
        assert_eq!(two_squares_pyx(25), Some((3, 4)));
        assert_eq!(two_squares_pyx(3), None);
    }

    #[test]
    fn test_is_sum_of_two_squares_pyx() {
        // Test SageMath examples
        assert!(is_sum_of_two_squares_pyx(0));
        assert!(is_sum_of_two_squares_pyx(1));
        assert!(is_sum_of_two_squares_pyx(2));
        assert!(!is_sum_of_two_squares_pyx(3));
        assert!(is_sum_of_two_squares_pyx(4));
        assert!(is_sum_of_two_squares_pyx(5));
        assert!(!is_sum_of_two_squares_pyx(6));

        // Verify range [0,30] matches SageMath
        let expected_sums: Vec<u64> = vec![0, 1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29];
        for n in 0..=30 {
            assert_eq!(
                is_sum_of_two_squares_pyx(n),
                expected_sums.contains(&n),
                "Failed for n = {}",
                n
            );
        }
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
        // Test SageMath examples with ascending order (a ≤ b ≤ c)
        assert_eq!(three_squares(0), Some((0, 0, 0)));
        assert_eq!(three_squares(5), Some((0, 1, 2)));
        assert_eq!(three_squares(107), Some((1, 5, 9)));
        assert_eq!(three_squares(7), None); // 7 = 4^0(8*0+7) - not a sum of 3 squares

        // Verify ordering and sum
        let (a, b, c) = three_squares(14).unwrap();
        assert_eq!(a * a + b * b + c * c, 14);
        assert!(a <= b && b <= c, "Expected a <= b <= c, got {} {} {}", a, b, c);

        let (a, b, c) = three_squares(3).unwrap();
        assert_eq!(a * a + b * b + c * c, 3);
        assert!(a <= b && b <= c);
    }

    #[test]
    fn test_three_squares_pyx() {
        // Test SageMath examples
        assert_eq!(three_squares_pyx(5), Some((0, 1, 2)));
        assert_eq!(three_squares_pyx(107), Some((1, 5, 9)));
        assert_eq!(three_squares_pyx(7), None);

        let result = three_squares_pyx(14);
        assert!(result.is_some());
        if let Some((a, b, c)) = result {
            assert_eq!(a * a + b * b + c * c, 14);
            assert!(a <= b && b <= c);
        }
    }

    #[test]
    fn test_four_squares() {
        // Test SageMath examples with ascending order (a ≤ b ≤ c ≤ d)
        assert_eq!(four_squares(15447), (2, 5, 17, 123));
        assert_eq!(four_squares(523439), (3, 5, 26, 723));

        // Test that every number can be expressed with correct ordering
        for n in 0..100 {
            let (a, b, c, d) = four_squares(n);
            assert_eq!(a * a + b * b + c * c + d * d, n);
            assert!(
                a <= b && b <= c && c <= d,
                "Failed ordering for n = {}: ({}, {}, {}, {})",
                n, a, b, c, d
            );
        }

        // Specific cases
        let (a, b, c, d) = four_squares(7);
        assert_eq!(a * a + b * b + c * c + d * d, 7);
        assert!(a <= b && b <= c && c <= d);

        let (a, b, c, d) = four_squares(15);
        assert_eq!(a * a + b * b + c * c + d * d, 15);
        assert!(a <= b && b <= c && c <= d);
    }

    #[test]
    fn test_four_squares_pyx() {
        // Test SageMath examples
        assert_eq!(four_squares_pyx(15447), (2, 5, 17, 123));
        assert_eq!(four_squares_pyx(523439), (3, 5, 26, 723));

        let (a, b, c, d) = four_squares_pyx(7);
        assert_eq!(a * a + b * b + c * c + d * d, 7);
        assert!(a <= b && b <= c && c <= d);
    }

    #[test]
    fn test_lagrange_theorem() {
        // Verify Lagrange's four-square theorem for first 200 positive integers
        // with correct ascending order
        for n in 1..=200 {
            let (a, b, c, d) = four_squares(n);
            assert_eq!(
                a * a + b * b + c * c + d * d,
                n,
                "Failed for n = {}",
                n
            );
            assert!(
                a <= b && b <= c && c <= d,
                "Failed ordering for n = {}: ({}, {}, {}, {})",
                n, a, b, c, d
            );
        }
    }
}
