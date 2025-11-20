//! Derangements - permutations with no fixed points
//!
//! A derangement is a permutation where no element appears in its original position.
//! For example, [1, 0, 3, 2] is a derangement of {0, 1, 2, 3} because:
//! - 0 is mapped to 1 (not to 0)
//! - 1 is mapped to 0 (not to 1)
//! - 2 is mapped to 3 (not to 2)
//! - 3 is mapped to 2 (not to 3)

use crate::permutations::Permutation;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// Count the number of derangements of n elements using inclusion-exclusion
///
/// The formula is: D(n) = n! * Σ(i=0 to n) (-1)^i / i!
///
/// This is approximately n!/e for large n.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::derangements::count_derangements;
/// use rustmath_integers::Integer;
///
/// assert_eq!(count_derangements(0), Integer::from(1));
/// assert_eq!(count_derangements(1), Integer::from(0));
/// assert_eq!(count_derangements(2), Integer::from(1));
/// assert_eq!(count_derangements(3), Integer::from(2));
/// assert_eq!(count_derangements(4), Integer::from(9));
/// ```
pub fn count_derangements(n: u32) -> Integer {
    if n == 0 {
        return Integer::one();
    }
    if n == 1 {
        return Integer::zero();
    }

    // Use the formula: D(n) = n! * Σ(i=0 to n) (-1)^i / i!
    // We can compute this exactly using rationals
    let mut sum = Rational::zero();

    for i in 0..=n {
        let sign = if i % 2 == 0 { 1 } else { -1 };
        let factorial_i = crate::factorial(i);
        let term = Rational::from_integer(Integer::from(sign)) / Rational::from_integer(factorial_i);
        sum = sum + term;
    }

    let factorial_n = crate::factorial(n);
    let result = Rational::from_integer(factorial_n) * sum;

    // The result should always be an integer
    result.numerator().clone()
}

/// Count derangements using the recurrence relation
///
/// D(n) = (n-1) * [D(n-1) + D(n-2)]
///
/// This is more efficient than inclusion-exclusion for computing a single value,
/// but inclusion-exclusion is better if you understand the mathematical foundation.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::derangements::count_derangements_recurrence;
/// use rustmath_integers::Integer;
///
/// assert_eq!(count_derangements_recurrence(4), Integer::from(9));
/// assert_eq!(count_derangements_recurrence(5), Integer::from(44));
/// ```
pub fn count_derangements_recurrence(n: u32) -> Integer {
    if n == 0 {
        return Integer::one();
    }
    if n == 1 {
        return Integer::zero();
    }
    if n == 2 {
        return Integer::one();
    }

    let mut d_prev2 = Integer::one(); // D(2)
    let mut d_prev1 = Integer::from(2); // D(3)

    for i in 4..=n {
        let d_i = Integer::from(i - 1) * (d_prev1.clone() + d_prev2.clone());
        d_prev2 = d_prev1;
        d_prev1 = d_i;
    }

    d_prev1
}

/// Check if a permutation is a derangement
///
/// Returns true if no element is in its original position
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::derangements::is_derangement;
/// use rustmath_combinatorics::Permutation;
///
/// let perm1 = Permutation::from_vec(vec![1, 0, 3, 2]).unwrap();
/// assert!(is_derangement(&perm1));
///
/// let perm2 = Permutation::from_vec(vec![0, 1, 2, 3]).unwrap();
/// assert!(!is_derangement(&perm2)); // Identity is not a derangement
///
/// let perm3 = Permutation::from_vec(vec![1, 0, 2]).unwrap();
/// assert!(!is_derangement(&perm3)); // 2 is in its original position
/// ```
pub fn is_derangement(perm: &Permutation) -> bool {
    for i in 0..perm.size() {
        if perm.apply(i) == Some(i) {
            return false;
        }
    }
    true
}

/// Generate all derangements of n elements
///
/// Uses a recursive construction algorithm that builds derangements efficiently
/// by ensuring no element is placed in its original position.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::derangements::all_derangements;
///
/// let derangements = all_derangements(3);
/// assert_eq!(derangements.len(), 2); // D(3) = 2
///
/// let derangements_4 = all_derangements(4);
/// assert_eq!(derangements_4.len(), 9); // D(4) = 9
/// ```
pub fn all_derangements(n: usize) -> Vec<Permutation> {
    if n == 0 {
        return vec![Permutation::from_vec(vec![]).unwrap()];
    }
    if n == 1 {
        return vec![]; // No derangements of a single element
    }

    let mut result = Vec::new();
    let mut perm = vec![0; n];
    generate_derangements(&mut perm, 0, &mut vec![false; n], &mut result);
    result
}

/// Recursive helper function to generate derangements
///
/// This uses backtracking to construct all valid derangements.
fn generate_derangements(
    perm: &mut Vec<usize>,
    pos: usize,
    used: &mut Vec<bool>,
    result: &mut Vec<Permutation>,
) {
    let n = perm.len();

    // Base case: we've filled all positions
    if pos == n {
        result.push(Permutation::from_vec(perm.clone()).unwrap());
        return;
    }

    // Try placing each unused element at position pos
    for val in 0..n {
        // Skip if already used or if it would be a fixed point
        if used[val] || val == pos {
            continue;
        }

        // Place val at position pos
        perm[pos] = val;
        used[val] = true;

        // Recursively fill the rest
        generate_derangements(perm, pos + 1, used, result);

        // Backtrack
        used[val] = false;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_derangements() {
        // D(0) = 1 (by convention, empty permutation)
        assert_eq!(count_derangements(0), Integer::from(1));

        // D(1) = 0 (no way to derange a single element)
        assert_eq!(count_derangements(1), Integer::from(0));

        // D(2) = 1 (only [1, 0])
        assert_eq!(count_derangements(2), Integer::from(1));

        // D(3) = 2 ([1, 2, 0] and [2, 0, 1])
        assert_eq!(count_derangements(3), Integer::from(2));

        // D(4) = 9
        assert_eq!(count_derangements(4), Integer::from(9));

        // D(5) = 44
        assert_eq!(count_derangements(5), Integer::from(44));

        // D(6) = 265
        assert_eq!(count_derangements(6), Integer::from(265));

        // D(10) = 1334961
        assert_eq!(count_derangements(10), Integer::from(1334961));
    }

    #[test]
    fn test_count_derangements_recurrence() {
        // Verify that recurrence formula gives same results
        for n in 0..=10 {
            assert_eq!(
                count_derangements(n),
                count_derangements_recurrence(n),
                "Mismatch at n={}",
                n
            );
        }
    }

    #[test]
    fn test_is_derangement() {
        // [1, 0] is a derangement
        let perm1 = Permutation::from_vec(vec![1, 0]).unwrap();
        assert!(is_derangement(&perm1));

        // [0, 1] is not a derangement (identity)
        let perm2 = Permutation::from_vec(vec![0, 1]).unwrap();
        assert!(!is_derangement(&perm2));

        // [1, 2, 0] is a derangement
        let perm3 = Permutation::from_vec(vec![1, 2, 0]).unwrap();
        assert!(is_derangement(&perm3));

        // [1, 0, 2] is not a derangement (2 is fixed)
        let perm4 = Permutation::from_vec(vec![1, 0, 2]).unwrap();
        assert!(!is_derangement(&perm4));

        // [2, 0, 1] is a derangement
        let perm5 = Permutation::from_vec(vec![2, 0, 1]).unwrap();
        assert!(is_derangement(&perm5));
    }

    #[test]
    fn test_all_derangements() {
        // D(0) = 1
        let d0 = all_derangements(0);
        assert_eq!(d0.len(), 1);

        // D(1) = 0
        let d1 = all_derangements(1);
        assert_eq!(d1.len(), 0);

        // D(2) = 1
        let d2 = all_derangements(2);
        assert_eq!(d2.len(), 1);
        assert!(is_derangement(&d2[0]));

        // D(3) = 2
        let d3 = all_derangements(3);
        assert_eq!(d3.len(), 2);
        for perm in &d3 {
            assert!(is_derangement(perm));
        }

        // D(4) = 9
        let d4 = all_derangements(4);
        assert_eq!(d4.len(), 9);
        for perm in &d4 {
            assert!(is_derangement(perm));
        }

        // Verify uniqueness
        for i in 0..d4.len() {
            for j in (i + 1)..d4.len() {
                assert_ne!(d4[i], d4[j], "Found duplicate derangements");
            }
        }
    }


    #[test]
    fn test_derangement_properties() {
        // All derangements of size n should have no fixed points
        for n in 2..=5 {
            let derangements = all_derangements(n);
            for perm in &derangements {
                for i in 0..perm.size() {
                    assert_ne!(
                        perm.apply(i),
                        Some(i),
                        "Found fixed point at position {} in {:?}",
                        i,
                        perm.as_slice()
                    );
                }
            }
        }
    }

    #[test]
    fn test_derangement_count_formula() {
        // Verify that the count matches the expected values
        let expected = vec![1, 0, 1, 2, 9, 44, 265, 1854, 14833];
        for (n, &expected_count) in expected.iter().enumerate() {
            assert_eq!(
                count_derangements(n as u32),
                Integer::from(expected_count),
                "Count mismatch at n={}",
                n
            );
        }
    }

    #[test]
    fn test_derangement_ratio() {
        // For large n, D(n)/n! should approach 1/e ≈ 0.367879...
        // Let's verify this for n=10
        let d10 = count_derangements(10);
        let fact10 = crate::factorial(10);

        // D(10) / 10! as a rational
        let ratio = Rational::from_integer(d10) / Rational::from_integer(fact10);

        // 1/e ≈ 0.3678794411714423
        // D(10)/10! should be very close to this
        // For n=10: D(10) = 1334961, 10! = 3628800
        // Ratio = 1334961/3628800 ≈ 0.36787918...

        // Convert to f64 for approximate comparison
        let ratio_f64 = ratio.numerator().to_f64().unwrap() / ratio.denominator().to_f64().unwrap();
        let one_over_e = 1.0 / std::f64::consts::E;

        assert!(
            (ratio_f64 - one_over_e).abs() < 0.0001,
            "D(10)/10! = {} is not close to 1/e = {}",
            ratio_f64,
            one_over_e
        );
    }

    #[test]
    fn test_empty_derangement() {
        // The empty permutation is considered a derangement by convention
        let empty = Permutation::from_vec(vec![]).unwrap();
        assert!(is_derangement(&empty));
    }
}
