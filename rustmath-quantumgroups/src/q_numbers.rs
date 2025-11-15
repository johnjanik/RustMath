//! Q-Numbers - Quantum Analogs of Integers and Combinatorial Functions
//!
//! This module implements q-analogs used in quantum group theory.
//! These differ from the standard q-analogs used in combinatorics.
//!
//! Corresponds to sage.algebras.quantum_groups.q_numbers

use rustmath_core::Ring;
use rustmath_polynomials::laurent::LaurentPolynomial;

/// The q-analog of a nonnegative integer n
///
/// Computes [n]_q = (q^n - q^{-n}) / (q - q^{-1})
///            = q^{n-1} + q^{n-3} + ... + q^{-n+3} + q^{-n+1}
///            = sum_{i=0}^{n-1} q^{n - 2i - 1}
///
/// # Arguments
///
/// * `n` - A nonnegative integer
///
/// # Returns
///
/// A Laurent polynomial in q representing [n]_q
///
/// # Examples
///
/// ```
/// use rustmath_quantumgroups::q_numbers::q_int;
/// use rustmath_integers::Integer;
///
/// // [2]_q = q^{-1} + q
/// let result = q_int::<Integer>(2);
/// ```
pub fn q_int<R: Ring>(n: u32) -> LaurentPolynomial<R> {
    if n == 0 {
        return LaurentPolynomial::new();
    }

    let mut result = LaurentPolynomial::new();

    for i in 0..n {
        let power = (n as i32) - 2 * (i as i32) - 1;
        result = result + LaurentPolynomial::monomial(R::one(), power);
    }

    result
}

/// The q-factorial [n]_q!
///
/// Computes [n]_q! = [n]_q * [n-1]_q * ... * [2]_q * [1]_q
///
/// # Arguments
///
/// * `n` - A nonnegative integer
///
/// # Returns
///
/// A Laurent polynomial in q representing [n]_q!
///
/// # Examples
///
/// ```
/// use rustmath_quantumgroups::q_numbers::q_factorial;
/// use rustmath_integers::Integer;
///
/// // [3]_q! = [3]_q * [2]_q * [1]_q
/// let result = q_factorial::<Integer>(3);
/// ```
pub fn q_factorial<R: Ring>(n: u32) -> LaurentPolynomial<R> {
    if n == 0 {
        return LaurentPolynomial::one();
    }

    let mut result = LaurentPolynomial::one();

    for i in 1..=n {
        result = result * q_int::<R>(i);
    }

    result
}

/// The q-binomial coefficient [n choose k]_q (Gaussian binomial coefficient)
///
/// Computes [n choose k]_q using the recurrence relation:
/// [n choose k]_q = [n-1 choose k]_q + q^{n-k} * [n-1 choose k-1]_q
///
/// # Arguments
///
/// * `n` - The upper index (nonnegative integer)
/// * `k` - The lower index (nonnegative integer with k <= n)
///
/// # Returns
///
/// A Laurent polynomial in q representing [n choose k]_q, or zero if k > n
///
/// # Examples
///
/// ```
/// use rustmath_quantumgroups::q_numbers::q_binomial;
/// use rustmath_integers::Integer;
///
/// // [n choose 0]_q = 1
/// let result = q_binomial::<Integer>(5, 0);
/// ```
pub fn q_binomial<R: Ring>(n: u32, k: u32) -> LaurentPolynomial<R> {
    if k > n {
        return LaurentPolynomial::new();
    }

    if k == 0 || k == n {
        return LaurentPolynomial::one();
    }

    // Special case: [n choose 1]_q = [n]_q
    if k == 1 {
        return q_int::<R>(n);
    }

    // Special case: [n choose n-1]_q = [n]_q (by symmetry)
    if k == n - 1 {
        return q_int::<R>(n);
    }

    // Use recurrence relation: [n choose k]_q = [n-1 choose k]_q + q^{n-k} * [n-1 choose k-1]_q
    q_binomial_recurrence(n, k)
}

/// Helper function using recurrence relation for q_binomial
///
/// Uses: [n choose k]_q = [n-1 choose k]_q + q^{n-k} * [n-1 choose k-1]_q
fn q_binomial_recurrence<R: Ring>(n: u32, k: u32) -> LaurentPolynomial<R> {
    // Base cases
    if k == 0 || k == n {
        return LaurentPolynomial::one();
    }

    if k > n {
        return LaurentPolynomial::new();
    }

    // Build Pascal's triangle style table for q-binomials
    // table[i] contains all [i choose j]_q for j = 0 to min(i, k)
    let mut table: Vec<Vec<LaurentPolynomial<R>>> = Vec::new();

    // Initialize first row: [0 choose 0]_q = 1
    table.push(vec![LaurentPolynomial::one()]);

    for i in 1..=(n as usize) {
        let max_j = k.min(i as u32) as usize;
        let mut row = Vec::new();

        for j in 0..=max_j {
            let val = if j == 0 {
                // [i choose 0]_q = 1
                LaurentPolynomial::one()
            } else if j == i {
                // [i choose i]_q = 1
                LaurentPolynomial::one()
            } else {
                // [i choose j]_q = [i-1 choose j]_q + q^{i-j} * [i-1 choose j-1]_q
                let term1 = if j < table[i - 1].len() {
                    table[i - 1][j].clone()
                } else {
                    LaurentPolynomial::new()
                };

                let term2 = if j > 0 && j - 1 < table[i - 1].len() {
                    table[i - 1][j - 1].clone() * LaurentPolynomial::monomial(R::one(), (i - j) as i32)
                } else {
                    LaurentPolynomial::new()
                };

                term1 + term2
            };

            row.push(val);
        }

        table.push(row);
    }

    table[n as usize][k as usize].clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_q_int_basic() {
        // [0]_q = 0
        let q0 = q_int::<Integer>(0);
        assert!(q0.is_zero());

        // [1]_q = q^0 = 1
        let q1 = q_int::<Integer>(1);
        assert_eq!(q1.coeff(0), Integer::one());
        assert!(!q1.is_zero());
    }

    #[test]
    fn test_q_int_2() {
        // [2]_q = q^{-1} + q = q^1 + q^{-1}
        let q2 = q_int::<Integer>(2);

        assert_eq!(q2.coeff(1), Integer::one());
        assert_eq!(q2.coeff(-1), Integer::one());
        assert_eq!(q2.coeff(0), Integer::zero());
    }

    #[test]
    fn test_q_int_3() {
        // [3]_q = q^2 + q^0 + q^{-2}
        let q3 = q_int::<Integer>(3);

        assert_eq!(q3.coeff(2), Integer::one());
        assert_eq!(q3.coeff(0), Integer::one());
        assert_eq!(q3.coeff(-2), Integer::one());
    }

    #[test]
    fn test_q_factorial_basic() {
        // [0]_q! = 1
        let qf0 = q_factorial::<Integer>(0);
        assert!(qf0.is_one());

        // [1]_q! = [1]_q = 1
        let qf1 = q_factorial::<Integer>(1);
        assert_eq!(qf1.coeff(0), Integer::one());
    }

    #[test]
    fn test_q_factorial_2() {
        // [2]_q! = [2]_q * [1]_q = (q + q^{-1}) * 1 = q + q^{-1}
        let qf2 = q_factorial::<Integer>(2);
        assert_eq!(qf2.coeff(1), Integer::one());
        assert_eq!(qf2.coeff(-1), Integer::one());
    }

    #[test]
    fn test_q_binomial_base_cases() {
        // [n choose 0]_q = 1
        let qb1 = q_binomial::<Integer>(5, 0);
        assert!(qb1.is_one());

        // [n choose n]_q = 1
        let qb2 = q_binomial::<Integer>(5, 5);
        assert!(qb2.is_one());

        // [n choose k]_q = 0 when k > n
        let qb3 = q_binomial::<Integer>(3, 5);
        assert!(qb3.is_zero());
    }

    #[test]
    fn test_q_binomial_small_values() {
        // [2 choose 1]_q = [2]_q = q + q^{-1}
        let qb = q_binomial::<Integer>(2, 1);
        assert_eq!(qb.coeff(1), Integer::one());
        assert_eq!(qb.coeff(-1), Integer::one());
    }

    #[test]
    fn test_q_binomial_3_1() {
        // [3 choose 1]_q = [3]_q = q^2 + 1 + q^{-2}
        let qb = q_binomial::<Integer>(3, 1);
        assert_eq!(qb.coeff(2), Integer::one());
        assert_eq!(qb.coeff(0), Integer::one());
        assert_eq!(qb.coeff(-2), Integer::one());
    }
}
