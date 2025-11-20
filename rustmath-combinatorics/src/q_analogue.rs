//! q-Analogues - q-deformed versions of combinatorial quantities
//!
//! This module provides q-analogues of common combinatorial functions, including:
//! - q-integers (also called q-numbers or quantum integers)
//! - q-factorials
//! - q-binomial coefficients (Gaussian binomial coefficients)
//! - Gaussian polynomials
//!
//! # Mathematical Background
//!
//! The q-analogue of an integer n is defined as:
//! ```text
//! [n]_q = (1 - q^n) / (1 - q) = 1 + q + q^2 + ... + q^{n-1}
//! ```
//!
//! When q = 1, we have [n]_q = n, recovering the classical integer.
//!
//! The q-factorial is:
//! ```text
//! [n]_q! = [1]_q · [2]_q · ... · [n]_q
//! ```
//!
//! The q-binomial coefficient (Gaussian binomial coefficient) is:
//! ```text
//! [n choose k]_q = [n]_q! / ([k]_q! · [n-k]_q!)
//! ```
//!
//! These coefficients count the number of k-dimensional subspaces of an n-dimensional
//! vector space over the finite field GF(q), and they are polynomials in q.

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_polynomials::{Polynomial, UnivariatePolynomial};

/// Compute the q-integer (q-number) [n]_q = 1 + q + q^2 + ... + q^{n-1}
///
/// This is a polynomial in q with integer coefficients.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::q_integer;
///
/// // [0]_q = 0
/// let q0 = q_integer(0);
///
/// // [1]_q = 1
/// let q1 = q_integer(1);
///
/// // [3]_q = 1 + q + q^2
/// let q3 = q_integer(3);
/// ```
///
/// # Mathematical Definition
///
/// [n]_q = (1 - q^n) / (1 - q) = 1 + q + q^2 + ... + q^{n-1}
///
/// When q = 1, this reduces to n (the classical integer).
pub fn q_integer(n: u32) -> UnivariatePolynomial<Integer> {
    if n == 0 {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }

    // [n]_q = 1 + q + q^2 + ... + q^{n-1}
    // Coefficients: [1, 1, 1, ..., 1] (n times)
    let coeffs = vec![Integer::one(); n as usize];
    UnivariatePolynomial::new(coeffs)
}

/// Compute the q-factorial [n]_q! = [1]_q · [2]_q · ... · [n]_q
///
/// This is a polynomial in q with integer coefficients.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::q_factorial;
///
/// // [0]_q! = 1
/// let qf0 = q_factorial(0);
///
/// // [1]_q! = [1]_q = 1
/// let qf1 = q_factorial(1);
///
/// // [3]_q! = [1]_q · [2]_q · [3]_q = 1 · (1+q) · (1+q+q^2)
/// let qf3 = q_factorial(3);
/// ```
///
/// # Mathematical Definition
///
/// [n]_q! = [1]_q · [2]_q · ... · [n]_q
///
/// where [k]_q = 1 + q + q^2 + ... + q^{k-1}
pub fn q_factorial(n: u32) -> UnivariatePolynomial<Integer> {
    if n == 0 {
        return UnivariatePolynomial::new(vec![Integer::one()]);
    }

    let mut result = UnivariatePolynomial::new(vec![Integer::one()]);

    for k in 1..=n {
        let q_k = q_integer(k);
        result = result * q_k;
    }

    result
}

/// Compute the q-binomial coefficient (Gaussian binomial coefficient)
///
/// [n choose k]_q = [n]_q! / ([k]_q! · [n-k]_q!)
///
/// This coefficient counts the number of k-dimensional subspaces of an
/// n-dimensional vector space over GF(q).
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::q_binomial;
///
/// // [n choose 0]_q = 1
/// let qb = q_binomial(5, 0);
///
/// // [n choose n]_q = 1
/// let qb = q_binomial(5, 5);
///
/// // [5 choose 2]_q
/// let qb = q_binomial(5, 2);
/// ```
///
/// # Properties
///
/// - [n choose 0]_q = [n choose n]_q = 1
/// - [n choose k]_q = 0 when k > n
/// - When q = 1, recovers the classical binomial coefficient
/// - Symmetric: [n choose k]_q = [n choose n-k]_q
///
/// # Algorithm
///
/// We compute this by direct polynomial multiplication and division.
/// An alternative is to use the recurrence:
/// [n choose k]_q = q^k · [n-1 choose k]_q + [n-1 choose k-1]_q
pub fn q_binomial(n: u32, k: u32) -> UnivariatePolynomial<Integer> {
    if k > n {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }

    if k == 0 || k == n {
        return UnivariatePolynomial::new(vec![Integer::one()]);
    }

    // Use symmetry to optimize computation
    let k = k.min(n - k);

    // Compute using the recurrence relation:
    // [n choose k]_q = q^k · [n-1 choose k]_q + [n-1 choose k-1]_q
    // This is more efficient than direct factorial division

    // Initialize DP table
    // dp[i][j] represents [i choose j]_q
    let mut dp: Vec<Vec<UnivariatePolynomial<Integer>>> = vec![vec![]; (n + 1) as usize];

    for i in 0..=n as usize {
        dp[i] = vec![UnivariatePolynomial::new(vec![Integer::zero()]); (k + 1) as usize];
        dp[i][0] = UnivariatePolynomial::new(vec![Integer::one()]);
    }

    for i in 1..=n as usize {
        for j in 1..=k.min(i as u32) as usize {
            if j == i {
                dp[i][j] = UnivariatePolynomial::new(vec![Integer::one()]);
            } else {
                // [i choose j]_q = q^j · [i-1 choose j]_q + [i-1 choose j-1]_q
                let q_power_j = if j > 0 {
                    // q^j is represented as polynomial with coefficient 1 at position j
                    let mut coeffs = vec![Integer::zero(); j + 1];
                    coeffs[j] = Integer::one();
                    UnivariatePolynomial::new(coeffs)
                } else {
                    UnivariatePolynomial::new(vec![Integer::one()])
                };

                let term1 = dp[i - 1][j].clone() * q_power_j;
                let term2 = dp[i - 1][j - 1].clone();
                dp[i][j] = term1 + term2;
            }
        }
    }

    dp[n as usize][k as usize].clone()
}

/// Alias for q-binomial coefficient
///
/// The Gaussian polynomial is another name for the q-binomial coefficient.
/// It's called "Gaussian" after Carl Friedrich Gauss.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::gaussian_polynomial;
///
/// let gp = gaussian_polynomial(5, 2);
/// ```
pub fn gaussian_polynomial(n: u32, k: u32) -> UnivariatePolynomial<Integer> {
    q_binomial(n, k)
}

/// Compute the q-binomial coefficient evaluated at a specific value of q
///
/// This evaluates [n choose k]_q at q = q_value.
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::q_binomial_eval;
/// use rustmath_integers::Integer;
///
/// // Evaluate [5 choose 2]_q at q = 2
/// let result = q_binomial_eval(5, 2, &Integer::from(2));
/// ```
pub fn q_binomial_eval(n: u32, k: u32, q_value: &Integer) -> Integer {
    let poly = q_binomial(n, k);
    poly.eval(q_value)
}

/// Compute the q-multinomial coefficient
///
/// This is a generalization of the q-binomial to multiple indices.
///
/// [n; k1, k2, ..., km]_q = [n]_q! / ([k1]_q! · [k2]_q! · ... · [km]_q!)
///
/// where k1 + k2 + ... + km = n
///
/// # Examples
///
/// ```
/// use rustmath_combinatorics::q_multinomial;
///
/// // [5; 2, 2, 1]_q
/// let qm = q_multinomial(5, &[2, 2, 1]);
/// ```
pub fn q_multinomial(n: u32, ks: &[u32]) -> UnivariatePolynomial<Integer> {
    let sum: u32 = ks.iter().sum();
    if sum != n {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }

    let mut result = q_factorial(n);

    for &k in ks {
        let q_k_factorial = q_factorial(k);
        // For polynomial division, we need to implement it properly
        // For now, we'll use a simplified approach
        // TODO: Implement proper polynomial division over integers
        result = divide_q_factorial_polynomials(result, q_k_factorial);
    }

    result
}

/// Helper function to divide two q-factorial polynomials
///
/// This function divides polynomial p by polynomial q, assuming the division
/// is exact (no remainder). This is guaranteed for q-factorials in the
/// q-multinomial computation.
fn divide_q_factorial_polynomials(
    p: UnivariatePolynomial<Integer>,
    q: UnivariatePolynomial<Integer>,
) -> UnivariatePolynomial<Integer> {
    // For q-factorials, we know the division is exact
    // We'll implement polynomial long division

    if q.is_zero() {
        panic!("Division by zero polynomial");
    }

    if p.is_zero() {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }

    let p_deg = p.degree().unwrap_or(0);
    let q_deg = q.degree().unwrap_or(0);

    if p_deg < q_deg {
        return UnivariatePolynomial::new(vec![Integer::zero()]);
    }

    let mut remainder = p.clone();
    let mut quotient_coeffs = vec![Integer::zero(); p_deg - q_deg + 1];

    let q_lead = q.coefficients()[q_deg].clone();

    while let Some(r_deg) = remainder.degree() {
        if r_deg < q_deg {
            break;
        }

        let r_lead = remainder.coefficients()[r_deg].clone();

        // For exact division with integers, we need r_lead to be divisible by q_lead
        if r_lead.clone() % q_lead.clone() != Integer::zero() {
            // Not exactly divisible - this shouldn't happen for q-factorials
            break;
        }

        let coeff = r_lead / q_lead.clone();
        let power = r_deg - q_deg;

        quotient_coeffs[power] = coeff.clone();

        // Subtract coeff * q * x^power from remainder
        let mut term_coeffs = vec![Integer::zero(); power];
        for (i, c) in q.coefficients().iter().enumerate() {
            term_coeffs.push(c.clone() * coeff.clone());
        }
        let term = UnivariatePolynomial::new(term_coeffs);
        remainder = remainder - term;
    }

    UnivariatePolynomial::new(quotient_coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::Ring;
    use rustmath_polynomials::Polynomial;

    #[test]
    fn test_q_integer() {
        // [0]_q = 0
        let q0 = q_integer(0);
        assert_eq!(q0.coefficients(), &[Integer::zero()]);

        // [1]_q = 1
        let q1 = q_integer(1);
        assert_eq!(q1.coefficients(), &[Integer::one()]);

        // [2]_q = 1 + q
        let q2 = q_integer(2);
        assert_eq!(q2.coefficients(), &[Integer::one(), Integer::one()]);

        // [3]_q = 1 + q + q^2
        let q3 = q_integer(3);
        assert_eq!(
            q3.coefficients(),
            &[Integer::one(), Integer::one(), Integer::one()]
        );

        // [5]_q = 1 + q + q^2 + q^3 + q^4
        let q5 = q_integer(5);
        assert_eq!(q5.coefficients().len(), 5);
        assert!(q5.coefficients().iter().all(|c| c == &Integer::one()));
    }

    #[test]
    fn test_q_integer_evaluation() {
        // Test that [n]_q evaluates to n when q = 1
        for n in 0..10 {
            let qn = q_integer(n);
            let result = qn.eval(&Integer::one());
            assert_eq!(result, Integer::from(n));
        }

        // Test evaluation at q = 2
        // [3]_2 = 1 + 2 + 4 = 7
        let q3 = q_integer(3);
        assert_eq!(q3.eval(&Integer::from(2)), Integer::from(7));

        // [4]_2 = 1 + 2 + 4 + 8 = 15
        let q4 = q_integer(4);
        assert_eq!(q4.eval(&Integer::from(2)), Integer::from(15));
    }

    #[test]
    fn test_q_factorial() {
        // [0]_q! = 1
        let qf0 = q_factorial(0);
        assert_eq!(qf0.coefficients(), &[Integer::one()]);

        // [1]_q! = [1]_q = 1
        let qf1 = q_factorial(1);
        assert_eq!(qf1.coefficients(), &[Integer::one()]);

        // [2]_q! = [1]_q · [2]_q = 1 · (1 + q) = 1 + q
        let qf2 = q_factorial(2);
        assert_eq!(qf2.coefficients(), &[Integer::one(), Integer::one()]);

        // [3]_q! = [1]_q · [2]_q · [3]_q = 1 · (1+q) · (1+q+q^2)
        //        = (1+q)(1+q+q^2) = 1 + q + q + q^2 + q^2 + q^3
        //        = 1 + 2q + 2q^2 + q^3
        let qf3 = q_factorial(3);
        assert_eq!(
            qf3.coefficients(),
            &[
                Integer::one(),
                Integer::from(2),
                Integer::from(2),
                Integer::one()
            ]
        );
    }

    #[test]
    fn test_q_factorial_evaluation() {
        // Test that [n]_q! evaluates to n! when q = 1
        let factorials = [1, 1, 2, 6, 24, 120];
        for (n, expected) in factorials.iter().enumerate() {
            let qfn = q_factorial(n as u32);
            let result = qfn.eval(&Integer::one());
            assert_eq!(result, Integer::from(*expected));
        }
    }

    #[test]
    fn test_q_binomial_base_cases() {
        // [n choose 0]_q = 1
        for n in 0..10 {
            let qb = q_binomial(n, 0);
            assert_eq!(qb.coefficients(), &[Integer::one()]);
        }

        // [n choose n]_q = 1
        for n in 0..10 {
            let qb = q_binomial(n, n);
            assert_eq!(qb.coefficients(), &[Integer::one()]);
        }

        // [n choose k]_q = 0 when k > n
        let qb = q_binomial(5, 6);
        assert_eq!(qb.coefficients(), &[Integer::zero()]);
    }

    #[test]
    fn test_q_binomial_small_values() {
        // [2 choose 1]_q = [2]_q = 1 + q
        let qb21 = q_binomial(2, 1);
        assert_eq!(qb21.coefficients(), &[Integer::one(), Integer::one()]);

        // [3 choose 1]_q = [3]_q = 1 + q + q^2
        let qb31 = q_binomial(3, 1);
        assert_eq!(
            qb31.coefficients(),
            &[Integer::one(), Integer::one(), Integer::one()]
        );
    }

    #[test]
    fn test_q_binomial_evaluation() {
        // Test that [n choose k]_q evaluates to C(n,k) when q = 1
        let test_cases = [
            (5, 2, 10),
            (6, 3, 20),
            (7, 2, 21),
            (8, 4, 70),
        ];

        for (n, k, expected) in test_cases.iter() {
            let qb = q_binomial(*n, *k);
            let result = qb.eval(&Integer::one());
            assert_eq!(result, Integer::from(*expected));
        }
    }

    #[test]
    fn test_q_binomial_symmetry() {
        // Test that [n choose k]_q = [n choose n-k]_q
        for n in 2..8 {
            for k in 0..=n {
                let qb1 = q_binomial(n, k);
                let qb2 = q_binomial(n, n - k);

                // Evaluate at q = 1 (should be equal)
                assert_eq!(
                    qb1.eval(&Integer::one()),
                    qb2.eval(&Integer::one())
                );

                // Evaluate at q = 2 (should be equal)
                assert_eq!(
                    qb1.eval(&Integer::from(2)),
                    qb2.eval(&Integer::from(2))
                );
            }
        }
    }

    #[test]
    fn test_q_binomial_eval() {
        // [5 choose 2]_2 should equal the number of 2-dimensional subspaces
        // of a 5-dimensional vector space over GF(2)
        let result = q_binomial_eval(5, 2, &Integer::from(2));
        // This equals (2^5 - 1)(2^5 - 2) / ((2^2 - 1)(2^2 - 2))
        // = (31)(30) / (3)(2) = 930 / 6 = 155
        assert_eq!(result, Integer::from(155));
    }

    #[test]
    fn test_gaussian_polynomial() {
        // Gaussian polynomial is just an alias for q-binomial
        let qb = q_binomial(5, 2);
        let gp = gaussian_polynomial(5, 2);

        assert_eq!(qb.coefficients(), gp.coefficients());
    }

    #[test]
    fn test_q_multinomial() {
        // [3; 1, 1, 1]_q should equal [3]_q! / ([1]_q!)^3 = [3]_q!
        // since [1]_q! = 1
        let qm = q_multinomial(3, &[1, 1, 1]);
        let qf3 = q_factorial(3);

        assert_eq!(
            qm.eval(&Integer::one()),
            qf3.eval(&Integer::one())
        );

        // Invalid case: sum doesn't equal n
        let qm_invalid = q_multinomial(5, &[2, 2]);
        assert_eq!(qm_invalid.coefficients(), &[Integer::zero()]);
    }
}
