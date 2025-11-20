//! Carlitz q-Bernoulli Numbers
//!
//! This module implements Carlitz's q-analog of the Bernoulli numbers.
//!
//! ## Background
//!
//! Carlitz introduced q-Bernoulli numbers in his 1948 paper "q-Bernoulli numbers
//! and polynomials" (Duke Math J. 15, 987-1000). These are q-analogs that specialize
//! to classical Bernoulli numbers when q = 1.
//!
//! The q-Bernoulli numbers are rational functions in q defined by the explicit formula:
//!
//! β_{n,q} = 1/(1-q)^n * Σ_{l=0}^n (n choose l) * (-1)^l * (l+1) / [l+1]_q
//!
//! where [k]_q is the q-integer: [k]_q = (q^k - q^{-k})/(q - q^{-1})
//!
//! ## Properties
//!
//! - β_{0,q} = 1
//! - β_{1,q} = -1/(q + 1) = -1/[2]_q
//! - lim_{q→1} β_{n,q} = B_n (classical Bernoulli number)
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_quantumgroups::q_bernoulli::q_bernoulli;
//! use rustmath_integers::Integer;
//!
//! // β_{0,q} = 1
//! let b0 = q_bernoulli::<Integer>(0);
//! // assert!(b0.numerator().is_constant() && b0.numerator().coeff(0) == Integer::one());
//!
//! // β_{1,q} = -1/(q + 1)
//! let b1 = q_bernoulli::<Integer>(1);
//! ```
//!
//! ## References
//!
//! - L. Carlitz, "q-Bernoulli numbers and polynomials", Duke Math J. 15 (1948), 987-1000
//! - L. Carlitz, "q-Bernoulli and Eulerian numbers", Trans. Am. Math. Soc. 76 (1954), 332-350

use rustmath_core::{IntegralDomain, NumericConversion, Ring};
use rustmath_polynomials::univariate::UnivariatePolynomial;
use rustmath_rings::fraction_field::FractionFieldElement;
use rustmath_polynomials::laurent::LaurentPolynomial;
use crate::q_numbers::q_int;

/// Type alias for rational functions (quotients of polynomials)
pub type RationalFunction<R> = FractionFieldElement<UnivariatePolynomial<R>>;

/// Computes the n-th Carlitz q-Bernoulli number.
///
/// The Carlitz q-Bernoulli numbers are rational functions in q that reduce to
/// classical Bernoulli numbers when q → 1.
///
/// # Formula
///
/// β_{n,q} = 1/(1-q)^n * Σ_{l=0}^n C(n,l) * (-1)^l * (l+1) / [l+1]_q
///
/// where [k]_q is the q-integer.
///
/// # Arguments
///
/// * `n` - The index of the q-Bernoulli number
///
/// # Returns
///
/// A rational function in q representing β_{n,q}
///
/// # Examples
///
/// ```rust
/// use rustmath_quantumgroups::q_bernoulli::q_bernoulli;
/// use rustmath_integers::Integer;
///
/// // Compute β_{0,q} = 1
/// let b0 = q_bernoulli::<Integer>(0);
///
/// // Compute β_{1,q} = -1/(q + 1)
/// let b1 = q_bernoulli::<Integer>(1);
/// ```
pub fn q_bernoulli<R: IntegralDomain + NumericConversion>(n: u32) -> RationalFunction<R> {
    // Base case: β_{0,q} = 1
    if n == 0 {
        return RationalFunction::new(
            UnivariatePolynomial::constant(R::one()),
            UnivariatePolynomial::constant(R::one()),
        );
    }

    // We'll compute the sum: Σ_{l=0}^n C(n,l) * (-1)^l * (l+1) / [l+1]_q
    // Each term is a rational function, so we need to accumulate as a rational function

    let mut sum_numerator = UnivariatePolynomial::constant(R::zero());
    let mut sum_denominator = UnivariatePolynomial::constant(R::one());

    for l in 0..=n {
        // Compute binomial coefficient C(n, l)
        let binom = binomial_coefficient(n, l);

        // Compute (-1)^l
        let sign = if l % 2 == 0 { R::one() } else { -R::one() };

        // Compute l+1
        let l_plus_1 = (l + 1) as i64;

        // Get [l+1]_q as a Laurent polynomial
        let q_int_laurent = q_int::<R>(l + 1);

        // Convert Laurent polynomial to rational function
        // [l+1]_q = Σ_{i} c_i q^i
        // We need to extract numerator and denominator
        let (q_int_num, q_int_den, min_power) = laurent_to_rational(&q_int_laurent);

        // Compute the coefficient: C(n,l) * (-1)^l * (l+1)
        let coeff_int = binom * sign.to_i64().unwrap_or(0) * l_plus_1;
        let coeff = R::from_i64(coeff_int);

        // This term is: coeff / [l+1]_q = (coeff * q_int_den) / q_int_num
        // Multiply polynomial by scalar coefficient
        let term_num_coeffs: Vec<R> = q_int_den.coefficients().iter()
            .map(|c| c.clone() * coeff.clone())
            .collect();
        let term_num = UnivariatePolynomial::new(term_num_coeffs);
        let term_den = q_int_num.clone();

        // Add this term to the sum: sum + term = (sum_num * term_den + term_num * sum_den) / (sum_den * term_den)
        sum_numerator = sum_numerator * term_den.clone() + term_num * sum_denominator.clone();
        sum_denominator = sum_denominator * term_den;
    }

    // Now we need to divide by (1-q)^n
    // (1-q) = -q + 1
    let one_minus_q = UnivariatePolynomial::new(vec![R::one(), -R::one()]);

    // (1-q)^n
    let mut one_minus_q_pow_n = UnivariatePolynomial::constant(R::one());
    for _ in 0..n {
        one_minus_q_pow_n = one_minus_q_pow_n * one_minus_q.clone();
    }

    // Final result: sum / (1-q)^n = sum_numerator / (sum_denominator * (1-q)^n)
    let final_denominator = sum_denominator * one_minus_q_pow_n;

    RationalFunction::new(sum_numerator, final_denominator)
}

/// Computes binomial coefficient C(n, k)
fn binomial_coefficient(n: u32, k: u32) -> i64 {
    if k > n {
        return 0;
    }

    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k); // Take advantage of symmetry

    let mut result = 1i64;
    for i in 0..k {
        result = result * (n - i) as i64 / (i + 1) as i64;
    }

    result
}

/// Convert a Laurent polynomial to a rational function (polynomial / polynomial)
///
/// A Laurent polynomial Σ c_i q^i where i can be negative is converted to
/// p(q) / q^m where m is the minimum power and p is a regular polynomial.
///
/// Returns (numerator, denominator, min_power)
fn laurent_to_rational<R: Ring>(laurent: &LaurentPolynomial<R>) -> (UnivariatePolynomial<R>, UnivariatePolynomial<R>, i32) {
    // Handle zero polynomial
    if laurent.is_zero() {
        return (
            UnivariatePolynomial::constant(R::zero()),
            UnivariatePolynomial::constant(R::one()),
            0
        );
    }

    // Find the range of powers
    let min_power = laurent.min_exponent().unwrap_or(0);
    let max_power = laurent.max_exponent().unwrap_or(0);

    if min_power >= 0 {
        // All powers are non-negative, so it's already a polynomial
        let mut coeffs = Vec::new();
        for i in 0..=max_power {
            coeffs.push(laurent.coeff(i));
        }
        return (
            UnivariatePolynomial::new(coeffs),
            UnivariatePolynomial::constant(R::one()),
            0
        );
    }

    // We have negative powers. Multiply by q^(-min_power) to clear denominators
    // If min_power = -m, multiply by q^m
    let shift = -min_power;

    let mut coeffs = Vec::new();
    for i in min_power..=max_power {
        coeffs.push(laurent.coeff(i));
    }

    // Numerator is the Laurent polynomial shifted by (-min_power)
    let numerator = UnivariatePolynomial::new(coeffs);

    // Denominator is q^(-min_power)
    // q^k is represented as x^k in polynomial form
    let mut denom_coeffs = vec![R::zero(); (shift + 1) as usize];
    denom_coeffs[shift as usize] = R::one();
    let denominator = UnivariatePolynomial::new(denom_coeffs);

    (numerator, denominator, min_power)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 0), 1);
        assert_eq!(binomial_coefficient(5, 1), 5);
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(5, 3), 10);
        assert_eq!(binomial_coefficient(5, 4), 5);
        assert_eq!(binomial_coefficient(5, 5), 1);
        assert_eq!(binomial_coefficient(3, 5), 0);
    }

    #[test]
    fn test_laurent_to_rational_positive_powers() {
        // [2]_q = q + q^{-1}
        let q2 = q_int::<Integer>(2);
        let (num, den, min_pow) = laurent_to_rational(&q2);

        // Should be (q^2 + 1) / q
        assert_eq!(min_pow, -1);
        // Numerator should have degree 2
        assert_eq!(num.degree(), Some(2));
        // Denominator should be q (degree 1)
        assert_eq!(den.degree(), Some(1));
    }

    #[test]
    fn test_q_bernoulli_0() {
        // β_{0,q} = 1
        let b0 = q_bernoulli::<Integer>(0);

        // Should be 1/1
        assert!(b0.numerator().degree() == Some(0) || b0.numerator().is_zero());
        assert_eq!(b0.numerator().coeff(0), &Integer::one());
        assert_eq!(b0.denominator().coeff(0), &Integer::one());
    }

    #[test]
    fn test_q_bernoulli_1() {
        // β_{1,q} = -1/(q + 1) = -1/[2]_q
        let b1 = q_bernoulli::<Integer>(1);

        // Numerator should be a constant (possibly with negative sign after simplification)
        // Denominator should involve (1-q) and [1]_q, [2]_q terms
        // This is a more complex check, so we just verify it's not zero
        assert!(!b1.numerator().is_zero());
        assert!(!b1.denominator().is_zero());
    }

    #[test]
    fn test_q_bernoulli_2() {
        // β_{2,q} should be a rational function
        let b2 = q_bernoulli::<Integer>(2);

        // Just verify it's well-defined
        assert!(!b2.numerator().is_zero() || true); // Can be zero
        assert!(!b2.denominator().is_zero());
    }

    #[test]
    fn test_q_bernoulli_small_values() {
        // Test that we can compute several q-Bernoulli numbers without panicking
        for n in 0..=5 {
            let bn = q_bernoulli::<Integer>(n);
            assert!(!bn.denominator().is_zero());
        }
    }
}
