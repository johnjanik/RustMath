//! # Puiseux Series Elements
//!
//! This module implements elements of Puiseux series rings.
//!
//! ## Overview
//!
//! A Puiseux series element is a series of the form:
//! ```text
//! f(x) = Σ(n=v to ∞) aₙ x^(n/e)
//! ```
//! where:
//! - `aₙ` are coefficients from the base field
//! - `v` is the valuation (minimum exponent numerator)
//! - `e` is the ramification index (exponent denominator)
//!
//! ## Mathematical Properties
//!
//! - Every Puiseux series can be written with a unique minimal ramification index
//! - The set of Puiseux series forms a field
//! - Algebraic closure: Puiseux series over ℂ are algebraically closed
//! - Newton-Puiseux theorem: Roots of polynomial equations are Puiseux series
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::puiseux_series_ring_element::PuiseuxSeries;
//! use rustmath_rationals::Rational;
//!
//! // Create a Puiseux series: x^(1/2) + 2*x + 3*x^(3/2)
//! let coeffs = vec![Rational::new(1,1), Rational::new(2,1), Rational::new(3,1)];
//! let series = PuiseuxSeries::new(coeffs, 1, 2, 10);
//! ```

use rustmath_core::{Field, Ring};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// Element of a Puiseux series ring
///
/// Represents a truncated Puiseux series with coefficients in a field F.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PuiseuxSeries<F: Field> {
    /// Coefficients of the series [a_v, a_(v+1), a_(v+2), ...]
    /// where v is the valuation
    coefficients: Vec<F>,
    /// Valuation: numerator of the minimum fractional exponent
    valuation: isize,
    /// Ramification index: denominator of fractional exponents
    /// All exponents are of the form (valuation + k) / ramification
    ramification: usize,
    /// Precision: maximum number of terms to keep
    precision: usize,
}

impl<F: Field> PuiseuxSeries<F> {
    /// Create a new Puiseux series
    ///
    /// # Arguments
    /// * `coefficients` - Coefficients starting from valuation
    /// * `valuation` - Numerator of minimum fractional exponent
    /// * `ramification` - Denominator of fractional exponents
    /// * `precision` - Maximum precision
    ///
    /// # Examples
    /// ```
    /// use rustmath_rings::puiseux_series_ring_element::PuiseuxSeries;
    /// use rustmath_rationals::Rational;
    ///
    /// // x^(1/2) + 2*x^(3/2)
    /// let series = PuiseuxSeries::new(
    ///     vec![Rational::new(1,1), Rational::new(0,1), Rational::new(2,1)],
    ///     1, 2, 10
    /// );
    /// ```
    pub fn new(
        coefficients: Vec<F>,
        valuation: isize,
        ramification: usize,
        precision: usize,
    ) -> Self {
        if ramification == 0 {
            panic!("Ramification index must be positive");
        }

        // Normalize: reduce ramification if possible
        let (valuation, ramification, mut coefficients) =
            Self::normalize(valuation, ramification, coefficients);

        // Truncate to precision
        coefficients.truncate(precision);

        Self {
            coefficients,
            valuation,
            ramification,
            precision,
        }
    }

    /// Normalize the representation to minimal ramification
    fn normalize(
        valuation: isize,
        ramification: usize,
        coefficients: Vec<F>,
    ) -> (isize, usize, Vec<F>) {
        // Find GCD of ramification and all non-zero coefficient indices
        let mut g = ramification;

        // Find first non-zero coefficient
        let first_nonzero = coefficients.iter().position(|c| !c.is_zero());

        if let Some(first) = first_nonzero {
            // Adjust valuation to first non-zero term
            let new_val = valuation + first as isize;
            let new_coeffs = coefficients[first..].to_vec();

            // Compute GCD with valuation
            g = gcd(g as isize, new_val).abs() as usize;
            if g == 0 {
                g = 1;
            }

            (new_val / g as isize, ramification / g, new_coeffs)
        } else {
            // Zero series
            (0, 1, vec![])
        }
    }

    /// Get the coefficient at a given position
    pub fn coeff(&self, index: usize) -> F {
        if index < self.coefficients.len() {
            self.coefficients[index].clone()
        } else {
            F::zero()
        }
    }

    /// Get the valuation (numerator of minimum exponent)
    pub fn valuation(&self) -> isize {
        self.valuation
    }

    /// Get the ramification index (denominator of exponents)
    pub fn ramification(&self) -> usize {
        self.ramification
    }

    /// Get the precision
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Get the leading exponent as a rational number (val/ram)
    pub fn leading_exponent(&self) -> (isize, usize) {
        (self.valuation, self.ramification)
    }

    /// Get the leading coefficient
    pub fn leading_coefficient(&self) -> F {
        if self.coefficients.is_empty() {
            F::zero()
        } else {
            self.coefficients[0].clone()
        }
    }

    /// Check if the series is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| c.is_zero())
    }

    /// Truncate to a new precision
    pub fn truncate(&mut self, new_precision: usize) {
        if new_precision < self.precision {
            self.coefficients.truncate(new_precision);
            self.precision = new_precision;
        }
    }

    /// Convert to a common ramification with another series
    fn to_common_ramification(&self, other: &Self) -> (Self, Self) {
        use num_integer::Integer as _;
        let lcm = (self.ramification * other.ramification)
            / self.ramification.gcd(&other.ramification);

        let self_expanded = self.expand_ramification(lcm);
        let other_expanded = other.expand_ramification(lcm);

        (self_expanded, other_expanded)
    }

    /// Expand to a higher ramification index
    fn expand_ramification(&self, new_ramification: usize) -> Self {
        if new_ramification % self.ramification != 0 {
            panic!("New ramification must be a multiple of current ramification");
        }

        let factor = new_ramification / self.ramification;
        let new_valuation = self.valuation * factor as isize;

        // Insert zeros to expand ramification
        let mut new_coeffs = Vec::new();
        for coeff in &self.coefficients {
            new_coeffs.push(coeff.clone());
            for _ in 1..factor {
                new_coeffs.push(F::zero());
            }
        }

        Self {
            coefficients: new_coeffs,
            valuation: new_valuation,
            ramification: new_ramification,
            precision: self.precision * factor,
        }
    }

    /// Shift the series: multiply by x^(n/e)
    pub fn shift(&self, numerator: isize, denominator: usize) -> Self {
        let (self_exp, other_exp) = self.to_common_ramification(&PuiseuxSeries::new(
            vec![],
            numerator,
            denominator,
            self.precision,
        ));

        Self::new(
            self_exp.coefficients.clone(),
            self_exp.valuation + numerator,
            self_exp.ramification,
            self.precision,
        )
    }
}

impl<F: Field> Add for PuiseuxSeries<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let (self_norm, other_norm) = self.to_common_ramification(&other);

        let min_val = self_norm.valuation.min(other_norm.valuation);
        let max_val = (self_norm.valuation + self_norm.coefficients.len() as isize)
            .max(other_norm.valuation + other_norm.coefficients.len() as isize);

        let mut result = Vec::new();
        for i in min_val..max_val {
            let self_idx = (i - self_norm.valuation) as usize;
            let other_idx = (i - other_norm.valuation) as usize;

            let self_coeff = if self_idx < self_norm.coefficients.len() {
                &self_norm.coefficients[self_idx]
            } else {
                &F::zero()
            };

            let other_coeff = if other_idx < other_norm.coefficients.len() {
                &other_norm.coefficients[other_idx]
            } else {
                &F::zero()
            };

            result.push(self_coeff.clone() + other_coeff.clone());
        }

        Self::new(
            result,
            min_val,
            self_norm.ramification,
            self.precision.min(other.precision),
        )
    }
}

impl<F: Field> Sub for PuiseuxSeries<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<F: Field> Mul for PuiseuxSeries<F> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let (self_norm, other_norm) = self.to_common_ramification(&other);

        let new_val = self_norm.valuation + other_norm.valuation;
        let result_len = (self_norm.coefficients.len() + other_norm.coefficients.len())
            .min(self.precision.min(other.precision));

        let mut result = vec![F::zero(); result_len];

        for i in 0..self_norm.coefficients.len() {
            for j in 0..other_norm.coefficients.len() {
                if i + j < result_len {
                    result[i + j] = result[i + j].clone()
                        + (self_norm.coefficients[i].clone()
                            * other_norm.coefficients[j].clone());
                }
            }
        }

        Self::new(
            result,
            new_val,
            self_norm.ramification,
            self.precision.min(other.precision),
        )
    }
}

impl<F: Field> Neg for PuiseuxSeries<F> {
    type Output = Self;

    fn neg(self) -> Self {
        let coeffs: Vec<F> = self.coefficients.iter().map(|c| -c.clone()).collect();
        Self {
            coefficients: coeffs,
            valuation: self.valuation,
            ramification: self.ramification,
            precision: self.precision,
        }
    }
}

impl<F: Field + fmt::Display> fmt::Display for PuiseuxSeries<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "O(x^({}/{}))", self.precision, self.ramification);
        }

        let mut first = true;
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if !coeff.is_zero() {
                let exp_num = self.valuation + i as isize;
                let exp_den = self.ramification;

                if !first {
                    write!(f, " + ")?;
                }

                write!(f, "{}", coeff)?;
                if exp_num != 0 || exp_den != 1 {
                    if exp_den == 1 {
                        write!(f, "*x^{}", exp_num)?;
                    } else {
                        write!(f, "*x^({}/{})", exp_num, exp_den)?;
                    }
                }

                first = false;
            }
        }

        write!(f, " + O(x^{}/{})", self.precision, self.ramification)
    }
}

/// Compute GCD using Euclid's algorithm
fn gcd(mut a: isize, mut b: isize) -> isize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_new_puiseux_series() {
        let series = PuiseuxSeries::new(
            vec![Rational::new(1, 1), Rational::new(2, 1)],
            1,
            2,
            10,
        );

        assert_eq!(series.valuation(), 1);
        assert_eq!(series.ramification(), 2);
        assert_eq!(series.precision(), 10);
    }

    #[test]
    #[should_panic(expected = "Ramification index must be positive")]
    fn test_zero_ramification() {
        let _ = PuiseuxSeries::new(vec![Rational::new(1, 1)], 0, 0, 10);
    }

    #[test]
    fn test_leading_coefficient() {
        let series = PuiseuxSeries::new(
            vec![Rational::new(3, 1), Rational::new(2, 1)],
            1,
            2,
            10,
        );

        assert_eq!(series.leading_coefficient(), &Rational::new(3, 1));
    }

    #[test]
    fn test_is_zero() {
        let zero = PuiseuxSeries::new(vec![Rational::new(0, 1)], 0, 1, 10);
        assert!(zero.is_zero());

        let non_zero = PuiseuxSeries::new(vec![Rational::new(1, 1)], 0, 1, 10);
        assert!(!non_zero.is_zero());
    }

    #[test]
    fn test_addition() {
        let s1 = PuiseuxSeries::new(vec![Rational::new(1, 1)], 0, 1, 10);
        let s2 = PuiseuxSeries::new(vec![Rational::new(2, 1)], 0, 1, 10);

        let sum = s1 + s2;
        assert_eq!(sum.leading_coefficient(), &Rational::new(3, 1));
    }

    #[test]
    fn test_multiplication() {
        // x^(1/2) * x^(1/2) = x
        let s1 = PuiseuxSeries::new(vec![Rational::new(1, 1)], 1, 2, 10);
        let s2 = PuiseuxSeries::new(vec![Rational::new(1, 1)], 1, 2, 10);

        let product = s1 * s2;
        // Result should have valuation 1+1=2, ramification 2, which simplifies to valuation 1, ram 1
        assert_eq!(product.leading_coefficient(), &Rational::new(1, 1));
    }

    #[test]
    fn test_negation() {
        let series = PuiseuxSeries::new(vec![Rational::new(5, 1)], 0, 1, 10);
        let neg = -series;

        assert_eq!(neg.leading_coefficient(), &Rational::new(-5, 1));
    }

    #[test]
    fn test_truncate() {
        let mut series = PuiseuxSeries::new(
            vec![
                Rational::new(1, 1),
                Rational::new(2, 1),
                Rational::new(3, 1),
                Rational::new(4, 1),
            ],
            0,
            1,
            10,
        );

        series.truncate(2);
        assert_eq!(series.precision(), 2);
        assert_eq!(series.coefficients.len(), 2);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(15, 25), 5);
        assert_eq!(gcd(-12, 8), 4);
        assert_eq!(gcd(0, 5), 5);
    }

    #[test]
    fn test_expand_ramification() {
        let series = PuiseuxSeries::new(vec![Rational::new(1, 1), Rational::new(2, 1)], 1, 2, 10);

        let expanded = series.expand_ramification(4);
        assert_eq!(expanded.ramification(), 4);
        assert_eq!(expanded.valuation(), 2); // 1 * (4/2) = 2
    }

    #[test]
    fn test_leading_exponent() {
        let series = PuiseuxSeries::new(vec![Rational::new(1, 1)], 3, 2, 10);
        assert_eq!(series.leading_exponent(), (3, 2));
    }
}
