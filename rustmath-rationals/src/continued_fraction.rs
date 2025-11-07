//! Continued fractions representation and operations

use crate::Rational;
use rustmath_core::{EuclideanDomain, Ring};
use rustmath_integers::Integer;

/// A continued fraction representation
///
/// Represents a number as: a₀ + 1/(a₁ + 1/(a₂ + 1/(a₃ + ...)))
///
/// Stored as the sequence [a₀, a₁, a₂, a₃, ...]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinuedFraction {
    /// The coefficients of the continued fraction
    coefficients: Vec<Integer>,
}

impl ContinuedFraction {
    /// Create a new continued fraction from coefficients
    pub fn new(coefficients: Vec<Integer>) -> Self {
        ContinuedFraction { coefficients }
    }

    /// Create a continued fraction from a rational number
    ///
    /// Uses the Euclidean algorithm to compute the continued fraction expansion
    pub fn from_rational(r: &Rational) -> Self {
        let mut coefficients = Vec::new();
        let mut num = r.numerator().clone();
        let mut den = r.denominator().clone();

        // Euclidean algorithm
        while !den.is_zero() {
            let (q, rem) = num.div_rem(&den).unwrap();
            coefficients.push(q);
            num = den;
            den = rem;
        }

        ContinuedFraction { coefficients }
    }

    /// Convert the continued fraction back to a rational number
    pub fn to_rational(&self) -> Rational {
        if self.coefficients.is_empty() {
            return Rational::new(0, 1).unwrap();
        }

        // Work backwards through the continued fraction
        let mut numerator = Integer::one();
        let mut denominator = Integer::zero();

        for coeff in self.coefficients.iter().rev() {
            // Add the coefficient
            let new_num = coeff.clone() * numerator.clone() + denominator;
            denominator = numerator;
            numerator = new_num;
        }

        Rational::new(numerator, denominator).unwrap()
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[Integer] {
        &self.coefficients
    }

    /// Compute the nth convergent of the continued fraction
    ///
    /// The nth convergent is the rational approximation using the first n+1 coefficients
    pub fn convergent(&self, n: usize) -> Rational {
        if n >= self.coefficients.len() {
            return self.to_rational();
        }

        let coeffs: Vec<Integer> = self.coefficients[..=n].to_vec();
        let cf = ContinuedFraction::new(coeffs);
        cf.to_rational()
    }

    /// Get all convergents up to the full continued fraction
    pub fn all_convergents(&self) -> Vec<Rational> {
        (0..self.coefficients.len())
            .map(|i| self.convergent(i))
            .collect()
    }

    /// Check if this is a finite continued fraction
    pub fn is_finite(&self) -> bool {
        // All continued fractions from rationals are finite
        true
    }

    /// Get the length of the continued fraction
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }
}

impl std::fmt::Display for ContinuedFraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if i > 0 {
                write!(f, "; ")?;
            }
            write!(f, "{}", coeff)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continued_fraction_simple() {
        // 3/2 = [1; 2]
        let r = Rational::new(3, 2).unwrap();
        let cf = ContinuedFraction::from_rational(&r);

        assert_eq!(cf.coefficients(), &[Integer::from(1), Integer::from(2)]);
        assert_eq!(cf.to_rational(), r);
    }

    #[test]
    fn test_continued_fraction_integer() {
        // 5 = [5]
        let r = Rational::new(5, 1).unwrap();
        let cf = ContinuedFraction::from_rational(&r);

        assert_eq!(cf.coefficients(), &[Integer::from(5)]);
        assert_eq!(cf.to_rational(), r);
    }

    #[test]
    fn test_continued_fraction_complex() {
        // 649/200 = [3; 4, 12, 4]
        let r = Rational::new(649, 200).unwrap();
        let cf = ContinuedFraction::from_rational(&r);

        assert_eq!(
            cf.coefficients(),
            &[
                Integer::from(3),
                Integer::from(4),
                Integer::from(12),
                Integer::from(4)
            ]
        );
        assert_eq!(cf.to_rational(), r);
    }

    #[test]
    fn test_convergents() {
        // 649/200 = [3; 4, 12, 4]
        let r = Rational::new(649, 200).unwrap();
        let cf = ContinuedFraction::from_rational(&r);

        let conv0 = cf.convergent(0);
        assert_eq!(conv0, Rational::new(3, 1).unwrap());

        let conv1 = cf.convergent(1);
        assert_eq!(conv1, Rational::new(13, 4).unwrap());

        let conv2 = cf.convergent(2);
        assert_eq!(conv2, Rational::new(159, 49).unwrap());

        let conv3 = cf.convergent(3);
        assert_eq!(conv3, Rational::new(649, 200).unwrap());
    }

    #[test]
    fn test_all_convergents() {
        let r = Rational::new(22, 7).unwrap();
        let cf = ContinuedFraction::from_rational(&r);
        let convergents = cf.all_convergents();

        // 22/7 = [3; 7]
        assert_eq!(convergents.len(), 2);
        assert_eq!(convergents[0], Rational::new(3, 1).unwrap());
        assert_eq!(convergents[1], Rational::new(22, 7).unwrap());
    }

    #[test]
    fn test_display() {
        let r = Rational::new(355, 113).unwrap();
        let cf = ContinuedFraction::from_rational(&r);
        let display = format!("{}", cf);

        // Should show the coefficients
        assert!(display.starts_with('['));
        assert!(display.ends_with(']'));
    }
}
