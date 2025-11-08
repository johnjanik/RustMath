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

/// A periodic continued fraction representation
///
/// Represents a number as: [a₀, a₁, ..., aₙ; repeating₀, repeating₁, ...]
/// where the "repeating" part repeats indefinitely.
///
/// This is used for quadratic irrationals like √2 = [1; 2, 2, 2, ...] = [1; (2)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeriodicContinuedFraction {
    /// The initial (non-repeating) coefficients
    initial: Vec<Integer>,
    /// The repeating coefficients
    repeating: Vec<Integer>,
}

impl PeriodicContinuedFraction {
    /// Create a new periodic continued fraction
    ///
    /// # Arguments
    /// * `initial` - The initial non-repeating coefficients
    /// * `repeating` - The repeating part
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::continued_fraction::PeriodicContinuedFraction;
    /// use rustmath_integers::Integer;
    ///
    /// // √2 = [1; (2)]
    /// let sqrt2 = PeriodicContinuedFraction::new(
    ///     vec![Integer::from(1)],
    ///     vec![Integer::from(2)]
    /// );
    /// ```
    pub fn new(initial: Vec<Integer>, repeating: Vec<Integer>) -> Self {
        if repeating.is_empty() {
            panic!("Repeating part cannot be empty for periodic continued fraction");
        }
        PeriodicContinuedFraction { initial, repeating }
    }

    /// Create a periodic continued fraction for √n
    ///
    /// Computes the periodic continued fraction representation of √n
    /// using the standard algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::continued_fraction::PeriodicContinuedFraction;
    /// use rustmath_integers::Integer;
    ///
    /// // √2 = [1; (2)]
    /// let sqrt2 = PeriodicContinuedFraction::from_sqrt(&Integer::from(2));
    /// assert_eq!(sqrt2.initial(), &[Integer::from(1)]);
    /// assert_eq!(sqrt2.repeating(), &[Integer::from(2)]);
    /// ```
    pub fn from_sqrt(n: &Integer) -> Option<Self> {
        // Check if n is a perfect square
        let sqrt_n = n.sqrt().ok()?;
        if &(sqrt_n.clone() * sqrt_n.clone()) == n {
            // Perfect square - not a quadratic irrational
            return None;
        }

        // Use the standard algorithm to compute the periodic continued fraction
        // √n = a₀ + (√n - a₀) where a₀ = floor(√n)
        let a0 = sqrt_n.clone();

        let mut coefficients = Vec::new();
        let mut m = Integer::zero();
        let mut d = Integer::one();
        let mut a = a0.clone();

        // Keep track of states to detect the period
        let mut seen_states = std::collections::HashMap::new();

        loop {
            // Record this state
            let state = (m.clone(), d.clone());
            if let Some(&start_idx) = seen_states.get(&state) {
                // We've found the period!
                let period: Vec<Integer> = coefficients[start_idx..].to_vec();
                let initial = if start_idx == 0 {
                    vec![a0]
                } else {
                    let mut init = vec![a0];
                    init.extend_from_slice(&coefficients[..start_idx]);
                    init
                };
                return Some(PeriodicContinuedFraction {
                    initial,
                    repeating: period,
                });
            }

            seen_states.insert(state, coefficients.len());

            // Compute next term
            m = &d * &a - m;
            d = (n.clone() - &m * &m) / d;

            if d.is_zero() {
                return None;
            }

            a = (&a0 + &m) / d.clone();
            coefficients.push(a.clone());

            // Safety check to prevent infinite loops
            if coefficients.len() > 1000 {
                return None;
            }
        }
    }

    /// Get the initial (non-repeating) coefficients
    pub fn initial(&self) -> &[Integer] {
        &self.initial
    }

    /// Get the repeating coefficients
    pub fn repeating(&self) -> &[Integer] {
        &self.repeating
    }

    /// Get the nth coefficient
    ///
    /// This includes both the initial and repeating parts
    pub fn get_coefficient(&self, n: usize) -> Integer {
        if n < self.initial.len() {
            self.initial[n].clone()
        } else {
            let idx = (n - self.initial.len()) % self.repeating.len();
            self.repeating[idx].clone()
        }
    }

    /// Compute the nth convergent
    ///
    /// Returns a rational approximation using the first n+1 terms
    pub fn convergent(&self, n: usize) -> Rational {
        if n == 0 {
            return Rational::new(self.get_coefficient(0), Integer::one()).unwrap();
        }

        // Build continued fraction with first n+1 terms
        let coeffs: Vec<Integer> = (0..=n).map(|i| self.get_coefficient(i)).collect();
        let cf = ContinuedFraction::new(coeffs);
        cf.to_rational()
    }

    /// Get convergents up to the nth term
    pub fn convergents(&self, n: usize) -> Vec<Rational> {
        (0..=n).map(|i| self.convergent(i)).collect()
    }

    /// Get the period length
    pub fn period_length(&self) -> usize {
        self.repeating.len()
    }
}

impl std::fmt::Display for PeriodicContinuedFraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;

        // Write initial part
        for (i, coeff) in self.initial.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coeff)?;
        }

        // Write repeating part
        write!(f, "; (")?;
        for (i, coeff) in self.repeating.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coeff)?;
        }
        write!(f, ")]")
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

    #[test]
    fn test_periodic_continued_fraction_sqrt2() {
        // √2 = [1; (2)]
        let sqrt2 = PeriodicContinuedFraction::from_sqrt(&Integer::from(2)).unwrap();

        assert_eq!(sqrt2.initial(), &[Integer::from(1)]);
        assert_eq!(sqrt2.repeating(), &[Integer::from(2)]);
        assert_eq!(sqrt2.period_length(), 1);

        // Check some coefficients
        assert_eq!(sqrt2.get_coefficient(0), Integer::from(1));
        assert_eq!(sqrt2.get_coefficient(1), Integer::from(2));
        assert_eq!(sqrt2.get_coefficient(2), Integer::from(2));
        assert_eq!(sqrt2.get_coefficient(3), Integer::from(2));
    }

    #[test]
    fn test_periodic_continued_fraction_sqrt3() {
        // √3 = [1; (1, 2)]
        let sqrt3 = PeriodicContinuedFraction::from_sqrt(&Integer::from(3)).unwrap();

        assert_eq!(sqrt3.initial(), &[Integer::from(1)]);
        assert_eq!(
            sqrt3.repeating(),
            &[Integer::from(1), Integer::from(2)]
        );
        assert_eq!(sqrt3.period_length(), 2);

        // Check coefficient pattern: 1, 1, 2, 1, 2, 1, 2, ...
        assert_eq!(sqrt3.get_coefficient(0), Integer::from(1));
        assert_eq!(sqrt3.get_coefficient(1), Integer::from(1));
        assert_eq!(sqrt3.get_coefficient(2), Integer::from(2));
        assert_eq!(sqrt3.get_coefficient(3), Integer::from(1));
        assert_eq!(sqrt3.get_coefficient(4), Integer::from(2));
    }

    #[test]
    fn test_periodic_continued_fraction_sqrt5() {
        // √5 = [2; (4)]
        let sqrt5 = PeriodicContinuedFraction::from_sqrt(&Integer::from(5)).unwrap();

        assert_eq!(sqrt5.initial(), &[Integer::from(2)]);
        assert_eq!(sqrt5.repeating(), &[Integer::from(4)]);
        assert_eq!(sqrt5.period_length(), 1);
    }

    #[test]
    fn test_periodic_continued_fraction_convergents() {
        // √2 = [1; (2)]
        let sqrt2 = PeriodicContinuedFraction::from_sqrt(&Integer::from(2)).unwrap();

        // First few convergents of √2:
        // p₀/q₀ = 1/1
        // p₁/q₁ = 3/2
        // p₂/q₂ = 7/5
        // p₃/q₃ = 17/12

        let conv = sqrt2.convergents(3);
        assert_eq!(conv[0], Rational::new(1, 1).unwrap());
        assert_eq!(conv[1], Rational::new(3, 2).unwrap());
        assert_eq!(conv[2], Rational::new(7, 5).unwrap());
        assert_eq!(conv[3], Rational::new(17, 12).unwrap());

        // Verify these are good approximations of √2 ≈ 1.41421356...
        // 1/1 = 1.0
        // 3/2 = 1.5
        // 7/5 = 1.4
        // 17/12 ≈ 1.41666...
    }

    #[test]
    fn test_periodic_continued_fraction_perfect_square() {
        // Perfect squares should return None
        let result = PeriodicContinuedFraction::from_sqrt(&Integer::from(4));
        assert!(result.is_none());

        let result = PeriodicContinuedFraction::from_sqrt(&Integer::from(9));
        assert!(result.is_none());

        let result = PeriodicContinuedFraction::from_sqrt(&Integer::from(16));
        assert!(result.is_none());
    }

    #[test]
    fn test_periodic_continued_fraction_display() {
        // √2 = [1; (2)]
        let sqrt2 = PeriodicContinuedFraction::from_sqrt(&Integer::from(2)).unwrap();
        let display = format!("{}", sqrt2);
        assert_eq!(display, "[1; (2)]");

        // √3 = [1; (1, 2)]
        let sqrt3 = PeriodicContinuedFraction::from_sqrt(&Integer::from(3)).unwrap();
        let display = format!("{}", sqrt3);
        assert_eq!(display, "[1; (1, 2)]");
    }
}
