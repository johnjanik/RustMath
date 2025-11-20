//! SageMath-compatible wrapper for Rational type
//!
//! This module provides a SageMath-compatible API for the RustMath Rational type,
//! allowing for easy migration from SageMath code and providing familiar method names.
//!
//! # Exact Arithmetic Guarantees
//!
//! All operations in this module use **exact arithmetic** - no floating-point approximations
//! are used unless explicitly requested (e.g., via `to_f64()`). This means:
//!
//! - All rational numbers are automatically reduced to lowest terms
//! - No precision is lost in arithmetic operations
//! - Comparisons are exact (no epsilon needed)
//! - Continued fractions are computed exactly using the Euclidean algorithm
//! - Convergents are exact rational approximations
//!
//! # Examples
//!
//! ```
//! use rustmath_rationals::sage_wrapper::SageRational;
//!
//! // Create a rational number
//! let r = SageRational::new(22, 7).unwrap();
//!
//! // Get numerator and denominator
//! assert_eq!(r.numerator().to_string(), "22");
//! assert_eq!(r.denominator().to_string(), "7");
//!
//! // Compute continued fraction expansion
//! let cf = r.continued_fraction();
//! println!("22/7 = {}", cf);  // [3; 7]
//!
//! // Get convergents
//! let convergents = r.convergents();
//! assert_eq!(convergents.len(), 2);
//! assert_eq!(convergents[0].to_string(), "3");
//! assert_eq!(convergents[1].to_string(), "22/7");
//! ```

use crate::continued_fraction::ContinuedFraction;
use crate::Rational;
use rustmath_core::{Result, Ring};
use rustmath_integers::Integer;

/// SageMath-compatible wrapper for Rational type
///
/// This type wraps the RustMath `Rational` type and provides a SageMath-compatible API.
/// All operations use exact arithmetic with no floating-point approximations.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SageRational(Rational);

impl SageRational {
    /// Create a new rational number from integers
    ///
    /// The rational is automatically reduced to lowest terms, and the denominator
    /// is guaranteed to be positive.
    ///
    /// # Arguments
    ///
    /// * `numerator` - The numerator (any type convertible to Integer)
    /// * `denominator` - The denominator (any type convertible to Integer)
    ///
    /// # Returns
    ///
    /// Returns `Ok(SageRational)` if the denominator is non-zero,
    /// otherwise returns `Err(MathError::DivisionByZero)`
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::new(4, 6).unwrap();
    /// assert_eq!(r.numerator().to_string(), "2");
    /// assert_eq!(r.denominator().to_string(), "3");
    /// ```
    pub fn new<T: Into<Integer>>(numerator: T, denominator: T) -> Result<Self> {
        Ok(SageRational(Rational::new(numerator, denominator)?))
    }

    /// Create a rational from an integer
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::from_integer(42);
    /// assert_eq!(r.numerator().to_string(), "42");
    /// assert_eq!(r.denominator().to_string(), "1");
    /// ```
    pub fn from_integer<T: Into<Integer>>(n: T) -> Self {
        SageRational(Rational::from_integer(n))
    }

    /// Get the numerator (in lowest terms)
    ///
    /// # Exact Arithmetic Guarantee
    ///
    /// The numerator is guaranteed to be part of the unique representation in lowest terms.
    /// For example, `SageRational::new(4, 6)` will have numerator 2, not 4.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    /// use rustmath_integers::Integer;
    ///
    /// let r = SageRational::new(4, 6).unwrap();
    /// assert_eq!(r.numerator(), &Integer::from(2));
    /// ```
    pub fn numerator(&self) -> &Integer {
        self.0.numerator()
    }

    /// Get the denominator (in lowest terms)
    ///
    /// # Exact Arithmetic Guarantee
    ///
    /// The denominator is guaranteed to be:
    /// 1. Part of the unique representation in lowest terms
    /// 2. Always positive (sign is carried by the numerator)
    /// 3. Non-zero
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    /// use rustmath_integers::Integer;
    ///
    /// let r = SageRational::new(4, 6).unwrap();
    /// assert_eq!(r.denominator(), &Integer::from(3));
    ///
    /// // Denominator is always positive
    /// let r = SageRational::new(-4, -6).unwrap();
    /// assert_eq!(r.denominator(), &Integer::from(3));
    /// assert_eq!(r.numerator(), &Integer::from(2));
    /// ```
    pub fn denominator(&self) -> &Integer {
        self.0.denominator()
    }

    /// Compute the continued fraction expansion
    ///
    /// Returns the finite continued fraction representation [a₀; a₁, a₂, ..., aₙ]
    /// using the Euclidean algorithm.
    ///
    /// # Exact Arithmetic Guarantee
    ///
    /// The continued fraction is computed using exact integer division (Euclidean algorithm).
    /// The expansion is unique and finite for all rational numbers.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// // 22/7 = [3; 7] (a famous approximation of π)
    /// let r = SageRational::new(22, 7).unwrap();
    /// let cf = r.continued_fraction();
    /// println!("{}", cf);  // [3; 7]
    ///
    /// // 649/200 = [3; 4, 12, 4]
    /// let r = SageRational::new(649, 200).unwrap();
    /// let cf = r.continued_fraction();
    /// println!("{}", cf);  // [3; 4, 12, 4]
    /// ```
    pub fn continued_fraction(&self) -> ContinuedFraction {
        ContinuedFraction::from_rational(&self.0)
    }

    /// Get all convergents of the continued fraction expansion
    ///
    /// Convergents are the sequence of "best rational approximations" obtained
    /// by truncating the continued fraction at each step.
    ///
    /// # Exact Arithmetic Guarantee
    ///
    /// All convergents are computed exactly as rational numbers. Each convergent
    /// is the best rational approximation with denominator up to that point.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// // 22/7 = [3; 7]
    /// let r = SageRational::new(22, 7).unwrap();
    /// let convergents = r.convergents();
    ///
    /// assert_eq!(convergents.len(), 2);
    /// assert_eq!(convergents[0].to_string(), "3");      // First convergent: 3/1
    /// assert_eq!(convergents[1].to_string(), "22/7");   // Second convergent: 22/7
    /// ```
    pub fn convergents(&self) -> Vec<SageRational> {
        let cf = self.continued_fraction();
        cf.all_convergents()
            .into_iter()
            .map(SageRational)
            .collect()
    }

    /// Get the nth convergent of the continued fraction
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the convergent (0-indexed)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::new(649, 200).unwrap();
    ///
    /// assert_eq!(r.convergent(0).to_string(), "3");
    /// assert_eq!(r.convergent(1).to_string(), "13/4");
    /// assert_eq!(r.convergent(2).to_string(), "159/49");
    /// assert_eq!(r.convergent(3).to_string(), "649/200");
    /// ```
    pub fn convergent(&self, n: usize) -> SageRational {
        let cf = self.continued_fraction();
        SageRational(cf.convergent(n))
    }

    /// Get the absolute value
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::new(-3, 4).unwrap();
    /// assert_eq!(r.abs().to_string(), "3/4");
    /// ```
    pub fn abs(&self) -> Self {
        SageRational(self.0.abs())
    }

    /// Get the sign of the rational number
    ///
    /// Returns:
    /// - 1 if positive
    /// - 0 if zero
    /// - -1 if negative
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// assert_eq!(SageRational::new(3, 4).unwrap().sign(), 1);
    /// assert_eq!(SageRational::new(0, 1).unwrap().sign(), 0);
    /// assert_eq!(SageRational::new(-3, 4).unwrap().sign(), -1);
    /// ```
    pub fn sign(&self) -> i32 {
        self.0.numerator().signum() as i32
    }

    /// Compute the reciprocal (multiplicative inverse)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::new(3, 4).unwrap();
    /// let inv = r.inverse().unwrap();
    /// assert_eq!(inv.to_string(), "4/3");
    /// ```
    pub fn inverse(&self) -> Result<Self> {
        Ok(SageRational(self.0.reciprocal()?))
    }

    /// Compute the floor (largest integer <= self)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::new(7, 3).unwrap();
    /// assert_eq!(r.floor().to_string(), "2");
    ///
    /// let r = SageRational::new(-7, 3).unwrap();
    /// assert_eq!(r.floor().to_string(), "-3");
    /// ```
    pub fn floor(&self) -> Integer {
        self.0.floor()
    }

    /// Compute the ceiling (smallest integer >= self)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::new(7, 3).unwrap();
    /// assert_eq!(r.ceil().to_string(), "3");
    ///
    /// let r = SageRational::new(-7, 3).unwrap();
    /// assert_eq!(r.ceil().to_string(), "-2");
    /// ```
    pub fn ceil(&self) -> Integer {
        self.0.ceil()
    }

    /// Round to the nearest integer
    ///
    /// Uses "round half up" rule: 0.5 rounds to 1, -0.5 rounds to -1
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::new(7, 3).unwrap();  // 2.333...
    /// assert_eq!(r.round().to_string(), "2");
    ///
    /// let r = SageRational::new(8, 3).unwrap();  // 2.666...
    /// assert_eq!(r.round().to_string(), "3");
    /// ```
    pub fn round(&self) -> Integer {
        self.0.round()
    }

    /// Check if this is an integer
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// assert!(SageRational::new(6, 3).unwrap().is_integer());
    /// assert!(!SageRational::new(7, 3).unwrap().is_integer());
    /// ```
    pub fn is_integer(&self) -> bool {
        self.0.is_integer()
    }

    /// Check if this is zero
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// assert!(SageRational::new(0, 1).unwrap().is_zero());
    /// assert!(!SageRational::new(1, 2).unwrap().is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Check if this is one
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// assert!(SageRational::new(3, 3).unwrap().is_one());
    /// assert!(!SageRational::new(2, 3).unwrap().is_one());
    /// ```
    pub fn is_one(&self) -> bool {
        self.0.is_one()
    }

    /// Compute the p-adic valuation
    ///
    /// Returns v_p(a/b) = v_p(a) - v_p(b), the exponent of p in the
    /// prime factorization when written in lowest terms.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    /// use rustmath_integers::Integer;
    ///
    /// // 8/3 = 2^3 / 3
    /// let r = SageRational::new(8, 3).unwrap();
    /// assert_eq!(r.valuation(&Integer::from(2)), 3);  // 2^3 in numerator
    /// assert_eq!(r.valuation(&Integer::from(3)), -1); // 3^1 in denominator
    /// ```
    pub fn valuation(&self, p: &Integer) -> i32 {
        self.0.valuation(p)
    }

    /// Get the height of the rational number
    ///
    /// Returns max(|numerator|, |denominator|) when in lowest terms.
    /// This is useful in Diophantine approximation theory.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    /// use rustmath_integers::Integer;
    ///
    /// let r = SageRational::new(3, 7).unwrap();
    /// assert_eq!(r.height(), Integer::from(7));
    ///
    /// let r = SageRational::new(22, 7).unwrap();
    /// assert_eq!(r.height(), Integer::from(22));
    /// ```
    pub fn height(&self) -> Integer {
        let num_abs = self.0.numerator().abs();
        let den_abs = self.0.denominator().abs();
        if num_abs > den_abs {
            num_abs
        } else {
            den_abs
        }
    }

    /// Get the content (for consistency with polynomial interface)
    ///
    /// For a rational number, this always returns the absolute value.
    /// This method exists for API compatibility with SageMath.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::new(-3, 4).unwrap();
    /// assert_eq!(r.content().to_string(), "3/4");
    /// ```
    pub fn content(&self) -> Self {
        self.abs()
    }

    /// Compute the nth root (if it exists as a rational)
    ///
    /// Returns Some(root) if the nth root is rational, None otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::new(8, 27).unwrap();  // (2/3)^3
    /// let root = r.nth_root(3).unwrap();
    /// assert_eq!(root.to_string(), "2/3");
    ///
    /// let r = SageRational::new(2, 1).unwrap();
    /// assert!(r.nth_root(2).is_none());  // √2 is irrational
    /// ```
    pub fn nth_root(&self, n: u32) -> Option<Self> {
        if n == 0 {
            return None;
        }

        // Try to compute nth root of numerator and denominator separately
        let num_root = self.0.numerator().nth_root(n).ok()?;
        let den_root = self.0.denominator().nth_root(n).ok()?;

        // Verify that these are actually nth roots
        let num_check = num_root.pow(n);
        let den_check = den_root.pow(n);

        if &num_check == self.0.numerator() && &den_check == self.0.denominator() {
            Some(SageRational(Rational::new(num_root, den_root).unwrap()))
        } else {
            None
        }
    }

    /// Compute the square root (if it exists as a rational)
    ///
    /// Returns Some(sqrt) if the square root is rational, None otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::new(4, 9).unwrap();  // (2/3)^2
    /// let sqrt = r.sqrt().unwrap();
    /// assert_eq!(sqrt.to_string(), "2/3");
    ///
    /// let r = SageRational::new(2, 1).unwrap();
    /// assert!(r.sqrt().is_none());  // √2 is irrational
    /// ```
    pub fn sqrt(&self) -> Option<Self> {
        self.nth_root(2)
    }

    /// Compute modulo an integer
    ///
    /// Returns the unique r with 0 <= r < m such that self = q*m + r for some integer q.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    /// use rustmath_integers::Integer;
    ///
    /// let r = SageRational::new(7, 3).unwrap();
    /// let m = r.mod_integer(&Integer::from(2));
    /// assert_eq!(m.to_string(), "1/3");  // 7/3 = 2*2 + 1/3
    /// ```
    pub fn mod_integer(&self, m: &Integer) -> Self {
        if m.is_zero() {
            return self.clone();
        }

        // self = q*m + r where r is the result
        // We need 0 <= r < m
        let q = self.floor() / m.clone();
        let r = self.0.clone() - Rational::from_integer(q * m.clone());
        SageRational(r)
    }

    /// Get the additive order
    ///
    /// Returns Some(1) if self is 0, otherwise None (infinite order in ℚ)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// assert_eq!(SageRational::new(0, 1).unwrap().additive_order(), Some(1));
    /// assert_eq!(SageRational::new(1, 2).unwrap().additive_order(), None);
    /// ```
    pub fn additive_order(&self) -> Option<u64> {
        if self.is_zero() {
            Some(1)
        } else {
            None // Infinite order
        }
    }

    /// Get the multiplicative order
    ///
    /// Returns Some(1) if self is 1, Some(2) if self is -1, otherwise None
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// assert_eq!(SageRational::new(1, 1).unwrap().multiplicative_order(), Some(1));
    /// assert_eq!(SageRational::new(-1, 1).unwrap().multiplicative_order(), Some(2));
    /// assert_eq!(SageRational::new(2, 1).unwrap().multiplicative_order(), None);
    /// ```
    pub fn multiplicative_order(&self) -> Option<u64> {
        if self.is_one() {
            Some(1)
        } else if self.0.clone() == -Rational::one() {
            Some(2)
        } else {
            None // Infinite order
        }
    }

    /// Convert to f64 (may lose precision)
    ///
    /// **WARNING**: This loses the exact arithmetic guarantee!
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let r = SageRational::new(1, 3).unwrap();
    /// let f = r.to_f64().unwrap();
    /// assert!((f - 0.333333333).abs() < 1e-6);
    /// ```
    pub fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }

    /// Get the underlying Rational object
    ///
    /// This is useful for interoperability with other RustMath types.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let sage_r = SageRational::new(3, 4).unwrap();
    /// let r = sage_r.as_rational();
    /// ```
    pub fn as_rational(&self) -> &Rational {
        &self.0
    }

    /// Convert to the underlying Rational object
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rationals::sage_wrapper::SageRational;
    ///
    /// let sage_r = SageRational::new(3, 4).unwrap();
    /// let r = sage_r.into_rational();
    /// ```
    pub fn into_rational(self) -> Rational {
        self.0
    }
}

// Implement From/Into for easy conversion
impl From<Rational> for SageRational {
    fn from(r: Rational) -> Self {
        SageRational(r)
    }
}

impl From<SageRational> for Rational {
    fn from(r: SageRational) -> Self {
        r.0
    }
}

impl From<i64> for SageRational {
    fn from(n: i64) -> Self {
        SageRational::from_integer(n)
    }
}

impl From<i32> for SageRational {
    fn from(n: i32) -> Self {
        SageRational::from_integer(n)
    }
}

// Display implementation
impl std::fmt::Display for SageRational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Debug for SageRational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SageRational({})", self.0)
    }
}

// Arithmetic operations (forward to Rational)
impl std::ops::Add for SageRational {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        SageRational(self.0 + other.0)
    }
}

impl std::ops::Sub for SageRational {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        SageRational(self.0 - other.0)
    }
}

impl std::ops::Mul for SageRational {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        SageRational(self.0 * other.0)
    }
}

impl std::ops::Div for SageRational {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        SageRational(self.0 / other.0)
    }
}

impl std::ops::Neg for SageRational {
    type Output = Self;

    fn neg(self) -> Self {
        SageRational(-self.0)
    }
}

// Reference arithmetic
impl<'a> std::ops::Add<&'a SageRational> for &SageRational {
    type Output = SageRational;

    fn add(self, other: &'a SageRational) -> SageRational {
        SageRational(&self.0 + &other.0)
    }
}

impl<'a> std::ops::Sub<&'a SageRational> for &SageRational {
    type Output = SageRational;

    fn sub(self, other: &'a SageRational) -> SageRational {
        SageRational(&self.0 - &other.0)
    }
}

impl<'a> std::ops::Mul<&'a SageRational> for &SageRational {
    type Output = SageRational;

    fn mul(self, other: &'a SageRational) -> SageRational {
        SageRational(&self.0 * &other.0)
    }
}

impl<'a> std::ops::Div<&'a SageRational> for &SageRational {
    type Output = SageRational;

    fn div(self, other: &'a SageRational) -> SageRational {
        SageRational(&self.0 / &other.0)
    }
}

impl<'a> std::ops::Neg for &SageRational {
    type Output = SageRational;

    fn neg(self) -> SageRational {
        SageRational(-&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::NumericConversion;

    #[test]
    fn test_creation_and_reduction() {
        let r = SageRational::new(4, 6).unwrap();
        assert_eq!(r.numerator(), &Integer::from(2));
        assert_eq!(r.denominator(), &Integer::from(3));
        assert_eq!(r.to_string(), "2/3");
    }

    #[test]
    fn test_numerator_denominator() {
        let r = SageRational::new(22, 7).unwrap();
        assert_eq!(r.numerator(), &Integer::from(22));
        assert_eq!(r.denominator(), &Integer::from(7));
    }

    #[test]
    fn test_continued_fraction() {
        // 22/7 = [3; 7]
        let r = SageRational::new(22, 7).unwrap();
        let cf = r.continued_fraction();
        assert_eq!(cf.coefficients(), &[Integer::from(3), Integer::from(7)]);

        // 649/200 = [3; 4, 12, 4]
        let r = SageRational::new(649, 200).unwrap();
        let cf = r.continued_fraction();
        assert_eq!(
            cf.coefficients(),
            &[
                Integer::from(3),
                Integer::from(4),
                Integer::from(12),
                Integer::from(4)
            ]
        );
    }

    #[test]
    fn test_convergents() {
        // 22/7 = [3; 7]
        let r = SageRational::new(22, 7).unwrap();
        let convergents = r.convergents();

        assert_eq!(convergents.len(), 2);
        assert_eq!(convergents[0].to_string(), "3");
        assert_eq!(convergents[1].to_string(), "22/7");
    }

    #[test]
    fn test_convergents_detailed() {
        // 649/200 = [3; 4, 12, 4]
        let r = SageRational::new(649, 200).unwrap();

        assert_eq!(r.convergent(0).to_string(), "3");
        assert_eq!(r.convergent(1).to_string(), "13/4");
        assert_eq!(r.convergent(2).to_string(), "159/49");
        assert_eq!(r.convergent(3).to_string(), "649/200");
    }

    #[test]
    fn test_arithmetic() {
        let a = SageRational::new(1, 2).unwrap();
        let b = SageRational::new(1, 3).unwrap();

        let sum = a.clone() + b.clone();
        assert_eq!(sum.to_string(), "5/6");

        let diff = a.clone() - b.clone();
        assert_eq!(diff.to_string(), "1/6");

        let prod = a.clone() * b.clone();
        assert_eq!(prod.to_string(), "1/6");

        let quot = a.clone() / b.clone();
        assert_eq!(quot.to_string(), "3/2");

        let neg = -a.clone();
        assert_eq!(neg.to_string(), "-1/2");
    }

    #[test]
    fn test_comparison() {
        let a = SageRational::new(1, 2).unwrap();
        let b = SageRational::new(2, 3).unwrap();

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, a.clone());
    }

    #[test]
    fn test_abs_and_sign() {
        let r = SageRational::new(-3, 4).unwrap();
        assert_eq!(r.abs().to_string(), "3/4");
        assert_eq!(r.sign(), -1);

        let r = SageRational::new(3, 4).unwrap();
        assert_eq!(r.abs().to_string(), "3/4");
        assert_eq!(r.sign(), 1);

        let r = SageRational::new(0, 1).unwrap();
        assert_eq!(r.abs().to_string(), "0");
        assert_eq!(r.sign(), 0);
    }

    #[test]
    fn test_inverse() {
        let r = SageRational::new(3, 4).unwrap();
        let inv = r.inverse().unwrap();
        assert_eq!(inv.to_string(), "4/3");

        let zero = SageRational::new(0, 1).unwrap();
        assert!(zero.inverse().is_err());
    }

    #[test]
    fn test_floor_ceil_round() {
        let r = SageRational::new(7, 3).unwrap();
        assert_eq!(r.floor(), Integer::from(2));
        assert_eq!(r.ceil(), Integer::from(3));
        assert_eq!(r.round(), Integer::from(2));

        let r = SageRational::new(8, 3).unwrap();
        assert_eq!(r.floor(), Integer::from(2));
        assert_eq!(r.ceil(), Integer::from(3));
        assert_eq!(r.round(), Integer::from(3));

        let r = SageRational::new(-7, 3).unwrap();
        assert_eq!(r.floor(), Integer::from(-3));
        assert_eq!(r.ceil(), Integer::from(-2));
    }

    #[test]
    fn test_is_predicates() {
        assert!(SageRational::new(6, 3).unwrap().is_integer());
        assert!(!SageRational::new(7, 3).unwrap().is_integer());

        assert!(SageRational::new(0, 1).unwrap().is_zero());
        assert!(!SageRational::new(1, 2).unwrap().is_zero());

        assert!(SageRational::new(3, 3).unwrap().is_one());
        assert!(!SageRational::new(2, 3).unwrap().is_one());
    }

    #[test]
    fn test_valuation() {
        // 8/3 = 2^3 / 3
        let r = SageRational::new(8, 3).unwrap();
        assert_eq!(r.valuation(&Integer::from(2)), 3);
        assert_eq!(r.valuation(&Integer::from(3)), -1);
        assert_eq!(r.valuation(&Integer::from(5)), 0);

        // 12/25 = (2^2 * 3) / 5^2
        let r = SageRational::new(12, 25).unwrap();
        assert_eq!(r.valuation(&Integer::from(2)), 2);
        assert_eq!(r.valuation(&Integer::from(3)), 1);
        assert_eq!(r.valuation(&Integer::from(5)), -2);
    }

    #[test]
    fn test_height() {
        let r = SageRational::new(3, 7).unwrap();
        assert_eq!(r.height(), Integer::from(7));

        let r = SageRational::new(22, 7).unwrap();
        assert_eq!(r.height(), Integer::from(22));

        let r = SageRational::new(-5, 3).unwrap();
        assert_eq!(r.height(), Integer::from(5));
    }

    #[test]
    fn test_content() {
        let r = SageRational::new(-3, 4).unwrap();
        assert_eq!(r.content().to_string(), "3/4");

        let r = SageRational::new(5, 2).unwrap();
        assert_eq!(r.content().to_string(), "5/2");
    }

    #[test]
    fn test_nth_root() {
        // (2/3)^3 = 8/27
        let r = SageRational::new(8, 27).unwrap();
        let root = r.nth_root(3).unwrap();
        assert_eq!(root.to_string(), "2/3");

        // √2 is irrational
        let r = SageRational::new(2, 1).unwrap();
        assert!(r.nth_root(2).is_none());

        // Perfect square
        let r = SageRational::new(4, 9).unwrap();
        let root = r.nth_root(2).unwrap();
        assert_eq!(root.to_string(), "2/3");
    }

    #[test]
    fn test_sqrt() {
        let r = SageRational::new(4, 9).unwrap();
        let sqrt = r.sqrt().unwrap();
        assert_eq!(sqrt.to_string(), "2/3");

        let r = SageRational::new(2, 1).unwrap();
        assert!(r.sqrt().is_none());
    }

    #[test]
    fn test_mod_integer() {
        let r = SageRational::new(7, 3).unwrap();
        let m = r.mod_integer(&Integer::from(2));
        assert_eq!(m.to_string(), "1/3");

        let r = SageRational::new(10, 3).unwrap();
        let m = r.mod_integer(&Integer::from(3));
        assert_eq!(m.to_string(), "1/3");
    }

    #[test]
    fn test_additive_order() {
        assert_eq!(SageRational::new(0, 1).unwrap().additive_order(), Some(1));
        assert_eq!(SageRational::new(1, 2).unwrap().additive_order(), None);
        assert_eq!(SageRational::new(-3, 4).unwrap().additive_order(), None);
    }

    #[test]
    fn test_multiplicative_order() {
        assert_eq!(
            SageRational::new(1, 1).unwrap().multiplicative_order(),
            Some(1)
        );
        assert_eq!(
            SageRational::new(-1, 1).unwrap().multiplicative_order(),
            Some(2)
        );
        assert_eq!(SageRational::new(2, 1).unwrap().multiplicative_order(), None);
        assert_eq!(
            SageRational::new(1, 2).unwrap().multiplicative_order(),
            None
        );
    }

    #[test]
    fn test_to_f64() {
        let r = SageRational::new(1, 3).unwrap();
        let f = r.to_f64().unwrap();
        assert!((f - 0.333333333).abs() < 1e-6);

        let r = SageRational::new(22, 7).unwrap();
        let f = r.to_f64().unwrap();
        assert!((f - 3.142857142).abs() < 1e-6);
    }

    #[test]
    fn test_conversion() {
        let r = Rational::new(3, 4).unwrap();
        let sage_r = SageRational::from(r.clone());
        assert_eq!(sage_r.to_string(), "3/4");

        let back: Rational = sage_r.into();
        assert_eq!(back, r);
    }

    #[test]
    fn test_from_integer() {
        let r = SageRational::from_integer(42);
        assert_eq!(r.to_string(), "42");
        assert!(r.is_integer());

        let r: SageRational = 42i64.into();
        assert_eq!(r.to_string(), "42");

        let r: SageRational = 42i32.into();
        assert_eq!(r.to_string(), "42");
    }

    #[test]
    fn test_exact_arithmetic_example() {
        // This test demonstrates exact arithmetic guarantees

        // Create 1/3
        let one_third = SageRational::new(1, 3).unwrap();

        // Add it three times
        let sum = &one_third + &one_third;
        let sum = &sum + &one_third;

        // Should be exactly 1, not 0.9999999...
        assert!(sum.is_one());
        assert_eq!(sum.to_string(), "1");

        // Verify continued fraction is exact
        let cf = sum.continued_fraction();
        assert_eq!(cf.coefficients(), &[Integer::from(1)]);
    }

    #[test]
    fn test_golden_ratio_convergents() {
        // The golden ratio φ = (1 + √5) / 2 has continued fraction [1; 1, 1, 1, ...]
        // Its convergents are ratios of Fibonacci numbers

        // Let's test with a high-precision rational approximation
        // Using Fibonacci numbers F(20)/F(19) = 6765/4181
        let approx = SageRational::new(6765, 4181).unwrap();

        let convergents = approx.convergents();

        // First few convergents should be Fibonacci ratios
        assert_eq!(convergents[0].to_string(), "1");
        assert_eq!(convergents[1].to_string(), "2");
        assert_eq!(convergents[2].to_string(), "3/2");
        assert_eq!(convergents[3].to_string(), "5/3");
        assert_eq!(convergents[4].to_string(), "8/5");
        assert_eq!(convergents[5].to_string(), "13/8");
    }

    #[test]
    fn test_display_debug() {
        let r = SageRational::new(3, 4).unwrap();
        assert_eq!(format!("{}", r), "3/4");
        assert_eq!(format!("{:?}", r), "SageRational(3/4)");

        let r = SageRational::new(5, 1).unwrap();
        assert_eq!(format!("{}", r), "5");
    }
}
