//! Algebraic real numbers (elements of AA)
//!
//! An algebraic real is a real number that is a root of a non-zero
//! polynomial with rational coefficients.

use crate::algebraic_number::AlgebraicNumber;
use crate::descriptor::{
    AlgebraicDescriptor, ANRoot, ANRational, ANUnaryExpr, ANBinaryExpr, BinaryOp, UnaryOp,
};
use rustmath_core::{Field, Ring, MathError, Result};
use rustmath_rationals::Rational;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_integers::Integer;
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::cmp::Ordering;
use std::fmt;

/// An algebraic real number (element of AA)
///
/// Represents a real number that is a root of a polynomial with rational coefficients.
/// This type is more restrictive than AlgebraicNumber - it only represents real values.
#[derive(Debug, Clone)]
pub struct AlgebraicReal {
    /// Internal representation using AlgebraicNumber
    inner: AlgebraicNumber,
}

impl AlgebraicReal {
    /// Create an algebraic real from a rational
    pub fn from_rational(r: Rational) -> Self {
        Self {
            inner: AlgebraicNumber::from_rational(r),
        }
    }

    /// Create an algebraic real from an integer
    pub fn from_i64(n: i64) -> Self {
        Self::from_rational(Rational::from_i64(n))
    }

    /// Create the square root of a positive integer
    ///
    /// # Arguments
    /// * `n` - A positive integer
    ///
    /// # Returns
    /// * sqrt(n) as an algebraic real number
    ///
    /// # Examples
    /// ```
    /// use rustmath_algebraic::AlgebraicReal;
    ///
    /// let sqrt2 = AlgebraicReal::sqrt(2);
    /// let sqrt3 = AlgebraicReal::sqrt(3);
    /// ```
    pub fn sqrt(n: i64) -> Self {
        if n == 0 {
            return Self::from_i64(0);
        }

        if n < 0 {
            panic!("Cannot take square root of negative number in AlgebraicReal");
        }

        // Check if n is a perfect square
        let sqrt_n = (n as f64).sqrt();
        if sqrt_n.fract() == 0.0 {
            return Self::from_i64(sqrt_n as i64);
        }

        // Create polynomial x^2 - n
        let poly = UnivariatePolynomial::new(vec![Integer::from(-n), Integer::zero(), Integer::one()]);

        // Create isolating interval using Newton's method approximation
        let approx = sqrt_n;
        let epsilon = 0.1;
        let lower = Rational::from_f64(approx - epsilon).unwrap();
        let upper = Rational::from_f64(approx + epsilon).unwrap();

        let root = ANRoot::new(poly, Some((lower, upper)), None);

        Self {
            inner: AlgebraicNumber::new(AlgebraicDescriptor::Root(root)),
        }
    }

    /// Create the nth root of a number
    pub fn nth_root(n: i64, degree: u32) -> Self {
        if degree == 0 {
            panic!("Cannot take 0th root");
        }

        if degree == 1 {
            return Self::from_i64(n);
        }

        if degree == 2 {
            return Self::sqrt(n);
        }

        if n < 0 && degree % 2 == 0 {
            panic!("Cannot take even root of negative number in AlgebraicReal");
        }

        // For odd degree, we can handle negative n
        let abs_n = n.abs();
        let sign = if n < 0 { -1 } else { 1 };

        // Check if abs_n is a perfect power
        let root_approx = (abs_n as f64).powf(1.0 / degree as f64);
        let root_int = root_approx.round() as i64;
        if root_int.pow(degree) == abs_n {
            return Self::from_i64(sign * root_int);
        }

        // Create polynomial x^degree - n
        let mut coeffs = vec![Integer::from(-n)];
        for _ in 1..degree {
            coeffs.push(Integer::zero());
        }
        coeffs.push(Integer::one());

        let poly = UnivariatePolynomial::new(coeffs);

        // Create isolating interval
        let approx = (n as f64).powf(1.0 / degree as f64);
        let epsilon = 0.1;
        let lower = Rational::from_f64(approx - epsilon).unwrap();
        let upper = Rational::from_f64(approx + epsilon).unwrap();

        let root = ANRoot::new(poly, Some((lower, upper)), None);

        Self {
            inner: AlgebraicNumber::new(AlgebraicDescriptor::Root(root)),
        }
    }

    /// Check if this is a rational number
    pub fn is_rational(&self) -> bool {
        self.inner.is_rational()
    }

    /// Try to convert to a rational number
    pub fn to_rational(&self) -> Option<Rational> {
        self.inner.to_rational()
    }

    /// Get a floating-point approximation
    pub fn to_f64(&self, precision: usize) -> f64 {
        if let Some(r) = self.to_rational() {
            r.to_f64()
        } else {
            // TODO: Implement proper evaluation
            0.0
        }
    }

    /// Simplify this algebraic real
    pub fn simplify(&self) -> Self {
        Self {
            inner: self.inner.simplify(),
        }
    }

    /// Check if this is exactly zero
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Check if this is exactly one
    pub fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    /// Compute the multiplicative inverse
    pub fn inverse(&self) -> Result<Self> {
        Ok(Self {
            inner: self.inner.inverse()?,
        })
    }

    /// Compute the absolute value
    pub fn abs(&self) -> Self {
        Self {
            inner: AlgebraicNumber::new(AlgebraicDescriptor::UnaryExpr(ANUnaryExpr::new(
                UnaryOp::Abs,
                self.inner.descriptor().clone(),
            ))),
        }
    }

    /// Convert to AlgebraicNumber
    pub fn to_algebraic_number(&self) -> AlgebraicNumber {
        self.inner.clone()
    }

    /// Get the sign of this number (-1, 0, or 1)
    pub fn sign(&self) -> i32 {
        if let Some(r) = self.to_rational() {
            if r.is_zero() {
                0
            } else if r.numerator().signum() > 0 {
                1
            } else {
                -1
            }
        } else {
            // TODO: Implement sign determination for irrational algebraic reals
            0
        }
    }
}

impl PartialEq for AlgebraicReal {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for AlgebraicReal {}

impl PartialOrd for AlgebraicReal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AlgebraicReal {
    fn cmp(&self, other: &Self) -> Ordering {
        // Simplify both
        let a = self.simplify();
        let b = other.simplify();

        // For rationals, compare exactly
        if let (Some(r1), Some(r2)) = (a.to_rational(), b.to_rational()) {
            if r1 < r2 {
                return Ordering::Less;
            } else if r1 > r2 {
                return Ordering::Greater;
            } else {
                return Ordering::Equal;
            }
        }

        // TODO: Implement exact comparison for irrational algebraic reals
        // using interval refinement
        Ordering::Equal
    }
}

impl fmt::Display for AlgebraicReal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

// Arithmetic operations

impl Add for AlgebraicReal {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            inner: self.inner + other.inner,
        }
    }
}

impl Sub for AlgebraicReal {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            inner: self.inner - other.inner,
        }
    }
}

impl Mul for AlgebraicReal {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            inner: self.inner * other.inner,
        }
    }
}

impl Div for AlgebraicReal {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            inner: self.inner / other.inner,
        }
    }
}

impl Neg for AlgebraicReal {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            inner: -self.inner,
        }
    }
}

// Ring trait implementation

impl Ring for AlgebraicReal {
    fn zero() -> Self {
        Self::from_i64(0)
    }

    fn one() -> Self {
        Self::from_i64(1)
    }

    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    fn is_one(&self) -> bool {
        self.is_one()
    }
}

impl rustmath_core::CommutativeRing for AlgebraicReal {}
impl rustmath_core::IntegralDomain for AlgebraicReal {}

impl Field for AlgebraicReal {
    fn inverse(&self) -> Result<Self> {
        self.inverse()
    }

    fn divide(&self, other: &Self) -> Result<Self> {
        if other.is_zero() {
            Err(MathError::DivisionByZero)
        } else {
            Ok(self.clone() / other.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_algebraic_reals() {
        let a = AlgebraicReal::from_rational(Rational::new(3, 4).unwrap());
        let b = AlgebraicReal::from_rational(Rational::new(1, 2).unwrap());

        assert!(a > b);
        assert!(b < a);
        assert_eq!(a.clone(), a.clone());
    }

    #[test]
    fn test_sqrt_perfect_square() {
        let sqrt4 = AlgebraicReal::sqrt(4);
        assert!(sqrt4.is_rational());
        assert_eq!(sqrt4.to_rational().unwrap(), Rational::new(2, 1).unwrap());
    }

    #[test]
    fn test_sqrt_irrational() {
        let sqrt2 = AlgebraicReal::sqrt(2);
        let sqrt2_squared = sqrt2.clone() * sqrt2.clone();

        assert!(sqrt2_squared.is_rational());
        assert_eq!(
            sqrt2_squared.to_rational().unwrap(),
            Rational::new(2, 1).unwrap()
        );
    }

    #[test]
    fn test_golden_ratio() {
        // φ = (1 + sqrt(5)) / 2
        let one = AlgebraicReal::from_i64(1);
        let two = AlgebraicReal::from_i64(2);
        let sqrt5 = AlgebraicReal::sqrt(5);

        let phi = (one + sqrt5) / two;

        // φ² = φ + 1
        let phi_squared = phi.clone() * phi.clone();
        let phi_plus_one = phi.clone() + AlgebraicReal::from_i64(1);

        // Both should simplify to the same value
        // (exact comparison will work once we implement it properly)
        assert_eq!(phi_squared, phi_plus_one);
    }

    #[test]
    fn test_nth_root() {
        let cube_root_8 = AlgebraicReal::nth_root(8, 3);
        assert!(cube_root_8.is_rational());
        assert_eq!(cube_root_8.to_rational().unwrap(), Rational::new(2, 1).unwrap());
    }

    #[test]
    fn test_ordering() {
        let a = AlgebraicReal::from_i64(3);
        let b = AlgebraicReal::from_i64(5);
        let c = AlgebraicReal::from_i64(3);

        assert!(a < b);
        assert!(b > a);
        assert!(a == c);
    }
}
