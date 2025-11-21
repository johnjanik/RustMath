//! Algebraic numbers (elements of QQbar)
//!
//! An algebraic number is a complex number that is a root of a non-zero
//! polynomial with rational coefficients.

use crate::descriptor::{AlgebraicDescriptor, ANRational, ANBinaryExpr, ANUnaryExpr, BinaryOp, UnaryOp};
use rustmath_core::{Field, Ring, MathError, Result};
use rustmath_rationals::Rational;
use rustmath_complex::Complex;
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::fmt;

/// An algebraic number (element of QQbar)
///
/// Represents a complex number that is a root of a polynomial with rational coefficients.
#[derive(Debug, Clone)]
pub struct AlgebraicNumber {
    /// Internal representation
    descriptor: AlgebraicDescriptor,
}

impl AlgebraicNumber {
    /// Create an algebraic number from a descriptor
    pub fn new(descriptor: AlgebraicDescriptor) -> Self {
        Self { descriptor }
    }

    /// Create an algebraic number from a rational
    pub fn from_rational(r: Rational) -> Self {
        Self {
            descriptor: AlgebraicDescriptor::Rational(ANRational::new(r)),
        }
    }

    /// Create an algebraic number from an integer
    pub fn from_i64(n: i64) -> Self {
        Self::from_rational(Rational::from_i64(n))
    }

    /// Check if this is a rational number
    pub fn is_rational(&self) -> bool {
        self.descriptor.simplify().is_rational()
    }

    /// Try to convert to a rational number
    pub fn to_rational(&self) -> Option<Rational> {
        self.descriptor.simplify().as_rational().cloned()
    }

    /// Get a complex approximation of this number
    pub fn to_complex(&self, precision: usize) -> Complex {
        // TODO: Implement proper evaluation
        // For now, just handle rationals
        if let Some(r) = self.to_rational() {
            Complex::new(r.to_f64().unwrap_or(0.0), 0.0)
        } else {
            // Placeholder for non-rational case
            Complex::new(0.0, 0.0)
        }
    }

    /// Simplify this algebraic number
    pub fn simplify(&self) -> Self {
        Self {
            descriptor: self.descriptor.simplify(),
        }
    }

    /// Get the internal descriptor
    pub fn descriptor(&self) -> &AlgebraicDescriptor {
        &self.descriptor
    }

    /// Create the zero element (0)
    pub fn zero() -> Self {
        Self::from_i64(0)
    }

    /// Create the one element (1)
    pub fn one() -> Self {
        Self::from_i64(1)
    }

    /// Check if this is exactly zero
    pub fn is_zero(&self) -> bool {
        if let Some(r) = self.to_rational() {
            r.is_zero()
        } else {
            false
        }
    }

    /// Check if this is exactly one
    pub fn is_one(&self) -> bool {
        if let Some(r) = self.to_rational() {
            r.is_one()
        } else {
            false
        }
    }

    /// Compute the multiplicative inverse
    pub fn inverse(&self) -> Result<Self> {
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        Ok(Self {
            descriptor: AlgebraicDescriptor::UnaryExpr(ANUnaryExpr::new(
                UnaryOp::Inv,
                self.descriptor.clone(),
            )),
        })
    }

    /// Compute the complex conjugate
    pub fn conjugate(&self) -> Self {
        Self {
            descriptor: AlgebraicDescriptor::UnaryExpr(ANUnaryExpr::new(
                UnaryOp::Conj,
                self.descriptor.clone(),
            )),
        }
    }

    /// Raise to a rational power
    pub fn pow_rational(&self, exponent: &Rational) -> Self {
        let exp_desc = AlgebraicDescriptor::Rational(ANRational::new(exponent.clone()));
        Self {
            descriptor: AlgebraicDescriptor::BinaryExpr(ANBinaryExpr::new(
                BinaryOp::Pow,
                self.descriptor.clone(),
                exp_desc,
            )),
        }
    }
}

impl PartialEq for AlgebraicNumber {
    fn eq(&self, other: &Self) -> bool {
        // Simplify both and compare
        let a = self.simplify();
        let b = other.simplify();

        // For rationals, compare exactly
        if let (Some(r1), Some(r2)) = (a.to_rational(), b.to_rational()) {
            return r1 == r2;
        }

        // For non-rationals, we need a more sophisticated comparison
        // TODO: Implement exact comparison using minimal polynomials
        false
    }
}

impl Eq for AlgebraicNumber {}

impl fmt::Display for AlgebraicNumber {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let simplified = self.simplify();
        if let Some(r) = simplified.to_rational() {
            write!(f, "{}", r)
        } else {
            write!(f, "{}", simplified.descriptor)
        }
    }
}

// Arithmetic operations

impl Add for AlgebraicNumber {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            descriptor: AlgebraicDescriptor::BinaryExpr(ANBinaryExpr::new(
                BinaryOp::Add,
                self.descriptor,
                other.descriptor,
            )),
        }
        .simplify()
    }
}

impl Sub for AlgebraicNumber {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            descriptor: AlgebraicDescriptor::BinaryExpr(ANBinaryExpr::new(
                BinaryOp::Sub,
                self.descriptor,
                other.descriptor,
            )),
        }
        .simplify()
    }
}

impl Mul for AlgebraicNumber {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            descriptor: AlgebraicDescriptor::BinaryExpr(ANBinaryExpr::new(
                BinaryOp::Mul,
                self.descriptor,
                other.descriptor,
            )),
        }
        .simplify()
    }
}

impl Div for AlgebraicNumber {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            descriptor: AlgebraicDescriptor::BinaryExpr(ANBinaryExpr::new(
                BinaryOp::Div,
                self.descriptor,
                other.descriptor,
            )),
        }
        .simplify()
    }
}

impl Neg for AlgebraicNumber {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            descriptor: AlgebraicDescriptor::UnaryExpr(ANUnaryExpr::new(
                UnaryOp::Neg,
                self.descriptor,
            )),
        }
        .simplify()
    }
}

// Ring trait implementation

impl Ring for AlgebraicNumber {
    fn zero() -> Self {
        AlgebraicNumber::zero()
    }

    fn one() -> Self {
        AlgebraicNumber::one()
    }

    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    fn is_one(&self) -> bool {
        self.is_one()
    }
}

impl rustmath_core::CommutativeRing for AlgebraicNumber {}
impl rustmath_core::IntegralDomain for AlgebraicNumber {}

impl Field for AlgebraicNumber {
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
    fn test_rational_arithmetic() {
        let a = AlgebraicNumber::from_rational(Rational::new(3, 4).unwrap());
        let b = AlgebraicNumber::from_rational(Rational::new(1, 2).unwrap());

        let sum = a.clone() + b.clone();
        assert_eq!(sum.to_rational().unwrap(), Rational::new(5, 4).unwrap());

        let diff = a.clone() - b.clone();
        assert_eq!(diff.to_rational().unwrap(), Rational::new(1, 4).unwrap());

        let prod = a.clone() * b.clone();
        assert_eq!(prod.to_rational().unwrap(), Rational::new(3, 8).unwrap());

        let quot = a.clone() / b.clone();
        assert_eq!(quot.to_rational().unwrap(), Rational::new(3, 2).unwrap());
    }

    #[test]
    fn test_negation() {
        let a = AlgebraicNumber::from_i64(5);
        let neg_a = -a;
        assert_eq!(neg_a.to_rational().unwrap(), Rational::new(-5, 1).unwrap());
    }

    #[test]
    fn test_inverse() {
        let a = AlgebraicNumber::from_rational(Rational::new(2, 3).unwrap());
        let inv = a.inverse().unwrap();
        assert_eq!(inv.to_rational().unwrap(), Rational::new(3, 2).unwrap());
    }

    #[test]
    fn test_ring_properties() {
        let zero = AlgebraicNumber::zero();
        let one = AlgebraicNumber::one();

        assert!(zero.is_zero());
        assert!(one.is_one());

        let a = AlgebraicNumber::from_i64(5);

        // a + 0 = a
        let sum = a.clone() + zero.clone();
        assert_eq!(sum.to_rational().unwrap(), a.to_rational().unwrap());

        // a * 1 = a
        let prod = a.clone() * one.clone();
        assert_eq!(prod.to_rational().unwrap(), a.to_rational().unwrap());

        // a * 0 = 0
        let prod_zero = a.clone() * zero.clone();
        assert!(prod_zero.is_zero());
    }
}
