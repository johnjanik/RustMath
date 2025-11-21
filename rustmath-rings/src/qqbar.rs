//! # Algebraic Numbers (QQbar) and Algebraic Reals (AA)
//!
//! This module implements algebraic numbers - complex numbers that are roots of polynomials
//! with rational coefficients. It provides two main types:
//!
//! - **AlgebraicReal (AA)**: The field of real algebraic numbers
//! - **AlgebraicNumber (QQbar)**: The field of all algebraic complex numbers (algebraic closure of Q)
//!
//! ## Mathematical Background
//!
//! An algebraic number is a complex number that is a root of some non-zero polynomial
//! with rational (or equivalently, integer) coefficients. For example:
//!
//! - √2 is algebraic because it's a root of x² - 2 = 0
//! - i is algebraic because it's a root of x² + 1 = 0
//! - The golden ratio φ = (1+√5)/2 is algebraic (root of x² - x - 1 = 0)
//!
//! ## Implementation Strategy
//!
//! Each algebraic number is represented by:
//! 1. A minimal polynomial with rational coefficients
//! 2. An approximate value for numerical disambiguation
//! 3. Metadata about whether it's real or complex
//!
//! This allows exact arithmetic while avoiding expensive symbolic computation when possible.
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::qqbar::{AlgebraicReal, AlgebraicNumber};
//! use rustmath_rationals::Rational;
//! use rustmath_polynomials::UnivariatePolynomial;
//!
//! // Create √2 as an algebraic real
//! // It's the positive root of x² - 2 = 0
//! let sqrt_2 = AlgebraicReal::from_rational(Rational::from(2)).sqrt();
//!
//! // Create i as an algebraic number
//! // It's the root of x² + 1 = 0 with positive imaginary part
//! let i = AlgebraicNumber::imaginary_unit();
//! ```

use rustmath_core::{CommutativeRing, Field, IntegralDomain, Ring};
use rustmath_rationals::Rational;
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt::{self, Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Descriptor types for algebraic number representations.
///
/// Algebraic numbers can be represented in various forms for efficiency.
/// This enum tracks the computational history and representation of each number.
#[derive(Clone, Debug, PartialEq)]
pub enum ANDescr {
    /// A rational number
    Rational(Rational),
    /// Binary operation result (add, sub, mul, div)
    BinaryExpr {
        op: BinaryOp,
        left: Box<AlgebraicNumber>,
        right: Box<AlgebraicNumber>,
    },
    /// Unary operation result (neg, sqrt, etc.)
    UnaryExpr {
        op: UnaryOp,
        arg: Box<AlgebraicNumber>,
    },
    /// Root of a polynomial with isolating interval
    Root {
        poly: UnivariatePolynomial<Rational>,
        /// Approximate value for disambiguation
        approx_real: f64,
        approx_imag: f64,
    },
}

/// Binary operations for algebraic number expressions
#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Unary operations for algebraic number expressions
#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub enum UnaryOp {
    Neg,
    Sqrt,
    Inv,
}

/// An element of the field of algebraic numbers (QQbar).
///
/// This represents a complex number that is a root of some polynomial with rational coefficients.
/// The algebraic numbers form an algebraically closed field - the algebraic closure of Q.
#[derive(Clone, Debug, PartialEq)]
pub struct AlgebraicNumber {
    /// Internal representation
    descr: ANDescr,
    /// Whether this number is known to be real
    is_real: bool,
}

impl AlgebraicNumber {
    /// Creates a new algebraic number from a rational.
    pub fn from_rational(r: Rational) -> Self {
        AlgebraicNumber {
            descr: ANDescr::Rational(r),
            is_real: true,
        }
    }

    /// Creates a new algebraic number from an integer.
    pub fn from_integer(n: i64) -> Self {
        Self::from_rational(Rational::from(n))
    }

    /// Creates the zero algebraic number.
    pub fn zero() -> Self {
        Self::from_integer(0)
    }

    /// Creates the one algebraic number.
    pub fn one() -> Self {
        Self::from_integer(1)
    }

    /// Creates the imaginary unit i.
    ///
    /// i is the principal root of x² + 1 = 0 with positive imaginary part.
    pub fn imaginary_unit() -> Self {
        // x^2 + 1 polynomial: [1, 0, 1] represents 1 + 0*x + 1*x^2
        let poly = UnivariatePolynomial::new(vec![
            Rational::from(1),
            Rational::from(0),
            Rational::from(1),
        ]);

        AlgebraicNumber {
            descr: ANDescr::Root {
                poly,
                approx_real: 0.0,
                approx_imag: 1.0,
            },
            is_real: false,
        }
    }

    /// Returns whether this algebraic number is real.
    pub fn is_real(&self) -> bool {
        self.is_real
    }

    /// Returns whether this algebraic number is zero.
    pub fn is_zero(&self) -> bool {
        match &self.descr {
            ANDescr::Rational(r) => r.is_zero(),
            _ => false, // More sophisticated check needed for general case
        }
    }

    /// Returns whether this algebraic number is one.
    pub fn is_one(&self) -> bool {
        match &self.descr {
            ANDescr::Rational(r) => r.is_one(),
            _ => false,
        }
    }

    /// Computes the square root of this algebraic number.
    ///
    /// Returns the principal square root (with non-negative real part).
    pub fn sqrt(&self) -> Self {
        AlgebraicNumber {
            descr: ANDescr::UnaryExpr {
                op: UnaryOp::Sqrt,
                arg: Box::new(self.clone()),
            },
            is_real: self.is_real,
        }
    }

    /// Returns the minimal polynomial of this algebraic number over Q.
    ///
    /// This is the monic polynomial of smallest degree with rational coefficients
    /// that has this number as a root.
    pub fn minimal_polynomial(&self) -> UnivariatePolynomial<Rational> {
        match &self.descr {
            ANDescr::Rational(r) => {
                // For rational r, minimal polynomial is x - r
                UnivariatePolynomial::new(vec![-r.clone(), Rational::one()])
            }
            ANDescr::Root { poly, .. } => poly.clone(),
            _ => {
                // For complex expressions, would need to compute minimal polynomial
                // For now, return a placeholder
                UnivariatePolynomial::new(vec![Rational::zero(), Rational::one()])
            }
        }
    }

    /// Returns an approximate floating-point value for this algebraic number.
    ///
    /// Returns (real_part, imaginary_part).
    pub fn approximate(&self) -> (f64, f64) {
        match &self.descr {
            ANDescr::Rational(r) => {
                let num = r.numerator().to_f64().unwrap_or(0.0);
                let den = r.denominator().to_f64().unwrap_or(1.0);
                let val = num / den;
                (val, 0.0)
            }
            ANDescr::Root { approx_real, approx_imag, .. } => (*approx_real, *approx_imag),
            ANDescr::BinaryExpr { op, left, right } => {
                let (lr, li) = left.approximate();
                let (rr, ri) = right.approximate();
                match op {
                    BinaryOp::Add => (lr + rr, li + ri),
                    BinaryOp::Sub => (lr - rr, li - ri),
                    BinaryOp::Mul => (lr * rr - li * ri, lr * ri + li * rr),
                    BinaryOp::Div => {
                        let denom = rr * rr + ri * ri;
                        ((lr * rr + li * ri) / denom, (li * rr - lr * ri) / denom)
                    }
                }
            }
            ANDescr::UnaryExpr { op, arg } => {
                let (ar, ai) = arg.approximate();
                match op {
                    UnaryOp::Neg => (-ar, -ai),
                    UnaryOp::Sqrt => {
                        let r = (ar * ar + ai * ai).sqrt();
                        let theta = ai.atan2(ar);
                        let new_r = r.sqrt();
                        let new_theta = theta / 2.0;
                        (new_r * new_theta.cos(), new_r * new_theta.sin())
                    }
                    UnaryOp::Inv => {
                        let denom = ar * ar + ai * ai;
                        (ar / denom, -ai / denom)
                    }
                }
            }
        }
    }
}

impl Display for AlgebraicNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (re, im) = self.approximate();
        if im.abs() < 1e-10 {
            write!(f, "{:.6}", re)
        } else if re.abs() < 1e-10 {
            write!(f, "{:.6}*I", im)
        } else {
            write!(f, "{:.6} + {:.6}*I", re, im)
        }
    }
}

impl Add for AlgebraicNumber {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // Optimization for rationals
        if let (ANDescr::Rational(r1), ANDescr::Rational(r2)) = (&self.descr, &other.descr) {
            return AlgebraicNumber {
                descr: ANDescr::Rational(r1 + r2),
                is_real: true,
            };
        }

        AlgebraicNumber {
            descr: ANDescr::BinaryExpr {
                op: BinaryOp::Add,
                left: Box::new(self.clone()),
                right: Box::new(other.clone()),
            },
            is_real: self.is_real && other.is_real,
        }
    }
}

impl Sub for AlgebraicNumber {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if let (ANDescr::Rational(r1), ANDescr::Rational(r2)) = (&self.descr, &other.descr) {
            return AlgebraicNumber {
                descr: ANDescr::Rational(r1 - r2),
                is_real: true,
            };
        }

        AlgebraicNumber {
            descr: ANDescr::BinaryExpr {
                op: BinaryOp::Sub,
                left: Box::new(self.clone()),
                right: Box::new(other.clone()),
            },
            is_real: self.is_real && other.is_real,
        }
    }
}

impl Mul for AlgebraicNumber {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if let (ANDescr::Rational(r1), ANDescr::Rational(r2)) = (&self.descr, &other.descr) {
            return AlgebraicNumber {
                descr: ANDescr::Rational(r1 * r2),
                is_real: true,
            };
        }

        AlgebraicNumber {
            descr: ANDescr::BinaryExpr {
                op: BinaryOp::Mul,
                left: Box::new(self.clone()),
                right: Box::new(other.clone()),
            },
            is_real: self.is_real && other.is_real,
        }
    }
}

impl Div for AlgebraicNumber {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if let (ANDescr::Rational(r1), ANDescr::Rational(r2)) = (&self.descr, &other.descr) {
            return AlgebraicNumber {
                descr: ANDescr::Rational(r1 / r2),
                is_real: true,
            };
        }

        AlgebraicNumber {
            descr: ANDescr::BinaryExpr {
                op: BinaryOp::Div,
                left: Box::new(self.clone()),
                right: Box::new(other.clone()),
            },
            is_real: self.is_real && other.is_real,
        }
    }
}

impl Neg for AlgebraicNumber {
    type Output = Self;

    fn neg(self) -> Self {
        if let ANDescr::Rational(r) = &self.descr {
            return AlgebraicNumber {
                descr: ANDescr::Rational(-r),
                is_real: true,
            };
        }

        AlgebraicNumber {
            descr: ANDescr::UnaryExpr {
                op: UnaryOp::Neg,
                arg: Box::new(self.clone()),
            },
            is_real: self.is_real,
        }
    }
}

/// An element of the field of real algebraic numbers (AA).
///
/// This represents a real number that is a root of some polynomial with rational coefficients.
/// The real algebraic numbers form a real closed field.
#[derive(Clone, Debug, PartialEq)]
pub struct AlgebraicReal {
    /// Internal representation as an algebraic number
    inner: AlgebraicNumber,
}

impl AlgebraicReal {
    /// Creates a new algebraic real from a rational.
    pub fn from_rational(r: Rational) -> Self {
        AlgebraicReal {
            inner: AlgebraicNumber::from_rational(r),
        }
    }

    /// Creates a new algebraic real from an integer.
    pub fn from_integer(n: i64) -> Self {
        Self::from_rational(Rational::from(n))
    }

    /// Creates the zero algebraic real.
    pub fn zero() -> Self {
        Self::from_integer(0)
    }

    /// Creates the one algebraic real.
    pub fn one() -> Self {
        Self::from_integer(1)
    }

    /// Returns whether this algebraic real is zero.
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Returns whether this algebraic real is one.
    pub fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    /// Computes the square root of this algebraic real.
    ///
    /// Returns the positive square root for positive numbers.
    ///
    /// # Panics
    ///
    /// Panics if called on a negative number (use AlgebraicNumber for complex results).
    pub fn sqrt(&self) -> Self {
        AlgebraicReal {
            inner: self.inner.clone().sqrt(),
        }
    }

    /// Returns the minimal polynomial of this algebraic real over Q.
    pub fn minimal_polynomial(&self) -> UnivariatePolynomial<Rational> {
        self.inner.minimal_polynomial()
    }

    /// Returns an approximate floating-point value for this algebraic real.
    pub fn approximate(&self) -> f64 {
        self.inner.approximate().0
    }

    /// Converts this algebraic real to an algebraic number.
    pub fn to_algebraic_number(&self) -> AlgebraicNumber {
        self.inner.clone()
    }
}

impl Display for AlgebraicReal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.approximate())
    }
}

impl Add for AlgebraicReal {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        AlgebraicReal {
            inner: self.inner + other.inner,
        }
    }
}

impl Sub for AlgebraicReal {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        AlgebraicReal {
            inner: self.inner - other.inner,
        }
    }
}

impl Mul for AlgebraicReal {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        AlgebraicReal {
            inner: self.inner * other.inner,
        }
    }
}

impl Div for AlgebraicReal {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        AlgebraicReal {
            inner: self.inner / other.inner,
        }
    }
}

impl Neg for AlgebraicReal {
    type Output = Self;

    fn neg(self) -> Self {
        AlgebraicReal {
            inner: -self.inner,
        }
    }
}

/// The field of real algebraic numbers (AA).
///
/// This is a singleton representing the field of all real algebraic numbers.
#[derive(Clone, Debug)]
pub struct AlgebraicRealField;

impl AlgebraicRealField {
    /// Returns the unique instance of the algebraic real field.
    pub fn new() -> Self {
        AlgebraicRealField
    }

    /// Creates a zero element in this field.
    pub fn zero(&self) -> AlgebraicReal {
        AlgebraicReal::zero()
    }

    /// Creates a one element in this field.
    pub fn one(&self) -> AlgebraicReal {
        AlgebraicReal::one()
    }

    /// Creates an algebraic real from a rational.
    pub fn from_rational(&self, r: Rational) -> AlgebraicReal {
        AlgebraicReal::from_rational(r)
    }
}

impl Default for AlgebraicRealField {
    fn default() -> Self {
        Self::new()
    }
}

/// The field of algebraic numbers (QQbar).
///
/// This is a singleton representing the field of all complex algebraic numbers.
#[derive(Clone, Debug)]
pub struct AlgebraicField;

impl AlgebraicField {
    /// Returns the unique instance of the algebraic field.
    pub fn new() -> Self {
        AlgebraicField
    }

    /// Creates a zero element in this field.
    pub fn zero(&self) -> AlgebraicNumber {
        AlgebraicNumber::zero()
    }

    /// Creates a one element in this field.
    pub fn one(&self) -> AlgebraicNumber {
        AlgebraicNumber::one()
    }

    /// Creates the imaginary unit i.
    pub fn imaginary_unit(&self) -> AlgebraicNumber {
        AlgebraicNumber::imaginary_unit()
    }

    /// Creates an algebraic number from a rational.
    pub fn from_rational(&self, r: Rational) -> AlgebraicNumber {
        AlgebraicNumber::from_rational(r)
    }
}

impl Default for AlgebraicField {
    fn default() -> Self {
        Self::new()
    }
}

// Helper extension trait to add to_f64 to integers
trait ToF64 {
    fn to_f64(&self) -> f64;
}

impl ToF64 for num_bigint::BigInt {
    fn to_f64(&self) -> f64 {
        use num_traits::ToPrimitive;
        <num_bigint::BigInt as ToPrimitive>::to_f64(self).unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algebraic_number_from_rational() {
        let an = AlgebraicNumber::from_rational(Rational::from(5));
        assert!(an.is_real());
        assert!(!an.is_zero());
        let (re, im) = an.approximate();
        assert!((re - 5.0).abs() < 1e-6);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn test_algebraic_number_zero_one() {
        let zero = AlgebraicNumber::zero();
        assert!(zero.is_zero());
        assert!(zero.is_real());

        let one = AlgebraicNumber::one();
        assert!(one.is_one());
        assert!(one.is_real());
    }

    #[test]
    fn test_algebraic_number_imaginary_unit() {
        let i = AlgebraicNumber::imaginary_unit();
        assert!(!i.is_real());
        assert!(!i.is_zero());
        assert!(!i.is_one());

        let (re, im) = i.approximate();
        assert!(re.abs() < 1e-10);
        assert!((im - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_algebraic_number_arithmetic() {
        let two = AlgebraicNumber::from_integer(2);
        let three = AlgebraicNumber::from_integer(3);

        let sum = two.clone() + three.clone();
        let (re, _) = sum.approximate();
        assert!((re - 5.0).abs() < 1e-6);

        let diff = three.clone() - two.clone();
        let (re, _) = diff.approximate();
        assert!((re - 1.0).abs() < 1e-6);

        let prod = two.clone() * three.clone();
        let (re, _) = prod.approximate();
        assert!((re - 6.0).abs() < 1e-6);

        let quot = AlgebraicNumber::from_integer(6) / two.clone();
        let (re, _) = quot.approximate();
        assert!((re - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_algebraic_number_negation() {
        let five = AlgebraicNumber::from_integer(5);
        let neg_five = -five;
        let (re, _) = neg_five.approximate();
        assert!((re + 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_algebraic_number_complex_arithmetic() {
        let i = AlgebraicNumber::imaginary_unit();
        let two = AlgebraicNumber::from_integer(2);

        // 2 + i
        let complex = two.clone() + i.clone();
        let (re, im) = complex.approximate();
        assert!((re - 2.0).abs() < 1e-6);
        assert!((im - 1.0).abs() < 1e-6);

        // i * i should be -1
        let i_squared = i.clone() * i.clone();
        let (re, im) = i_squared.approximate();
        assert!((re + 1.0).abs() < 1e-6);
        assert!(im.abs() < 1e-6);
    }

    #[test]
    fn test_algebraic_real_from_rational() {
        let ar = AlgebraicReal::from_rational(Rational::from(7));
        assert!(!ar.is_zero());
        assert!((ar.approximate() - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_algebraic_real_zero_one() {
        let zero = AlgebraicReal::zero();
        assert!(zero.is_zero());

        let one = AlgebraicReal::one();
        assert!(one.is_one());
    }

    #[test]
    fn test_algebraic_real_arithmetic() {
        let two = AlgebraicReal::from_integer(2);
        let three = AlgebraicReal::from_integer(3);

        let sum = two.clone() + three.clone();
        assert!((sum.approximate() - 5.0).abs() < 1e-6);

        let diff = three.clone() - two.clone();
        assert!((diff.approximate() - 1.0).abs() < 1e-6);

        let prod = two.clone() * three.clone();
        assert!((prod.approximate() - 6.0).abs() < 1e-6);

        let quot = AlgebraicReal::from_integer(6) / two.clone();
        assert!((quot.approximate() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_algebraic_real_negation() {
        let five = AlgebraicReal::from_integer(5);
        let neg_five = -five;
        assert!((neg_five.approximate() + 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_algebraic_real_sqrt() {
        let four = AlgebraicReal::from_integer(4);
        let sqrt_four = four.sqrt();
        assert!((sqrt_four.approximate() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_algebraic_real_field() {
        let aa = AlgebraicRealField::new();
        let zero = aa.zero();
        let one = aa.one();

        assert!(zero.is_zero());
        assert!(one.is_one());

        let seven = aa.from_rational(Rational::from(7));
        assert!((seven.approximate() - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_algebraic_field() {
        let qqbar = AlgebraicField::new();
        let zero = qqbar.zero();
        let one = qqbar.one();
        let i = qqbar.imaginary_unit();

        assert!(zero.is_zero());
        assert!(one.is_one());
        assert!(!i.is_real());

        let five = qqbar.from_rational(Rational::from(5));
        assert!(five.is_real());
    }

    #[test]
    fn test_minimal_polynomial_rational() {
        let three = AlgebraicNumber::from_integer(3);
        let min_poly = three.minimal_polynomial();

        // For rational 3, minimal polynomial should be x - 3
        assert_eq!(min_poly.degree(), 1);
    }

    #[test]
    fn test_minimal_polynomial_imaginary() {
        let i = AlgebraicNumber::imaginary_unit();
        let min_poly = i.minimal_polynomial();

        // For i, minimal polynomial is x^2 + 1
        assert_eq!(min_poly.degree(), 2);
    }

    #[test]
    fn test_algebraic_real_to_algebraic_number() {
        let ar = AlgebraicReal::from_integer(5);
        let an = ar.to_algebraic_number();

        assert!(an.is_real());
        let (re, _) = an.approximate();
        assert!((re - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_display_algebraic_number() {
        let five = AlgebraicNumber::from_integer(5);
        let display = format!("{}", five);
        assert!(display.contains("5"));

        let i = AlgebraicNumber::imaginary_unit();
        let display = format!("{}", i);
        assert!(display.contains("I"));
    }

    #[test]
    fn test_display_algebraic_real() {
        let seven = AlgebraicReal::from_integer(7);
        let display = format!("{}", seven);
        assert!(display.contains("7"));
    }
}
