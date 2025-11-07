//! Rational numbers (fractions)

use rustmath_core::{CommutativeRing, EuclideanDomain, Field, IntegralDomain, MathError, NumericConversion, Result, Ring};
use rustmath_integers::Integer;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Rational number (fraction)
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Rational {
    numerator: Integer,
    denominator: Integer,
}

impl Rational {
    /// Create a new rational number from integers, automatically simplifying
    pub fn new<T: Into<Integer>>(numerator: T, denominator: T) -> Result<Self> {
        let num = numerator.into();
        let den = denominator.into();

        if den.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        let mut rational = Rational {
            numerator: num,
            denominator: den,
        };

        rational.simplify();
        Ok(rational)
    }

    /// Create a rational from an integer
    pub fn from_integer<T: Into<Integer>>(n: T) -> Self {
        Rational {
            numerator: n.into(),
            denominator: Integer::one(),
        }
    }

    /// Get the numerator
    pub fn numerator(&self) -> &Integer {
        &self.numerator
    }

    /// Get the denominator
    pub fn denominator(&self) -> &Integer {
        &self.denominator
    }

    /// Simplify to lowest terms
    fn simplify(&mut self) {
        let gcd = self.numerator.gcd(&self.denominator);

        if !gcd.is_one() {
            self.numerator = self.numerator.clone() / gcd.clone();
            self.denominator = self.denominator.clone() / gcd;
        }

        // Ensure denominator is positive
        if self.denominator.signum() < 0 {
            self.numerator = -self.numerator.clone();
            self.denominator = -self.denominator.clone();
        }
    }

    /// Get the absolute value
    pub fn abs(&self) -> Self {
        Rational {
            numerator: self.numerator.abs(),
            denominator: self.denominator.clone(),
        }
    }

    /// Get the reciprocal
    pub fn reciprocal(&self) -> Result<Self> {
        if self.numerator.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        Ok(Rational {
            numerator: self.denominator.clone(),
            denominator: self.numerator.clone(),
        })
    }

    /// Compute the floor (largest integer <= self)
    pub fn floor(&self) -> Integer {
        if self.numerator.signum() >= 0 {
            self.numerator.clone() / self.denominator.clone()
        } else {
            // For negative numbers, need to round down
            let (q, r) = self.numerator.div_rem(&self.denominator).unwrap();
            if r.is_zero() {
                q
            } else {
                q - Integer::one()
            }
        }
    }

    /// Compute the ceiling (smallest integer >= self)
    pub fn ceil(&self) -> Integer {
        -(-self.clone()).floor()
    }

    /// Convert to float (may lose precision)
    pub fn to_f64(&self) -> Option<f64> {
        let num = self.numerator.to_f64()?;
        let den = self.denominator.to_f64()?;
        Some(num / den)
    }

    /// Check if this is an integer
    pub fn is_integer(&self) -> bool {
        self.denominator.is_one()
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        let left = self.numerator.clone() * other.denominator.clone();
        let right = other.numerator.clone() * self.denominator.clone();
        left.cmp(&right)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator.is_one() {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rational({}/{})", self.numerator, self.denominator)
    }
}

// Arithmetic operations
impl Add for Rational {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let num = self.numerator.clone() * other.denominator.clone()
            + other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        Rational::new(num, den).unwrap()
    }
}

impl<'b> Add<&'b Rational> for &Rational {
    type Output = Rational;

    fn add(self, other: &'b Rational) -> Rational {
        let num = &self.numerator * &other.denominator + &other.numerator * &self.denominator;
        let den = &self.denominator * &other.denominator;

        Rational::new(num, den).unwrap()
    }
}

impl Sub for Rational {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let num = self.numerator.clone() * other.denominator.clone()
            - other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        Rational::new(num, den).unwrap()
    }
}

impl<'b> Sub<&'b Rational> for &Rational {
    type Output = Rational;

    fn sub(self, other: &'b Rational) -> Rational {
        let num = &self.numerator * &other.denominator - &other.numerator * &self.denominator;
        let den = &self.denominator * &other.denominator;

        Rational::new(num, den).unwrap()
    }
}

impl Mul for Rational {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let num = self.numerator.clone() * other.numerator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        Rational::new(num, den).unwrap()
    }
}

impl<'b> Mul<&'b Rational> for &Rational {
    type Output = Rational;

    fn mul(self, other: &'b Rational) -> Rational {
        let num = &self.numerator * &other.numerator;
        let den = &self.denominator * &other.denominator;

        Rational::new(num, den).unwrap()
    }
}

impl Div for Rational {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let num = self.numerator.clone() * other.denominator.clone();
        let den = self.denominator.clone() * other.numerator.clone();

        Rational::new(num, den).unwrap()
    }
}

impl<'b> Div<&'b Rational> for &Rational {
    type Output = Rational;

    fn div(self, other: &'b Rational) -> Rational {
        let num = &self.numerator * &other.denominator;
        let den = &self.denominator * &other.numerator;

        Rational::new(num, den).unwrap()
    }
}

impl Neg for Rational {
    type Output = Self;

    fn neg(self) -> Self {
        Rational {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}

impl Neg for &Rational {
    type Output = Rational;

    fn neg(self) -> Rational {
        Rational {
            numerator: -&self.numerator,
            denominator: self.denominator.clone(),
        }
    }
}

// Ring trait implementation
impl Ring for Rational {
    fn zero() -> Self {
        Rational {
            numerator: Integer::zero(),
            denominator: Integer::one(),
        }
    }

    fn one() -> Self {
        Rational {
            numerator: Integer::one(),
            denominator: Integer::one(),
        }
    }

    fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    fn is_one(&self) -> bool {
        self.numerator.is_one() && self.denominator.is_one()
    }
}

impl CommutativeRing for Rational {}
impl IntegralDomain for Rational {}

impl Field for Rational {
    fn inverse(&self) -> Result<Self> {
        self.reciprocal()
    }

    fn divide(&self, other: &Self) -> Result<Self> {
        if other.is_zero() {
            Err(MathError::DivisionByZero)
        } else {
            Ok(self.clone() / other.clone())
        }
    }
}

impl NumericConversion for Rational {
    fn from_i64(n: i64) -> Self {
        Rational::from_integer(n)
    }

    fn from_u64(n: u64) -> Self {
        Rational::from_integer(n)
    }

    fn to_i64(&self) -> Option<i64> {
        if self.is_integer() {
            self.numerator.to_i64()
        } else {
            None
        }
    }

    fn to_u64(&self) -> Option<u64> {
        if self.is_integer() {
            self.numerator.to_u64()
        } else {
            None
        }
    }

    fn to_f64(&self) -> Option<f64> {
        self.to_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation_and_simplification() {
        let r = Rational::new(4, 6).unwrap();
        assert_eq!(r.numerator(), &Integer::from(2));
        assert_eq!(r.denominator(), &Integer::from(3));
    }

    #[test]
    fn test_arithmetic() {
        let a = Rational::new(1, 2).unwrap();
        let b = Rational::new(1, 3).unwrap();

        let sum = a.clone() + b.clone();
        assert_eq!(sum, Rational::new(5, 6).unwrap());

        let diff = a.clone() - b.clone();
        assert_eq!(diff, Rational::new(1, 6).unwrap());

        let prod = a.clone() * b.clone();
        assert_eq!(prod, Rational::new(1, 6).unwrap());

        let quot = a.clone() / b.clone();
        assert_eq!(quot, Rational::new(3, 2).unwrap());
    }

    #[test]
    fn test_comparison() {
        let a = Rational::new(1, 2).unwrap();
        let b = Rational::new(2, 3).unwrap();

        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn test_floor_ceil() {
        let r = Rational::new(7, 3).unwrap();
        assert_eq!(r.floor(), Integer::from(2));
        assert_eq!(r.ceil(), Integer::from(3));

        let r = Rational::new(-7, 3).unwrap();
        assert_eq!(r.floor(), Integer::from(-3));
        assert_eq!(r.ceil(), Integer::from(-2));
    }
}
