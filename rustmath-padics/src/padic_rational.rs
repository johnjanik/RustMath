//! p-adic rationals Qp

use crate::PadicInteger;
use rustmath_core::{CommutativeRing, Field, MathError, NumericConversion, Result, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// p-adic rational number
///
/// Represented as a*p^v where a is a p-adic integer (unit) and v is the valuation
#[derive(Clone, Debug)]
pub struct PadicRational {
    /// p-adic integer part (unit in Zp)
    unit: PadicInteger,
    /// Valuation (power of p)
    valuation: i32,
}

impl PadicRational {
    /// Create a p-adic rational from a rational number
    ///
    /// # Arguments
    ///
    /// * `rational` - Rational number to convert
    /// * `prime` - Prime p
    /// * `precision` - Precision for the p-adic integer part
    pub fn from_rational(rational: Rational, prime: Integer, precision: usize) -> Result<Self> {
        // Compute p-adic valuation of numerator and denominator
        let num_val = rational.numerator().valuation(&prime) as i32;
        let den_val = rational.denominator().valuation(&prime) as i32;
        let valuation = num_val - den_val;

        // Extract the unit part (remove all factors of p)
        let mut num = rational.numerator().clone();
        let mut den = rational.denominator().clone();

        // Remove factors of p from numerator
        for _ in 0..num_val {
            num = num / prime.clone();
        }

        // Remove factors of p from denominator
        for _ in 0..den_val {
            den = den / prime.clone();
        }

        // Now num and den are coprime to p
        // Compute num/den in Zp
        let num_padic = PadicInteger::from_integer(num, prime.clone(), precision)?;
        let den_padic = PadicInteger::from_integer(den, prime, precision)?;

        let unit = num_padic * den_padic.inverse()?;

        Ok(PadicRational { unit, valuation })
    }

    /// Create from a p-adic integer
    pub fn from_padic_integer(padic: PadicInteger) -> Self {
        PadicRational {
            unit: padic,
            valuation: 0,
        }
    }

    /// Get the unit part
    pub fn unit(&self) -> &PadicInteger {
        &self.unit
    }

    /// Get the valuation
    pub fn valuation(&self) -> i32 {
        self.valuation
    }

    /// Get the prime
    pub fn prime(&self) -> &Integer {
        self.unit.prime()
    }

    /// Get the precision
    pub fn precision(&self) -> usize {
        self.unit.precision()
    }

    /// Compute absolute value (p-adic norm): |x|_p = p^(-v) where v is the valuation
    pub fn abs(&self) -> f64 {
        let p = self.prime().to_f64().unwrap();
        p.powi(-self.valuation)
    }

    /// Lift to higher precision
    pub fn lift(&self, new_precision: usize) -> Result<Self> {
        Ok(PadicRational {
            unit: self.unit.lift(new_precision)?,
            valuation: self.valuation,
        })
    }
}

impl fmt::Display for PadicRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.valuation == 0 {
            write!(f, "{}", self.unit)
        } else if self.valuation > 0 {
            write!(f, "{}*{}^{}", self.unit, self.prime(), self.valuation)
        } else {
            write!(f, "{}/{}^{}", self.unit, self.prime(), -self.valuation)
        }
    }
}

impl PartialEq for PadicRational {
    fn eq(&self, other: &Self) -> bool {
        self.unit == other.unit && self.valuation == other.valuation
    }
}

impl Ring for PadicRational {
    fn zero() -> Self {
        panic!("Cannot create PadicRational::zero() without parameters");
    }

    fn one() -> Self {
        panic!("Cannot create PadicRational::one() without parameters");
    }

    fn is_zero(&self) -> bool {
        self.unit.is_zero()
    }

    fn is_one(&self) -> bool {
        self.unit.is_one() && self.valuation == 0
    }
}

impl CommutativeRing for PadicRational {}

impl Field for PadicRational {
    fn inverse(&self) -> Result<Self> {
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        Ok(PadicRational {
            unit: self.unit.inverse()?,
            valuation: -self.valuation,
        })
    }
}

impl Neg for PadicRational {
    type Output = Self;

    fn neg(self) -> Self::Output {
        PadicRational {
            unit: -self.unit,
            valuation: self.valuation,
        }
    }
}

impl Neg for &PadicRational {
    type Output = PadicRational;

    fn neg(self) -> Self::Output {
        PadicRational {
            unit: -self.unit.clone(),
            valuation: self.valuation,
        }
    }
}

impl Add for PadicRational {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        if self.valuation == other.valuation {
            // Same valuation: just add the units
            PadicRational {
                unit: self.unit + other.unit,
                valuation: self.valuation,
            }
        } else if self.valuation < other.valuation {
            // self has smaller valuation (larger p-adic norm)
            let diff = (other.valuation - self.valuation) as usize;
            let p = self.prime().clone();
            let mut shifted_other = other.unit;
            for _ in 0..diff {
                shifted_other = shifted_other * PadicInteger::from_integer(p.clone(), p.clone(), self.precision()).unwrap();
            }
            PadicRational {
                unit: self.unit + shifted_other,
                valuation: self.valuation,
            }
        } else {
            // other has smaller valuation
            let diff = (self.valuation - other.valuation) as usize;
            let p = self.prime().clone();
            let precision = self.precision();
            let mut shifted_self = self.unit;
            for _ in 0..diff {
                shifted_self = shifted_self * PadicInteger::from_integer(p.clone(), p.clone(), precision).unwrap();
            }
            PadicRational {
                unit: shifted_self + other.unit,
                valuation: other.valuation,
            }
        }
    }
}

impl Sub for PadicRational {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl Mul for PadicRational {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        PadicRational {
            unit: self.unit * other.unit,
            valuation: self.valuation + other.valuation,
        }
    }
}

impl Div for PadicRational {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        if other.is_zero() {
            panic!("Division by zero");
        }
        PadicRational {
            unit: self.unit * other.unit.inverse().unwrap(),
            valuation: self.valuation - other.valuation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_rational() {
        let p = Integer::from(5);
        let prec = 10;

        // 3/5 in Q_5
        let rat = Rational::new(Integer::from(3), Integer::from(5)).unwrap();
        let padic = PadicRational::from_rational(rat, p, prec).unwrap();

        // Valuation should be -1 (one factor of 5 in denominator)
        assert_eq!(padic.valuation(), -1);
    }

    #[test]
    fn test_padic_norm() {
        let p = Integer::from(5);
        let prec = 10;

        // 25 = 5^2 has 5-adic norm 1/25
        let rat = Rational::new(Integer::from(25), Integer::from(1)).unwrap();
        let padic = PadicRational::from_rational(rat, p, prec).unwrap();

        assert_eq!(padic.valuation(), 2);
        assert!((padic.abs() - 0.04).abs() < 1e-10);
    }
}
