//! Modular arithmetic

use crate::Integer;
use rustmath_core::{MathError, Result};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// Integer modulo n
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ModularInteger {
    value: Integer,
    modulus: Integer,
}

impl ModularInteger {
    /// Create a new modular integer
    pub fn new(value: Integer, modulus: Integer) -> Result<Self> {
        if modulus.is_zero() || modulus == Integer::from(1) {
            return Err(MathError::InvalidArgument(
                "Modulus must be > 1".to_string(),
            ));
        }

        let normalized = value % modulus.clone();
        let normalized = if normalized.signum() < 0 {
            normalized + modulus.clone()
        } else {
            normalized
        };

        Ok(ModularInteger {
            value: normalized,
            modulus,
        })
    }

    /// Get the value
    pub fn value(&self) -> &Integer {
        &self.value
    }

    /// Get the modulus
    pub fn modulus(&self) -> &Integer {
        &self.modulus
    }

    /// Compute the multiplicative inverse using extended Euclidean algorithm
    pub fn inverse(&self) -> Result<Self> {
        if self.value.is_zero() {
            return Err(MathError::NotInvertible);
        }

        let (gcd, s, _) = self.value.extended_gcd(&self.modulus);

        if !gcd.is_one() {
            return Err(MathError::NotInvertible);
        }

        ModularInteger::new(s, self.modulus.clone())
    }

    /// Modular exponentiation
    pub fn pow(&self, exp: &Integer) -> Result<Self> {
        let result = self.value.mod_pow(exp, &self.modulus)?;
        Ok(ModularInteger {
            value: result,
            modulus: self.modulus.clone(),
        })
    }
}

impl fmt::Display for ModularInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.value, self.modulus)
    }
}

impl fmt::Debug for ModularInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModularInteger({} mod {})", self.value, self.modulus)
    }
}

impl Add for ModularInteger {
    type Output = Result<Self>;

    fn add(self, other: Self) -> Result<Self> {
        if self.modulus != other.modulus {
            return Err(MathError::InvalidArgument(
                "Cannot add modular integers with different moduli".to_string(),
            ));
        }

        let sum = (self.value + other.value) % self.modulus.clone();
        Ok(ModularInteger {
            value: sum,
            modulus: self.modulus,
        })
    }
}

impl Sub for ModularInteger {
    type Output = Result<Self>;

    fn sub(self, other: Self) -> Result<Self> {
        if self.modulus != other.modulus {
            return Err(MathError::InvalidArgument(
                "Cannot subtract modular integers with different moduli".to_string(),
            ));
        }

        let mut diff = (self.value - other.value) % self.modulus.clone();
        if diff.signum() < 0 {
            diff = diff + self.modulus.clone();
        }

        Ok(ModularInteger {
            value: diff,
            modulus: self.modulus,
        })
    }
}

impl Mul for ModularInteger {
    type Output = Result<Self>;

    fn mul(self, other: Self) -> Result<Self> {
        if self.modulus != other.modulus {
            return Err(MathError::InvalidArgument(
                "Cannot multiply modular integers with different moduli".to_string(),
            ));
        }

        let product = (self.value * other.value) % self.modulus.clone();
        Ok(ModularInteger {
            value: product,
            modulus: self.modulus,
        })
    }
}

impl Neg for ModularInteger {
    type Output = Self;

    fn neg(self) -> Self {
        let value = if self.value.is_zero() {
            Integer::zero()
        } else {
            self.modulus.clone() - self.value
        };

        ModularInteger {
            value,
            modulus: self.modulus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_arithmetic() {
        let a = ModularInteger::new(Integer::from(10), Integer::from(7)).unwrap();
        let b = ModularInteger::new(Integer::from(5), Integer::from(7)).unwrap();

        let sum = (a.clone() + b.clone()).unwrap();
        assert_eq!(sum.value(), &Integer::from(1));

        let prod = (a.clone() * b.clone()).unwrap();
        assert_eq!(prod.value(), &Integer::from(1));
    }

    #[test]
    fn test_inverse() {
        let a = ModularInteger::new(Integer::from(3), Integer::from(11)).unwrap();
        let a_inv = a.inverse().unwrap();

        let prod = (a * a_inv).unwrap();
        assert_eq!(prod.value(), &Integer::from(1));
    }

    #[test]
    fn test_pow() {
        let base = ModularInteger::new(Integer::from(2), Integer::from(13)).unwrap();
        let result = base.pow(&Integer::from(10)).unwrap();
        assert_eq!(result.value(), &Integer::from(10));
    }
}
