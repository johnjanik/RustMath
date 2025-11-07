//! Prime finite fields GF(p)

use rustmath_core::{EuclideanDomain, Field, MathError, Result, Ring};
use rustmath_integers::Integer;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Element of a prime finite field GF(p)
///
/// Represents integers modulo a prime p
#[derive(Clone, Debug)]
pub struct PrimeField {
    value: Integer,
    modulus: Integer,
}

impl PrimeField {
    /// Create a new element in GF(p)
    ///
    /// Requires p to be prime (not checked here for performance)
    pub fn new(value: Integer, modulus: Integer) -> Result<Self> {
        if modulus <= Integer::one() {
            return Err(MathError::InvalidArgument(
                "Modulus must be > 1".to_string(),
            ));
        }

        // Reduce value modulo p
        let (_, reduced) = value.div_rem(&modulus)?;
        let value = if reduced.signum() < 0 {
            reduced + modulus.clone()
        } else {
            reduced
        };

        Ok(PrimeField { value, modulus })
    }

    /// Get the value
    pub fn value(&self) -> &Integer {
        &self.value
    }

    /// Get the modulus (characteristic of the field)
    pub fn modulus(&self) -> &Integer {
        &self.modulus
    }

    /// Compute the multiplicative order of this element
    ///
    /// Returns the smallest k > 0 such that self^k = 1
    pub fn multiplicative_order(&self) -> Option<Integer> {
        if self.value.is_zero() {
            return None;
        }

        let mut power = self.clone();
        let mut k = Integer::one();

        let one = PrimeField::new(Integer::one(), self.modulus.clone()).unwrap();

        while power != one {
            power = power * self.clone();
            k = k + Integer::one();

            // Safety check to prevent infinite loops
            if k > self.modulus.clone() {
                return None;
            }
        }

        Some(k)
    }

    /// Check if this is a generator (primitive element) of the multiplicative group
    ///
    /// An element g is a generator if its order equals p-1
    pub fn is_generator(&self) -> bool {
        if let Some(order) = self.multiplicative_order() {
            order == self.modulus.clone() - Integer::one()
        } else {
            false
        }
    }

    /// Compute the Legendre symbol (a/p)
    ///
    /// Returns 0 if a ≡ 0 (mod p), 1 if a is a quadratic residue, -1 otherwise
    pub fn legendre_symbol(&self) -> Integer {
        self.value.legendre_symbol(&self.modulus)
    }

    /// Check if this element is a quadratic residue (perfect square in the field)
    pub fn is_quadratic_residue(&self) -> bool {
        if self.value.is_zero() {
            return true;
        }

        let leg = self.legendre_symbol();
        leg == Integer::one()
    }
}

impl fmt::Display for PrimeField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.value, self.modulus)
    }
}

impl PartialEq for PrimeField {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.modulus == other.modulus
    }
}

impl Add for PrimeField {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.modulus, other.modulus, "Moduli must match");

        let sum = self.value + other.value;
        PrimeField::new(sum, self.modulus).unwrap()
    }
}

impl Sub for PrimeField {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.modulus, other.modulus, "Moduli must match");

        let diff = self.value - other.value;
        PrimeField::new(diff, self.modulus).unwrap()
    }
}

impl Mul for PrimeField {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.modulus, other.modulus, "Moduli must match");

        let prod = self.value * other.value;
        PrimeField::new(prod, self.modulus).unwrap()
    }
}

impl Div for PrimeField {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        assert_eq!(self.modulus, other.modulus, "Moduli must match");

        let inv = other.inverse().unwrap();
        self * inv
    }
}

impl Neg for PrimeField {
    type Output = Self;

    fn neg(self) -> Self {
        let neg_val = self.modulus.clone() - self.value;
        PrimeField::new(neg_val, self.modulus).unwrap()
    }
}

impl Ring for PrimeField {
    fn zero() -> Self {
        // Can't create without modulus, this is a limitation
        // In practice, elements should be created with new()
        panic!("Cannot create PrimeField::zero() without modulus");
    }

    fn one() -> Self {
        panic!("Cannot create PrimeField::one() without modulus");
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value.is_one()
    }
}

impl Field for PrimeField {
    fn inverse(&self) -> Result<Self> {
        if self.value.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        // Use extended GCD to find multiplicative inverse
        let inv = self.value.mod_inverse(&self.modulus)?;

        Ok(PrimeField {
            value: inv,
            modulus: self.modulus.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let p = Integer::from(7);

        let a = PrimeField::new(Integer::from(3), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(5), p.clone()).unwrap();

        // Addition: 3 + 5 = 8 ≡ 1 (mod 7)
        let sum = a.clone() + b.clone();
        assert_eq!(sum.value(), &Integer::from(1));

        // Multiplication: 3 * 5 = 15 ≡ 1 (mod 7)
        let prod = a.clone() * b.clone();
        assert_eq!(prod.value(), &Integer::from(1));

        // Subtraction: 3 - 5 = -2 ≡ 5 (mod 7)
        let diff = a.clone() - b.clone();
        assert_eq!(diff.value(), &Integer::from(5));
    }

    #[test]
    fn test_inverse() {
        let p = Integer::from(7);
        let a = PrimeField::new(Integer::from(3), p.clone()).unwrap();

        let inv = a.inverse().unwrap();
        // 3 * 5 = 15 ≡ 1 (mod 7), so inverse of 3 is 5
        assert_eq!(inv.value(), &Integer::from(5));

        // Verify: a * a^(-1) = 1
        let prod = a * inv;
        assert!(prod.is_one());
    }

    #[test]
    fn test_division() {
        let p = Integer::from(7);
        let a = PrimeField::new(Integer::from(6), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(2), p.clone()).unwrap();

        // 6 / 2 = 3 (mod 7)
        let quot = a / b;
        assert_eq!(quot.value(), &Integer::from(3));
    }

    #[test]
    fn test_multiplicative_order() {
        let p = Integer::from(7);

        // Order of 2 in GF(7)
        let a = PrimeField::new(Integer::from(2), p.clone()).unwrap();
        let order = a.multiplicative_order().unwrap();

        // 2^1 = 2, 2^2 = 4, 2^3 = 1 (mod 7)
        assert_eq!(order, Integer::from(3));
    }

    #[test]
    fn test_generator() {
        let p = Integer::from(7);

        // 3 is a generator of GF(7)*
        let g = PrimeField::new(Integer::from(3), p.clone()).unwrap();
        assert!(g.is_generator());

        // 2 is not a generator
        let not_g = PrimeField::new(Integer::from(2), p.clone()).unwrap();
        assert!(!not_g.is_generator());
    }
}
