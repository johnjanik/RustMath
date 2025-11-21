//! Modular arithmetic

use crate::Integer;
use rustmath_core::{MathError, NumericConversion, Result};
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

    /// Check if this element is a unit (has a multiplicative inverse)
    ///
    /// Returns true if gcd(value, modulus) = 1
    pub fn is_unit(&self) -> bool {
        let gcd = self.value.gcd(&self.modulus);
        gcd.is_one()
    }

    /// Compute the multiplicative order of this element
    ///
    /// Returns the smallest positive integer k such that self^k ≡ 1 (mod n).
    /// Returns None if the element is not a unit.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_integers::{Integer, ModularInteger};
    ///
    /// let a = ModularInteger::new(Integer::from(2), Integer::from(7)).unwrap();
    /// let order = a.multiplicative_order();
    /// assert_eq!(order, Some(3)); // 2^3 ≡ 1 (mod 7)
    /// ```
    pub fn multiplicative_order(&self) -> Option<usize> {
        if !self.is_unit() {
            return None;
        }

        let mut power = ModularInteger::new(Integer::one(), self.modulus.clone()).ok()?;
        let mut order = 0usize;

        // The order must divide φ(n), so we have an upper bound
        // For simplicity, we just iterate up to the modulus
        let max_order = self.modulus.to_usize().unwrap_or(10000);

        loop {
            order += 1;
            power = (power * self.clone()).ok()?;

            if power.value.is_one() {
                return Some(order);
            }

            if order >= max_order {
                // Safety check to prevent infinite loops
                return None;
            }
        }
    }
}

/// Find all primitive roots modulo n
///
/// A primitive root modulo n is an integer g such that every integer coprime to n
/// is congruent to a power of g modulo n. Equivalently, g has multiplicative order φ(n).
///
/// Primitive roots exist if and only if n is 1, 2, 4, p^k, or 2p^k for odd prime p.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::modular::primitive_roots;
///
/// let roots = primitive_roots(&Integer::from(7));
/// // 7 is prime, so it has primitive roots: [3, 5]
/// ```
pub fn primitive_roots(n: &Integer) -> Vec<Integer> {
    if *n <= Integer::one() {
        return vec![];
    }

    // Calculate φ(n)
    let phi_n = match n.euler_phi() {
        Ok(val) => val,
        Err(_) => return vec![],
    };

    let mut roots = Vec::new();

    // Try all values from 1 to n-1
    for candidate_val in 1..n.to_usize().unwrap_or(1000).min(1000) {
        let candidate = Integer::from(candidate_val as i64);

        if let Ok(mod_int) = ModularInteger::new(candidate.clone(), n.clone()) {
            if let Some(order) = mod_int.multiplicative_order() {
                let order_int = Integer::from(order as i64);
                if order_int == phi_n {
                    roots.push(candidate);
                }
            }
        }
    }

    roots
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

    #[test]
    fn test_is_unit() {
        // 3 is a unit mod 7 (gcd(3,7) = 1)
        let a = ModularInteger::new(Integer::from(3), Integer::from(7)).unwrap();
        assert!(a.is_unit());

        // 6 is not a unit mod 9 (gcd(6,9) = 3)
        let b = ModularInteger::new(Integer::from(6), Integer::from(9)).unwrap();
        assert!(!b.is_unit());

        // 0 is never a unit
        let c = ModularInteger::new(Integer::from(0), Integer::from(7)).unwrap();
        assert!(!c.is_unit());
    }

    #[test]
    fn test_multiplicative_order() {
        // Order of 2 mod 7
        let a = ModularInteger::new(Integer::from(2), Integer::from(7)).unwrap();
        assert_eq!(a.multiplicative_order(), Some(3)); // 2^3 = 8 ≡ 1 (mod 7)

        // Order of 3 mod 7 (primitive root)
        let b = ModularInteger::new(Integer::from(3), Integer::from(7)).unwrap();
        assert_eq!(b.multiplicative_order(), Some(6)); // φ(7) = 6

        // Non-unit has no order
        let c = ModularInteger::new(Integer::from(6), Integer::from(9)).unwrap();
        assert_eq!(c.multiplicative_order(), None);
    }

    #[test]
    fn test_primitive_roots() {
        // Primitive roots of 7 (prime)
        let roots = primitive_roots(&Integer::from(7));
        assert_eq!(roots.len(), 2); // φ(φ(7)) = φ(6) = 2
        assert!(roots.contains(&Integer::from(3)));
        assert!(roots.contains(&Integer::from(5)));

        // Primitive roots of 14 = 2 * 7
        let roots = primitive_roots(&Integer::from(14));
        assert!(!roots.is_empty());

        // 8 has no primitive roots (not 1, 2, 4, p^k, or 2p^k)
        let roots = primitive_roots(&Integer::from(8));
        assert!(roots.is_empty());
    }
}
