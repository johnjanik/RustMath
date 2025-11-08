//! Prime finite fields GF(p)

use rustmath_core::{CommutativeRing, EuclideanDomain, Field, MathError, NumericConversion, Result, Ring};
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
    pub fn legendre_symbol(&self) -> Result<Integer> {
        let symbol = self.value.legendre_symbol(&self.modulus)?;
        Ok(Integer::from(symbol as i32))
    }

    /// Check if this element is a quadratic residue (perfect square in the field)
    pub fn is_quadratic_residue(&self) -> bool {
        if self.value.is_zero() {
            return true;
        }

        let leg = self.legendre_symbol().unwrap_or(Integer::zero());
        leg == Integer::one()
    }

    /// Compute discrete logarithm: given g^x = h, find x
    ///
    /// Uses the baby-step giant-step algorithm, which runs in O(√p) time and space.
    ///
    /// # Arguments
    ///
    /// * `base` - The base g (should be a generator for guaranteed solution)
    /// * `target` - The target value h
    ///
    /// # Returns
    ///
    /// The discrete logarithm x such that base^x = target, if it exists.
    ///
    /// # Algorithm
    ///
    /// Baby-step giant-step:
    /// 1. Compute m = ceil(sqrt(p-1))
    /// 2. Baby steps: Store g^j for j = 0, 1, ..., m-1
    /// 3. Giant steps: Compute h * g^(-im) for i = 0, 1, 2, ... until match found
    pub fn discrete_log(base: &PrimeField, target: &PrimeField) -> Result<Integer> {
        assert_eq!(base.modulus, target.modulus, "Moduli must match");

        // Handle special cases
        if base.value.is_zero() || base.value.is_one() {
            return Err(MathError::InvalidArgument(
                "Base must be neither 0 nor 1".to_string(),
            ));
        }

        if target.value.is_one() {
            return Ok(Integer::zero());
        }

        if target.value.is_zero() {
            return Err(MathError::InvalidArgument(
                "Logarithm of zero does not exist".to_string(),
            ));
        }

        // Order of the multiplicative group is p - 1
        let group_order = base.modulus.clone() - Integer::one();

        // Compute m = ceil(sqrt(p-1))
        let m = group_order.sqrt()? + Integer::one();
        let m_usize = m.to_usize().ok_or_else(|| {
            MathError::NumericalError("Group order too large for discrete log".to_string())
        })?;

        // Baby steps: compute and store g^j for j = 0, 1, ..., m-1
        use std::collections::HashMap;
        let mut baby_steps: HashMap<Integer, usize> = HashMap::new();

        let mut power = PrimeField::new(Integer::one(), base.modulus.clone())?;
        for j in 0..m_usize {
            baby_steps.insert(power.value.clone(), j);
            power = power * base.clone();
        }

        // Compute g^(-m) for giant steps
        let base_power_m = {
            let mut result = PrimeField::new(Integer::one(), base.modulus.clone())?;
            let mut b = base.clone();
            let mut exp = m.clone();

            // Fast exponentiation
            while exp > Integer::zero() {
                if exp.clone() % Integer::from(2) == Integer::one() {
                    result = result * b.clone();
                }
                b = b.clone() * b.clone();
                exp = exp / Integer::from(2);
            }
            result
        };

        let giant_step_multiplier = base_power_m.inverse()?;

        // Giant steps: compute h * g^(-im) for i = 0, 1, 2, ...
        let mut gamma = target.clone();
        for i in 0..m_usize {
            if let Some(&j) = baby_steps.get(&gamma.value) {
                // Found! h = g^(im + j)
                let result = Integer::from(i as i64) * m.clone() + Integer::from(j as i64);
                return Ok(result % group_order);
            }

            gamma = gamma * giant_step_multiplier.clone();
        }

        Err(MathError::InvalidArgument(
            "Discrete logarithm not found (target may not be in the subgroup generated by base)"
                .to_string(),
        ))
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

impl CommutativeRing for PrimeField {
    // Marker trait, no methods to implement
}

impl Field for PrimeField {
    fn inverse(&self) -> Result<Self> {
        if self.value.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        // Use extended GCD to find multiplicative inverse
        let (gcd, x, _) = self.value.extended_gcd(&self.modulus);
        if !gcd.is_one() {
            return Err(MathError::InvalidArgument("No inverse exists".to_string()));
        }

        // x is the inverse, but may be negative - normalize to [0, modulus)
        let inv = if x < Integer::zero() {
            x + self.modulus.clone()
        } else {
            x
        };

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

    #[test]
    fn test_discrete_log() {
        let p = Integer::from(11);
        let g = PrimeField::new(Integer::from(2), p.clone()).unwrap(); // 2 is a generator of GF(11)*

        // Test: 2^7 = 128 ≡ 7 (mod 11)
        let target = PrimeField::new(Integer::from(7), p.clone()).unwrap();
        let log = PrimeField::discrete_log(&g, &target).unwrap();

        // Verify: g^log = target
        let mut verification = PrimeField::new(Integer::one(), p.clone()).unwrap();
        let mut temp_g = g.clone();
        let mut exp = log.clone();

        while exp > Integer::zero() {
            if exp.clone() % Integer::from(2) == Integer::one() {
                verification = verification * temp_g.clone();
            }
            temp_g = temp_g.clone() * temp_g.clone();
            exp = exp / Integer::from(2);
        }

        assert_eq!(verification.value(), target.value());
    }

    #[test]
    fn test_discrete_log_small() {
        let p = Integer::from(7);
        let g = PrimeField::new(Integer::from(3), p.clone()).unwrap(); // 3 is a generator of GF(7)*

        // 3^0 = 1, 3^1 = 3, 3^2 = 2, 3^3 = 6, 3^4 = 4, 3^5 = 5, 3^6 = 1 (mod 7)
        // Test: log_3(2) = 2
        let target = PrimeField::new(Integer::from(2), p.clone()).unwrap();
        let log = PrimeField::discrete_log(&g, &target).unwrap();
        assert_eq!(log, Integer::from(2));

        // Test: log_3(6) = 3
        let target = PrimeField::new(Integer::from(6), p.clone()).unwrap();
        let log = PrimeField::discrete_log(&g, &target).unwrap();
        assert_eq!(log, Integer::from(3));
    }
}
