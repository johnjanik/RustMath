//! p-adic integers Zp

use rustmath_core::{MathError, Result, Ring};
use rustmath_integers::Integer;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// p-adic integer with finite precision
///
/// Represents an element of Zp truncated to precision n.
/// Stored as an integer modulo p^n.
#[derive(Clone, Debug)]
pub struct PadicInteger {
    /// Value modulo p^precision
    value: Integer,
    /// Prime p
    prime: Integer,
    /// Precision (number of p-adic digits)
    precision: usize,
}

impl PadicInteger {
    /// Create a new p-adic integer from an integer
    ///
    /// # Arguments
    ///
    /// * `value` - Integer to convert to p-adic form
    /// * `prime` - Prime p
    /// * `precision` - Number of p-adic digits to keep
    pub fn from_integer(value: Integer, prime: Integer, precision: usize) -> Result<Self> {
        if prime <= Integer::one() {
            return Err(MathError::InvalidArgument(
                "Prime must be > 1".to_string(),
            ));
        }

        if precision == 0 {
            return Err(MathError::InvalidArgument(
                "Precision must be > 0".to_string(),
            ));
        }

        // Compute p^precision
        let modulus = prime.pow(precision as u32);

        // Reduce value modulo p^precision
        let reduced = value.clone() % modulus.clone();
        let value = if reduced.signum() < 0 {
            reduced + modulus
        } else {
            reduced
        };

        Ok(PadicInteger {
            value,
            prime,
            precision,
        })
    }

    /// Create zero
    pub fn zero(prime: Integer, precision: usize) -> Result<Self> {
        Self::from_integer(Integer::zero(), prime, precision)
    }

    /// Create one
    pub fn one(prime: Integer, precision: usize) -> Result<Self> {
        Self::from_integer(Integer::one(), prime, precision)
    }

    /// Get the value modulo p^precision
    pub fn value(&self) -> &Integer {
        &self.value
    }

    /// Get the prime p
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// Get the precision
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Get the p-adic valuation (largest k such that p^k divides the value)
    pub fn valuation(&self) -> u32 {
        if self.value.is_zero() {
            return u32::MAX; // Infinite valuation
        }

        self.value.valuation(&self.prime)
    }

    /// Get the residue (value modulo p)
    pub fn residue(&self) -> Integer {
        let residue = self.value.clone() % self.prime.clone();
        if residue.signum() < 0 {
            residue + self.prime.clone()
        } else {
            residue
        }
    }

    /// Lift precision to a higher value
    pub fn lift(&self, new_precision: usize) -> Result<Self> {
        if new_precision < self.precision {
            return Err(MathError::InvalidArgument(
                "New precision must be >= current precision".to_string(),
            ));
        }

        Ok(PadicInteger {
            value: self.value.clone(),
            prime: self.prime.clone(),
            precision: new_precision,
        })
    }

    /// Truncate to lower precision
    pub fn truncate(&self, new_precision: usize) -> Result<Self> {
        if new_precision > self.precision {
            return Err(MathError::InvalidArgument(
                "New precision must be <= current precision".to_string(),
            ));
        }

        if new_precision == 0 {
            return Err(MathError::InvalidArgument(
                "Precision must be > 0".to_string(),
            ));
        }

        let modulus = self.prime.pow(new_precision as u32);
        let reduced = self.value.clone() % modulus.clone();

        Ok(PadicInteger {
            value: reduced,
            prime: self.prime.clone(),
            precision: new_precision,
        })
    }

    /// Compute multiplicative inverse (if it exists)
    ///
    /// Inverse exists if and only if gcd(value, p) = 1
    pub fn inverse(&self) -> Result<Self> {
        let modulus = self.prime.pow(self.precision as u32);

        // Check if inverse exists
        let gcd = self.value.gcd(&self.prime);
        if !gcd.is_one() {
            return Err(MathError::NotInvertible);
        }

        // Use extended GCD to find inverse
        let (gcd_check, s, _) = self.value.extended_gcd(&modulus);
        if !gcd_check.is_one() {
            return Err(MathError::NotInvertible);
        }

        // s is the inverse, but might be negative
        let inv = if s.signum() < 0 {
            s + modulus.clone()
        } else {
            s % modulus.clone()
        };

        Ok(PadicInteger {
            value: inv,
            prime: self.prime.clone(),
            precision: self.precision,
        })
    }

    /// Hensel lifting for linear congruences: ax ≡ b (mod p^k)
    ///
    /// Lifts a solution from precision n to precision n+1
    /// This is a simplified version that works for the inverse operation
    pub fn hensel_lift_linear(&self, a: &Self, b: &Self) -> Result<Self> {
        // Newton's method: x_{n+1} = x_n - f(x_n)/f'(x_n)
        // For f(x) = ax - b, we have f'(x) = a
        // So x_{n+1} = x_n - (ax_n - b)/a = x_n(2 - ax_n/b)

        if self.precision >= 50 {
            // Already at high precision
            return Ok(self.clone());
        }

        let new_precision = self.precision + 1;

        // Compute correction term
        let ax = a.clone() * self.clone();
        let residual = b.clone() - ax;

        // Lift: x' = x + residual * a^(-1)
        let a_inv = a.inverse()?;
        let correction = residual * a_inv;
        let lifted = self.clone() + correction;

        // Convert to higher precision
        PadicInteger::from_integer(lifted.value, self.prime.clone(), new_precision)
    }
}

/// Hensel lifting for polynomial roots
///
/// Given a polynomial f and an approximate root r such that f(r) ≡ 0 (mod p^k),
/// lift r to a root modulo p^(k+1).
///
/// This uses Newton's method: r' = r - f(r)/f'(r) mod p^(k+1)
///
/// Note: This is a simplified version. Full implementation would work with
/// polynomial types from rustmath-polynomials.
pub fn hensel_lift_root(
    root: &PadicInteger,
    f_value: &PadicInteger,
    f_prime_value: &PadicInteger,
) -> Result<PadicInteger> {
    if f_prime_value.is_zero() {
        return Err(MathError::InvalidArgument(
            "Derivative is zero at root".to_string(),
        ));
    }

    // Newton step: r' = r - f(r)/f'(r)
    let f_prime_inv = f_prime_value.inverse()?;
    let correction = f_value.clone() * f_prime_inv;
    let new_root = root.clone() - correction;

    // Lift precision
    let new_precision = root.precision() + 1;
    PadicInteger::from_integer(new_root.value, root.prime().clone(), new_precision)
}

impl fmt::Display for PadicInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} + O({}^{})",
            self.value, self.prime, self.precision
        )
    }
}

impl PartialEq for PadicInteger {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.prime == other.prime
            && self.precision == other.precision
    }
}

impl Add for PadicInteger {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.prime, other.prime, "Primes must match");

        let precision = self.precision.min(other.precision);
        let sum = self.value + other.value;

        PadicInteger::from_integer(sum, self.prime, precision).unwrap()
    }
}

impl Sub for PadicInteger {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.prime, other.prime, "Primes must match");

        let precision = self.precision.min(other.precision);
        let diff = self.value - other.value;

        PadicInteger::from_integer(diff, self.prime, precision).unwrap()
    }
}

impl Mul for PadicInteger {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.prime, other.prime, "Primes must match");

        let precision = self.precision.min(other.precision);
        let prod = self.value * other.value;

        PadicInteger::from_integer(prod, self.prime, precision).unwrap()
    }
}

impl Neg for PadicInteger {
    type Output = Self;

    fn neg(self) -> Self {
        let modulus = self.prime.pow(self.precision as u32);
        let neg_val = modulus - self.value;

        PadicInteger::from_integer(neg_val, self.prime, self.precision).unwrap()
    }
}

impl Ring for PadicInteger {
    fn zero() -> Self {
        panic!("Cannot create PadicInteger::zero() without parameters");
    }

    fn one() -> Self {
        panic!("Cannot create PadicInteger::one() without parameters");
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value.is_one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let p = Integer::from(5);
        let prec = 3;

        let a = PadicInteger::from_integer(Integer::from(7), p.clone(), prec).unwrap();
        let b = PadicInteger::from_integer(Integer::from(11), p.clone(), prec).unwrap();

        // 7 + 11 = 18 in Z_5
        let sum = a.clone() + b.clone();
        assert_eq!(sum.value(), &Integer::from(18));

        // 7 * 11 = 77 in Z_5
        let prod = a.clone() * b.clone();
        assert_eq!(prod.value(), &Integer::from(77));
    }

    #[test]
    fn test_valuation() {
        let p = Integer::from(5);
        let prec = 10;

        // 25 = 5^2
        let a = PadicInteger::from_integer(Integer::from(25), p.clone(), prec).unwrap();
        assert_eq!(a.valuation(), 2);

        // 7 has valuation 0 (not divisible by 5)
        let b = PadicInteger::from_integer(Integer::from(7), p.clone(), prec).unwrap();
        assert_eq!(b.valuation(), 0);
    }

    #[test]
    fn test_inverse() {
        let p = Integer::from(5);
        let prec = 3;

        let a = PadicInteger::from_integer(Integer::from(2), p.clone(), prec).unwrap();
        let inv = a.inverse().unwrap();

        // Verify: a * a^(-1) ≡ 1 (mod 5^3)
        let prod = a * inv;
        assert!(prod.is_one());
    }

    #[test]
    fn test_truncate() {
        let p = Integer::from(5);
        let a = PadicInteger::from_integer(Integer::from(123), p.clone(), 5).unwrap();

        let truncated = a.truncate(2).unwrap();
        assert_eq!(truncated.precision(), 2);
    }
}
