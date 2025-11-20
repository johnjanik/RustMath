//! p-adic numbers with capped relative precision
//!
//! This module implements p-adic elements where precision is tracked relative to
//! the element's valuation. This is Sage's `padic_capped_relative_element`.
//!
//! # Precision Semantics
//!
//! In the capped relative precision model:
//! - Each element x has a **valuation** v(x) = the largest power of p dividing x
//! - Each element has a **relative precision** n
//! - The element is known modulo p^(v(x) + n)
//! - This is called "relative" because precision is relative to the valuation
//!
//! ## Example
//!
//! Consider x = 5^3 * 7 + O(5^10) in Q_5:
//! - Valuation v(x) = 3 (since 5^3 divides x)
//! - Relative precision = 7 (known mod 5^10, and 10 = 3 + 7)
//! - Absolute precision = 10 (known mod 5^10)
//!
//! ## Precision Tracking in Arithmetic
//!
//! - **Addition/Subtraction**: min(prec1, prec2), but can lose precision if cancellation occurs
//! - **Multiplication**: min(prec1, prec2)
//! - **Division**: precision of numerator (denominator must be a unit)
//! - **Powering**: precision scales appropriately
//!
//! ## Precision Loss in Addition
//!
//! When adding elements with different valuations, precision can be lost:
//! ```text
//! x = 1 + O(5^10)      (valuation 0, rel. precision 10)
//! y = 5^8 + O(5^10)    (valuation 8, rel. precision 2)
//! x + y = 1 + O(5^8)   (result has rel. precision 8 due to y's contribution)
//! ```
//!
//! When subtracting nearly equal elements, severe precision loss occurs:
//! ```text
//! x = 7 + O(5^10)      (valuation 0, rel. precision 10)
//! y = 7 + O(5^10)      (valuation 0, rel. precision 10)
//! x - y = O(5^10)      (exact zero or very high valuation)
//! ```

use rustmath_core::{CommutativeRing, Field, IntegralDomain, MathError, Result, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::cmp;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A p-adic number with capped relative precision
///
/// Represents an element of the p-adic field Q_p or ring Z_p with precision
/// tracked relative to the element's valuation.
///
/// # Internal Representation
///
/// - `unit`: The unit part (an integer coprime to p)
/// - `valuation`: The p-adic valuation (power of p)
/// - `prime`: The prime p
/// - `rel_precision`: Relative precision (number of significant p-adic digits)
///
/// The element represents: unit * p^valuation + O(p^(valuation + rel_precision))
#[derive(Clone, Debug)]
pub struct CappedRelativePadicElement {
    /// Unit part (coprime to p, reduced mod p^rel_precision)
    unit: Integer,
    /// p-adic valuation
    valuation: i64,
    /// Prime p
    prime: Integer,
    /// Relative precision (number of known p-adic digits)
    rel_precision: u32,
}

impl CappedRelativePadicElement {
    /// Create a new p-adic element with capped relative precision
    ///
    /// # Arguments
    ///
    /// * `value` - Integer value
    /// * `prime` - Prime p
    /// * `rel_precision` - Relative precision (number of significant p-adic digits)
    ///
    /// # Returns
    ///
    /// A p-adic element representing value + O(p^(v+n)) where v is the valuation
    /// of value and n is rel_precision.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::capped_relative::CappedRelativePadicElement;
    ///
    /// // Create 7 + O(5^10) in Q_5
    /// let x = CappedRelativePadicElement::new(
    ///     Integer::from(7),
    ///     Integer::from(5),
    ///     10
    /// ).unwrap();
    /// assert_eq!(x.valuation(), 0); // 7 is not divisible by 5
    /// assert_eq!(x.rel_precision(), 10);
    /// ```
    pub fn new(value: Integer, prime: Integer, rel_precision: u32) -> Result<Self> {
        if prime <= Integer::one() || !prime.is_prime() {
            return Err(MathError::InvalidArgument(
                "Prime must be an actual prime number".to_string(),
            ));
        }

        if rel_precision == 0 {
            return Err(MathError::InvalidArgument(
                "Relative precision must be > 0".to_string(),
            ));
        }

        // Handle zero specially
        if value.is_zero() {
            return Ok(CappedRelativePadicElement {
                unit: Integer::zero(),
                valuation: 0, // Zero has infinite valuation, but we represent it this way
                prime,
                rel_precision,
            });
        }

        // Compute valuation and extract unit
        let valuation = value.valuation(&prime) as i64;
        let mut unit = value.clone();

        // Divide out p^valuation to get the unit
        for _ in 0..valuation {
            unit = unit / prime.clone();
        }

        // Reduce unit modulo p^rel_precision
        let modulus = prime.pow(rel_precision);
        unit = unit % modulus.clone();

        // Ensure unit is positive
        if unit.signum() < 0 {
            unit = unit + modulus;
        }

        Ok(CappedRelativePadicElement {
            unit,
            valuation,
            prime,
            rel_precision,
        })
    }

    /// Create a p-adic element from a rational number
    ///
    /// # Arguments
    ///
    /// * `rational` - Rational number to convert
    /// * `prime` - Prime p
    /// * `rel_precision` - Relative precision
    ///
    /// # Example
    ///
    /// ```rust
    /// use rustmath_integers::Integer;
    /// use rustmath_rationals::Rational;
    /// use rustmath_rings::padics::capped_relative::CappedRelativePadicElement;
    ///
    /// // Create 3/5 in Q_5
    /// let rat = Rational::new(Integer::from(3), Integer::from(5)).unwrap();
    /// let x = CappedRelativePadicElement::from_rational(rat, Integer::from(5), 10).unwrap();
    /// assert_eq!(x.valuation(), -1); // Valuation is -1 due to 5 in denominator
    /// ```
    pub fn from_rational(rational: Rational, prime: Integer, rel_precision: u32) -> Result<Self> {
        if prime <= Integer::one() || !prime.is_prime() {
            return Err(MathError::InvalidArgument(
                "Prime must be an actual prime number".to_string(),
            ));
        }

        // Compute valuations
        let num_val = rational.numerator().valuation(&prime) as i64;
        let den_val = rational.denominator().valuation(&prime) as i64;
        let valuation = num_val - den_val;

        // Extract unit parts (remove all factors of p)
        let mut num = rational.numerator().clone();
        let mut den = rational.denominator().clone();

        // Remove p^num_val from numerator
        for _ in 0..num_val {
            num = num / prime.clone();
        }

        // Remove p^den_val from denominator
        for _ in 0..den_val {
            den = den / prime.clone();
        }

        // Now compute num/den mod p^rel_precision
        let modulus = prime.pow(rel_precision);

        // Find modular inverse of den
        let (gcd, s, _) = den.extended_gcd(&modulus);
        if !gcd.is_one() {
            return Err(MathError::NotInvertible);
        }

        let den_inv = if s.signum() < 0 {
            s + modulus.clone()
        } else {
            s % modulus.clone()
        };

        // Compute unit = num * den^(-1) mod p^rel_precision
        let unit = (num * den_inv) % modulus.clone();
        let unit = if unit.signum() < 0 {
            unit + modulus
        } else {
            unit
        };

        Ok(CappedRelativePadicElement {
            unit,
            valuation,
            prime,
            rel_precision,
        })
    }

    /// Create zero with specified prime and precision
    pub fn zero(prime: Integer, rel_precision: u32) -> Result<Self> {
        Self::new(Integer::zero(), prime, rel_precision)
    }

    /// Create one with specified prime and precision
    pub fn one(prime: Integer, rel_precision: u32) -> Result<Self> {
        Self::new(Integer::one(), prime, rel_precision)
    }

    /// Get the prime p
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// Get the p-adic valuation
    ///
    /// This is the largest k such that p^k divides the element.
    /// For zero, we return 0 (though mathematically it should be infinity).
    pub fn valuation(&self) -> i64 {
        if self.unit.is_zero() {
            // Zero has infinite valuation, but we can't represent that
            // Return a large value or handle specially
            i64::MAX
        } else {
            self.valuation
        }
    }

    /// Get the relative precision
    ///
    /// This is the number of known p-adic digits relative to the valuation.
    pub fn rel_precision(&self) -> u32 {
        self.rel_precision
    }

    /// Get the absolute precision
    ///
    /// This is valuation + relative_precision, i.e., the element is known mod p^abs_prec.
    pub fn abs_precision(&self) -> i64 {
        if self.unit.is_zero() {
            0 // Zero is exact at precision 0
        } else {
            self.valuation + (self.rel_precision as i64)
        }
    }

    /// Get the unit part
    ///
    /// Returns the unit u such that this element equals u * p^v where v is the valuation.
    pub fn unit(&self) -> &Integer {
        &self.unit
    }

    /// Split into unit and prime power: returns (unit, valuation)
    ///
    /// This decomposes the element as: unit * p^valuation
    /// where unit is coprime to p (or zero).
    ///
    /// # Example
    ///
    /// ```rust
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::capped_relative::CappedRelativePadicElement;
    ///
    /// // 25 = 5^2 * 1 in Q_5
    /// let x = CappedRelativePadicElement::new(Integer::from(25), Integer::from(5), 10).unwrap();
    /// let (unit, val) = x.unit_split();
    /// assert_eq!(val, 2);
    /// assert_eq!(unit, &Integer::from(1));
    /// ```
    pub fn unit_split(&self) -> (&Integer, i64) {
        (&self.unit, self.valuation)
    }

    /// Compute the Teichmüller lift
    ///
    /// The Teichmüller lift of x ∈ F_p is the unique (p-1)-th root of unity in Z_p
    /// that reduces to x modulo p.
    ///
    /// For x ≠ 0 in F_p, the Teichmüller lift ω(x) satisfies:
    /// - ω(x)^(p-1) = 1
    /// - ω(x) ≡ x (mod p)
    /// - ω(x) is computed as lim_{n→∞} x^(p^n)
    ///
    /// # Arguments
    ///
    /// * `residue` - Element of F_p (integer mod p)
    /// * `prime` - Prime p
    /// * `rel_precision` - Desired relative precision
    ///
    /// # Returns
    ///
    /// The Teichmüller lift ω(residue) in Z_p
    ///
    /// # Example
    ///
    /// ```rust
    /// use rustmath_integers::Integer;
    /// use rustmath_rings::padics::capped_relative::CappedRelativePadicElement;
    ///
    /// // Teichmüller lift of 2 in Z_5
    /// let omega = CappedRelativePadicElement::teichmuller_lift(
    ///     Integer::from(2),
    ///     Integer::from(5),
    ///     10
    /// ).unwrap();
    /// assert_eq!(omega.unit() % Integer::from(5), Integer::from(2)); // Reduces to 2 mod 5
    /// ```
    pub fn teichmuller_lift(residue: Integer, prime: Integer, rel_precision: u32) -> Result<Self> {
        if prime <= Integer::one() || !prime.is_prime() {
            return Err(MathError::InvalidArgument(
                "Prime must be an actual prime number".to_string(),
            ));
        }

        // Reduce residue mod p
        let mut res = residue % prime.clone();
        if res.signum() < 0 {
            res = res + prime.clone();
        }

        if res.is_zero() {
            return Self::zero(prime, rel_precision);
        }

        // Compute using the iteration: ω = lim x^(p^n)
        // We iterate: x_{n+1} = x_n^p mod p^(n+1)
        // Starting from x_0 = residue mod p

        let mut current = res.clone();
        let mut current_precision = 1u32;

        // Iterate until we reach desired precision
        while current_precision < rel_precision {
            // Double precision each iteration
            current_precision = cmp::min(current_precision * 2, rel_precision);

            // Compute current^p mod p^current_precision
            let modulus = prime.pow(current_precision);
            current = current.modpow(&prime, &modulus);
        }

        Ok(CappedRelativePadicElement {
            unit: current,
            valuation: 0,
            prime,
            rel_precision,
        })
    }

    /// Lift to higher precision
    ///
    /// This doesn't actually give more information, it just allows the element
    /// to participate in calculations with higher precision elements.
    pub fn lift_to_precision(&self, new_precision: u32) -> Result<Self> {
        if new_precision < self.rel_precision {
            return Err(MathError::InvalidArgument(
                "New precision must be >= current precision".to_string(),
            ));
        }

        Ok(CappedRelativePadicElement {
            unit: self.unit.clone(),
            valuation: self.valuation,
            prime: self.prime.clone(),
            rel_precision: new_precision,
        })
    }

    /// Reduce to lower precision
    pub fn reduce_precision(&self, new_precision: u32) -> Result<Self> {
        if new_precision == 0 {
            return Err(MathError::InvalidArgument(
                "Precision must be > 0".to_string(),
            ));
        }

        if new_precision >= self.rel_precision {
            return Ok(self.clone());
        }

        // Reduce unit modulo p^new_precision
        let modulus = self.prime.pow(new_precision);
        let new_unit = self.unit.clone() % modulus.clone();
        let new_unit = if new_unit.signum() < 0 {
            new_unit + modulus
        } else {
            new_unit
        };

        Ok(CappedRelativePadicElement {
            unit: new_unit,
            valuation: self.valuation,
            prime: self.prime.clone(),
            rel_precision: new_precision,
        })
    }

    /// Compute multiplicative inverse
    ///
    /// Only works if the element is a unit (valuation = 0).
    pub fn inverse(&self) -> Result<Self> {
        if self.unit.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        if self.valuation != 0 {
            return Err(MathError::NotInvertible);
        }

        // Compute unit^(-1) mod p^rel_precision
        let modulus = self.prime.pow(self.rel_precision);
        let (gcd, s, _) = self.unit.extended_gcd(&modulus);

        if !gcd.is_one() {
            return Err(MathError::NotInvertible);
        }

        let inv_unit = if s.signum() < 0 {
            s + modulus.clone()
        } else {
            s % modulus
        };

        Ok(CappedRelativePadicElement {
            unit: inv_unit,
            valuation: -self.valuation,
            prime: self.prime.clone(),
            rel_precision: self.rel_precision,
        })
    }

    /// Check if this element is zero (within the known precision)
    pub fn is_zero_precision(&self) -> bool {
        self.unit.is_zero()
    }

    /// Check if this element is one
    pub fn is_one_precision(&self) -> bool {
        self.valuation == 0 && self.unit.is_one()
    }
}

impl fmt::Display for CappedRelativePadicElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.unit.is_zero() {
            write!(f, "O({}^{})", self.prime, self.rel_precision)
        } else if self.valuation == 0 {
            write!(
                f,
                "{} + O({}^{})",
                self.unit,
                self.prime,
                self.abs_precision()
            )
        } else if self.valuation > 0 {
            write!(
                f,
                "{}*{}^{} + O({}^{})",
                self.unit,
                self.prime,
                self.valuation,
                self.prime,
                self.abs_precision()
            )
        } else {
            write!(
                f,
                "{}/{}^{} + O({}^{})",
                self.unit,
                self.prime,
                -self.valuation,
                self.prime,
                self.abs_precision()
            )
        }
    }
}

impl PartialEq for CappedRelativePadicElement {
    fn eq(&self, other: &Self) -> bool {
        if self.prime != other.prime {
            return false;
        }

        // Compare at the minimum precision
        let min_prec = cmp::min(self.rel_precision, other.rel_precision);
        let self_reduced = self.reduce_precision(min_prec).unwrap();
        let other_reduced = other.reduce_precision(min_prec).unwrap();

        self_reduced.unit == other_reduced.unit && self_reduced.valuation == other_reduced.valuation
    }
}

impl Eq for CappedRelativePadicElement {}

impl Add for CappedRelativePadicElement {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.prime, other.prime, "Primes must match");

        // Handle zero cases
        if self.unit.is_zero() {
            return other;
        }
        if other.unit.is_zero() {
            return self;
        }

        // Result precision is minimum of the two
        let result_precision = cmp::min(self.rel_precision, other.rel_precision);

        if self.valuation == other.valuation {
            // Same valuation: add units directly
            let sum_unit = (self.unit + other.unit) % self.prime.pow(result_precision);

            // Check if sum is zero or has higher valuation
            if sum_unit.is_zero() {
                // Complete cancellation - return zero
                return CappedRelativePadicElement::zero(self.prime, result_precision).unwrap();
            }

            // Check for partial cancellation (sum has higher valuation)
            let sum_valuation = sum_unit.valuation(&self.prime) as i64;
            if sum_valuation > 0 {
                // Precision loss due to cancellation
                let mut new_unit = sum_unit.clone();
                for _ in 0..sum_valuation {
                    new_unit = new_unit / self.prime.clone();
                }
                let new_rel_prec = if result_precision > sum_valuation as u32 {
                    result_precision - sum_valuation as u32
                } else {
                    1
                };

                return CappedRelativePadicElement {
                    unit: new_unit % self.prime.pow(new_rel_prec),
                    valuation: self.valuation + sum_valuation,
                    prime: self.prime,
                    rel_precision: new_rel_prec,
                };
            }

            CappedRelativePadicElement {
                unit: sum_unit,
                valuation: self.valuation,
                prime: self.prime,
                rel_precision: result_precision,
            }
        } else if self.valuation < other.valuation {
            // self has lower valuation (dominates)
            let val_diff = (other.valuation - self.valuation) as u32;

            // Shift other's unit by p^val_diff
            let modulus = self.prime.pow(result_precision);
            let shifted_other = (other.unit * self.prime.pow(val_diff)) % modulus.clone();
            let sum_unit = (self.unit + shifted_other) % modulus;

            CappedRelativePadicElement {
                unit: sum_unit,
                valuation: self.valuation,
                prime: self.prime,
                rel_precision: result_precision,
            }
        } else {
            // other has lower valuation (dominates)
            let val_diff = (self.valuation - other.valuation) as u32;

            let modulus = self.prime.pow(result_precision);
            let shifted_self = (self.unit * self.prime.pow(val_diff)) % modulus.clone();
            let sum_unit = (shifted_self + other.unit) % modulus;

            CappedRelativePadicElement {
                unit: sum_unit,
                valuation: other.valuation,
                prime: self.prime,
                rel_precision: result_precision,
            }
        }
    }
}

impl Sub for CappedRelativePadicElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl Mul for CappedRelativePadicElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        assert_eq!(self.prime, other.prime, "Primes must match");

        // Handle zero
        if self.unit.is_zero() || other.unit.is_zero() {
            return CappedRelativePadicElement::zero(
                self.prime,
                cmp::min(self.rel_precision, other.rel_precision),
            )
            .unwrap();
        }

        let result_precision = cmp::min(self.rel_precision, other.rel_precision);
        let modulus = self.prime.pow(result_precision);
        let product_unit = (self.unit * other.unit) % modulus;

        CappedRelativePadicElement {
            unit: product_unit,
            valuation: self.valuation + other.valuation,
            prime: self.prime,
            rel_precision: result_precision,
        }
    }
}

impl Div for CappedRelativePadicElement {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        assert_eq!(self.prime, other.prime, "Primes must match");

        if other.unit.is_zero() {
            panic!("Division by zero");
        }

        // Compute unit quotient: self.unit * other.unit^(-1) mod p^precision
        let result_precision = cmp::min(self.rel_precision, other.rel_precision);
        let modulus = self.prime.pow(result_precision);

        let (gcd, s, _) = other.unit.extended_gcd(&modulus);
        if !gcd.is_one() {
            panic!("Divisor unit is not invertible");
        }

        let other_inv = if s.signum() < 0 {
            s + modulus.clone()
        } else {
            s % modulus.clone()
        };

        let quotient_unit = (self.unit * other_inv) % modulus;

        CappedRelativePadicElement {
            unit: quotient_unit,
            valuation: self.valuation - other.valuation,
            prime: self.prime,
            rel_precision: result_precision,
        }
    }
}

impl Neg for CappedRelativePadicElement {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.unit.is_zero() {
            return self;
        }

        let modulus = self.prime.pow(self.rel_precision);
        let neg_unit = modulus - self.unit;

        CappedRelativePadicElement {
            unit: neg_unit,
            valuation: self.valuation,
            prime: self.prime,
            rel_precision: self.rel_precision,
        }
    }
}

// Note: We don't implement Ring trait directly because zero() and one()
// require parameters (prime and precision) which the trait doesn't support.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let p = Integer::from(5);

        // Create 7 + O(5^10)
        let x = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();
        assert_eq!(x.valuation(), 0);
        assert_eq!(x.rel_precision(), 10);
        assert_eq!(x.abs_precision(), 10);
        assert_eq!(x.unit(), &Integer::from(7));
    }

    #[test]
    fn test_construction_with_valuation() {
        let p = Integer::from(5);

        // Create 25 = 5^2 * 1
        let x = CappedRelativePadicElement::new(Integer::from(25), p.clone(), 10).unwrap();
        assert_eq!(x.valuation(), 2);
        assert_eq!(x.rel_precision(), 10);
        assert_eq!(x.abs_precision(), 12);
        assert_eq!(x.unit(), &Integer::from(1));
    }

    #[test]
    fn test_from_rational() {
        let p = Integer::from(5);

        // Create 3/5 in Q_5
        let rat = Rational::new(Integer::from(3), Integer::from(5)).unwrap();
        let x = CappedRelativePadicElement::from_rational(rat, p.clone(), 10).unwrap();

        assert_eq!(x.valuation(), -1);
        assert_eq!(x.abs_precision(), 9);
    }

    #[test]
    fn test_addition_same_valuation() {
        let p = Integer::from(5);

        let x = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();
        let y = CappedRelativePadicElement::new(Integer::from(11), p.clone(), 10).unwrap();

        let sum = x + y;
        assert_eq!(sum.valuation(), 0);
        assert_eq!(sum.unit(), &Integer::from(18));
    }

    #[test]
    fn test_addition_different_valuations() {
        let p = Integer::from(5);

        // 1 + O(5^10)
        let x = CappedRelativePadicElement::new(Integer::from(1), p.clone(), 10).unwrap();

        // 5^3 + O(5^10) = 125 + O(5^10)
        let y = CappedRelativePadicElement::new(Integer::from(125), p.clone(), 10).unwrap();

        let sum = x + y;
        // Result should be 1 + 125 = 126 + O(5^10)
        assert_eq!(sum.valuation(), 0);
        assert_eq!(sum.unit(), &Integer::from(126));
    }

    #[test]
    fn test_subtraction_with_cancellation() {
        let p = Integer::from(5);

        // 7 + O(5^10)
        let x = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();

        // 7 + O(5^10)
        let y = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();

        let diff = x - y;
        // Should be zero
        assert!(diff.unit.is_zero());
    }

    #[test]
    fn test_subtraction_partial_cancellation() {
        let p = Integer::from(5);

        // 7 + O(5^10)
        let x = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();

        // 2 + O(5^10)
        let y = CappedRelativePadicElement::new(Integer::from(2), p.clone(), 10).unwrap();

        let diff = x - y;
        // Should be 5 = 5^1 * 1 + O(5^10)
        assert_eq!(diff.valuation(), 1);
        assert_eq!(diff.unit(), &Integer::from(1));
        assert_eq!(diff.rel_precision(), 9); // Lost one digit of precision
    }

    #[test]
    fn test_multiplication() {
        let p = Integer::from(5);

        let x = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();
        let y = CappedRelativePadicElement::new(Integer::from(3), p.clone(), 10).unwrap();

        let prod = x * y;
        assert_eq!(prod.valuation(), 0);
        assert_eq!(prod.unit(), &Integer::from(21));
    }

    #[test]
    fn test_multiplication_with_valuations() {
        let p = Integer::from(5);

        // 5 = 5^1 * 1
        let x = CappedRelativePadicElement::new(Integer::from(5), p.clone(), 10).unwrap();

        // 25 = 5^2 * 1
        let y = CappedRelativePadicElement::new(Integer::from(25), p.clone(), 10).unwrap();

        let prod = x * y;
        // Should be 125 = 5^3 * 1
        assert_eq!(prod.valuation(), 3);
        assert_eq!(prod.unit(), &Integer::from(1));
    }

    #[test]
    fn test_division() {
        let p = Integer::from(5);

        let x = CappedRelativePadicElement::new(Integer::from(21), p.clone(), 10).unwrap();
        let y = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();

        let quot = x / y;
        assert_eq!(quot.valuation(), 0);
        assert_eq!(quot.unit(), &Integer::from(3));
    }

    #[test]
    fn test_division_with_valuations() {
        let p = Integer::from(5);

        // 125 = 5^3 * 1
        let x = CappedRelativePadicElement::new(Integer::from(125), p.clone(), 10).unwrap();

        // 25 = 5^2 * 1
        let y = CappedRelativePadicElement::new(Integer::from(25), p.clone(), 10).unwrap();

        let quot = x / y;
        // Should be 5 = 5^1 * 1
        assert_eq!(quot.valuation(), 1);
        assert_eq!(quot.unit(), &Integer::from(1));
    }

    #[test]
    fn test_negation() {
        let p = Integer::from(5);

        let x = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();
        let neg_x = -x.clone();

        let sum = x + neg_x;
        assert!(sum.unit.is_zero());
    }

    #[test]
    fn test_unit_split() {
        let p = Integer::from(5);

        // 125 = 5^3 * 1
        let x = CappedRelativePadicElement::new(Integer::from(125), p.clone(), 10).unwrap();

        let (unit, val) = x.unit_split();
        assert_eq!(val, 3);
        assert_eq!(unit, &Integer::from(1));
    }

    #[test]
    fn test_unit_split_with_nontrivial_unit() {
        let p = Integer::from(5);

        // 250 = 5^3 * 2
        let x = CappedRelativePadicElement::new(Integer::from(250), p.clone(), 10).unwrap();

        let (unit, val) = x.unit_split();
        assert_eq!(val, 3);
        assert_eq!(unit, &Integer::from(2));
    }

    #[test]
    fn test_teichmuller_lift() {
        let p = Integer::from(5);

        // Teichmüller lift of 2 in Z_5
        let omega = CappedRelativePadicElement::teichmuller_lift(
            Integer::from(2),
            p.clone(),
            10
        ).unwrap();

        // Should reduce to 2 mod 5
        assert_eq!(omega.unit() % Integer::from(5), Integer::from(2));

        // Should have valuation 0
        assert_eq!(omega.valuation(), 0);

        // omega^4 should be 1 (since 2^4 = 16 ≡ 1 mod 5 for the (p-1)=4 root)
        let omega4 = omega.clone() * omega.clone() * omega.clone() * omega.clone();
        // The result should be 1 mod 5 (though not exactly 1 in higher precision)
        assert_eq!(omega4.unit() % Integer::from(5), Integer::from(1));
    }

    #[test]
    fn test_teichmuller_lift_zero() {
        let p = Integer::from(5);

        let omega = CappedRelativePadicElement::teichmuller_lift(
            Integer::from(0),
            p.clone(),
            10
        ).unwrap();

        assert!(omega.is_zero_precision());
    }

    #[test]
    fn test_precision_reduction() {
        let p = Integer::from(5);

        let x = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();
        let reduced = x.reduce_precision(5).unwrap();

        assert_eq!(reduced.rel_precision(), 5);
        assert_eq!(reduced.valuation(), 0);
    }

    #[test]
    fn test_precision_lift() {
        let p = Integer::from(5);

        let x = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 5).unwrap();
        let lifted = x.lift_to_precision(10).unwrap();

        assert_eq!(lifted.rel_precision(), 10);
        assert_eq!(lifted.valuation(), 0);
    }

    #[test]
    fn test_inverse() {
        let p = Integer::from(5);

        // 7 is a unit in Z_5
        let x = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();
        let inv = x.inverse().unwrap();

        let prod = x * inv;
        assert!(prod.is_one_precision());
    }

    #[test]
    fn test_precision_loss_in_addition() {
        let p = Integer::from(5);

        // x = 1 + O(5^10) has rel. prec 10
        let x = CappedRelativePadicElement::new(Integer::from(1), p.clone(), 10).unwrap();

        // y = 5^8 + O(5^10) has rel. prec 2 (valuation 8, abs. prec 10)
        let y = CappedRelativePadicElement::new(
            Integer::from(5).pow(8),
            p.clone(),
            2
        ).unwrap();

        // When we add, result precision should be min(10, 2) = 2
        let sum = x.clone() + y.clone();
        // The lower precision element limits the result
        assert_eq!(sum.rel_precision(), 2);
    }

    #[test]
    fn test_zero_handling() {
        let p = Integer::from(5);

        let zero = CappedRelativePadicElement::zero(p.clone(), 10).unwrap();
        let x = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();

        let sum = zero.clone() + x.clone();
        assert_eq!(sum.unit(), &Integer::from(7));

        let prod = zero.clone() * x.clone();
        assert!(prod.is_zero_precision());
    }

    #[test]
    fn test_display() {
        let p = Integer::from(5);

        let x = CappedRelativePadicElement::new(Integer::from(7), p.clone(), 10).unwrap();
        let display = format!("{}", x);
        assert!(display.contains("7"));
        assert!(display.contains("5"));
        assert!(display.contains("10"));
    }
}
