//! # Non-Negative Integer Semiring
//!
//! This module implements the semiring of non-negative integers ℕ₀ = {0, 1, 2, 3, ...}.
//!
//! ## Overview
//!
//! The non-negative integers form a semiring (also called a rig) under standard
//! addition and multiplication:
//! - Addition: a + b (always defined for non-negative integers)
//! - Multiplication: a × b
//! - Additive identity: 0
//! - Multiplicative identity: 1
//!
//! ## Theory
//!
//! Properties of ℕ₀:
//! - **Semiring**: Has addition and multiplication with identities
//! - **Commutative**: Both operations are commutative
//! - **Cancellative**: Can cancel in multiplication (if ab = ac and a ≠ 0, then b = c)
//! - **No additive inverses**: Not a ring (can't subtract in general)
//! - **Ordered**: Natural ordering 0 < 1 < 2 < ...
//!
//! ## Use Cases
//!
//! - Combinatorics (counting problems)
//! - Tropical geometry (where min/+ or max/+ replace +/×)
//! - Formal language theory
//! - Optimization problems
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::semirings::non_negative_integer_semiring::NonNegativeIntegerSemiring;
//!
//! let semiring = NonNegativeIntegerSemiring::new();
//! let a = semiring.from_usize(5);
//! let b = semiring.from_usize(3);
//! ```

use rustmath_core::Ring;
use std::fmt;
use std::ops::{Add, Mul};

/// The semiring of non-negative integers
///
/// Represents ℕ₀ = {0, 1, 2, 3, ...} with standard addition and multiplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonNegativeIntegerSemiring {
    // Empty struct - the semiring structure is determined by the type
}

impl NonNegativeIntegerSemiring {
    /// Create a new non-negative integer semiring
    ///
    /// # Examples
    /// ```
    /// use rustmath_rings::semirings::non_negative_integer_semiring::NonNegativeIntegerSemiring;
    ///
    /// let nn = NonNegativeIntegerSemiring::new();
    /// ```
    pub fn new() -> Self {
        Self {}
    }

    /// Create an element from a usize
    pub fn from_usize(&self, value: usize) -> NonNegativeInteger {
        NonNegativeInteger { value }
    }

    /// Get the zero element
    pub fn zero(&self) -> NonNegativeInteger {
        NonNegativeInteger { value: 0 }
    }

    /// Get the one element
    pub fn one(&self) -> NonNegativeInteger {
        NonNegativeInteger { value: 1 }
    }

    /// Check if an element is in this semiring
    pub fn contains(&self, _element: &NonNegativeInteger) -> bool {
        true // All NonNegativeInteger values are in ℕ₀
    }
}

impl Default for NonNegativeIntegerSemiring {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for NonNegativeIntegerSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Non-negative Integer Semiring")
    }
}

/// An element of the non-negative integer semiring
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NonNegativeInteger {
    /// The value (guaranteed to be non-negative)
    value: usize,
}

impl NonNegativeInteger {
    /// Create a new non-negative integer
    ///
    /// # Examples
    /// ```
    /// use rustmath_rings::semirings::non_negative_integer_semiring::NonNegativeInteger;
    ///
    /// let n = NonNegativeInteger::new(42);
    /// ```
    pub fn new(value: usize) -> Self {
        Self { value }
    }

    /// Get the value
    pub fn value(&self) -> usize {
        self.value
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    /// Check if this is one
    pub fn is_one(&self) -> bool {
        self.value == 1
    }

    /// Factorial of this number
    pub fn factorial(&self) -> Self {
        let mut result = 1;
        for i in 2..=self.value {
            result *= i;
        }
        Self { value: result }
    }

    /// Binomial coefficient C(self, k)
    pub fn binomial(&self, k: usize) -> Self {
        if k > self.value {
            return Self { value: 0 };
        }

        let k = k.min(self.value - k); // Optimization: C(n,k) = C(n,n-k)

        let mut result = 1;
        for i in 0..k {
            result = result * (self.value - i) / (i + 1);
        }

        Self { value: result }
    }

    /// GCD with another non-negative integer
    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.value;
        let mut b = other.value;

        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }

        Self { value: a }
    }

    /// LCM with another non-negative integer
    pub fn lcm(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self { value: 0 };
        }

        let gcd = self.gcd(other);
        Self {
            value: (self.value / gcd.value) * other.value,
        }
    }

    /// Power (self^exp)
    pub fn pow(&self, exp: usize) -> Self {
        Self {
            value: self.value.pow(exp as u32),
        }
    }
}

impl Add for NonNegativeInteger {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
        }
    }
}

impl Mul for NonNegativeInteger {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
        }
    }
}

impl From<usize> for NonNegativeInteger {
    fn from(value: usize) -> Self {
        Self { value }
    }
}

impl From<NonNegativeInteger> for usize {
    fn from(n: NonNegativeInteger) -> Self {
        n.value
    }
}

impl fmt::Display for NonNegativeInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semiring_creation() {
        let nn = NonNegativeIntegerSemiring::new();
        assert_eq!(nn.zero().value(), 0);
        assert_eq!(nn.one().value(), 1);
    }

    #[test]
    fn test_from_usize() {
        let nn = NonNegativeIntegerSemiring::new();
        let n = nn.from_usize(42);
        assert_eq!(n.value(), 42);
    }

    #[test]
    fn test_addition() {
        let a = NonNegativeInteger::new(5);
        let b = NonNegativeInteger::new(3);
        let sum = a + b;
        assert_eq!(sum.value(), 8);
    }

    #[test]
    fn test_multiplication() {
        let a = NonNegativeInteger::new(5);
        let b = NonNegativeInteger::new(3);
        let prod = a * b;
        assert_eq!(prod.value(), 15);
    }

    #[test]
    fn test_is_zero_one() {
        let zero = NonNegativeInteger::new(0);
        assert!(zero.is_zero());
        assert!(!zero.is_one());

        let one = NonNegativeInteger::new(1);
        assert!(!one.is_zero());
        assert!(one.is_one());
    }

    #[test]
    fn test_factorial() {
        assert_eq!(NonNegativeInteger::new(0).factorial().value(), 1);
        assert_eq!(NonNegativeInteger::new(1).factorial().value(), 1);
        assert_eq!(NonNegativeInteger::new(5).factorial().value(), 120);
        assert_eq!(NonNegativeInteger::new(10).factorial().value(), 3628800);
    }

    #[test]
    fn test_binomial() {
        let n = NonNegativeInteger::new(5);

        assert_eq!(n.binomial(0).value(), 1);
        assert_eq!(n.binomial(1).value(), 5);
        assert_eq!(n.binomial(2).value(), 10);
        assert_eq!(n.binomial(3).value(), 10);
        assert_eq!(n.binomial(4).value(), 5);
        assert_eq!(n.binomial(5).value(), 1);
        assert_eq!(n.binomial(6).value(), 0);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(
            NonNegativeInteger::new(12).gcd(&NonNegativeInteger::new(8)).value(),
            4
        );
        assert_eq!(
            NonNegativeInteger::new(15).gcd(&NonNegativeInteger::new(25)).value(),
            5
        );
        assert_eq!(
            NonNegativeInteger::new(7).gcd(&NonNegativeInteger::new(13)).value(),
            1
        );
    }

    #[test]
    fn test_lcm() {
        assert_eq!(
            NonNegativeInteger::new(12).lcm(&NonNegativeInteger::new(8)).value(),
            24
        );
        assert_eq!(
            NonNegativeInteger::new(4).lcm(&NonNegativeInteger::new(6)).value(),
            12
        );
    }

    #[test]
    fn test_pow() {
        assert_eq!(NonNegativeInteger::new(2).pow(3).value(), 8);
        assert_eq!(NonNegativeInteger::new(5).pow(2).value(), 25);
        assert_eq!(NonNegativeInteger::new(10).pow(0).value(), 1);
    }

    #[test]
    fn test_ordering() {
        let a = NonNegativeInteger::new(5);
        let b = NonNegativeInteger::new(10);

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a.max(b), b);
        assert_eq!(a.min(b), a);
    }

    #[test]
    fn test_from_into() {
        let n: NonNegativeInteger = 42.into();
        assert_eq!(n.value(), 42);

        let v: usize = n.into();
        assert_eq!(v, 42);
    }

    #[test]
    fn test_display() {
        let n = NonNegativeInteger::new(123);
        assert_eq!(format!("{}", n), "123");

        let nn = NonNegativeIntegerSemiring::new();
        assert_eq!(format!("{}", nn), "Non-negative Integer Semiring");
    }
}
