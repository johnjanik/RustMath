//! Arbitrary precision integers

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use rustmath_core::{
    CommutativeRing, EuclideanDomain, IntegralDomain, MathError, NumericConversion, Result, Ring,
};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Arbitrary precision integer
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Integer {
    value: BigInt,
}

impl Integer {
    /// Create a new integer from a BigInt
    pub fn new(value: BigInt) -> Self {
        Integer { value }
    }

    /// Get the absolute value
    pub fn abs(&self) -> Self {
        Integer::new(self.value.abs())
    }

    /// Get the sign: -1 for negative, 0 for zero, 1 for positive
    pub fn signum(&self) -> i8 {
        if self.value.is_zero() {
            0
        } else if self.value.is_positive() {
            1
        } else {
            -1
        }
    }

    /// Compute GCD using Euclidean algorithm
    pub fn gcd(&self, other: &Self) -> Self {
        EuclideanDomain::gcd(self, other)
    }

    /// Compute LCM
    pub fn lcm(&self, other: &Self) -> Self {
        EuclideanDomain::lcm(self, other)
    }

    /// Extended Euclidean algorithm
    pub fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        EuclideanDomain::extended_gcd(self, other)
    }

    /// Modular exponentiation: compute (self^exp) % modulus efficiently
    pub fn mod_pow(&self, exp: &Self, modulus: &Self) -> Result<Self> {
        if modulus.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        let mut base = self.value.clone() % &modulus.value;
        let mut exp = exp.value.clone();
        let mut result = BigInt::one();

        while !exp.is_zero() {
            if (&exp % 2) == BigInt::one() {
                result = (result * &base) % &modulus.value;
            }
            base = (&base * &base) % &modulus.value;
            exp >>= 1;
        }

        Ok(Integer::new(result))
    }

    /// Check if the number is even
    pub fn is_even(&self) -> bool {
        (&self.value % 2) == BigInt::zero()
    }

    /// Check if the number is odd
    pub fn is_odd(&self) -> bool {
        !self.is_even()
    }

    /// Get the bit length
    pub fn bit_length(&self) -> u64 {
        self.value.bits()
    }

    /// Convert to BigInt reference
    pub fn as_bigint(&self) -> &BigInt {
        &self.value
    }

    /// Convert into BigInt
    pub fn into_bigint(self) -> BigInt {
        self.value
    }
}

// Implement From traits for convenient construction
impl From<i64> for Integer {
    fn from(n: i64) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<u64> for Integer {
    fn from(n: u64) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<i32> for Integer {
    fn from(n: i32) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<u32> for Integer {
    fn from(n: u32) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<BigInt> for Integer {
    fn from(n: BigInt) -> Self {
        Integer::new(n)
    }
}

// Display implementation
impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Debug for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Integer({})", self.value)
    }
}

// Arithmetic operations
impl Add for Integer {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Integer::new(self.value + other.value)
    }
}

impl<'b> Add<&'b Integer> for &Integer {
    type Output = Integer;

    fn add(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value + &other.value)
    }
}

impl Sub for Integer {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Integer::new(self.value - other.value)
    }
}

impl<'b> Sub<&'b Integer> for &Integer {
    type Output = Integer;

    fn sub(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value - &other.value)
    }
}

impl Mul for Integer {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Integer::new(self.value * other.value)
    }
}

impl<'b> Mul<&'b Integer> for &Integer {
    type Output = Integer;

    fn mul(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value * &other.value)
    }
}

impl Div for Integer {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Integer::new(self.value / other.value)
    }
}

impl<'b> Div<&'b Integer> for &Integer {
    type Output = Integer;

    fn div(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value / &other.value)
    }
}

impl Rem for Integer {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        Integer::new(self.value % other.value)
    }
}

impl<'b> Rem<&'b Integer> for &Integer {
    type Output = Integer;

    fn rem(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value % &other.value)
    }
}

impl Neg for Integer {
    type Output = Self;

    fn neg(self) -> Self {
        Integer::new(-self.value)
    }
}

impl Neg for &Integer {
    type Output = Integer;

    fn neg(self) -> Integer {
        Integer::new(-&self.value)
    }
}

// Ring trait implementation
impl Ring for Integer {
    fn zero() -> Self {
        Integer::new(BigInt::zero())
    }

    fn one() -> Self {
        Integer::new(BigInt::one())
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value == BigInt::one()
    }

    fn pow(&self, n: u32) -> Self {
        Integer::new(num_traits::pow(self.value.clone(), n as usize))
    }
}

impl CommutativeRing for Integer {}
impl IntegralDomain for Integer {}

impl EuclideanDomain for Integer {
    fn norm(&self) -> u64 {
        // For integers, use absolute value as norm
        // This is a simplification; true norm could be arbitrarily large
        self.abs().to_u64().unwrap_or(u64::MAX)
    }

    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        let q = &self.value / &other.value;
        let r = &self.value % &other.value;

        Ok((Integer::new(q), Integer::new(r)))
    }
}

impl NumericConversion for Integer {
    fn from_i64(n: i64) -> Self {
        Integer::from(n)
    }

    fn from_u64(n: u64) -> Self {
        Integer::from(n)
    }

    fn to_i64(&self) -> Option<i64> {
        use num_traits::ToPrimitive;
        self.value.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        use num_traits::ToPrimitive;
        self.value.to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        use num_traits::ToPrimitive;
        self.value.to_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = Integer::from(42);
        let b = Integer::from(17);

        assert_eq!(a.clone() + b.clone(), Integer::from(59));
        assert_eq!(a.clone() - b.clone(), Integer::from(25));
        assert_eq!(a.clone() * b.clone(), Integer::from(714));
        assert_eq!(a.clone() / b.clone(), Integer::from(2));
        assert_eq!(a.clone() % b.clone(), Integer::from(8));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(Integer::from(48).gcd(&Integer::from(18)), Integer::from(6));
        assert_eq!(Integer::from(100).gcd(&Integer::from(35)), Integer::from(5));
    }

    #[test]
    fn test_lcm() {
        assert_eq!(Integer::from(12).lcm(&Integer::from(18)), Integer::from(36));
        assert_eq!(Integer::from(21).lcm(&Integer::from(6)), Integer::from(42));
    }

    #[test]
    fn test_extended_gcd() {
        let a = Integer::from(240);
        let b = Integer::from(46);
        let (gcd, s, t) = a.extended_gcd(&b);

        assert_eq!(gcd, Integer::from(2));
        assert_eq!(&a * &s + &b * &t, gcd);
    }

    #[test]
    fn test_mod_pow() {
        let base = Integer::from(2);
        let exp = Integer::from(10);
        let modulus = Integer::from(1000);

        let result = base.mod_pow(&exp, &modulus).unwrap();
        assert_eq!(result, Integer::from(24));
    }

    #[test]
    fn test_pow() {
        assert_eq!(Integer::from(2).pow(10), Integer::from(1024));
        assert_eq!(Integer::from(3).pow(5), Integer::from(243));
    }

    #[test]
    fn test_large_numbers() {
        // Test with numbers larger than 64-bit
        let a = Integer::from(1u64 << 63);
        let b = Integer::from(1u64 << 62);
        let c = a.clone() + b.clone();

        assert!(c > a);
        assert!(c > b);
    }
}
