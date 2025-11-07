//! Core algebraic traits
//!
//! This module defines the fundamental algebraic structures used in mathematics:
//! - Magma: Set with binary operation
//! - Semigroup: Magma with associativity
//! - Monoid: Semigroup with identity
//! - Group: Monoid with inverses
//! - Ring: Abelian group with multiplication
//! - Field: Ring with multiplicative inverses
//! - EuclideanDomain: Ring with division algorithm
//! - Module: Generalization of vector spaces

use crate::error::{MathError, Result};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A magma is a set with a binary operation
pub trait Magma: Sized {
    /// The binary operation
    fn op(&self, other: &Self) -> Self;
}

/// A semigroup is a magma where the operation is associative
pub trait Semigroup: Magma {}

/// A monoid is a semigroup with an identity element
pub trait Monoid: Semigroup {
    /// The identity element
    fn identity() -> Self;

    /// Check if this is the identity
    fn is_identity(&self) -> bool;
}

/// A group is a monoid where every element has an inverse
pub trait Group: Monoid {
    /// The inverse element
    fn inverse(&self) -> Self;
}

/// An abelian (commutative) group
pub trait AbelianGroup: Group {}

/// A ring is an abelian group under addition with an associative multiplication
pub trait Ring:
    Sized
    + Clone
    + Debug
    + Display
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    /// The additive identity (zero)
    fn zero() -> Self;

    /// The multiplicative identity (one)
    fn one() -> Self;

    /// Check if this element is zero
    fn is_zero(&self) -> bool;

    /// Check if this element is one
    fn is_one(&self) -> bool;

    /// Compute self raised to the power n (for non-negative n)
    fn pow(&self, n: u32) -> Self {
        if n == 0 {
            Self::one()
        } else if n == 1 {
            self.clone()
        } else {
            let half = self.pow(n / 2);
            let result = half.clone() * half;
            if n % 2 == 0 {
                result
            } else {
                result * self.clone()
            }
        }
    }
}

/// A commutative ring (multiplication is commutative)
pub trait CommutativeRing: Ring {}

/// An integral domain is a commutative ring with no zero divisors
pub trait IntegralDomain: CommutativeRing {}

/// A Euclidean domain is an integral domain with a division algorithm
pub trait EuclideanDomain: IntegralDomain {
    /// The Euclidean function (e.g., absolute value for integers, degree for polynomials)
    fn norm(&self) -> u64;

    /// Division with remainder: returns (quotient, remainder)
    fn div_rem(&self, other: &Self) -> Result<(Self, Self)>;

    /// Greatest common divisor using Euclidean algorithm
    fn gcd(&self, other: &Self) -> Self {
        let mut a = self.clone();
        let mut b = other.clone();

        while !b.is_zero() {
            let (_, r) = a.div_rem(&b).unwrap();
            a = b;
            b = r;
        }

        a
    }

    /// Least common multiple
    fn lcm(&self, other: &Self) -> Self {
        if self.is_zero() && other.is_zero() {
            return Self::zero();
        }
        let gcd = self.gcd(other);
        let (quotient, _) = (self.clone() * other.clone()).div_rem(&gcd).unwrap();
        quotient
    }

    /// Extended Euclidean algorithm: returns (gcd, s, t) where gcd = s*a + t*b
    fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        let mut old_r = self.clone();
        let mut r = other.clone();
        let mut old_s = Self::one();
        let mut s = Self::zero();
        let mut old_t = Self::zero();
        let mut t = Self::one();

        while !r.is_zero() {
            let (quotient, remainder) = old_r.div_rem(&r).unwrap();
            old_r = r;
            r = remainder;

            let temp_s = s.clone();
            s = old_s.clone() - quotient.clone() * s;
            old_s = temp_s;

            let temp_t = t.clone();
            t = old_t.clone() - quotient * t;
            old_t = temp_t;
        }

        (old_r, old_s, old_t)
    }
}

/// A field is a commutative ring where every non-zero element has a multiplicative inverse
pub trait Field: CommutativeRing + Div<Output = Self> {
    /// Multiplicative inverse (reciprocal)
    fn inverse(&self) -> Result<Self>;

    /// Division
    fn divide(&self, other: &Self) -> Result<Self> {
        if other.is_zero() {
            Err(MathError::DivisionByZero)
        } else {
            Ok(self.clone() * other.inverse()?)
        }
    }
}

/// A module over a ring (generalization of vector space)
pub trait Module<R: Ring>:
    Sized
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
{
    /// Scalar multiplication
    fn scalar_mul(&self, scalar: &R) -> Self;

    /// The zero element
    fn zero() -> Self;

    /// Check if this is zero
    fn is_zero(&self) -> bool;
}

/// A vector space is a module over a field
pub trait VectorSpace<F: Field>: Module<F> {
    /// The dimension of the vector space (if finite)
    fn dimension() -> Option<usize>;
}

/// An algebra over a field (vector space with multiplication)
pub trait Algebra<F: Field>: VectorSpace<F> + Ring {}

/// Trait for types that can be converted to/from standard Rust numeric types
pub trait NumericConversion {
    /// Convert from i64
    fn from_i64(n: i64) -> Self;

    /// Convert from u64
    fn from_u64(n: u64) -> Self;

    /// Try to convert to i64
    fn to_i64(&self) -> Option<i64>;

    /// Try to convert to u64
    fn to_u64(&self) -> Option<u64>;

    /// Try to convert to f64 (may lose precision)
    fn to_f64(&self) -> Option<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test implementation for i32
    impl Ring for i32 {
        fn zero() -> Self { 0 }
        fn one() -> Self { 1 }
        fn is_zero(&self) -> bool { *self == 0 }
        fn is_one(&self) -> bool { *self == 1 }
    }

    impl CommutativeRing for i32 {}
    impl IntegralDomain for i32 {}

    #[test]
    fn test_pow() {
        assert_eq!(2i32.pow(0), 1);
        assert_eq!(2i32.pow(1), 2);
        assert_eq!(2i32.pow(10), 1024);
        assert_eq!(3i32.pow(4), 81);
    }
}
