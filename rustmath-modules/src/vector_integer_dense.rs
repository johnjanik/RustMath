//! Dense integer vector implementation

use num_bigint::BigInt;
use num_traits::{Zero, One, Signed};
use std::ops::{Add, Sub, Mul, Neg, Index, IndexMut};

/// A dense vector with integer (BigInt) entries
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorIntegerDense {
    data: Vec<BigInt>,
}

impl VectorIntegerDense {
    /// Create a new vector from data
    pub fn new(data: Vec<BigInt>) -> Self {
        Self { data }
    }

    /// Create a zero vector of given dimension
    pub fn zero(dim: usize) -> Self {
        Self {
            data: vec![BigInt::zero(); dim],
        }
    }

    /// Create a basis vector (all zeros except 1 at position i)
    pub fn basis(dim: usize, i: usize) -> Self {
        assert!(i < dim, "Index out of bounds");
        let mut data = vec![BigInt::zero(); dim];
        data[i] = BigInt::one();
        Self { data }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Get the data as a slice
    pub fn as_slice(&self) -> &[BigInt] {
        &self.data
    }

    /// Check if this is the zero vector
    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|x| x.is_zero())
    }

    /// Compute the dot product with another vector
    pub fn dot(&self, other: &Self) -> BigInt {
        assert_eq!(self.dimension(), other.dimension());
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .fold(BigInt::zero(), |acc, x| acc + x)
    }

    /// Compute the L1 norm (sum of absolute values)
    pub fn norm_l1(&self) -> BigInt {
        self.data.iter().map(|x| x.abs()).fold(BigInt::zero(), |acc, x| acc + x)
    }

    /// Compute the squared L2 norm
    pub fn norm_l2_squared(&self) -> BigInt {
        self.data.iter().map(|x| x * x).fold(BigInt::zero(), |acc, x| acc + x)
    }

    /// Scale by a scalar
    pub fn scale(&self, scalar: &BigInt) -> Self {
        Self {
            data: self.data.iter().map(|x| x * scalar).collect(),
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.dimension(), other.dimension());
        Self {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.dimension(), other.dimension());
        Self {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Negate all elements
    pub fn negate(&self) -> Self {
        Self {
            data: self.data.iter().map(|x| -x).collect(),
        }
    }

    /// Iterator over elements
    pub fn iter(&self) -> impl Iterator<Item = &BigInt> {
        self.data.iter()
    }

    /// Mutable iterator over elements
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut BigInt> {
        self.data.iter_mut()
    }
}

impl Index<usize> for VectorIntegerDense {
    type Output = BigInt;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for VectorIntegerDense {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Add for VectorIntegerDense {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        VectorIntegerDense::add(&self, &other)
    }
}

impl Sub for VectorIntegerDense {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        VectorIntegerDense::sub(&self, &other)
    }
}

impl Neg for VectorIntegerDense {
    type Output = Self;

    fn neg(self) -> Self {
        self.negate()
    }
}

impl Mul<BigInt> for VectorIntegerDense {
    type Output = Self;

    fn mul(self, scalar: BigInt) -> Self {
        self.scale(&scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let v = VectorIntegerDense::new(vec![BigInt::from(1), BigInt::from(2), BigInt::from(3)]);
        assert_eq!(v.dimension(), 3);
        assert_eq!(v[0], BigInt::from(1));
        assert_eq!(v[2], BigInt::from(3));
    }

    #[test]
    fn test_zero() {
        let v = VectorIntegerDense::zero(5);
        assert_eq!(v.dimension(), 5);
        assert!(v.is_zero());
    }

    #[test]
    fn test_basis() {
        let v = VectorIntegerDense::basis(4, 2);
        assert_eq!(v[0], BigInt::zero());
        assert_eq!(v[1], BigInt::zero());
        assert_eq!(v[2], BigInt::one());
        assert_eq!(v[3], BigInt::zero());
    }

    #[test]
    fn test_dot_product() {
        let v1 = VectorIntegerDense::new(vec![BigInt::from(1), BigInt::from(2), BigInt::from(3)]);
        let v2 = VectorIntegerDense::new(vec![BigInt::from(4), BigInt::from(5), BigInt::from(6)]);
        assert_eq!(v1.dot(&v2), BigInt::from(32)); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_addition() {
        let v1 = VectorIntegerDense::new(vec![BigInt::from(1), BigInt::from(2)]);
        let v2 = VectorIntegerDense::new(vec![BigInt::from(3), BigInt::from(4)]);
        let v3 = v1 + v2;
        assert_eq!(v3[0], BigInt::from(4));
        assert_eq!(v3[1], BigInt::from(6));
    }

    #[test]
    fn test_scaling() {
        let v = VectorIntegerDense::new(vec![BigInt::from(2), BigInt::from(3)]);
        let v2 = v.scale(&BigInt::from(5));
        assert_eq!(v2[0], BigInt::from(10));
        assert_eq!(v2[1], BigInt::from(15));
    }

    #[test]
    fn test_norms() {
        let v = VectorIntegerDense::new(vec![BigInt::from(3), BigInt::from(-4)]);
        assert_eq!(v.norm_l1(), BigInt::from(7));
        assert_eq!(v.norm_l2_squared(), BigInt::from(25));
    }
}
