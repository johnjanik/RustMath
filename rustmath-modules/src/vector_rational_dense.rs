//! Dense rational vector implementation

use num_rational::BigRational;
use num_bigint::BigInt;
use num_traits::{Zero, One};
use std::ops::{Add, Sub, Mul, Neg, Index, IndexMut};

/// A dense vector with rational (BigRational) entries
#[derive(Clone, Debug, PartialEq)]
pub struct VectorRationalDense {
    data: Vec<BigRational>,
}

impl VectorRationalDense {
    pub fn new(data: Vec<BigRational>) -> Self {
        Self { data }
    }

    pub fn zero(dim: usize) -> Self {
        Self { data: vec![BigRational::zero(); dim] }
    }

    pub fn basis(dim: usize, i: usize) -> Self {
        assert!(i < dim);
        let mut data = vec![BigRational::zero(); dim];
        data[i] = BigRational::one();
        Self { data }
    }

    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[BigRational] {
        &self.data
    }

    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|x| x.is_zero())
    }

    pub fn dot(&self, other: &Self) -> BigRational {
        assert_eq!(self.dimension(), other.dimension());
        self.data.iter().zip(&other.data)
            .map(|(a, b)| a * b)
            .fold(BigRational::zero(), |acc, x| acc + x)
    }

    pub fn scale(&self, scalar: &BigRational) -> Self {
        Self { data: self.data.iter().map(|x| x * scalar).collect() }
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.dimension(), other.dimension());
        Self {
            data: self.data.iter().zip(&other.data)
                .map(|(a, b)| a + b).collect()
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.dimension(), other.dimension());
        Self {
            data: self.data.iter().zip(&other.data)
                .map(|(a, b)| a - b).collect()
        }
    }

    pub fn negate(&self) -> Self {
        Self { data: self.data.iter().map(|x| -x).collect() }
    }
}

impl Index<usize> for VectorRationalDense {
    type Output = BigRational;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for VectorRationalDense {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
