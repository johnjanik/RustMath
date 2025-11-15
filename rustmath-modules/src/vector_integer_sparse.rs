//! Sparse integer vector implementation

use num_bigint::BigInt;
use num_traits::{Zero, One};
use std::collections::HashMap;

/// A sparse vector with integer entries (stores only non-zero elements)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorIntegerSparse {
    dimension: usize,
    data: HashMap<usize, BigInt>,
}

impl VectorIntegerSparse {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            data: HashMap::new(),
        }
    }

    pub fn zero(dimension: usize) -> Self {
        Self::new(dimension)
    }

    pub fn basis(dimension: usize, i: usize) -> Self {
        assert!(i < dimension);
        let mut vec = Self::new(dimension);
        vec.set(i, BigInt::one());
        vec
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn get(&self, index: usize) -> BigInt {
        self.data.get(&index).cloned().unwrap_or_else(BigInt::zero)
    }

    pub fn set(&mut self, index: usize, value: BigInt) {
        assert!(index < self.dimension);
        if value.is_zero() {
            self.data.remove(&index);
        } else {
            self.data.insert(index, value);
        }
    }

    pub fn is_zero(&self) -> bool {
        self.data.is_empty()
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn dot(&self, other: &Self) -> BigInt {
        assert_eq!(self.dimension, other.dimension);
        self.data.iter()
            .filter_map(|(idx, val)| other.data.get(idx).map(|v| val * v))
            .fold(BigInt::zero(), |acc, x| acc + x)
    }

    pub fn scale(&self, scalar: &BigInt) -> Self {
        if scalar.is_zero() {
            return Self::zero(self.dimension);
        }
        let mut result = Self::new(self.dimension);
        for (idx, val) in &self.data {
            result.data.insert(*idx, val * scalar);
        }
        result
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.dimension, other.dimension);
        let mut result = Self::new(self.dimension);
        for (idx, val) in &self.data {
            result.set(*idx, val.clone());
        }
        for (idx, val) in &other.data {
            let current = result.get(*idx);
            result.set(*idx, current + val);
        }
        result
    }

    pub fn negate(&self) -> Self {
        let mut result = Self::new(self.dimension);
        for (idx, val) in &self.data {
            result.data.insert(*idx, -val);
        }
        result
    }

    pub fn iter_nonzero(&self) -> impl Iterator<Item = (usize, &BigInt)> {
        self.data.iter().map(|(idx, val)| (*idx, val))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector() {
        let mut v = VectorIntegerSparse::new(100);
        v.set(10, BigInt::from(42));
        v.set(50, BigInt::from(-17));
        assert_eq!(v.nnz(), 2);
        assert_eq!(v.get(10), BigInt::from(42));
        assert_eq!(v.get(25), BigInt::zero());
    }

    #[test]
    fn test_sparse_dot() {
        let mut v1 = VectorIntegerSparse::new(100);
        v1.set(5, BigInt::from(3));
        v1.set(10, BigInt::from(4));
        
        let mut v2 = VectorIntegerSparse::new(100);
        v2.set(5, BigInt::from(2));
        v2.set(15, BigInt::from(7));
        
        assert_eq!(v1.dot(&v2), BigInt::from(6)); // only index 5 overlaps: 3*2=6
    }
}
