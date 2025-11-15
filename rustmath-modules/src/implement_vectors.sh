#!/bin/bash

# Implement vector_rational_dense
cat > vector_rational_dense.rs << 'EOF'
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
EOF

# Implement vector_integer_sparse
cat > vector_integer_sparse.rs << 'EOF'
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
EOF

# Implement vector_double_dense
cat > vector_double_dense.rs << 'EOF'
//! Dense floating-point vector implementation

use std::ops::{Add, Sub, Mul, Neg, Index, IndexMut};

/// A dense vector with f64 entries
#[derive(Clone, Debug, PartialEq)]
pub struct VectorDoubleDense {
    data: Vec<f64>,
}

impl VectorDoubleDense {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn zero(dim: usize) -> Self {
        Self { data: vec![0.0; dim] }
    }

    pub fn basis(dim: usize, i: usize) -> Self {
        assert!(i < dim);
        let mut data = vec![0.0; dim];
        data[i] = 1.0;
        Self { data }
    }

    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn is_zero(&self, tol: f64) -> bool {
        self.data.iter().all(|x| x.abs() < tol)
    }

    pub fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.dimension(), other.dimension());
        self.data.iter().zip(&other.data)
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn norm_l1(&self) -> f64 {
        self.data.iter().map(|x| x.abs()).sum()
    }

    pub fn norm_l2(&self) -> f64 {
        self.norm_l2_squared().sqrt()
    }

    pub fn norm_l2_squared(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum()
    }

    pub fn normalize(&self) -> Option<Self> {
        let norm = self.norm_l2();
        if norm > 1e-10 {
            Some(self.scale(1.0 / norm))
        } else {
            None
        }
    }

    pub fn scale(&self, scalar: f64) -> Self {
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

impl Index<usize> for VectorDoubleDense {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for VectorDoubleDense {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_vector() {
        let v = VectorDoubleDense::new(vec![3.0, 4.0]);
        assert_eq!(v.norm_l2(), 5.0);
    }

    #[test]
    fn test_normalization() {
        let v = VectorDoubleDense::new(vec![3.0, 4.0]);
        let normalized = v.normalize().unwrap();
        assert!((normalized.norm_l2() - 1.0).abs() < 1e-10);
    }
}
EOF

echo "Vector implementations created"
