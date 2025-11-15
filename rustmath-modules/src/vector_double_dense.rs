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
