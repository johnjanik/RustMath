//! Vector operations

use rustmath_core::{MathError, Result, Ring};
use std::ops::{Add, Sub};

/// Generic vector over a ring R
#[derive(Clone, PartialEq, Debug)]
pub struct Vector<R: Ring> {
    data: Vec<R>,
}

impl<R: Ring> Vector<R> {
    /// Create a new vector
    pub fn new(data: Vec<R>) -> Self {
        Vector { data }
    }

    /// Create a zero vector of given dimension
    pub fn zeros(n: usize) -> Self {
        Vector {
            data: (0..n).map(|_| R::zero()).collect(),
        }
    }

    /// Get the dimension
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Get element at index i
    pub fn get(&self, i: usize) -> Result<&R> {
        self.data
            .get(i)
            .ok_or_else(|| MathError::InvalidArgument("Index out of bounds".to_string()))
    }

    /// Set element at index i
    pub fn set(&mut self, i: usize, value: R) -> Result<()> {
        if i >= self.data.len() {
            return Err(MathError::InvalidArgument("Index out of bounds".to_string()));
        }
        self.data[i] = value;
        Ok(())
    }

    /// Dot product with another vector
    pub fn dot(&self, other: &Self) -> Result<R> {
        if self.dim() != other.dim() {
            return Err(MathError::InvalidArgument(
                "Vectors must have same dimension for dot product".to_string(),
            ));
        }

        let mut sum = R::zero();
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            sum = sum + a.clone() * b.clone();
        }

        Ok(sum)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        Vector {
            data: self.data.iter().map(|x| x.clone() * scalar.clone()).collect(),
        }
    }
}

impl<R: Ring> Add for Vector<R> {
    type Output = Result<Self>;

    fn add(self, other: Self) -> Result<Self> {
        if self.dim() != other.dim() {
            return Err(MathError::InvalidArgument(
                "Vectors must have same dimension for addition".to_string(),
            ));
        }

        let data = self
            .data
            .into_iter()
            .zip(other.data)
            .map(|(a, b)| a + b)
            .collect();

        Ok(Vector { data })
    }
}

impl<R: Ring> Sub for Vector<R> {
    type Output = Result<Self>;

    fn sub(self, other: Self) -> Result<Self> {
        if self.dim() != other.dim() {
            return Err(MathError::InvalidArgument(
                "Vectors must have same dimension for subtraction".to_string(),
            ));
        }

        let data = self
            .data
            .into_iter()
            .zip(other.data)
            .map(|(a, b)| a - b)
            .collect();

        Ok(Vector { data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let v = Vector::new(vec![1, 2, 3]);
        assert_eq!(v.dim(), 3);
        assert_eq!(*v.get(0).unwrap(), 1);
        assert_eq!(*v.get(2).unwrap(), 3);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector::new(vec![1, 2, 3]);
        let v2 = Vector::new(vec![4, 5, 6]);

        let dot = v1.dot(&v2).unwrap();
        assert_eq!(dot, 32); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_scalar_mul() {
        let v = Vector::new(vec![1, 2, 3]);
        let scaled = v.scalar_mul(&2);

        assert_eq!(*scaled.get(0).unwrap(), 2);
        assert_eq!(*scaled.get(1).unwrap(), 4);
        assert_eq!(*scaled.get(2).unwrap(), 6);
    }
}
