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

    /// Cross product (3D vectors only)
    pub fn cross(&self, other: &Self) -> Result<Self> {
        if self.dim() != 3 || other.dim() != 3 {
            return Err(MathError::InvalidArgument(
                "Cross product is only defined for 3D vectors".to_string(),
            ));
        }

        let a = &self.data;
        let b = &other.data;

        // Cross product formula: (a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1)
        let x = a[1].clone() * b[2].clone() - a[2].clone() * b[1].clone();
        let y = a[2].clone() * b[0].clone() - a[0].clone() * b[2].clone();
        let z = a[0].clone() * b[1].clone() - a[1].clone() * b[0].clone();

        Ok(Vector {
            data: vec![x, y, z],
        })
    }

    /// Compute the p-norm of the vector
    /// For p = 1: sum of absolute values
    /// For p = 2: Euclidean norm (sqrt of sum of squares)
    /// For p = infinity: maximum absolute value
    pub fn norm(&self, p: f64) -> f64
    where
        R: rustmath_core::NumericConversion,
    {
        if p == f64::INFINITY {
            // Infinity norm: max absolute value
            self.data
                .iter()
                .map(|x| x.to_f64().unwrap_or(0.0).abs())
                .fold(0.0, f64::max)
        } else if p == 1.0 {
            // 1-norm: sum of absolute values
            self.data
                .iter()
                .map(|x| x.to_f64().unwrap_or(0.0).abs())
                .sum()
        } else if p == 2.0 {
            // 2-norm: Euclidean norm
            let sum_of_squares: f64 = self
                .data
                .iter()
                .map(|x| {
                    let val = x.to_f64().unwrap_or(0.0);
                    val * val
                })
                .sum();
            sum_of_squares.sqrt()
        } else {
            // General p-norm
            let sum: f64 = self
                .data
                .iter()
                .map(|x| x.to_f64().unwrap_or(0.0).abs().powf(p))
                .sum();
            sum.powf(1.0 / p)
        }
    }

    /* // Commented out: Requires from_f64
    /// Normalize the vector to unit length (using 2-norm)
    /// Returns None if the vector has zero norm
    pub fn normalize(&self) -> Option<Vector<R>>
    where
        R: rustmath_core::NumericConversion + rustmath_core::Field,
    {
        let norm_val = self.norm(2.0);

        if norm_val == 0.0 {
            return None;
        }

        // Convert norm to R type
        let norm_r = R::from_f64(norm_val)?;

        Some(Vector {
            data: self
                .data
                .iter()
                .map(|x| x.clone() / norm_r.clone())
                .collect(),
        })
    }
    */

    /// Get inner reference to data
    pub fn data(&self) -> &[R] {
        &self.data
    }

    /// Convert to owned data
    pub fn into_data(self) -> Vec<R> {
        self.data
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

    #[test]
    fn test_cross_product() {
        let v1 = Vector::new(vec![1, 0, 0]);
        let v2 = Vector::new(vec![0, 1, 0]);

        let cross = v1.cross(&v2).unwrap();
        assert_eq!(*cross.get(0).unwrap(), 0);
        assert_eq!(*cross.get(1).unwrap(), 0);
        assert_eq!(*cross.get(2).unwrap(), 1); // i × j = k

        // Test another case
        let v3 = Vector::new(vec![1, 2, 3]);
        let v4 = Vector::new(vec![4, 5, 6]);
        let cross2 = v3.cross(&v4).unwrap();
        // [1,2,3] × [4,5,6] = [2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4] = [-3, 6, -3]
        assert_eq!(*cross2.get(0).unwrap(), -3);
        assert_eq!(*cross2.get(1).unwrap(), 6);
        assert_eq!(*cross2.get(2).unwrap(), -3);
    }

    #[test]
    fn test_norm() {
        use rustmath_rationals::Rational;

        let v = Vector::new(vec![
            Rational::from_integer(3),
            Rational::from_integer(4),
            Rational::from_integer(0),
        ]);

        // 2-norm (Euclidean): sqrt(9 + 16 + 0) = 5
        let norm2 = v.norm(2.0);
        assert!((norm2 - 5.0).abs() < 1e-10);

        // 1-norm: |3| + |4| + |0| = 7
        let norm1 = v.norm(1.0);
        assert!((norm1 - 7.0).abs() < 1e-10);

        // Infinity norm: max(|3|, |4|, |0|) = 4
        let norm_inf = v.norm(f64::INFINITY);
        assert!((norm_inf - 4.0).abs() < 1e-10);
    }

    /* // Commented out: normalize method is commented out
    #[test]
    fn test_normalize() {
        use rustmath_rationals::Rational;

        let v = Vector::new(vec![
            Rational::from_integer(3),
            Rational::from_integer(4),
            Rational::from_integer(0),
        ]);

        let normalized = v.normalize().unwrap();
        let norm = normalized.norm(2.0);
        assert!((norm - 1.0).abs() < 1e-10); // Should be unit vector
    }
    */
}
