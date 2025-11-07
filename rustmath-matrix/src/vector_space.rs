//! Vector space structures and operations

use crate::{Matrix, Vector};
use rustmath_core::{Field, MathError, Result};

/// A vector space over a field F
///
/// Represents a finite-dimensional vector space with a basis.
pub struct VectorSpace<F: Field> {
    /// The field over which this vector space is defined
    _field: std::marker::PhantomData<F>,
    /// Dimension of the vector space
    dimension: usize,
    /// Basis vectors (each vector is a column)
    basis: Vec<Vec<F>>,
}

impl<F: Field> VectorSpace<F> {
    /// Create a vector space with given dimension and standard basis
    ///
    /// The standard basis for F^n is {e_1, e_2, ..., e_n} where e_i has 1 in position i.
    pub fn new(dimension: usize) -> Self {
        let mut basis = Vec::with_capacity(dimension);

        for i in 0..dimension {
            let mut v = vec![F::zero(); dimension];
            v[i] = F::one();
            basis.push(v);
        }

        VectorSpace {
            _field: std::marker::PhantomData,
            dimension,
            basis,
        }
    }

    /// Create a vector space from a given basis
    ///
    /// The basis vectors should be linearly independent.
    /// This method doesn't verify independence - use with caution.
    pub fn from_basis(basis: Vec<Vec<F>>) -> Result<Self> {
        if basis.is_empty() {
            return Err(MathError::InvalidArgument(
                "Basis cannot be empty".to_string(),
            ));
        }

        let dimension = basis.len();
        let vec_dim = basis[0].len();

        // Check all basis vectors have same dimension
        for v in &basis {
            if v.len() != vec_dim {
                return Err(MathError::InvalidArgument(
                    "All basis vectors must have same dimension".to_string(),
                ));
            }
        }

        Ok(VectorSpace {
            _field: std::marker::PhantomData,
            dimension,
            basis,
        })
    }

    /// Get the dimension of this vector space
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get a reference to the basis vectors
    pub fn basis(&self) -> &[Vec<F>] {
        &self.basis
    }

    /// Express a vector in terms of the basis (find coordinates)
    ///
    /// Given a vector v, find coefficients c_i such that v = Σ c_i * b_i
    /// where b_i are the basis vectors.
    pub fn coordinates(&self, v: &[F]) -> Result<Vec<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        if v.len() != self.basis[0].len() {
            return Err(MathError::InvalidArgument(
                "Vector dimension doesn't match vector space ambient dimension".to_string(),
            ));
        }

        // Build matrix with basis vectors as columns
        let ambient_dim = self.basis[0].len();
        let mut data = Vec::with_capacity(ambient_dim * self.dimension);

        for i in 0..ambient_dim {
            for j in 0..self.dimension {
                data.push(self.basis[j][i].clone());
            }
        }

        let basis_matrix = Matrix::from_vec(ambient_dim, self.dimension, data)?;

        // Solve basis_matrix * c = v for c
        let solution = basis_matrix.solve(v)?;

        solution.ok_or_else(|| {
            MathError::InvalidArgument("Vector is not in the span of the basis".to_string())
        })
    }

    /// Compute the direct sum with another vector space
    ///
    /// The direct sum V ⊕ W has dimension dim(V) + dim(W) and
    /// basis {v_1, ..., v_n, w_1, ..., w_m} where v_i ∈ V and w_j ∈ W.
    pub fn direct_sum(&self, other: &Self) -> Self {
        let new_dim = self.dimension + other.dimension;
        let ambient_dim = self.basis[0].len();
        let other_ambient_dim = other.basis[0].len();

        let combined_ambient_dim = ambient_dim + other_ambient_dim;

        let mut new_basis = Vec::with_capacity(new_dim);

        // Add vectors from first space (padded with zeros)
        for v in &self.basis {
            let mut padded = v.clone();
            padded.extend(vec![F::zero(); other_ambient_dim]);
            new_basis.push(padded);
        }

        // Add vectors from second space (padded with zeros at start)
        for w in &other.basis {
            let mut padded = vec![F::zero(); ambient_dim];
            padded.extend(w.clone());
            new_basis.push(padded);
        }

        VectorSpace {
            _field: std::marker::PhantomData,
            dimension: new_dim,
            basis: new_basis,
        }
    }

    /// Check if a vector is in this vector space
    pub fn contains(&self, v: &[F]) -> bool
    where
        F: rustmath_core::NumericConversion,
    {
        self.coordinates(v).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_vector_space_creation() {
        let v: VectorSpace<Rational> = VectorSpace::new(3);
        assert_eq!(v.dimension(), 3);
        assert_eq!(v.basis().len(), 3);
    }

    #[test]
    fn test_standard_basis() {
        let v: VectorSpace<Rational> = VectorSpace::new(2);
        let basis = v.basis();

        // e_1 = [1, 0]
        assert_eq!(basis[0][0], Rational::from(1));
        assert_eq!(basis[0][1], Rational::from(0));

        // e_2 = [0, 1]
        assert_eq!(basis[1][0], Rational::from(0));
        assert_eq!(basis[1][1], Rational::from(1));
    }

    #[test]
    fn test_coordinates() {
        let v: VectorSpace<Rational> = VectorSpace::new(2);

        // Vector [3, 4] should have coordinates [3, 4] in standard basis
        let vec = vec![Rational::from(3), Rational::from(4)];
        let coords = v.coordinates(&vec).unwrap();

        assert_eq!(coords[0], Rational::from(3));
        assert_eq!(coords[1], Rational::from(4));
    }

    #[test]
    fn test_direct_sum() {
        let v1: VectorSpace<Rational> = VectorSpace::new(2);
        let v2: VectorSpace<Rational> = VectorSpace::new(3);

        let sum = v1.direct_sum(&v2);

        assert_eq!(sum.dimension(), 5);
        assert_eq!(sum.basis()[0].len(), 5);
    }

    #[test]
    fn test_contains() {
        let v: VectorSpace<Rational> = VectorSpace::new(2);

        // Any 2D vector should be in R^2
        let vec = vec![Rational::from(5), Rational::from(7)];
        assert!(v.contains(&vec));
    }
}
