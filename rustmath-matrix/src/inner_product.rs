//! Inner product spaces
//!
//! This module provides structures and operations for inner product spaces,
//! which are vector spaces equipped with an inner product (bilinear form).

use crate::{Matrix, Vector};
use rustmath_core::{Field, MathError, Result};

/// An inner product space is a vector space with an inner product
///
/// The inner product is a bilinear map <·,·>: V × V → F satisfying:
/// - Linearity in first argument: <au + bv, w> = a<u,w> + b<v,w>
/// - Conjugate symmetry: <u,v> = conj(<v,u>) (for real fields, <u,v> = <v,u>)
/// - Positive-definiteness: <v,v> ≥ 0 and <v,v> = 0 iff v = 0
pub struct InnerProductSpace<F: Field> {
    /// Dimension of the space
    dimension: usize,
    /// The Gram matrix defining the inner product
    /// <u,v> = u^T G v where G is the Gram matrix
    gram_matrix: Matrix<F>,
}

impl<F: Field> InnerProductSpace<F> {
    /// Create an inner product space with the standard (Euclidean) inner product
    ///
    /// The standard inner product is <u,v> = u₁v₁ + u₂v₂ + ... + uₙvₙ
    /// This corresponds to the identity Gram matrix.
    pub fn euclidean(dimension: usize) -> Self {
        InnerProductSpace {
            dimension,
            gram_matrix: Matrix::identity(dimension),
        }
    }

    /// Create an inner product space with a custom Gram matrix
    ///
    /// The Gram matrix G defines the inner product as <u,v> = u^T G v.
    /// For a valid inner product, G should be symmetric and positive-definite.
    pub fn from_gram_matrix(gram_matrix: Matrix<F>) -> Result<Self> {
        if !gram_matrix.is_square() {
            return Err(MathError::InvalidArgument(
                "Gram matrix must be square".to_string(),
            ));
        }

        Ok(InnerProductSpace {
            dimension: gram_matrix.rows(),
            gram_matrix,
        })
    }

    /// Get the dimension of this inner product space
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Compute the inner product of two vectors
    ///
    /// <u,v> = u^T G v where G is the Gram matrix
    pub fn inner_product(&self, u: &[F], v: &[F]) -> Result<F> {
        if u.len() != self.dimension || v.len() != self.dimension {
            return Err(MathError::InvalidArgument(
                "Vector dimensions don't match inner product space dimension".to_string(),
            ));
        }

        // Compute G * v
        let mut gv = vec![F::zero(); self.dimension];
        for i in 0..self.dimension {
            let mut sum = F::zero();
            for j in 0..self.dimension {
                sum = sum + self.gram_matrix.data[i * self.dimension + j].clone() * v[j].clone();
            }
            gv[i] = sum;
        }

        // Compute u^T * (G * v)
        let mut result = F::zero();
        for i in 0..self.dimension {
            result = result + u[i].clone() * gv[i].clone();
        }

        Ok(result)
    }

    /// Compute the norm (length) of a vector
    ///
    /// ||v|| = sqrt(<v,v>)
    ///
    /// Note: This returns the square of the norm since we're working over
    /// general fields that may not have square roots.
    pub fn norm_squared(&self, v: &[F]) -> Result<F> {
        self.inner_product(v, v)
    }

    /// Check if two vectors are orthogonal
    ///
    /// Vectors u and v are orthogonal if <u,v> = 0
    pub fn are_orthogonal(&self, u: &[F], v: &[F]) -> Result<bool> {
        let ip = self.inner_product(u, v)?;
        Ok(ip.is_zero())
    }

    /// Gram-Schmidt orthogonalization process
    ///
    /// Given a list of linearly independent vectors, returns an orthogonal
    /// (or orthonormal) basis for the same subspace.
    ///
    /// Note: For fields without division (like integers), this may not work.
    pub fn gram_schmidt(&self, vectors: &[Vec<F>]) -> Result<Vec<Vec<F>>>
    where
        F: rustmath_core::NumericConversion,
    {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let mut orthogonal_basis = Vec::new();

        for v in vectors {
            if v.len() != self.dimension {
                return Err(MathError::InvalidArgument(
                    "All vectors must have the same dimension".to_string(),
                ));
            }

            // Start with v
            let mut u = v.clone();

            // Subtract projections onto previously computed orthogonal vectors
            for prev in &orthogonal_basis {
                // proj_prev(v) = <v, prev> / <prev, prev> * prev
                let numerator = self.inner_product(v, prev)?;
                let denominator = self.inner_product(prev, prev)?;

                if !denominator.is_zero() {
                    let coefficient = numerator / denominator;

                    // u = u - coefficient * prev
                    for i in 0..self.dimension {
                        u[i] = u[i].clone() - coefficient.clone() * prev[i].clone();
                    }
                }
            }

            // Check if u is non-zero (vectors must be linearly independent)
            let norm_sq = self.norm_squared(&u)?;
            if norm_sq.is_zero() {
                return Err(MathError::InvalidArgument(
                    "Input vectors are not linearly independent".to_string(),
                ));
            }

            orthogonal_basis.push(u);
        }

        Ok(orthogonal_basis)
    }

    /// Orthonormal basis via Gram-Schmidt
    ///
    /// Returns an orthonormal basis (each vector has norm 1 and they are mutually orthogonal)
    pub fn orthonormalize(&self, vectors: &[Vec<F>]) -> Result<Vec<Vec<F>>>
    where
        F: rustmath_core::NumericConversion,
    {
        let orthogonal = self.gram_schmidt(vectors)?;

        let mut orthonormal = Vec::new();

        for v in orthogonal {
            let norm_sq = self.norm_squared(&v)?;

            // For true orthonormalization, we'd need sqrt
            // For now, we return vectors with norm² = 1 scaling
            // In a field with square roots, we'd compute norm = sqrt(norm_sq)

            // Check if we can normalize
            if let Some(norm_sq_f64) = norm_sq.to_f64() {
                if norm_sq_f64 > 0.0 {
                    if let Some(norm_val) = F::from_f64(norm_sq_f64.sqrt()) {
                        let mut normalized = Vec::with_capacity(self.dimension);
                        for val in v {
                            normalized.push(val / norm_val.clone());
                        }
                        orthonormal.push(normalized);
                        continue;
                    }
                }
            }

            // If we can't compute square root, just use the orthogonal vector
            orthonormal.push(v);
        }

        Ok(orthonormal)
    }

    /// Project a vector onto a subspace
    ///
    /// Given a vector v and a basis for a subspace, compute the projection
    /// of v onto that subspace.
    pub fn project(&self, v: &[F], subspace_basis: &[Vec<F>]) -> Result<Vec<F>>
    where
        F: rustmath_core::NumericConversion,
    {
        // First orthogonalize the subspace basis
        let orthogonal_basis = self.gram_schmidt(subspace_basis)?;

        // Initialize projection as zero vector
        let mut projection = vec![F::zero(); self.dimension];

        // Sum up projections onto each basis vector
        for basis_vec in &orthogonal_basis {
            let numerator = self.inner_product(v, basis_vec)?;
            let denominator = self.inner_product(basis_vec, basis_vec)?;

            if !denominator.is_zero() {
                let coefficient = numerator / denominator;

                for i in 0..self.dimension {
                    projection[i] =
                        projection[i].clone() + coefficient.clone() * basis_vec[i].clone();
                }
            }
        }

        Ok(projection)
    }

    /// Compute the angle between two vectors (cosine of angle)
    ///
    /// cos(θ) = <u,v> / (||u|| ||v||)
    ///
    /// Returns a value that can be converted to f64 for interpretation
    pub fn cosine_angle(&self, u: &[F], v: &[F]) -> Result<F>
    where
        F: rustmath_core::NumericConversion,
    {
        let inner_prod = self.inner_product(u, v)?;
        let norm_u_sq = self.norm_squared(u)?;
        let norm_v_sq = self.norm_squared(v)?;

        // cos(θ) = <u,v> / sqrt(||u||² ||v||²)
        let denominator_sq = norm_u_sq * norm_v_sq;

        if let Some(denom_f64) = denominator_sq.to_f64() {
            if denom_f64 > 0.0 {
                if let Some(denom) = F::from_f64(denom_f64.sqrt()) {
                    return Ok(inner_prod / denom);
                }
            }
        }

        Err(MathError::InvalidArgument(
            "Cannot compute angle for zero vectors or in this field".to_string(),
        ))
    }

    /// Get a reference to the Gram matrix
    pub fn gram_matrix(&self) -> &Matrix<F> {
        &self.gram_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_euclidean_inner_product() {
        let space: InnerProductSpace<Rational> = InnerProductSpace::euclidean(3);

        let u = vec![Rational::from(1), Rational::from(2), Rational::from(3)];
        let v = vec![Rational::from(4), Rational::from(5), Rational::from(6)];

        // <u,v> = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let ip = space.inner_product(&u, &v).unwrap();
        assert_eq!(ip, Rational::from(32));
    }

    #[test]
    fn test_norm_squared() {
        let space: InnerProductSpace<Rational> = InnerProductSpace::euclidean(2);

        let v = vec![Rational::from(3), Rational::from(4)];

        // ||v||² = 3² + 4² = 9 + 16 = 25
        let norm_sq = space.norm_squared(&v).unwrap();
        assert_eq!(norm_sq, Rational::from(25));
    }

    #[test]
    fn test_orthogonal() {
        let space: InnerProductSpace<Rational> = InnerProductSpace::euclidean(2);

        let u = vec![Rational::from(1), Rational::from(0)];
        let v = vec![Rational::from(0), Rational::from(1)];

        assert!(space.are_orthogonal(&u, &v).unwrap());
    }

    #[test]
    fn test_gram_schmidt() {
        let space: InnerProductSpace<Rational> = InnerProductSpace::euclidean(2);

        // Start with non-orthogonal vectors
        let v1 = vec![Rational::from(1), Rational::from(1)];
        let v2 = vec![Rational::from(1), Rational::from(0)];

        let orthogonal = space.gram_schmidt(&[v1, v2]).unwrap();

        // Check that result vectors are orthogonal
        assert_eq!(orthogonal.len(), 2);

        let ip = space.inner_product(&orthogonal[0], &orthogonal[1]).unwrap();
        assert_eq!(ip, Rational::from(0));
    }

    #[test]
    fn test_custom_gram_matrix() {
        // Create inner product space with custom Gram matrix
        // G = [2 1]
        //     [1 2]
        let gram = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from(2),
                Rational::from(1),
                Rational::from(1),
                Rational::from(2),
            ],
        )
        .unwrap();

        let space = InnerProductSpace::from_gram_matrix(gram).unwrap();

        let u = vec![Rational::from(1), Rational::from(0)];
        let v = vec![Rational::from(0), Rational::from(1)];

        // <u,v> = u^T G v = [1 0] [2 1] [0] = [1 0] [1] = 1
        //                           [1 2] [1]         [2]
        let ip = space.inner_product(&u, &v).unwrap();
        assert_eq!(ip, Rational::from(1));
    }
}
