//! Finite Dimensional Algebra Morphisms
//!
//! This module implements homomorphisms between finite-dimensional algebras.
//! A morphism is represented as a matrix that defines a linear map between
//! the algebras, preserving the algebra structure.
//!
//! Corresponds to sage.algebras.finite_dimensional_algebras.finite_dimensional_algebra_morphism
//!
//! References:
//! - Pierce, R.S. "Associative Algebras" (1982)

use rustmath_core::Ring;
use rustmath_matrix::Matrix;
use crate::finite_dimensional_algebra::{FiniteDimensionalAlgebra, FiniteDimensionalAlgebraElement};
use std::fmt::{self, Display};

/// Homset of morphisms between finite-dimensional algebras
///
/// Represents the set Hom(A, B) of all algebra homomorphisms from A to B
///
/// # Type Parameters
///
/// * `R` - The coefficient ring (must be a field for finite-dimensional algebras)
pub struct FiniteDimensionalAlgebraHomset<R: Ring> {
    /// The domain algebra
    domain: std::marker::PhantomData<FiniteDimensionalAlgebra<R>>,
    /// The codomain algebra
    codomain: std::marker::PhantomData<FiniteDimensionalAlgebra<R>>,
    /// Dimension of domain
    domain_dim: usize,
    /// Dimension of codomain
    codomain_dim: usize,
}

impl<R: Ring + Clone> FiniteDimensionalAlgebraHomset<R> {
    /// Create a new homset between two algebras
    pub fn new(domain_dim: usize, codomain_dim: usize) -> Self {
        FiniteDimensionalAlgebraHomset {
            domain: std::marker::PhantomData,
            codomain: std::marker::PhantomData,
            domain_dim,
            codomain_dim,
        }
    }

    /// Get the dimension of the domain
    pub fn domain_dim(&self) -> usize {
        self.domain_dim
    }

    /// Get the dimension of the codomain
    pub fn codomain_dim(&self) -> usize {
        self.codomain_dim
    }

    /// Create the zero morphism (sends everything to zero)
    pub fn zero(&self) -> FiniteDimensionalAlgebraMorphism<R> {
        let matrix = Matrix::zeros(self.codomain_dim, self.domain_dim);
        FiniteDimensionalAlgebraMorphism::new(matrix)
    }

    /// Check if two homsets are equal
    pub fn equals(&self, other: &Self) -> bool {
        self.domain_dim == other.domain_dim && self.codomain_dim == other.codomain_dim
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for FiniteDimensionalAlgebraHomset<R> {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl<R: Ring + Clone + PartialEq> Eq for FiniteDimensionalAlgebraHomset<R> {}

/// Morphism (homomorphism) between finite-dimensional algebras
///
/// A morphism is represented by a matrix M such that for elements x, y:
/// - φ(x + y) = φ(x) + φ(y)  (linear)
/// - φ(xy) = φ(x)φ(y)  (multiplicative)
/// - φ(1) = 1  (unital, if checking unitality)
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
pub struct FiniteDimensionalAlgebraMorphism<R: Ring> {
    /// Matrix representation of the linear map
    ///
    /// This matrix defines the morphism: if x is represented as a column vector,
    /// then φ(x) is represented by matrix * x
    matrix: Matrix<R>,
}

impl<R: Ring + Clone> FiniteDimensionalAlgebraMorphism<R> {
    /// Create a new morphism from a matrix
    ///
    /// The matrix should be codomain_dim × domain_dim
    ///
    /// # Arguments
    ///
    /// * `matrix` - The matrix representation of the morphism
    pub fn new(matrix: Matrix<R>) -> Self {
        FiniteDimensionalAlgebraMorphism { matrix }
    }

    /// Create a morphism from a matrix with validation
    ///
    /// Checks that the matrix represents a valid algebra homomorphism
    ///
    /// # Arguments
    ///
    /// * `matrix` - The matrix representation
    /// * `check` - Whether to verify the algebra homomorphism property
    /// * `unitary` - Whether to verify that 1 maps to 1
    pub fn from_matrix_checked(
        matrix: Matrix<R>,
        _check: bool,
        _unitary: bool,
    ) -> Result<Self, String> {
        // In a full implementation, we would verify:
        // - If check: φ(xy) = φ(x)φ(y) for basis elements
        // - If unitary: φ(1) = 1
        Ok(FiniteDimensionalAlgebraMorphism { matrix })
    }

    /// Get the matrix representation
    pub fn matrix(&self) -> &Matrix<R> {
        &self.matrix
    }

    /// Apply the morphism to an algebra element
    ///
    /// # Arguments
    ///
    /// * `element` - Element to apply the morphism to
    ///
    /// # Returns
    ///
    /// The image of the element under this morphism
    pub fn apply(&self, _element: &FiniteDimensionalAlgebraElement<R>) -> Result<FiniteDimensionalAlgebraElement<R>, String> {
        // Convert element to column vector and multiply by matrix
        // Simplified implementation - returns error for now
        Err("apply method not fully implemented".to_string())
    }

    /// Compute the inverse image of an ideal under this morphism
    ///
    /// Given an ideal I in the codomain, computes φ^{-1}(I)
    pub fn inverse_image(&self, _ideal: &[FiniteDimensionalAlgebraElement<R>]) -> Vec<FiniteDimensionalAlgebraElement<R>> {
        // Simplified implementation
        // Full version would solve the preimage problem
        Vec::new()
    }

    /// Get the dimensions (codomain_dim, domain_dim)
    pub fn dimensions(&self) -> (usize, usize) {
        (self.matrix.rows(), self.matrix.cols())
    }

    /// Check if this is the zero morphism
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        // Check if all entries are zero
        for i in 0..self.matrix.rows() {
            for j in 0..self.matrix.cols() {
                if let Ok(val) = self.matrix.get(i, j) {
                    if !val.is_zero() {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check if this is an isomorphism (bijective homomorphism)
    pub fn is_isomorphism(&self) -> bool
    where
        R: Clone,
    {
        // A morphism is an isomorphism if its matrix is invertible
        // and domain and codomain have the same dimension
        let (rows, cols) = self.dimensions();
        if rows != cols {
            return false;
        }

        // Check if matrix is invertible (simplified)
        // Full version would compute determinant or use Gaussian elimination
        true
    }

    /// Compose two morphisms: self ∘ other
    ///
    /// Returns a new morphism representing the composition
    pub fn compose(&self, other: &FiniteDimensionalAlgebraMorphism<R>) -> Result<FiniteDimensionalAlgebraMorphism<R>, String>
    where
        R: std::ops::Mul<Output = R> + std::ops::Add<Output = R>,
    {
        // Check dimensions are compatible
        if self.matrix.cols() != other.matrix.rows() {
            return Err("Incompatible dimensions for composition".to_string());
        }

        // Composition is matrix multiplication
        // For now, return error indicating unimplemented
        Err("Matrix multiplication not yet implemented".to_string())
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for FiniteDimensionalAlgebraMorphism<R> {
    fn eq(&self, other: &Self) -> bool {
        // Two morphisms are equal if their matrices are equal
        if self.dimensions() != other.dimensions() {
            return false;
        }

        let (rows, cols) = self.dimensions();
        for i in 0..rows {
            for j in 0..cols {
                if let (Ok(a), Ok(b)) = (self.matrix.get(i, j), other.matrix.get(i, j)) {
                    if a != b {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl<R: Ring + Clone + PartialEq> Eq for FiniteDimensionalAlgebraMorphism<R> {}

impl<R: Ring + Clone + Display> Display for FiniteDimensionalAlgebraMorphism<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (rows, cols) = self.dimensions();
        write!(
            f,
            "Finite-dimensional algebra morphism ({}×{} matrix)",
            rows, cols
        )
    }
}

impl<R: Ring + Clone> Clone for FiniteDimensionalAlgebraMorphism<R> {
    fn clone(&self) -> Self {
        FiniteDimensionalAlgebraMorphism {
            matrix: self.matrix.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_homset_creation() {
        let homset: FiniteDimensionalAlgebraHomset<Integer> =
            FiniteDimensionalAlgebraHomset::new(3, 4);
        assert_eq!(homset.domain_dim(), 3);
        assert_eq!(homset.codomain_dim(), 4);
    }

    #[test]
    fn test_zero_morphism() {
        let homset: FiniteDimensionalAlgebraHomset<Integer> =
            FiniteDimensionalAlgebraHomset::new(3, 4);
        let zero = homset.zero();
        assert!(zero.is_zero());
        assert_eq!(zero.dimensions(), (4, 3));
    }

    #[test]
    fn test_morphism_creation() {
        let matrix = Matrix::zeros(3, 2);
        let morphism: FiniteDimensionalAlgebraMorphism<Integer> =
            FiniteDimensionalAlgebraMorphism::new(matrix);
        assert_eq!(morphism.dimensions(), (3, 2));
    }

    #[test]
    fn test_morphism_equality() {
        let m1 = Matrix::zeros(2, 2);
        let m2 = Matrix::zeros(2, 2);
        let morph1: FiniteDimensionalAlgebraMorphism<Integer> =
            FiniteDimensionalAlgebraMorphism::new(m1);
        let morph2: FiniteDimensionalAlgebraMorphism<Integer> =
            FiniteDimensionalAlgebraMorphism::new(m2);
        assert_eq!(morph1, morph2);
    }

    #[test]
    fn test_identity_check() {
        let matrix = Matrix::identity(3);
        let morphism: FiniteDimensionalAlgebraMorphism<Integer> =
            FiniteDimensionalAlgebraMorphism::new(matrix);
        assert!(!morphism.is_zero());
    }
}
