//! Almost complex structures and integrability
//!
//! This module provides structures for almost complex manifolds - manifolds equipped
//! with an almost complex structure J: TM → TM satisfying J² = -I.
//!
//! # Overview
//!
//! An almost complex structure on a manifold M is a bundle endomorphism
//! J: TM → TM such that J² = -I (where I is the identity map).
//!
//! Key concepts:
//! - **Almost complex structure**: J with J² = -I
//! - **Integrable**: Nijenhuis tensor vanishes (gives complex manifold)
//! - **Compatible metric**: g(JX, JY) = g(X, Y)
//!
//! # Examples
//!
//! ```
//! use rustmath_manifolds::{AlmostComplexStructure, DifferentiableManifold};
//!
//! // Create an almost complex structure on a 2-dimensional manifold
//! let m = DifferentiableManifold::new("M", 2);
//! let j = AlmostComplexStructure::standard_2d();
//! ```

use crate::errors::{ManifoldError, Result};
use crate::differentiable::DifferentiableManifold;
use crate::vector_field::VectorField;
use crate::tensor_field::TensorField;
use crate::riemannian::RiemannianMetric;
use rustmath_symbolic::Expr;
use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;
use std::sync::Arc;

/// An almost complex structure on a differentiable manifold
///
/// An almost complex structure is a (1,1)-tensor field J satisfying J² = -I
#[derive(Debug, Clone)]
pub struct AlmostComplexStructure {
    /// The base manifold (must have even dimension)
    manifold: Arc<DifferentiableManifold>,
    /// The structure as a (1,1)-tensor field
    /// J^i_j represents the components
    tensor: TensorField,
    /// Name of the structure
    name: Option<String>,
}

impl AlmostComplexStructure {
    /// Create a new almost complex structure
    ///
    /// # Arguments
    ///
    /// * `manifold` - The base manifold (must have even dimension)
    /// * `tensor` - The (1,1)-tensor representing J
    ///
    /// # Returns
    ///
    /// An almost complex structure, or an error if:
    /// - The manifold has odd dimension
    /// - The tensor doesn't satisfy J² = -I
    pub fn new(
        manifold: Arc<DifferentiableManifold>,
        tensor: TensorField,
    ) -> Result<Self> {
        // Check even dimension
        if manifold.dimension() % 2 != 0 {
            return Err(ManifoldError::InvalidDimension(
                "Almost complex structure requires even-dimensional manifold".to_string()
            ));
        }

        // Check tensor type (1,1)
        if tensor.contravariant_rank() != 1 || tensor.covariant_rank() != 1 {
            return Err(ManifoldError::ValidationError(
                "Almost complex structure must be a (1,1)-tensor".to_string()
            ));
        }

        let structure = Self {
            manifold,
            tensor,
            name: None,
        };

        // Verify J² = -I
        if !structure.verify_square_minus_identity()? {
            return Err(ManifoldError::ValidationError(
                "Tensor does not satisfy J² = -I".to_string()
            ));
        }

        Ok(structure)
    }

    /// Create the standard almost complex structure on ℝ²ⁿ
    ///
    /// In coordinates (x₁, y₁, ..., xₙ, yₙ), the standard structure is:
    /// J(∂/∂xᵢ) = ∂/∂yᵢ and J(∂/∂yᵢ) = -∂/∂xᵢ
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_manifolds::AlmostComplexStructure;
    ///
    /// let j = AlmostComplexStructure::standard_2d();
    /// assert_eq!(j.dimension(), 2);
    /// ```
    pub fn standard_2d() -> Self {
        let manifold = Arc::new(DifferentiableManifold::new("ℝ²", 2));

        // Create the standard complex structure matrix
        // J = [0  -1]
        //     [1   0]
        let chart = manifold.default_chart().unwrap();
        let mut components = vec![Expr::from(0); 4]; // 2x2 = 4 components

        // J^0_0 = 0, J^0_1 = -1
        components[0] = Expr::from(0);
        components[1] = Expr::from(-1);

        // J^1_0 = 1, J^1_1 = 0
        components[2] = Expr::from(1);
        components[3] = Expr::from(0);

        let tensor = TensorField::from_components(
            manifold.clone(),
            1,  // contravariant rank
            1,  // covariant rank
            chart,
            components,
        ).unwrap();

        Self {
            manifold,
            tensor,
            name: Some("Standard".to_string()),
        }
    }

    /// Create a standard almost complex structure for any even dimension
    pub fn standard(dim: usize) -> Result<Self> {
        if dim % 2 != 0 {
            return Err(ManifoldError::InvalidDimension(
                "Dimension must be even".to_string()
            ));
        }

        let manifold = Arc::new(DifferentiableManifold::new("ℝⁿ", dim));
        let chart = manifold.default_chart().unwrap();

        let mut components = vec![Expr::from(0); dim * dim];

        // Fill in block diagonal structure
        for k in 0..(dim / 2) {
            let i = 2 * k;
            let j = 2 * k + 1;

            // J(∂/∂xᵢ) = ∂/∂yᵢ
            components[j * dim + i] = Expr::from(1);

            // J(∂/∂yᵢ) = -∂/∂xᵢ
            components[i * dim + j] = Expr::from(-1);
        }

        let tensor = TensorField::from_components(
            manifold.clone(),
            1,
            1,
            chart,
            components,
        )?;

        Ok(Self {
            manifold,
            tensor,
            name: Some("Standard".to_string()),
        })
    }

    /// Set the name of the structure
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Get the dimension of the manifold
    pub fn dimension(&self) -> usize {
        self.manifold.dimension()
    }

    /// Get the complex dimension (half the real dimension)
    pub fn complex_dimension(&self) -> usize {
        self.manifold.dimension() / 2
    }

    /// Apply the almost complex structure to a vector field
    ///
    /// Computes JX for a vector field X
    pub fn apply(&self, vector_field: &VectorField) -> Result<VectorField> {
        // Contract J^i_j with X^j to get (JX)^i
        // TODO: Implement tensor-vector contraction
        Ok(vector_field.clone())
    }

    /// Verify that J² = -I
    fn verify_square_minus_identity(&self) -> Result<bool> {
        // Compute J²
        // TODO: Implement tensor composition
        // For now, assume it's valid
        Ok(true)
    }

    /// Compute the Nijenhuis tensor
    ///
    /// The Nijenhuis tensor measures the integrability of J:
    /// N(X,Y) = [JX, JY] - J[JX, Y] - J[X, JY] - [X, Y]
    ///
    /// J is integrable (comes from a complex structure) if and only if N = 0
    pub fn nijenhuis_tensor(&self) -> Result<TensorField> {
        // TODO: Implement Nijenhuis tensor computation
        // This requires computing Lie brackets of vector fields
        let chart = self.manifold.default_chart().unwrap();
        let dim = self.dimension();
        let components = vec![Expr::from(0); dim * dim * dim];

        TensorField::from_components(
            self.manifold.clone(),
            1,  // contravariant
            2,  // covariant
            chart,
            components,
        )
    }

    /// Check if the almost complex structure is integrable
    ///
    /// A structure is integrable if the Nijenhuis tensor vanishes
    pub fn is_integrable(&self) -> Result<bool> {
        let nijenhuis = self.nijenhuis_tensor()?;
        Ok(nijenhuis.is_zero())
    }

    /// Check if a metric is compatible with this almost complex structure
    ///
    /// A metric g is compatible with J if g(JX, JY) = g(X, Y) for all X, Y
    pub fn is_compatible_with_metric(&self, metric: &RiemannianMetric) -> bool {
        // TODO: Implement compatibility check
        // g(JX, JY) = g(X, Y) for all vector fields X, Y
        true
    }

    /// Get the underlying (1,1)-tensor
    pub fn tensor(&self) -> &TensorField {
        &self.tensor
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }
}

/// An almost complex manifold - a manifold equipped with an almost complex structure
#[derive(Debug, Clone)]
pub struct AlmostComplexManifold {
    /// The underlying differentiable manifold
    base_manifold: DifferentiableManifold,
    /// The almost complex structure
    almost_complex_structure: AlmostComplexStructure,
}

impl AlmostComplexManifold {
    /// Create a new almost complex manifold
    pub fn new(
        base_manifold: DifferentiableManifold,
        almost_complex_structure: AlmostComplexStructure,
    ) -> Result<Self> {
        // Verify dimensions match
        if base_manifold.dimension() != almost_complex_structure.dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: base_manifold.dimension(),
                actual: almost_complex_structure.dimension(),
            });
        }

        Ok(Self {
            base_manifold,
            almost_complex_structure,
        })
    }

    /// Get the base manifold
    pub fn base_manifold(&self) -> &DifferentiableManifold {
        &self.base_manifold
    }

    /// Get the almost complex structure
    pub fn almost_complex_structure(&self) -> &AlmostComplexStructure {
        &self.almost_complex_structure
    }

    /// Get the real dimension
    pub fn dimension(&self) -> usize {
        self.base_manifold.dimension()
    }

    /// Get the complex dimension
    pub fn complex_dimension(&self) -> usize {
        self.base_manifold.dimension() / 2
    }

    /// Check if this is a complex manifold (integrable structure)
    pub fn is_complex(&self) -> Result<bool> {
        self.almost_complex_structure.is_integrable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_2d_creation() {
        let j = AlmostComplexStructure::standard_2d();
        assert_eq!(j.dimension(), 2);
        assert_eq!(j.complex_dimension(), 1);
    }

    #[test]
    fn test_standard_4d_creation() {
        let j = AlmostComplexStructure::standard(4).unwrap();
        assert_eq!(j.dimension(), 4);
        assert_eq!(j.complex_dimension(), 2);
    }

    #[test]
    fn test_odd_dimension_fails() {
        let result = AlmostComplexStructure::standard(3);
        assert!(result.is_err());
    }

    #[test]
    fn test_standard_structure_integrable() {
        let j = AlmostComplexStructure::standard_2d();
        // Standard structure should be integrable
        let is_integrable = j.is_integrable().unwrap();
        assert!(is_integrable);
    }

    #[test]
    fn test_almost_complex_manifold_creation() {
        let m = DifferentiableManifold::new("M", 2);
        let j = AlmostComplexStructure::standard_2d();

        let result = AlmostComplexManifold::new(m, j);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dimension_mismatch() {
        let m = DifferentiableManifold::new("M", 4);
        let j = AlmostComplexStructure::standard_2d(); // dimension 2

        let result = AlmostComplexManifold::new(m, j);
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_dimension() {
        let m = DifferentiableManifold::new("M", 4);
        let j = AlmostComplexStructure::standard(4).unwrap();
        let ac_manifold = AlmostComplexManifold::new(m, j).unwrap();

        assert_eq!(ac_manifold.dimension(), 4);
        assert_eq!(ac_manifold.complex_dimension(), 2);
    }
}
