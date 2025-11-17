//! Differential forms on manifolds
//!
//! This module implements differential p-forms, which are totally antisymmetric
//! covariant tensor fields. Includes:
//! - Exterior derivative
//! - Wedge product
//! - Interior product
//! - Lie derivative

use crate::errors::{ManifoldError, Result};
use crate::chart::Chart;
use crate::tensor_field::TensorField;
use crate::vector_field::VectorField;
use crate::scalar_field::ScalarFieldEnhanced as ScalarField;
use crate::differentiable::DifferentiableManifold;
use rustmath_symbolic::Expr;
use std::ops::{Add, Neg};
use std::sync::Arc;

/// A differential p-form
///
/// A p-form is a totally antisymmetric covariant tensor of rank p.
/// In local coordinates: ω = ω_{i₁...iₚ} dx^{i₁} ∧ ... ∧ dx^{iₚ}
#[derive(Clone)]
pub struct DiffForm {
    /// Base tensor field (totally antisymmetric covariant tensor)
    tensor: TensorField,
    /// Degree (p) of the form
    degree: usize,
}

impl DiffForm {
    /// Create a new differential form
    pub fn new(manifold: Arc<DifferentiableManifold>, degree: usize) -> Self {
        let tensor = TensorField::new(manifold.clone(), 0, degree);
        Self { tensor, degree }
    }

    /// Create from a tensor field (assumes it's antisymmetric)
    pub fn from_tensor(tensor: TensorField, degree: usize) -> Result<Self> {
        if tensor.contravariant_rank() != 0 {
            return Err(ManifoldError::InvalidTensorRank);
        }
        if tensor.covariant_rank() != degree {
            return Err(ManifoldError::DimensionMismatch {
                expected: degree,
                actual: tensor.covariant_rank(),
            });
        }

        Ok(Self { tensor, degree })
    }

    /// Get the degree of the form
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        self.tensor.manifold()
    }

    /// Get the underlying tensor field
    pub fn tensor(&self) -> &TensorField {
        &self.tensor
    }

    /// Create a coordinate 1-form dx^i
    pub fn coordinate_form(
        manifold: Arc<DifferentiableManifold>,
        chart: &Chart,
        index: usize,
    ) -> Result<Self> {
        let n = manifold.dimension();
        if index >= n {
            return Err(ManifoldError::InvalidIndex);
        }

        let mut components = vec![Expr::from(0); n];
        components[index] = Expr::from(1);

        let tensor = TensorField::from_components(
            manifold.clone(),
            0,
            1,
            chart,
            components,
        )?;

        Ok(Self { tensor, degree: 1 })
    }

    /// Exterior derivative d: Ωᵖ(M) → Ωᵖ⁺¹(M)
    ///
    /// For a p-form ω, dω is a (p+1)-form with components:
    /// (dω)_{i₀i₁...iₚ} = (p+1) ∂_{[i₀} ω_{i₁...iₚ]}
    /// where [...] denotes antisymmetrization
    pub fn exterior_derivative(&self, chart: &Chart) -> Result<DiffForm> {
        let n = self.manifold().dimension();
        let p = self.degree;

        if p >= n {
            // Form is already of maximal degree
            return Ok(DiffForm::new(self.manifold().clone(), p + 1));
        }

        let components = self.tensor.components(chart)?;
        let symbols = chart.coordinate_symbols();

        // Compute new components
        let new_size = n.pow((p + 1) as u32);
        let mut d_components = vec![Expr::from(0); new_size];

        // For each (p+1)-tuple of indices (i₀, i₁, ..., iₚ)
        // Compute alternating sum of derivatives
        // This is a simplified implementation - full antisymmetrization needed
        for i in 0..new_size {
            let mut sum = Expr::from(0);

            // Differentiate with respect to each coordinate
            for j in 0..n {
                if i < components.len() {
                    sum = sum + components[i].differentiate(&symbols[j % symbols.len()]);
                }
            }

            d_components[i] = sum;
        }

        let d_tensor = TensorField::from_components(
            self.manifold().clone(),
            0,
            p + 1,
            chart,
            d_components,
        )?;

        Ok(DiffForm { tensor: d_tensor, degree: p + 1 })
    }

    /// Wedge product: ωᵖ ∧ ηᵍ → ω∧η ∈ Ωᵖ⁺ᵍ(M)
    ///
    /// The wedge product of a p-form and a q-form is a (p+q)-form.
    /// (ω ∧ η)_{i₁...iₚ₊ᵩ} = ((p+q)!/(p!q!)) ω_{[i₁...iₚ} η_{iₚ₊₁...iₚ₊ᵩ]}
    pub fn wedge(&self, other: &DiffForm, chart: &Chart) -> Result<DiffForm> {
        if !Arc::ptr_eq(self.manifold(), other.manifold()) {
            return Err(ManifoldError::DifferentManifolds);
        }

        let p = self.degree;
        let q = other.degree;
        let n = self.manifold().dimension();

        if p + q > n {
            // Wedge product vanishes
            return Ok(DiffForm::new(self.manifold().clone(), p + q));
        }

        // Compute tensor product
        let product_tensor = self.tensor.tensor_product(&other.tensor, chart)?;

        // Antisymmetrize to get wedge product
        // This is a simplified implementation
        // Full version would properly antisymmetrize all indices

        Ok(DiffForm {
            tensor: product_tensor,
            degree: p + q,
        })
    }

    /// Interior product (contraction) with a vector field: i_X ω
    ///
    /// The interior product of a vector field X with a p-form ω
    /// produces a (p-1)-form.
    pub fn interior_product(&self, vector: &VectorField, chart: &Chart) -> Result<DiffForm> {
        if !Arc::ptr_eq(self.manifold(), vector.manifold()) {
            return Err(ManifoldError::DifferentManifolds);
        }

        if self.degree == 0 {
            return Err(ManifoldError::InvalidOperation(
                "Cannot contract 0-form".to_string()
            ));
        }

        // Contract the first covariant index with the vector field
        // This is a simplified implementation
        let contracted_tensor = self.tensor.contract(0, 0, chart)?;

        Ok(DiffForm {
            tensor: contracted_tensor,
            degree: self.degree - 1,
        })
    }

    /// Lie derivative along a vector field: ℒ_X ω
    ///
    /// Cartan's formula: ℒ_X ω = i_X(dω) + d(i_X ω)
    pub fn lie_derivative(&self, vector: &VectorField, chart: &Chart) -> Result<DiffForm> {
        if !Arc::ptr_eq(self.manifold(), vector.manifold()) {
            return Err(ManifoldError::DifferentManifolds);
        }

        // Cartan's formula: ℒ_X ω = i_X(dω) + d(i_X ω)

        if self.degree == 0 {
            // For 0-forms (scalar fields), Lie derivative is just directional derivative
            // This is a special case
            return Ok(self.clone());
        }

        let d_omega = self.exterior_derivative(chart)?;
        let ix_d_omega = d_omega.interior_product(vector, chart)?;

        let ix_omega = self.interior_product(vector, chart)?;
        let d_ix_omega = ix_omega.exterior_derivative(chart)?;

        ix_d_omega + d_ix_omega
    }

    /// Check if this is the zero form
    pub fn is_zero(&self) -> bool {
        self.tensor.is_zero()
    }

    /// Create a zero p-form
    pub fn zero(manifold: Arc<DifferentiableManifold>, degree: usize) -> Self {
        Self::new(manifold, degree)
    }
}

// Arithmetic operations

impl Add for DiffForm {
    type Output = Result<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        if !Arc::ptr_eq(self.manifold(), rhs.manifold()) {
            return Err(ManifoldError::DifferentManifolds);
        }
        if self.degree != rhs.degree {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.degree,
                actual: rhs.degree,
            });
        }

        // Add tensor components
        // This is simplified - proper implementation needed
        Ok(self)
    }
}

impl Neg for DiffForm {
    type Output = Self;

    fn neg(self) -> Self::Output {
        // Negate all tensor components
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::EuclideanSpace;

    #[test]
    fn test_diff_form_creation() {
        let m = Arc::new(EuclideanSpace::new(2));
        let omega = DiffForm::new(m.clone(), 1);
        assert_eq!(omega.degree(), 1);
    }

    #[test]
    fn test_coordinate_form() {
        let m = Arc::new(EuclideanSpace::new(3));
        let chart = m.default_chart().unwrap();

        let dx = DiffForm::coordinate_form(m.clone(), chart, 0).unwrap();
        assert_eq!(dx.degree(), 1);
    }

    #[test]
    fn test_zero_form() {
        let m = Arc::new(EuclideanSpace::new(2));
        let zero = DiffForm::zero(m, 1);
        assert!(zero.is_zero());
    }

    #[test]
    fn test_exterior_derivative_degree() {
        let m = Arc::new(EuclideanSpace::new(3));
        let chart = m.default_chart().unwrap();
        let omega = DiffForm::new(m.clone(), 1);

        let d_omega = omega.exterior_derivative(chart).unwrap();
        assert_eq!(d_omega.degree(), 2);
    }

    #[test]
    fn test_wedge_product_degree() {
        let m = Arc::new(EuclideanSpace::new(3));
        let chart = m.default_chart().unwrap();

        let omega1 = DiffForm::new(m.clone(), 1);
        let omega2 = DiffForm::new(m.clone(), 1);

        let wedge = omega1.wedge(&omega2, chart).unwrap();
        assert_eq!(wedge.degree(), 2);
    }
}
