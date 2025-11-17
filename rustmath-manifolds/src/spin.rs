//! Spin structures and spinor bundles
//!
//! This module provides structures for spin geometry - the geometry of spinors
//! on Riemannian manifolds. Spin structures allow us to define spinor fields,
//! which are fundamental in physics (fermions in quantum field theory).
//!
//! # Overview
//!
//! A spin structure on an oriented Riemannian manifold M is a lift of the
//! oriented orthonormal frame bundle SO(M) to the spin group Spin(n).
//!
//! Key concepts:
//! - **Spin group**: The double cover of SO(n), denoted Spin(n)
//! - **Spin structure**: A principal Spin(n)-bundle covering SO(M)
//! - **Spinor bundle**: Associated vector bundle for spinor representations
//! - **Dirac operator**: First-order differential operator on spinors
//!
//! # Existence
//!
//! Not all manifolds admit spin structures. A necessary and sufficient condition
//! is that the second Stiefel-Whitney class w₂(M) vanishes.
//!
//! # Examples
//!
//! ```
//! use rustmath_manifolds::{SpinStructure, RiemannianMetric};
//!
//! // Some manifolds that admit spin structures:
//! // - All oriented surfaces (2D manifolds)
//! // - S^n for all n
//! // - Any oriented 3-manifold
//! ```

use crate::errors::{ManifoldError, Result};
use crate::differentiable::DifferentiableManifold;
use crate::riemannian::RiemannianMetric;
use crate::vector_bundle::VectorBundle;
use crate::tensor_field::TensorField;
use rustmath_symbolic::Expr;
use rustmath_complex::Complex;
use std::sync::Arc;

/// The spin group Spin(n) as a double cover of SO(n)
///
/// Spin(n) → SO(n) with kernel {±1}
#[derive(Debug, Clone)]
pub struct SpinGroup {
    /// Dimension n (for Spin(n))
    dimension: usize,
}

impl SpinGroup {
    /// Create the spin group Spin(n)
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The covering map π: Spin(n) → SO(n)
    pub fn covering_map(&self) -> Result<()> {
        // TODO: Implement covering map representation
        Ok(())
    }

    /// The spinor representation
    ///
    /// For Spin(n), the spinor representation has dimension 2^(n/2) (even n)
    /// or 2^((n-1)/2) (odd n)
    pub fn spinor_dimension(&self) -> usize {
        if self.dimension % 2 == 0 {
            2_usize.pow((self.dimension / 2) as u32)
        } else {
            2_usize.pow(((self.dimension - 1) / 2) as u32)
        }
    }
}

/// A spin structure on an oriented Riemannian manifold
///
/// This is a principal Spin(n)-bundle P → M that double-covers the
/// oriented orthonormal frame bundle SO(M).
#[derive(Debug, Clone)]
pub struct SpinStructure {
    /// The base manifold (must be oriented)
    manifold: Arc<DifferentiableManifold>,
    /// The Riemannian metric
    metric: RiemannianMetric,
    /// The dimension n
    dimension: usize,
    /// The spin group
    spin_group: SpinGroup,
}

impl SpinStructure {
    /// Create a spin structure
    ///
    /// # Arguments
    ///
    /// * `manifold` - The base manifold (must be oriented)
    /// * `metric` - The Riemannian metric
    ///
    /// # Returns
    ///
    /// A spin structure, or an error if:
    /// - The manifold is not orientable
    /// - The second Stiefel-Whitney class doesn't vanish
    pub fn new(
        manifold: Arc<DifferentiableManifold>,
        metric: RiemannianMetric,
    ) -> Result<Self> {
        let dimension = manifold.dimension();

        // TODO: Check orientability
        // TODO: Check w₂(M) = 0

        let spin_group = SpinGroup::new(dimension);

        Ok(Self {
            manifold,
            metric,
            dimension,
            spin_group,
        })
    }

    /// Create the canonical spin structure on ℝⁿ
    pub fn euclidean(n: usize) -> Self {
        let manifold = Arc::new(DifferentiableManifold::new("ℝⁿ", n));
        let metric = RiemannianMetric::euclidean(manifold.clone(), n);
        let spin_group = SpinGroup::new(n);

        Self {
            manifold,
            metric,
            dimension: n,
            spin_group,
        }
    }

    /// Create the spin structure on the n-sphere Sⁿ
    ///
    /// All spheres admit spin structures
    pub fn sphere(n: usize) -> Result<Self> {
        let manifold = Arc::new(DifferentiableManifold::new(&format!("S{}", n), n));
        let metric = RiemannianMetric::euclidean(manifold.clone(), n); // TODO: Round metric
        let spin_group = SpinGroup::new(n);

        Ok(Self {
            manifold,
            metric,
            dimension: n,
            spin_group,
        })
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the metric
    pub fn metric(&self) -> &RiemannianMetric {
        &self.metric
    }

    /// Get the spin group
    pub fn spin_group(&self) -> &SpinGroup {
        &self.spin_group
    }

    /// Get the spinor bundle dimension
    pub fn spinor_dimension(&self) -> usize {
        self.spin_group.spinor_dimension()
    }

    /// Construct the spinor bundle S → M
    ///
    /// This is the associated bundle P ×_ρ V where ρ is the spinor representation
    pub fn spinor_bundle(&self) -> SpinorBundle {
        SpinorBundle::new(self.clone())
    }

    /// Check if two spin structures are equivalent
    pub fn is_equivalent(&self, _other: &SpinStructure) -> bool {
        // TODO: Check if spin structures differ by automorphism
        false
    }
}

/// The spinor bundle - vector bundle of spinors
///
/// Sections of this bundle are spinor fields
#[derive(Debug, Clone)]
pub struct SpinorBundle {
    /// The spin structure
    spin_structure: SpinStructure,
    /// Dimension of the fiber (spinor space)
    fiber_dimension: usize,
}

impl SpinorBundle {
    /// Create a spinor bundle from a spin structure
    pub fn new(spin_structure: SpinStructure) -> Self {
        let fiber_dimension = spin_structure.spinor_dimension();

        Self {
            spin_structure,
            fiber_dimension,
        }
    }

    /// Get the spin structure
    pub fn spin_structure(&self) -> &SpinStructure {
        &self.spin_structure
    }

    /// Get the fiber dimension
    pub fn fiber_dimension(&self) -> usize {
        self.fiber_dimension
    }

    /// Get the base manifold
    pub fn base_manifold(&self) -> &Arc<DifferentiableManifold> {
        self.spin_structure.manifold()
    }
}

/// A spinor field - section of the spinor bundle
///
/// Spinor fields are acted upon by the Dirac operator
#[derive(Debug, Clone)]
pub struct SpinorField {
    /// The spinor bundle
    bundle: Arc<SpinorBundle>,
    /// Components of the spinor (complex-valued)
    components: Vec<Complex>,
    /// Name of the field
    name: Option<String>,
}

impl SpinorField {
    /// Create a new spinor field
    pub fn new(bundle: Arc<SpinorBundle>) -> Self {
        let dim = bundle.fiber_dimension();
        let components = vec![Complex::new(0.0, 0.0); dim];

        Self {
            bundle,
            components,
            name: None,
        }
    }

    /// Create a spinor field with specific components
    pub fn from_components(
        bundle: Arc<SpinorBundle>,
        components: Vec<Complex>,
    ) -> Result<Self> {
        if components.len() != bundle.fiber_dimension() {
            return Err(ManifoldError::DimensionMismatch {
                expected: bundle.fiber_dimension(),
                actual: components.len(),
            });
        }

        Ok(Self {
            bundle,
            components,
            name: None,
        })
    }

    /// Set the name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Get the components
    pub fn components(&self) -> &[Complex] {
        &self.components
    }

    /// Get the bundle
    pub fn bundle(&self) -> &Arc<SpinorBundle> {
        &self.bundle
    }

    /// Compute the norm of the spinor
    pub fn norm(&self) -> f64 {
        self.components
            .iter()
            .map(|z| z.abs() * z.abs())
            .sum::<f64>()
            .sqrt()
    }
}

/// Clifford multiplication - multiplication of vectors with spinors
///
/// The Clifford algebra Cl(M, g) acts on spinors via Clifford multiplication
#[derive(Debug, Clone)]
pub struct CliffordMultiplication {
    /// The dimension
    dimension: usize,
}

impl CliffordMultiplication {
    /// Create Clifford multiplication structure
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Multiply a tangent vector with a spinor: v · ψ
    pub fn multiply(&self, _vector: &[f64], spinor: &SpinorField) -> Result<SpinorField> {
        // TODO: Implement Clifford multiplication using gamma matrices
        Ok(spinor.clone())
    }

    /// Get the gamma matrices (Clifford algebra generators)
    pub fn gamma_matrices(&self) -> Vec<Vec<Complex>> {
        // TODO: Construct gamma matrices for Clifford algebra
        // These satisfy {γᵢ, γⱼ} = 2gᵢⱼ
        let dim = self.spinor_dimension();
        vec![vec![Complex::new(0.0, 0.0); dim]; self.dimension]
    }

    /// Get the spinor representation dimension
    fn spinor_dimension(&self) -> usize {
        if self.dimension % 2 == 0 {
            2_usize.pow((self.dimension / 2) as u32)
        } else {
            2_usize.pow(((self.dimension - 1) / 2) as u32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spin_group_creation() {
        let spin3 = SpinGroup::new(3);
        assert_eq!(spin3.dimension(), 3);
    }

    #[test]
    fn test_spin_group_spinor_dimension() {
        // Spin(2) has spinor dimension 2^1 = 2
        let spin2 = SpinGroup::new(2);
        assert_eq!(spin2.spinor_dimension(), 2);

        // Spin(3) has spinor dimension 2^1 = 2
        let spin3 = SpinGroup::new(3);
        assert_eq!(spin3.spinor_dimension(), 2);

        // Spin(4) has spinor dimension 2^2 = 4
        let spin4 = SpinGroup::new(4);
        assert_eq!(spin4.spinor_dimension(), 4);
    }

    #[test]
    fn test_euclidean_spin_structure() {
        let spin = SpinStructure::euclidean(3);
        assert_eq!(spin.dimension, 3);
        assert_eq!(spin.spinor_dimension(), 2);
    }

    #[test]
    fn test_sphere_spin_structure() {
        let spin = SpinStructure::sphere(2).unwrap();
        assert_eq!(spin.dimension, 2);
    }

    #[test]
    fn test_spinor_bundle_creation() {
        let spin = SpinStructure::euclidean(3);
        let bundle = spin.spinor_bundle();
        assert_eq!(bundle.fiber_dimension(), 2);
    }

    #[test]
    fn test_spinor_field_creation() {
        let spin = SpinStructure::euclidean(2);
        let bundle = Arc::new(spin.spinor_bundle());
        let field = SpinorField::new(bundle);
        assert_eq!(field.components().len(), 2);
    }

    #[test]
    fn test_spinor_field_from_components() {
        let spin = SpinStructure::euclidean(2);
        let bundle = Arc::new(spin.spinor_bundle());

        let components = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
        ];

        let field = SpinorField::from_components(bundle, components);
        assert!(field.is_ok());
    }

    #[test]
    fn test_spinor_field_norm() {
        let spin = SpinStructure::euclidean(2);
        let bundle = Arc::new(spin.spinor_bundle());

        let components = vec![
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];

        let field = SpinorField::from_components(bundle, components).unwrap();
        assert!((field.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_clifford_multiplication() {
        let cliff = CliffordMultiplication::new(3);
        assert_eq!(cliff.dimension, 3);
    }
}
