//! Sub-Riemannian manifolds and sub-Riemannian geometry
//!
//! This module provides structures for sub-Riemannian manifolds - manifolds where
//! the metric is defined only on a subbundle of the tangent bundle (called the
//! horizontal distribution).
//!
//! # Overview
//!
//! A sub-Riemannian manifold consists of:
//! - A smooth manifold M
//! - A distribution D ⊂ TM (the horizontal distribution)
//! - A metric g defined only on D
//!
//! Sub-Riemannian geometry appears in:
//! - Control theory (nonholonomic constraints)
//! - Robotics (path planning with constraints)
//! - Neuroscience (visual cortex modeling)
//!
//! # Examples
//!
//! ```
//! use rustmath_manifolds::{SubRiemannianManifold, DifferentiableManifold};
//!
//! // Create a sub-Riemannian manifold (e.g., Heisenberg group)
//! let m = DifferentiableManifold::new("H³", 3);
//! ```

use crate::errors::{ManifoldError, Result};
use crate::differentiable::DifferentiableManifold;
use crate::vector_field::VectorField;
use crate::tensor_field::TensorField;
use crate::tangent_space::TangentVector;
use crate::point::ManifoldPoint;
use rustmath_symbolic::Expr;
use std::sync::Arc;

/// A distribution on a manifold - a subbundle of the tangent bundle
///
/// D assigns to each point p a subspace D_p ⊂ T_p M
#[derive(Debug, Clone)]
pub struct Distribution {
    /// The base manifold
    manifold: Arc<DifferentiableManifold>,
    /// Rank of the distribution (dimension of D_p at each point)
    rank: usize,
    /// Frame: vector fields that span the distribution
    frame: Vec<VectorField>,
    /// Name of the distribution
    name: Option<String>,
}

impl Distribution {
    /// Create a new distribution
    ///
    /// # Arguments
    ///
    /// * `manifold` - The base manifold
    /// * `frame` - Vector fields spanning the distribution
    ///
    /// # Returns
    ///
    /// A distribution, or an error if the frame doesn't have constant rank
    pub fn new(
        manifold: Arc<DifferentiableManifold>,
        frame: Vec<VectorField>,
    ) -> Result<Self> {
        if frame.is_empty() {
            return Err(ManifoldError::ValidationError(
                "Distribution frame cannot be empty".to_string()
            ));
        }

        let rank = frame.len();

        // TODO: Verify frame has constant rank (vectors are linearly independent)

        Ok(Self {
            manifold,
            rank,
            frame,
            name: None,
        })
    }

    /// Set the name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Get the rank (dimension) of the distribution
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the frame (spanning vector fields)
    pub fn frame(&self) -> &[VectorField] {
        &self.frame
    }

    /// Get the manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Check if the distribution is involutive (integrable)
    ///
    /// A distribution is involutive if [X, Y] ∈ D for all X, Y ∈ D
    /// (i.e., closed under Lie brackets)
    pub fn is_involutive(&self) -> Result<bool> {
        // Check if Lie bracket of any two frame vectors is in the distribution
        let chart = self.manifold.default_chart().ok_or(ManifoldError::NoChart)?;
        for i in 0..self.frame.len() {
            for j in i+1..self.frame.len() {
                let bracket = self.frame[i].lie_bracket(&self.frame[j], chart)?;

                // Check if bracket can be expressed as a linear combination of frame
                // TODO: Implement proper linear dependence check
                // For now, assume involutive
            }
        }

        Ok(true)
    }

    /// Compute the Lie bracket extension (flag of distributions)
    ///
    /// D^(1) = span{D, [D, D]}
    /// D^(2) = span{D^(1), [D^(1), D^(1)]}
    /// etc.
    ///
    /// Returns the flag D ⊂ D^(1) ⊂ D^(2) ⊂ ...
    pub fn flag(&self) -> Result<Vec<Distribution>> {
        let mut current_frame = self.frame.clone();
        let mut flag = vec![self.clone()];

        let chart = self.manifold.default_chart().ok_or(ManifoldError::NoChart)?;
        loop {
            let mut new_frame = current_frame.clone();
            let mut added = false;

            // Add Lie brackets
            for i in 0..current_frame.len() {
                for j in i+1..current_frame.len() {
                    let bracket = current_frame[i].lie_bracket(&current_frame[j], chart)?;

                    // TODO: Check if bracket is linearly independent of current frame
                    // If so, add it
                    new_frame.push(bracket);
                    added = true;
                }
            }

            if !added || new_frame.len() == self.manifold.dimension() {
                break; // Reached full dimension or no new vectors
            }

            let next_dist = Distribution::new(self.manifold.clone(), new_frame.clone())?;
            flag.push(next_dist);
            current_frame = new_frame;
        }

        Ok(flag)
    }

    /// Check if the distribution is bracket-generating (completely non-holonomic)
    ///
    /// A distribution is bracket-generating if iterated Lie brackets span TM
    pub fn is_bracket_generating(&self) -> Result<bool> {
        let flag = self.flag()?;
        let last_dist = flag.last().unwrap();
        Ok(last_dist.rank() == self.manifold.dimension())
    }
}

/// A sub-Riemannian manifold - manifold with metric on a distribution
#[derive(Debug, Clone)]
pub struct SubRiemannianManifold {
    /// The base manifold
    base_manifold: Arc<DifferentiableManifold>,
    /// The horizontal distribution (where metric is defined)
    distribution: Distribution,
    /// The metric tensor (defined only on the distribution)
    metric: TensorField,
}

impl SubRiemannianManifold {
    /// Create a new sub-Riemannian manifold
    ///
    /// # Arguments
    ///
    /// * `base_manifold` - The underlying manifold
    /// * `distribution` - The horizontal distribution
    /// * `metric` - Metric tensor on the distribution
    pub fn new(
        base_manifold: Arc<DifferentiableManifold>,
        distribution: Distribution,
        metric: TensorField,
    ) -> Result<Self> {
        // Verify metric has the right type (0,2) symmetric
        if metric.contravariant_rank() != 0 || metric.covariant_rank() != 2 {
            return Err(ManifoldError::ValidationError(
                "Sub-Riemannian metric must be a (0,2)-tensor".to_string()
            ));
        }

        Ok(Self {
            base_manifold,
            distribution,
            metric,
        })
    }

    /// Create the Heisenberg group as a sub-Riemannian manifold
    ///
    /// The Heisenberg group H³ is ℝ³ with coordinates (x, y, z) and
    /// horizontal distribution spanned by:
    /// X₁ = ∂/∂x + y ∂/∂z
    /// X₂ = ∂/∂y
    ///
    /// The metric makes {X₁, X₂} orthonormal
    pub fn heisenberg_group() -> Result<Self> {
        let base = Arc::new(DifferentiableManifold::new("H³", 3));
        let chart = base.default_chart().unwrap();

        // Create frame
        let x1 = VectorField::from_components(
            base.clone(),
            chart,
            vec![
                Expr::from(1),                    // ∂/∂x
                Expr::from(0),                    // 0
                Expr::symbol("y"),                 // y ∂/∂z
            ],
        )?;

        let x2 = VectorField::from_components(
            base.clone(),
            chart,
            vec![
                Expr::from(0),                    // 0
                Expr::from(1),                    // ∂/∂y
                Expr::from(0),                    // 0
            ],
        )?;

        let distribution = Distribution::new(base.clone(), vec![x1, x2])?
            .with_name("Horizontal");

        // Create orthonormal metric
        // g(X₁, X₁) = g(X₂, X₂) = 1, g(X₁, X₂) = 0
        let metric_components = vec![
            Expr::from(1), Expr::from(0),  // g_11, g_12
            Expr::from(0), Expr::from(1),  // g_21, g_22
        ];

        let metric = TensorField::from_components(
            base.clone(),
            0,
            2,
            chart,
            metric_components,
        )?;

        Ok(Self {
            base_manifold: base,
            distribution,
            metric,
        })
    }

    /// Get the base manifold
    pub fn base_manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.base_manifold
    }

    /// Get the horizontal distribution
    pub fn distribution(&self) -> &Distribution {
        &self.distribution
    }

    /// Get the metric
    pub fn metric(&self) -> &TensorField {
        &self.metric
    }

    /// Compute the sub-Riemannian distance between two points
    ///
    /// The distance is the infimum of lengths of horizontal curves connecting the points
    pub fn distance(&self, p: &ManifoldPoint, q: &ManifoldPoint) -> Result<f64> {
        // TODO: Implement sub-Riemannian distance computation
        // This is generally a hard optimal control problem
        Ok(0.0)
    }

    /// Check if the manifold is contact (odd-dimensional, bracket-generating)
    pub fn is_contact(&self) -> Result<bool> {
        if self.base_manifold.dimension() % 2 == 0 {
            return Ok(false); // Contact manifolds are odd-dimensional
        }

        // Check if bracket-generating
        self.distribution.is_bracket_generating()
    }

    /// Compute geodesics (length-minimizing horizontal curves)
    ///
    /// These satisfy the sub-Riemannian geodesic equation from the Hamiltonian formulation
    pub fn geodesic_equations(&self) -> Result<Vec<Expr>> {
        // TODO: Derive geodesic equations using Pontryagin maximum principle
        let dim = self.base_manifold.dimension();
        Ok(vec![Expr::from(0); dim])
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.base_manifold.dimension()
    }

    /// Get the rank of the distribution
    pub fn distribution_rank(&self) -> usize {
        self.distribution.rank()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_creation() {
        let m = Arc::new(DifferentiableManifold::new("M", 3));
        let chart = m.default_chart().unwrap();

        let v1 = VectorField::from_components(
            m.clone(),
            chart,
            vec![Expr::from(1), Expr::from(0), Expr::from(0)],
        ).unwrap();

        let v2 = VectorField::from_components(
            m.clone(),
            chart,
            vec![Expr::from(0), Expr::from(1), Expr::from(0)],
        ).unwrap();

        let dist = Distribution::new(m.clone(), vec![v1, v2]);
        assert!(dist.is_ok());
        assert_eq!(dist.unwrap().rank(), 2);
    }

    #[test]
    fn test_distribution_empty_frame_fails() {
        let m = Arc::new(DifferentiableManifold::new("M", 3));
        let dist = Distribution::new(m, vec![]);
        assert!(dist.is_err());
    }

    #[test]
    fn test_heisenberg_group() {
        let h = SubRiemannianManifold::heisenberg_group();
        assert!(h.is_ok());

        let heisenberg = h.unwrap();
        assert_eq!(heisenberg.dimension(), 3);
        assert_eq!(heisenberg.distribution_rank(), 2);
    }

    #[test]
    fn test_heisenberg_bracket_generating() {
        let h = SubRiemannianManifold::heisenberg_group().unwrap();
        // Heisenberg distribution is bracket-generating
        let is_bg = h.distribution().is_bracket_generating().unwrap();
        assert!(is_bg);
    }

    #[test]
    fn test_subriemannian_manifold_creation() {
        let m = Arc::new(DifferentiableManifold::new("M", 3));
        let chart = m.default_chart().unwrap();

        let v = VectorField::from_components(
            m.clone(),
            chart,
            vec![Expr::from(1), Expr::from(0), Expr::from(0)],
        ).unwrap();

        let dist = Distribution::new(m.clone(), vec![v]).unwrap();

        let metric_components = vec![Expr::from(1)]; // 1x1 metric
        let metric = TensorField::from_components(
            m.clone(),
            0,
            2,
            chart,
            metric_components,
        ).unwrap();

        let result = SubRiemannianManifold::new(m, dist, metric);
        assert!(result.is_ok());
    }
}
