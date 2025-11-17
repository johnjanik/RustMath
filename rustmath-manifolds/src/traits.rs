//! Trait hierarchy for differential manifolds
//!
//! This module defines the trait hierarchy that mirrors SageMath's manifold class structure.
//! The design uses trait composition instead of class inheritance to achieve similar flexibility.
//!
//! NOTE: Parent and UniqueRepresentation have been removed from the base traits
//! to ensure object safety (allow usage as dyn Trait). If Parent functionality
//! is needed for specific implementations, it can be added separately.

use crate::chart::Chart;
use crate::point::ManifoldPoint;
use crate::errors::Result;
use std::sync::Arc;
use std::hash::Hash;

// ============================================================================
// DOMAIN TRAITS (manifolds, subsets, points)
// ============================================================================

/// Base trait for any subset of a manifold
///
/// This trait is object-safe to allow usage as `Arc<dyn ManifoldSubsetTrait>`.
/// If you need Parent or UniqueRepresentation functionality, implement those
/// traits separately on your concrete types.
pub trait ManifoldSubsetTrait {
    /// Dimension of the ambient space
    fn dimension(&self) -> usize;

    /// Get the ambient manifold (if this is a proper subset)
    fn ambient_manifold(&self) -> Option<Arc<dyn TopologicalManifoldTrait>>;

    /// Check if this subset is open
    fn is_open(&self) -> bool;

    /// Check if this subset is closed
    fn is_closed(&self) -> bool;
}

/// Topological manifold
pub trait TopologicalManifoldTrait: ManifoldSubsetTrait {
    /// Get all charts in the atlas
    fn atlas(&self) -> &[Chart];

    /// Get the default chart
    fn default_chart(&self) -> Option<&Chart>;

    /// Add a chart to the atlas
    fn add_chart(&mut self, chart: Chart) -> Result<()>;

    /// Get a chart by ID
    fn get_chart(&self, id: &str) -> Option<&Chart>;
}

/// Differentiable (smooth) manifold
pub trait DifferentiableManifoldTrait: TopologicalManifoldTrait {
    /// Verify that the atlas is C^∞-compatible
    fn verify_smoothness(&self) -> Result<()>;

    /// Get the scalar field algebra C^∞(M)
    fn scalar_field_algebra(&self) -> Arc<dyn ScalarFieldAlgebraTrait>;

    /// Get the vector field module 𝔛(M)
    fn vector_field_module(&self) -> Arc<dyn VectorFieldModuleTrait>;
}

/// Parallelizable manifold (has a global frame)
pub trait ParallelizableManifoldTrait: DifferentiableManifoldTrait {
    /// Get the vector field free module (parallelizable case)
    fn vector_field_free_module(&self) -> Arc<dyn VectorFieldFreeModuleTrait>;
}

// ============================================================================
// SCALAR FIELD TRAITS
// ============================================================================

/// Algebra of scalar fields C^∞(M)
///
/// Object-safe trait (Parent removed for trait object compatibility).
pub trait ScalarFieldAlgebraTrait {
    /// Get the underlying manifold
    fn manifold(&self) -> Arc<dyn DifferentiableManifoldTrait>;
}

/// Differentiable scalar field algebra (specialization)
pub trait DiffScalarFieldAlgebraTrait: ScalarFieldAlgebraTrait {
    // All fields are C^∞
}

// ============================================================================
// MODULE TRAITS (tensor fields, vector fields)
// ============================================================================

/// Module of vector fields over C^∞(M)
///
/// Object-safe trait (Parent removed for trait object compatibility).
pub trait VectorFieldModuleTrait {
    /// Get the underlying manifold
    fn manifold(&self) -> Arc<dyn DifferentiableManifoldTrait>;

    /// Get the rank (contravariant, covariant)
    fn rank(&self) -> (usize, usize);
}

/// Free module of vector fields (parallelizable case)
pub trait VectorFieldFreeModuleTrait: VectorFieldModuleTrait {
    /// Get the number of basis elements
    fn rank_value(&self) -> usize;
}

/// Tensor field module T^(p,q)(M)
///
/// Object-safe trait (Parent removed for trait object compatibility).
pub trait TensorFieldModuleTrait {
    /// Contravariant rank (p)
    fn contravariant_rank(&self) -> usize;

    /// Covariant rank (q)
    fn covariant_rank(&self) -> usize;

    /// Total rank
    fn total_rank(&self) -> usize {
        self.contravariant_rank() + self.covariant_rank()
    }
}

/// Tangent space at a point T_p(M)
pub trait TangentSpaceTrait: VectorFieldFreeModuleTrait {
    /// Get the base point
    fn base_point(&self) -> &ManifoldPoint;
}

// ============================================================================
// ELEMENT TRAITS (fields, vectors, tensors)
// ============================================================================

/// A scalar field f: M → ℝ
pub trait ScalarFieldTrait: Clone {
    /// Get the underlying manifold
    fn manifold(&self) -> Arc<dyn DifferentiableManifoldTrait>;

    /// Get the name of the field
    fn name(&self) -> Option<&str>;
}

/// A vector field X ∈ 𝔛(M)
pub trait VectorFieldTrait: Clone {
    /// Get the underlying manifold
    fn manifold(&self) -> Arc<dyn DifferentiableManifoldTrait>;

    /// Get the name of the field
    fn name(&self) -> Option<&str>;
}

/// Vector field on parallelizable manifold (free module element)
pub trait VectorFieldParalTrait: VectorFieldTrait {
    // Specialized operations for parallelizable case
}

/// Tensor field of type (p, q)
pub trait TensorFieldTrait: Clone {
    /// Contravariant rank
    fn contravariant_rank(&self) -> usize;

    /// Covariant rank
    fn covariant_rank(&self) -> usize;
}

/// Tangent vector at a point
pub trait TangentVectorTrait: Clone {
    /// Get the base point
    fn base_point(&self) -> &ManifoldPoint;
}

/// Differential form (totally antisymmetric covariant tensor)
pub trait DiffFormTrait: TensorFieldTrait {
    /// Get the degree of the form
    fn degree(&self) -> usize;
}
