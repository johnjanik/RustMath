//! Vector bundles over manifolds
//!
//! A vector bundle is a topological construction that makes precise the idea
//! of a family of vector spaces parameterized by another space. This module
//! provides structures for working with vector bundles in differential geometry.

use crate::{TopologicalManifold, ManifoldPoint, ManifoldError, Result};
use std::marker::PhantomData;

/// A topological vector bundle over a base manifold
///
/// A vector bundle E → M consists of:
/// - A base manifold M
/// - A total space E
/// - A projection π: E → M
/// - For each point p ∈ M, a vector space Ep (the fiber over p)
///
/// # Type Parameters
///
/// * `M` - The base manifold type
/// * `F` - The fiber type (typically a vector space)
#[derive(Clone)]
pub struct TopologicalVectorBundle<M, F> {
    /// The base manifold
    base_manifold: M,
    /// Rank (dimension) of each fiber
    rank: usize,
    /// Name of the vector bundle
    name: String,
    /// Field over which the vector spaces are defined (e.g., "R" for real, "C" for complex)
    field: String,
    _fiber: PhantomData<F>,
}

impl<M: TopologicalManifold, F> TopologicalVectorBundle<M, F> {
    /// Create a new vector bundle
    ///
    /// # Arguments
    ///
    /// * `base_manifold` - The base manifold
    /// * `rank` - The dimension of each fiber vector space
    /// * `name` - Name for this vector bundle
    /// * `field` - The field over which the vector spaces are defined
    pub fn new(base_manifold: M, rank: usize, name: String, field: String) -> Self {
        TopologicalVectorBundle {
            base_manifold,
            rank,
            name,
            field,
            _fiber: PhantomData,
        }
    }

    /// Get the base manifold
    pub fn base_manifold(&self) -> &M {
        &self.base_manifold
    }

    /// Get the rank (dimension of fibers)
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the name of this vector bundle
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the field
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the dimension of the total space (base dimension + rank)
    pub fn total_dimension(&self) -> usize {
        self.base_manifold.dimension() + self.rank
    }
}

/// The tangent bundle of a differentiable manifold
///
/// For an n-dimensional manifold M, the tangent bundle TM is a 2n-dimensional
/// manifold whose fiber over each point p is the tangent space TpM.
pub struct TangentBundle<M> {
    bundle: TopologicalVectorBundle<M, Vec<f64>>,
}

impl<M: TopologicalManifold> TangentBundle<M> {
    /// Create the tangent bundle for a manifold
    pub fn new(base_manifold: M) -> Self {
        let rank = base_manifold.dimension();
        let name = format!("T{}", base_manifold.name());
        let bundle = TopologicalVectorBundle::new(
            base_manifold,
            rank,
            name,
            "R".to_string(),
        );

        TangentBundle { bundle }
    }

    /// Get the underlying vector bundle
    pub fn bundle(&self) -> &TopologicalVectorBundle<M, Vec<f64>> {
        &self.bundle
    }

    /// Get the base manifold
    pub fn base_manifold(&self) -> &M {
        self.bundle.base_manifold()
    }
}

/// The cotangent bundle of a differentiable manifold
///
/// For an n-dimensional manifold M, the cotangent bundle T*M is a 2n-dimensional
/// manifold whose fiber over each point p is the cotangent space T*pM (dual to TpM).
pub struct CotangentBundle<M> {
    bundle: TopologicalVectorBundle<M, Vec<f64>>,
}

impl<M: TopologicalManifold> CotangentBundle<M> {
    /// Create the cotangent bundle for a manifold
    pub fn new(base_manifold: M) -> Self {
        let rank = base_manifold.dimension();
        let name = format!("T*{}", base_manifold.name());
        let bundle = TopologicalVectorBundle::new(
            base_manifold,
            rank,
            name,
            "R".to_string(),
        );

        CotangentBundle { bundle }
    }

    /// Get the underlying vector bundle
    pub fn bundle(&self) -> &TopologicalVectorBundle<M, Vec<f64>> {
        &self.bundle
    }

    /// Get the base manifold
    pub fn base_manifold(&self) -> &M {
        self.bundle.base_manifold()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::{RealLine, EuclideanSpace};

    #[test]
    fn test_vector_bundle_creation() {
        let base = RealLine::new();
        let bundle = TopologicalVectorBundle::<_, Vec<f64>>::new(
            base,
            2,
            "E".to_string(),
            "R".to_string(),
        );

        assert_eq!(bundle.rank(), 2);
        assert_eq!(bundle.name(), "E");
        assert_eq!(bundle.field(), "R");
        assert_eq!(bundle.base_manifold().dimension(), 1);
        assert_eq!(bundle.total_dimension(), 3); // 1 (base) + 2 (fiber)
    }

    #[test]
    fn test_tangent_bundle() {
        let manifold = EuclideanSpace::new(3);
        let tangent = TangentBundle::new(manifold);

        assert_eq!(tangent.bundle().rank(), 3);
        assert_eq!(tangent.base_manifold().dimension(), 3);
        assert_eq!(tangent.bundle().total_dimension(), 6);
    }

    #[test]
    fn test_cotangent_bundle() {
        let manifold = EuclideanSpace::new(2);
        let cotangent = CotangentBundle::new(manifold);

        assert_eq!(cotangent.bundle().rank(), 2);
        assert_eq!(cotangent.base_manifold().dimension(), 2);
        assert_eq!(cotangent.bundle().total_dimension(), 4);
    }

    #[test]
    fn test_bundle_names() {
        let manifold = RealLine::new();
        let tangent = TangentBundle::new(manifold.clone());
        let cotangent = CotangentBundle::new(manifold);

        assert!(tangent.bundle().name().starts_with("T"));
        assert!(cotangent.bundle().name().starts_with("T*"));
    }
}
