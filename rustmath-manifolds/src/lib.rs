//! RustMath Manifolds - Differential geometry and manifold theory
//!
//! This crate provides structures and operations for working with mathematical manifolds,
//! including topological manifolds, differentiable manifolds, charts, scalar fields,
//! and various geometric objects.
//!
//! # Overview
//!
//! A manifold is a topological space that locally resembles Euclidean space. This crate
//! implements the foundational structures needed for differential geometry:
//!
//! - **ManifoldSubset**: Base type representing subsets of manifolds
//! - **TopologicalManifold**: Manifolds with a topological structure
//! - **DifferentiableManifold**: Smooth manifolds with differentiable structure
//! - **Chart**: Local coordinate systems on manifolds
//! - **Point**: Points on manifolds
//! - **ScalarField**: Smooth scalar-valued functions on manifolds
//!
//! # Examples
//!
//! ```
//! use rustmath_manifolds::{TopologicalManifold, RealLine};
//!
//! // Create a 1-dimensional real line manifold
//! let real_line = RealLine::new();
//! ```

pub mod subset;
pub mod manifold;
pub mod chart;
pub mod point;
pub mod scalar_field;
pub mod differentiable;
pub mod examples;
pub mod errors;

pub use subset::ManifoldSubset;
pub use manifold::TopologicalManifold;
pub use chart::{Chart, CoordinateFunction};
pub use point::ManifoldPoint;
pub use scalar_field::ScalarField;
pub use differentiable::DifferentiableManifold;
pub use examples::{RealLine, EuclideanSpace};
pub use errors::{ManifoldError, Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_line_creation() {
        let real_line = RealLine::new();
        assert_eq!(real_line.dimension(), 1);
    }

    #[test]
    fn test_euclidean_space_creation() {
        let euclidean_2d = EuclideanSpace::new(2);
        assert_eq!(euclidean_2d.dimension(), 2);

        let euclidean_3d = EuclideanSpace::new(3);
        assert_eq!(euclidean_3d.dimension(), 3);
    }
}
