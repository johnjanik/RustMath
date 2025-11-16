//! Abstract Base Classes for Geometry Types
//!
//! This module provides marker traits corresponding to SageMath's sage.geometry.abc module.
//! These traits are used for type constraints and trait bounds, similar to how SageMath
//! uses ABC classes for isinstance() checks.
//!
//! The traits in this module are intentionally minimal, serving primarily as markers
//! in the type system rather than defining specific behavior.

use std::fmt::Debug;

/// Marker trait for lattice polytope types
///
/// This trait corresponds to SageMath's `sage.geometry.abc.LatticePolytope`.
/// It serves as an abstract base for lattice polytope implementations.
///
/// A lattice polytope is the convex hull of a finite set of integer points.
/// In the context of algebraic geometry, these are fundamental objects in
/// toric geometry and mirror symmetry.
///
/// # Purpose
///
/// This trait is defined primarily for type constraints and trait bounds.
/// It should be implemented by concrete lattice polytope types but is not
/// intended to define specific methods.
///
/// # Example
///
/// ```
/// use rustmath_geometry::abc::LatticePolytope;
///
/// // A concrete implementation would implement this trait
/// #[derive(Debug, Clone)]
/// struct MyLatticePolytope {
///     vertices: Vec<Vec<i64>>,
/// }
///
/// impl LatticePolytope for MyLatticePolytope {}
///
/// fn process_polytope<T: LatticePolytope>(polytope: &T) {
///     // Generic function that works with any LatticePolytope
/// }
/// ```
pub trait LatticePolytope: Debug + Clone {}

/// Marker trait for convex rational polyhedral cone types
///
/// This trait corresponds to SageMath's `sage.geometry.abc.ConvexRationalPolyhedralCone`.
/// It serves as an abstract base for cone implementations.
///
/// A convex rational polyhedral cone is a subset of ℝⁿ defined as the non-negative
/// linear combinations of a finite set of vectors with rational coordinates.
/// These are fundamental objects in toric geometry and optimization.
///
/// # Mathematical Definition
///
/// A cone C is a set of the form:
/// ```text
/// C = { λ₁v₁ + λ₂v₂ + ... + λₖvₖ : λᵢ ≥ 0, vᵢ ∈ ℚⁿ }
/// ```
///
/// # Purpose
///
/// This trait is defined primarily for type constraints and trait bounds.
/// Concrete cone types should implement this trait.
///
/// # Example
///
/// ```
/// use rustmath_geometry::abc::ConvexRationalPolyhedralCone;
///
/// #[derive(Debug, Clone)]
/// struct MyCone {
///     rays: Vec<Vec<i64>>,
///     dimension: usize,
/// }
///
/// impl ConvexRationalPolyhedralCone for MyCone {}
///
/// fn process_cone<T: ConvexRationalPolyhedralCone>(cone: &T) {
///     // Generic function that works with any ConvexRationalPolyhedralCone
/// }
/// ```
pub trait ConvexRationalPolyhedralCone: Debug + Clone {}

/// Marker trait for polyhedron types
///
/// This trait corresponds to SageMath's `sage.geometry.abc.Polyhedron`.
/// It serves as an abstract base for polyhedron implementations.
///
/// A polyhedron is a region in space defined by a finite system of linear
/// inequalities and equations. It can be represented either as:
/// - V-representation: convex hull of vertices plus conic hull of rays/lines
/// - H-representation: intersection of half-spaces
///
/// # Mathematical Definition
///
/// A polyhedron P can be defined as:
/// ```text
/// P = { x ∈ ℝⁿ : Ax ≤ b, Cx = d }
/// ```
/// or equivalently as:
/// ```text
/// P = conv(v₁, ..., vₖ) + cone(r₁, ..., rₘ) + span(l₁, ..., lₚ)
/// ```
///
/// # Purpose
///
/// This trait is defined primarily for type constraints and trait bounds.
/// Concrete polyhedron types (over various base rings) should implement this trait.
///
/// # Example
///
/// ```
/// use rustmath_geometry::abc::Polyhedron;
///
/// #[derive(Debug, Clone)]
/// struct MyPolyhedron {
///     vertices: Vec<Vec<f64>>,
///     inequalities: Vec<Vec<f64>>,
/// }
///
/// impl Polyhedron for MyPolyhedron {}
///
/// fn compute_volume<T: Polyhedron>(poly: &T) -> f64 {
///     // Generic function that works with any Polyhedron
///     0.0 // placeholder
/// }
/// ```
pub trait Polyhedron: Debug + Clone {}

/// Extension trait providing common geometry operations
///
/// This trait can be implemented alongside the marker traits to provide
/// shared functionality across different geometry types.
pub trait GeometryBase: Debug + Clone {
    /// Returns the dimension of the object
    fn dim(&self) -> usize;

    /// Returns the ambient dimension (dimension of the containing space)
    fn ambient_dim(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test implementation for LatticePolytope
    #[derive(Debug, Clone)]
    struct TestLatticePolytope {
        dim: usize,
    }

    impl LatticePolytope for TestLatticePolytope {}

    impl GeometryBase for TestLatticePolytope {
        fn dim(&self) -> usize {
            self.dim
        }

        fn ambient_dim(&self) -> usize {
            self.dim
        }
    }

    // Test implementation for ConvexRationalPolyhedralCone
    #[derive(Debug, Clone)]
    struct TestCone {
        dim: usize,
    }

    impl ConvexRationalPolyhedralCone for TestCone {}

    impl GeometryBase for TestCone {
        fn dim(&self) -> usize {
            self.dim
        }

        fn ambient_dim(&self) -> usize {
            self.dim
        }
    }

    // Test implementation for Polyhedron
    #[derive(Debug, Clone)]
    struct TestPolyhedron {
        dim: usize,
    }

    impl Polyhedron for TestPolyhedron {}

    impl GeometryBase for TestPolyhedron {
        fn dim(&self) -> usize {
            self.dim
        }

        fn ambient_dim(&self) -> usize {
            self.dim
        }
    }

    #[test]
    fn test_lattice_polytope_trait() {
        let polytope = TestLatticePolytope { dim: 3 };
        assert_eq!(polytope.dim(), 3);
    }

    #[test]
    fn test_cone_trait() {
        let cone = TestCone { dim: 2 };
        assert_eq!(cone.dim(), 2);
    }

    #[test]
    fn test_polyhedron_trait() {
        let poly = TestPolyhedron { dim: 4 };
        assert_eq!(poly.dim(), 4);
    }

    // Test generic functions using the traits
    fn process_lattice_polytope<T: LatticePolytope + GeometryBase>(obj: &T) -> usize {
        obj.dim()
    }

    fn process_cone<T: ConvexRationalPolyhedralCone + GeometryBase>(obj: &T) -> usize {
        obj.dim()
    }

    fn process_polyhedron<T: Polyhedron + GeometryBase>(obj: &T) -> usize {
        obj.dim()
    }

    #[test]
    fn test_generic_functions() {
        let polytope = TestLatticePolytope { dim: 3 };
        let cone = TestCone { dim: 2 };
        let poly = TestPolyhedron { dim: 4 };

        assert_eq!(process_lattice_polytope(&polytope), 3);
        assert_eq!(process_cone(&cone), 2);
        assert_eq!(process_polyhedron(&poly), 4);
    }

    #[test]
    fn test_trait_bounds_composition() {
        fn process_geometry<T: GeometryBase>(obj: &T) -> (usize, usize) {
            (obj.dim(), obj.ambient_dim())
        }

        let polytope = TestLatticePolytope { dim: 3 };
        let (d, ad) = process_geometry(&polytope);
        assert_eq!(d, 3);
        assert_eq!(ad, 3);
    }
}
