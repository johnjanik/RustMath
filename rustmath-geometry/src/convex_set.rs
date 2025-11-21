//! Convex Set Trait Hierarchy
//!
//! This module defines abstract traits for convex sets in geometry,
//! corresponding to SageMath's sage.geometry.convex_set module.
//!
//! The trait hierarchy mirrors the topological classification:
//! - ConvexSetBase: Foundation for all convex sets
//! - ConvexSetClosed: Closed convex sets
//! - ConvexSetCompact: Compact (bounded closed) convex sets
//! - ConvexSetRelativelyOpen: Relatively open convex sets
//! - ConvexSetOpen: Open convex sets

use std::fmt::Debug;

/// Base trait for all convex sets
///
/// A convex set is a subset S of a vector space such that for any two points
/// x, y in S and any t in [0, 1], the point (1-t)x + ty is also in S.
///
/// This trait defines the fundamental operations that all convex sets must support.
pub trait ConvexSetBase: Debug + Clone {
    /// Type representing points in the ambient space
    type Point: Clone + Debug;

    /// Type representing vectors/displacements
    type Vector: Clone + Debug;

    /// Type representing scalar values
    type Scalar: Clone + Debug;

    /// Returns the dimension of the convex set
    ///
    /// This is the dimension of the smallest affine subspace containing the set.
    /// Returns None if the set is empty (dimension -∞ in mathematical convention).
    fn dim(&self) -> Option<usize>;

    /// Returns the dimension of the ambient vector space
    fn ambient_dim(&self) -> usize;

    /// Returns the codimension relative to the ambient space
    ///
    /// This is `ambient_dim() - dim()`, or None if the set is empty.
    fn codimension(&self) -> Option<usize> {
        self.dim().map(|d| self.ambient_dim() - d)
    }

    /// Tests whether the set is empty
    fn is_empty(&self) -> bool {
        self.dim().is_none()
    }

    /// Tests whether the set is finite (contains finitely many points)
    ///
    /// A convex set is finite if and only if it has dimension 0 and is non-empty.
    fn is_finite(&self) -> bool {
        self.dim() == Some(0)
    }

    /// Tests whether the set represents the entire ambient space
    fn is_universe(&self) -> bool {
        false // Most sets are not the entire space; override if needed
    }

    /// Tests whether the set is full-dimensional (dim == ambient_dim)
    fn is_full_dimensional(&self) -> bool {
        self.dim().map_or(false, |d| d == self.ambient_dim())
    }

    /// Tests whether a point is contained in the set
    fn contains(&self, point: &Self::Point) -> bool;

    /// Returns a representative point from the relative interior of the set
    ///
    /// This method should return None if the set is empty.
    fn representative_point(&self) -> Option<Self::Point>;

    /// Returns whether this is a closed set in the topological sense
    fn is_closed(&self) -> bool {
        false // Default: unknown topology
    }

    /// Returns whether this is an open set in the topological sense
    fn is_open(&self) -> bool {
        false // Default: unknown topology
    }

    /// Returns whether this is a relatively open set
    ///
    /// A set is relatively open if it is open in its affine hull.
    fn is_relatively_open(&self) -> bool {
        false // Default: unknown
    }

    /// Returns whether this is a compact set
    ///
    /// A convex set is compact if and only if it is closed and bounded.
    fn is_compact(&self) -> bool {
        false // Default: not compact
    }
}

/// Trait for closed convex sets
///
/// A closed convex set contains all its limit points. In finite dimensions,
/// this is equivalent to being defined by a finite system of non-strict
/// linear inequalities.
pub trait ConvexSetClosed: ConvexSetBase {
    /// Closed sets are always closed
    fn is_closed_set(&self) -> bool {
        true
    }

    /// A closed set is open only if it's empty or the entire space
    fn is_open_set(&self) -> bool {
        self.is_empty() || self.is_universe()
    }
}

/// Trait for compact convex sets
///
/// A compact convex set is both closed and bounded. In finite dimensions,
/// these are exactly the convex hulls of finite point sets (polytopes).
pub trait ConvexSetCompact: ConvexSetClosed {
    /// Compact sets are always compact
    fn is_compact_set(&self) -> bool {
        true
    }

    /// A compact set is the universe only if it's zero-dimensional and non-empty
    fn is_universe_compact(&self) -> bool {
        self.dim() == Some(0) && !self.is_empty()
    }
}

/// Trait for relatively open convex sets
///
/// A relatively open convex set is open in its affine hull.
/// This means it contains no points from its relative boundary.
pub trait ConvexSetRelativelyOpen: ConvexSetBase {
    /// Relatively open sets are always relatively open
    fn is_relatively_open_set(&self) -> bool {
        true
    }

    /// A relatively open set is open (in ambient space) if it's empty or full-dimensional
    fn is_open_set_relatively(&self) -> bool {
        self.is_empty() || self.is_full_dimensional()
    }
}

/// Trait for open convex sets
///
/// An open convex set is open in the ambient vector space topology.
/// In finite dimensions, these cannot be defined by non-strict inequalities alone.
pub trait ConvexSetOpen: ConvexSetRelativelyOpen {
    /// Open sets are always open
    fn is_open_set_open(&self) -> bool {
        true
    }

    /// An open set is closed only if it's empty or the entire space
    fn is_closed_set_open(&self) -> bool {
        self.is_empty() || self.is_universe()
    }
}

/// Helper trait for affine hull computations
pub trait AffineHull: ConvexSetBase {
    /// Type representing the affine hull (typically a polyhedron or affine space)
    type AffineHullType;

    /// Computes the affine hull of the convex set
    ///
    /// The affine hull is the smallest affine subspace containing the set.
    fn affine_hull(&self) -> Self::AffineHullType;

    /// Returns an affine basis for the set
    ///
    /// An affine basis is a minimal set of points whose affine hull equals
    /// the affine hull of the entire set. The basis contains at most `dim() + 1` points.
    fn affine_basis(&self) -> Option<Vec<Self::Point>> {
        None // Optional method
    }
}

/// Helper trait for topological operations
pub trait TopologicalOperations: ConvexSetBase {
    /// Type representing the closure
    type ClosureType;

    /// Type representing the interior
    type InteriorType;

    /// Type representing the relative interior
    type RelativeInteriorType;

    /// Returns the topological closure of the set
    fn closure(&self) -> Self::ClosureType;

    /// Returns the interior of the set in the ambient space
    ///
    /// May be empty for lower-dimensional sets.
    fn interior(&self) -> Option<Self::InteriorType>;

    /// Returns the relative interior of the set
    ///
    /// The relative interior is the interior with respect to the affine hull.
    /// Non-empty for all non-empty convex sets.
    fn relative_interior(&self) -> Option<Self::RelativeInteriorType>;
}

/// Helper trait for geometric transformations
pub trait ConvexSetTransformations: ConvexSetBase {
    /// Returns the result of dilating (scaling) the set by a scalar
    ///
    /// For a set S and scalar c, dilation returns {c * x : x in S}.
    fn dilation(&self, scalar: &Self::Scalar) -> Self;

    /// Translates the set by a vector
    ///
    /// For a set S and vector v, translation returns {x + v : x in S}.
    fn translation(&self, vector: &Self::Vector) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example implementation for testing: A simple interval [a, b] in R
    #[derive(Debug, Clone)]
    struct Interval {
        a: f64,
        b: f64,
    }

    impl ConvexSetBase for Interval {
        type Point = f64;
        type Vector = f64;
        type Scalar = f64;

        fn dim(&self) -> Option<usize> {
            if self.a > self.b {
                None // Empty interval
            } else if self.a == self.b {
                Some(0) // Single point
            } else {
                Some(1) // Line segment
            }
        }

        fn ambient_dim(&self) -> usize {
            1
        }

        fn contains(&self, point: &Self::Point) -> bool {
            self.a <= *point && *point <= self.b
        }

        fn representative_point(&self) -> Option<Self::Point> {
            if self.is_empty() {
                None
            } else {
                Some((self.a + self.b) / 2.0)
            }
        }

        fn is_closed(&self) -> bool {
            true
        }

        fn is_compact(&self) -> bool {
            !self.is_empty() // Closed and bounded in R
        }
    }

    impl ConvexSetClosed for Interval {}
    impl ConvexSetCompact for Interval {}

    #[test]
    fn test_interval_basic_properties() {
        let interval = Interval { a: 1.0, b: 3.0 };

        assert_eq!(interval.dim(), Some(1));
        assert_eq!(interval.ambient_dim(), 1);
        assert_eq!(interval.codimension(), Some(0));
        assert!(!interval.is_empty());
        assert!(!interval.is_finite());
        assert!(interval.is_full_dimensional());
    }

    #[test]
    fn test_interval_containment() {
        let interval = Interval { a: 1.0, b: 3.0 };

        assert!(interval.contains(&2.0));
        assert!(interval.contains(&1.0));
        assert!(interval.contains(&3.0));
        assert!(!interval.contains(&0.0));
        assert!(!interval.contains(&4.0));
    }

    #[test]
    fn test_interval_representative_point() {
        let interval = Interval { a: 1.0, b: 3.0 };
        let pt = interval.representative_point();

        assert!(pt.is_some());
        assert!(interval.contains(&pt.unwrap()));
    }

    #[test]
    fn test_empty_interval() {
        let empty = Interval { a: 3.0, b: 1.0 };

        assert!(empty.is_empty());
        assert_eq!(empty.dim(), None);
        assert_eq!(empty.representative_point(), None);
    }

    #[test]
    fn test_point_interval() {
        let point = Interval { a: 2.0, b: 2.0 };

        assert!(!point.is_empty());
        assert!(point.is_finite());
        assert_eq!(point.dim(), Some(0));
        assert!(point.contains(&2.0));
        assert!(!point.contains(&2.1));
    }

    #[test]
    fn test_interval_topology() {
        let interval = Interval { a: 1.0, b: 3.0 };

        assert!(interval.is_closed());
        assert!(interval.is_compact());
        assert!(!interval.is_open());
    }
}
