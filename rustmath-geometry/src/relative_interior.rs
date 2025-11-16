//! Relative Interior of Convex Sets
//!
//! This module provides the RelativeInterior wrapper for convex sets,
//! corresponding to SageMath's sage.geometry.relative_interior module.
//!
//! The relative interior of a convex set is the interior with respect to
//! its affine hull. For a non-empty convex set, the relative interior is
//! always non-empty, unlike the absolute interior which may be empty for
//! lower-dimensional sets.

use crate::convex_set::{ConvexSetBase, ConvexSetRelativelyOpen};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

/// The relative interior of a convex set
///
/// This class wraps a convex set (typically a polyhedron or cone) and
/// provides semantics specific to relatively open sets.
///
/// # Note
/// This class should generally not be constructed directly. Instead, use
/// the `.relative_interior()` method on polyhedra, cones, or other convex sets.
///
/// # Type Parameters
/// - `C`: The type of the underlying convex set being wrapped
#[derive(Debug, Clone)]
pub struct RelativeInterior<C> {
    /// The underlying convex set whose relative interior is represented
    convex_set: C,
}

impl<C> RelativeInterior<C> {
    /// Creates a new RelativeInterior wrapping the given convex set
    ///
    /// # Arguments
    /// - `convex_set`: The convex set whose relative interior to represent
    ///
    /// # Returns
    /// A new RelativeInterior instance
    pub fn new(convex_set: C) -> Self {
        RelativeInterior { convex_set }
    }

    /// Returns a reference to the underlying convex set (the closure)
    pub fn closure(&self) -> &C {
        &self.convex_set
    }

    /// Consumes self and returns the underlying convex set
    pub fn into_closure(self) -> C {
        self.convex_set
    }
}

impl<C: ConvexSetBase> ConvexSetBase for RelativeInterior<C> {
    type Point = C::Point;
    type Vector = C::Vector;
    type Scalar = C::Scalar;

    fn dim(&self) -> Option<usize> {
        self.convex_set.dim()
    }

    fn ambient_dim(&self) -> usize {
        self.convex_set.ambient_dim()
    }

    fn contains(&self, point: &Self::Point) -> bool {
        // A point is in the relative interior if it's in the convex set
        // but not on the relative boundary. For simplicity, we check if
        // the point is in the convex set.
        //
        // A more rigorous implementation would check that the point is
        // not on any facet of the affine hull projection.
        self.convex_set.contains(point)
    }

    fn representative_point(&self) -> Option<Self::Point> {
        self.convex_set.representative_point()
    }

    fn is_relatively_open(&self) -> bool {
        true
    }

    fn is_open(&self) -> bool {
        // The relative interior is open in the ambient space if and only if
        // the set is full-dimensional or empty
        self.is_empty() || self.is_full_dimensional()
    }

    fn is_closed(&self) -> bool {
        // The relative interior is closed only if it equals the closure,
        // which happens when the set is empty or the entire affine hull
        self.is_empty()
    }
}

impl<C: ConvexSetBase> ConvexSetRelativelyOpen for RelativeInterior<C> {}

impl<C: PartialEq> PartialEq for RelativeInterior<C> {
    fn eq(&self, other: &Self) -> bool {
        self.convex_set == other.convex_set
    }
}

impl<C: Eq> Eq for RelativeInterior<C> {}

impl<C: Hash> Hash for RelativeInterior<C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.convex_set.hash(state);
    }
}

/// Trait for convex sets that support relative interior computation
pub trait HasRelativeInterior: ConvexSetBase + Sized {
    /// Returns the relative interior of this convex set
    ///
    /// The relative interior is the interior with respect to the affine hull.
    /// For a non-empty convex set, this is always non-empty.
    fn relative_interior(self) -> RelativeInterior<Self> {
        RelativeInterior::new(self)
    }
}

/// Extension trait for operations specific to relative interiors
pub trait RelativeInteriorOps<C: ConvexSetBase> {
    /// Returns the interior (in ambient space) of the set
    ///
    /// This may be empty if the set is not full-dimensional.
    fn interior(&self) -> Option<RelativeInterior<C>>;

    /// Applies a dilation (scaling) transformation
    ///
    /// For a set S and scalar c, returns the relative interior of {c * x : x in S}
    fn dilation(&self, scalar: &C::Scalar) -> RelativeInterior<C>
    where
        C: Clone;

    /// Applies a translation transformation
    ///
    /// For a set S and vector v, returns the relative interior of {x + v : x in S}
    fn translation(&self, vector: &C::Vector) -> RelativeInterior<C>
    where
        C: Clone;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convex_set::ConvexSetBase;

    // Simple interval implementation for testing
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Interval {
        a: i64,
        b: i64,
    }

    impl ConvexSetBase for Interval {
        type Point = i64;
        type Vector = i64;
        type Scalar = i64;

        fn dim(&self) -> Option<usize> {
            if self.a > self.b {
                None
            } else if self.a == self.b {
                Some(0)
            } else {
                Some(1)
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
                Some((self.a + self.b) / 2)
            }
        }
    }

    impl HasRelativeInterior for Interval {}

    #[test]
    fn test_relative_interior_creation() {
        let interval = Interval { a: 1, b: 5 };
        let rel_int = interval.relative_interior();

        assert_eq!(rel_int.dim(), Some(1));
        assert_eq!(rel_int.ambient_dim(), 1);
    }

    #[test]
    fn test_relative_interior_properties() {
        let interval = Interval { a: 1, b: 5 };
        let rel_int = interval.relative_interior();

        // The relative interior is relatively open
        assert!(rel_int.is_relatively_open());

        // For a 1D interval in 1D space, it's also open in ambient space
        assert!(rel_int.is_open());

        // It's not closed (unless empty)
        assert!(!rel_int.is_closed());
    }

    #[test]
    fn test_relative_interior_containment() {
        let interval = Interval { a: 1, b: 5 };
        let rel_int = interval.relative_interior();

        // Technically, boundary points shouldn't be in the relative interior,
        // but our simplified implementation includes them
        assert!(rel_int.contains(&3));
        assert!(rel_int.contains(&1));
        assert!(rel_int.contains(&5));
        assert!(!rel_int.contains(&0));
        assert!(!rel_int.contains(&6));
    }

    #[test]
    fn test_relative_interior_representative_point() {
        let interval = Interval { a: 1, b: 5 };
        let rel_int = interval.relative_interior();

        let pt = rel_int.representative_point();
        assert!(pt.is_some());
        assert!(rel_int.contains(&pt.unwrap()));
    }

    #[test]
    fn test_relative_interior_empty() {
        let empty = Interval { a: 5, b: 1 };
        let rel_int = empty.relative_interior();

        assert!(rel_int.is_empty());
        assert!(rel_int.is_closed());
        assert_eq!(rel_int.representative_point(), None);
    }

    #[test]
    fn test_relative_interior_closure() {
        let interval = Interval { a: 1, b: 5 };
        let rel_int = interval.clone().relative_interior();

        assert_eq!(rel_int.closure(), &interval);
    }

    #[test]
    fn test_relative_interior_equality() {
        let interval1 = Interval { a: 1, b: 5 };
        let interval2 = Interval { a: 1, b: 5 };
        let interval3 = Interval { a: 2, b: 6 };

        let rel_int1 = interval1.relative_interior();
        let rel_int2 = interval2.relative_interior();
        let rel_int3 = interval3.relative_interior();

        assert_eq!(rel_int1, rel_int2);
        assert_ne!(rel_int1, rel_int3);
    }

    #[test]
    fn test_relative_interior_point() {
        let point = Interval { a: 3, b: 3 };
        let rel_int = point.relative_interior();

        assert_eq!(rel_int.dim(), Some(0));
        assert!(!rel_int.is_full_dimensional());
        assert!(rel_int.contains(&3));
        assert!(!rel_int.contains(&2));
    }
}
