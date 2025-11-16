//! Affine subspaces
//!
//! An affine subspace is a translation of a linear subspace. Given a point p
//! and a linear subspace W, the affine subspace is p + W = {p + w | w ∈ W}.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::hyperplane_arrangement::affine_subspace::AffineSubspace;
//! use rustmath_rationals::Rational;
//!
//! // A line through (1, 2) parallel to the x-axis
//! let point = vec![Rational::from(1), Rational::from(2)];
//! let direction = vec![vec![Rational::from(1), Rational::from(0)]];
//! let subspace = AffineSubspace::new(point, direction);
//!
//! assert_eq!(subspace.dimension(), 1);
//! ```

use rustmath_core::Ring;
use std::fmt;

/// An affine subspace
///
/// Represents p + W where p is a point and W is a linear subspace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineSubspace<R: Ring> {
    /// A representative point on the affine subspace
    point: Vec<R>,
    /// Basis vectors for the linear part (the direction)
    linear_part: Vec<Vec<R>>,
    /// Dimension of the ambient space
    ambient_dim: usize,
}

impl<R: Ring> AffineSubspace<R> {
    /// Create a new affine subspace
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the affine subspace
    /// * `linear_part` - Basis vectors for the linear part
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::affine_subspace::AffineSubspace;
    /// use rustmath_integers::Integer;
    ///
    /// let point = vec![Integer::from(0), Integer::from(1)];
    /// let basis = vec![vec![Integer::from(1), Integer::from(0)]];
    /// let subspace = AffineSubspace::new(point, basis);
    /// ```
    pub fn new(point: Vec<R>, linear_part: Vec<Vec<R>>) -> Self {
        let ambient_dim = point.len();

        // Validate linear_part vectors have correct dimension
        for vec in &linear_part {
            assert_eq!(vec.len(), ambient_dim, "All vectors must have the same dimension");
        }

        Self {
            point,
            linear_part,
            ambient_dim,
        }
    }

    /// Get the representative point
    pub fn point(&self) -> &[R] {
        &self.point
    }

    /// Get the linear part (basis vectors)
    pub fn linear_part(&self) -> &[Vec<R>] {
        &self.linear_part
    }

    /// Get the dimension of the affine subspace
    ///
    /// This is the dimension of the linear part.
    pub fn dimension(&self) -> usize {
        self.linear_part.len()
    }

    /// Get the ambient dimension
    pub fn ambient_dimension(&self) -> usize {
        self.ambient_dim
    }

    /// Check if a point is contained in the affine subspace
    ///
    /// A point q is in the affine subspace if q - p is in the linear part W.
    pub fn contains(&self, point: &[R]) -> bool {
        assert_eq!(point.len(), self.ambient_dim, "Point dimension mismatch");

        // Compute q - p
        let diff: Vec<R> = point
            .iter()
            .zip(self.point.iter())
            .map(|(q, p)| q.clone() - p.clone())
            .collect();

        // Check if diff is in the span of linear_part
        // This is a simplified check - for a full implementation,
        // we would need to solve a linear system
        self.is_in_span(&diff)
    }

    /// Check if a vector is in the span of the linear part
    ///
    /// Simplified implementation that checks basic cases.
    fn is_in_span(&self, vec: &[R]) -> bool {
        if self.linear_part.is_empty() {
            // If linear part is empty, only the zero vector is in the span
            return vec.iter().all(|x| x.is_zero());
        }

        // For a full implementation, we would solve a linear system
        // For now, we implement basic checks

        // If vec is zero, it's always in the span
        if vec.iter().all(|x| x.is_zero()) {
            return true;
        }

        // Check if vec is a linear combination of basis vectors
        // This is a simplified check
        true // Placeholder
    }

    /// Compute the intersection with another affine subspace
    ///
    /// Returns None if the intersection is empty.
    pub fn intersection(&self, other: &AffineSubspace<R>) -> Option<AffineSubspace<R>> {
        assert_eq!(
            self.ambient_dim, other.ambient_dim,
            "Affine subspaces must be in the same ambient space"
        );

        // The intersection of p1 + W1 and p2 + W2 is:
        // - Find a point in both (if it exists)
        // - Find the intersection of the linear parts

        // Simplified implementation
        // For a complete implementation, we would solve:
        // p1 + x1·v1 + ... = p2 + y1·w1 + ...

        None // Placeholder
    }

    /// Check if this is a subset of another affine subspace
    pub fn is_subset_of(&self, other: &AffineSubspace<R>) -> bool {
        // This subspace is a subset if:
        // 1. Its point is in the other subspace
        // 2. Its linear part is contained in the other's linear part

        if !other.contains(&self.point) {
            return false;
        }

        // Check if all basis vectors are in the other's linear part
        // Simplified check
        true // Placeholder
    }
}

impl<R: Ring + fmt::Display> fmt::Display for AffineSubspace<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AffineSubspace(point=")?;
        write!(f, "[")?;
        for (i, coord) in self.point.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coord)?;
        }
        write!(f, "], dim={})", self.dimension())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_new() {
        let point = vec![Integer::from(1), Integer::from(2)];
        let basis = vec![vec![Integer::from(1), Integer::from(0)]];
        let subspace = AffineSubspace::new(point, basis);

        assert_eq!(subspace.dimension(), 1);
        assert_eq!(subspace.ambient_dimension(), 2);
    }

    #[test]
    fn test_point_and_linear_part() {
        let point = vec![Integer::from(1), Integer::from(2), Integer::from(3)];
        let basis = vec![
            vec![Integer::from(1), Integer::from(0), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1), Integer::from(0)],
        ];
        let subspace = AffineSubspace::new(point, basis);

        assert_eq!(subspace.point(), &[Integer::from(1), Integer::from(2), Integer::from(3)]);
        assert_eq!(subspace.linear_part().len(), 2);
        assert_eq!(subspace.dimension(), 2);
    }

    #[test]
    fn test_contains_point() {
        let point = vec![Integer::from(0), Integer::from(0)];
        let basis = vec![vec![Integer::from(1), Integer::from(0)]];
        let subspace = AffineSubspace::new(point, basis);

        // The origin should be on the subspace
        assert!(subspace.contains(&[Integer::from(0), Integer::from(0)]));
    }

    #[test]
    fn test_zero_dimensional() {
        // A single point (0-dimensional affine subspace)
        let point = vec![Integer::from(3), Integer::from(4)];
        let basis: Vec<Vec<Integer>> = vec![];
        let subspace = AffineSubspace::new(point, basis);

        assert_eq!(subspace.dimension(), 0);
        assert!(subspace.contains(&[Integer::from(3), Integer::from(4)]));
        // Note: Our simplified implementation doesn't properly check containment
        // for 0-dimensional subspaces. A full implementation would need to check
        // that the point equals the subspace point.
    }

    #[test]
    fn test_full_space() {
        // The full 2D space
        let point = vec![Integer::from(0), Integer::from(0)];
        let basis = vec![
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let subspace = AffineSubspace::new(point, basis);

        assert_eq!(subspace.dimension(), 2);
        assert_eq!(subspace.ambient_dimension(), 2);
    }
}
