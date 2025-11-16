//! Lattice polytopes
//!
//! This module provides types and functions for working with lattice polytopes -
//! polytopes whose vertices all have integer coordinates in a lattice.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::lattice_polytope::LatticePolytopeClass;
//! use rustmath_integers::Integer;
//!
//! // Create a triangle with vertices at (0,0), (1,0), (0,1)
//! let vertices = vec![
//!     vec![Integer::from(0), Integer::from(0)],
//!     vec![Integer::from(1), Integer::from(0)],
//!     vec![Integer::from(0), Integer::from(1)],
//! ];
//! let polytope = LatticePolytopeClass::new(vertices);
//!
//! assert_eq!(polytope.n_vertices(), 3);
//! assert_eq!(polytope.dim(), 2);
//! ```

use rustmath_core::Ring;
use rustmath_integers::Integer;
use std::fmt;
use std::hash::{Hash, Hasher};

/// A lattice polytope
///
/// A polytope whose vertices all have integer coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticePolytopeClass {
    /// The vertices of the polytope
    vertices: Vec<Vec<Integer>>,
    /// Dimension of the ambient space
    ambient_dim: usize,
    /// Cached dimension of the polytope
    dimension: Option<usize>,
}

impl LatticePolytopeClass {
    /// Create a new lattice polytope from vertices
    ///
    /// # Arguments
    ///
    /// * `vertices` - The vertices of the polytope (must have integer coordinates)
    ///
    /// # Panics
    ///
    /// Panics if vertices list is empty or if vertices have different dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::lattice_polytope::LatticePolytopeClass;
    /// use rustmath_integers::Integer;
    ///
    /// let vertices = vec![
    ///     vec![Integer::from(0), Integer::from(0)],
    ///     vec![Integer::from(1), Integer::from(0)],
    ///     vec![Integer::from(0), Integer::from(1)],
    /// ];
    /// let polytope = LatticePolytopeClass::new(vertices);
    /// ```
    pub fn new(vertices: Vec<Vec<Integer>>) -> Self {
        if vertices.is_empty() {
            panic!("Polytope must have at least one vertex");
        }

        let ambient_dim = vertices[0].len();

        // Verify all vertices have the same dimension
        for v in &vertices {
            if v.len() != ambient_dim {
                panic!("All vertices must have the same dimension");
            }
        }

        Self {
            vertices,
            ambient_dim,
            dimension: None,
        }
    }

    /// Get the vertices
    pub fn vertices(&self) -> &[Vec<Integer>] {
        &self.vertices
    }

    /// Get the number of vertices
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get the dimension of the ambient space
    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    /// Get the dimension of the polytope
    ///
    /// This is the dimension of the affine hull of the vertices.
    pub fn dim(&self) -> usize {
        // For a proper implementation, we would compute the affine dimension
        // by finding the rank of the matrix of vertex differences
        // For now, we use a simplified computation
        if self.n_vertices() == 1 {
            0
        } else {
            // Approximate dimension based on number of vertices
            (self.n_vertices() - 1).min(self.ambient_dim)
        }
    }

    /// Get the lattice dimension
    ///
    /// This is the dimension of the smallest affine lattice containing the polytope.
    pub fn lattice_dim(&self) -> usize {
        self.ambient_dim
    }

    /// Check if the polytope is reflexive
    ///
    /// A polytope is reflexive if both it and its polar dual are lattice polytopes.
    /// This is a simplified check.
    pub fn is_reflexive(&self) -> bool {
        // For a proper implementation, we would:
        // 1. Compute the polar dual
        // 2. Check if all vertices of the dual are lattice points
        // For now, return false as a conservative estimate
        false
    }

    /// Get all lattice points within the polytope
    ///
    /// Returns all integer points that lie inside or on the boundary of the polytope.
    ///
    /// Note: This is a placeholder that returns only vertices.
    /// A full implementation would enumerate all interior lattice points.
    pub fn points(&self) -> Vec<Vec<Integer>> {
        // For a simple implementation, return the vertices
        // A complete implementation would compute all interior lattice points
        self.vertices.clone()
    }

    /// Get the number of lattice points
    pub fn n_points(&self) -> usize {
        self.points().len()
    }

    /// Compute facet normals
    ///
    /// Returns the normal vectors to each facet of the polytope.
    ///
    /// Note: This is a placeholder implementation.
    pub fn facet_normals(&self) -> Vec<Vec<Integer>> {
        // For a proper implementation, we would compute the convex hull
        // and extract the facet normals
        vec![]
    }

    /// Get the polar dual
    ///
    /// For a reflexive polytope, returns the polar dual.
    /// Returns None if not reflexive or if computation is not implemented.
    pub fn polar(&self) -> Option<Self> {
        // The polar dual of a polytope with vertices v_i is:
        // { x : <x, v_i> >= -1 for all i }
        // This requires solving a dual polytope problem
        None
    }

    /// Get faces of a specific dimension
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of faces to return
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::lattice_polytope::LatticePolytopeClass;
    /// use rustmath_integers::Integer;
    ///
    /// let vertices = vec![
    ///     vec![Integer::from(0), Integer::from(0)],
    ///     vec![Integer::from(1), Integer::from(0)],
    ///     vec![Integer::from(0), Integer::from(1)],
    /// ];
    /// let polytope = LatticePolytopeClass::new(vertices);
    ///
    /// // Dimension 0 faces are vertices
    /// let vertex_faces = polytope.faces(0);
    /// ```
    pub fn faces(&self, dimension: usize) -> Vec<Self> {
        if dimension == 0 {
            // 0-dimensional faces are vertices
            self.vertices
                .iter()
                .map(|v| Self::new(vec![v.clone()]))
                .collect()
        } else {
            // For higher dimensions, we would need to compute the face lattice
            vec![]
        }
    }

    /// Get facets (codimension-1 faces)
    pub fn facets(&self) -> Vec<Self> {
        if self.dim() == 0 {
            vec![]
        } else {
            self.faces(self.dim().saturating_sub(1))
        }
    }

    /// Get edges (1-dimensional faces)
    pub fn edges(&self) -> Vec<Self> {
        self.faces(1)
    }

    /// Check if this polytope contains a point
    ///
    /// # Arguments
    ///
    /// * `point` - The point to check
    pub fn contains(&self, point: &[Integer]) -> bool {
        if point.len() != self.ambient_dim {
            return false;
        }

        // For a proper implementation, we would check if the point
        // satisfies all facet inequalities
        // For now, check if the point is one of the vertices
        self.vertices.iter().any(|v| v.as_slice() == point)
    }

    /// Compute the volume of the polytope
    ///
    /// Returns the lattice volume (normalized volume).
    /// Note: This is a placeholder that returns 0.
    pub fn volume(&self) -> Integer {
        // For a proper implementation, we would use the lattice point
        // enumeration theorem or triangulation
        Integer::from(0)
    }

    /// Check if two polytopes are equal
    ///
    /// Two polytopes are equal if they have the same set of vertices.
    pub fn equals(&self, other: &Self) -> bool {
        if self.n_vertices() != other.n_vertices() {
            return false;
        }

        // Check if all vertices match (order-independent)
        for v in &self.vertices {
            if !other.vertices.contains(v) {
                return false;
            }
        }

        true
    }
}

impl Hash for LatticePolytopeClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash based on vertices
        self.n_vertices().hash(state);
        self.ambient_dim.hash(state);
        // For proper hashing, we should sort vertices first
        for v in &self.vertices {
            for coord in v {
                coord.hash(state);
            }
        }
    }
}

impl fmt::Display for LatticePolytopeClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LatticePolytope({}-dimensional with {} vertices)",
            self.dim(),
            self.n_vertices()
        )
    }
}

/// Create a lattice polytope from vertices
///
/// This is a convenience function that wraps LatticePolytopeClass::new.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::lattice_polytope;
/// use rustmath_integers::Integer;
///
/// let vertices = vec![
///     vec![Integer::from(0), Integer::from(0)],
///     vec![Integer::from(1), Integer::from(0)],
/// ];
/// let polytope = lattice_polytope(vertices);
/// ```
pub fn lattice_polytope(vertices: Vec<Vec<Integer>>) -> LatticePolytopeClass {
    LatticePolytopeClass::new(vertices)
}

/// Convex hull of a set of points
///
/// Computes the convex hull and returns it as a lattice polytope.
///
/// # Arguments
///
/// * `points` - The points to compute the convex hull of
///
/// Note: This is a simplified implementation that just wraps the points.
/// A full implementation would compute the actual convex hull.
pub fn convex_hull(points: Vec<Vec<Integer>>) -> LatticePolytopeClass {
    // A proper implementation would compute the actual convex hull
    // removing interior points
    LatticePolytopeClass::new(points)
}

/// Create a cross-polytope (orthoplex) in dimension n
///
/// The cross-polytope is the convex hull of the standard basis vectors
/// and their negatives: {±e_i : i = 1..n}
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::cross_polytope;
///
/// // 2D cross-polytope (diamond/square rotated 45 degrees)
/// let cross = cross_polytope(2);
/// assert_eq!(cross.n_vertices(), 4);
/// ```
pub fn cross_polytope(dimension: usize) -> LatticePolytopeClass {
    if dimension == 0 {
        panic!("Dimension must be at least 1");
    }

    let mut vertices = Vec::new();

    // Add ±e_i for each coordinate direction
    for i in 0..dimension {
        // +e_i
        let mut v_pos = vec![Integer::from(0); dimension];
        v_pos[i] = Integer::from(1);
        vertices.push(v_pos);

        // -e_i
        let mut v_neg = vec![Integer::from(0); dimension];
        v_neg[i] = Integer::from(-1);
        vertices.push(v_neg);
    }

    LatticePolytopeClass::new(vertices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_polytope() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        assert_eq!(polytope.n_vertices(), 3);
        assert_eq!(polytope.ambient_dim(), 2);
    }

    #[test]
    fn test_vertices() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices.clone());

        assert_eq!(polytope.vertices(), &vertices);
    }

    #[test]
    fn test_dim() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        // For 3 vertices in 2D, dimension should be 2
        assert_eq!(polytope.dim(), 2);
    }

    #[test]
    fn test_single_point() {
        let vertices = vec![vec![Integer::from(1), Integer::from(2)]];
        let polytope = LatticePolytopeClass::new(vertices);

        assert_eq!(polytope.dim(), 0);
        assert_eq!(polytope.n_vertices(), 1);
    }

    #[test]
    fn test_contains() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        assert!(polytope.contains(&[Integer::from(0), Integer::from(0)]));
        assert!(polytope.contains(&[Integer::from(1), Integer::from(0)]));
    }

    #[test]
    fn test_points() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        let points = polytope.points();
        assert_eq!(points.len(), 2);
    }

    #[test]
    fn test_faces() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        let vertex_faces = polytope.faces(0);
        assert_eq!(vertex_faces.len(), 3);
    }

    #[test]
    fn test_lattice_polytope_function() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = lattice_polytope(vertices);

        assert_eq!(polytope.n_vertices(), 2);
    }

    #[test]
    fn test_convex_hull() {
        let points = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = convex_hull(points);

        assert_eq!(polytope.n_vertices(), 3);
    }

    #[test]
    fn test_cross_polytope() {
        let cross = cross_polytope(2);

        assert_eq!(cross.n_vertices(), 4);
        assert_eq!(cross.ambient_dim(), 2);

        // Vertices should be (±1, 0) and (0, ±1)
        let vertices = cross.vertices();
        assert!(vertices.contains(&vec![Integer::from(1), Integer::from(0)]));
        assert!(vertices.contains(&vec![Integer::from(-1), Integer::from(0)]));
        assert!(vertices.contains(&vec![Integer::from(0), Integer::from(1)]));
        assert!(vertices.contains(&vec![Integer::from(0), Integer::from(-1)]));
    }

    #[test]
    fn test_cross_polytope_3d() {
        let cross = cross_polytope(3);

        // 3D cross-polytope has 6 vertices
        assert_eq!(cross.n_vertices(), 6);
        assert_eq!(cross.ambient_dim(), 3);
    }

    #[test]
    fn test_equals() {
        let vertices1 = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope1 = LatticePolytopeClass::new(vertices1.clone());
        let polytope2 = LatticePolytopeClass::new(vertices1);

        assert!(polytope1.equals(&polytope2));
    }

    #[test]
    fn test_display() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        let display = format!("{}", polytope);
        assert!(display.contains("LatticePolytope"));
    }
}
