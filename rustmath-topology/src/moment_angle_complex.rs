//! Moment-Angle Complex Module
//!
//! Implements moment-angle complexes from toric topology.
//!
//! Moment-angle complexes are topological spaces associated with simplicial complexes,
//! important in toric topology and geometric combinatorics.
//!
//! This mirrors SageMath's `sage.topology.moment_angle_complex`.

use rustmath_integers::Integer;
use std::collections::HashSet;
use std::fmt;

/// A moment-angle complex Z_K associated with a simplicial complex K.
///
/// For a simplicial complex K on n vertices, Z_K is a subspace of (S^1)^n
/// or (D^2)^n defined by the combinatorics of K.
#[derive(Debug, Clone)]
pub struct MomentAngleComplex {
    /// Number of vertices in the underlying simplicial complex
    n_vertices: usize,
    /// Facets of the underlying simplicial complex
    facets: Vec<HashSet<usize>>,
    /// Name of the complex
    name: Option<String>,
}

impl MomentAngleComplex {
    /// Create a new moment-angle complex from a simplicial complex.
    ///
    /// # Arguments
    /// * `n_vertices` - Number of vertices
    /// * `facets` - Maximal faces of the simplicial complex
    pub fn new(n_vertices: usize, facets: Vec<HashSet<usize>>) -> Self {
        Self {
            n_vertices,
            facets,
            name: None,
        }
    }

    /// Create a moment-angle complex with a name.
    pub fn with_name(n_vertices: usize, facets: Vec<HashSet<usize>>, name: &str) -> Self {
        Self {
            n_vertices,
            facets,
            name: Some(name.to_string()),
        }
    }

    /// Get the number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    /// Get the facets.
    pub fn facets(&self) -> &[HashSet<usize>] {
        &self.facets
    }

    /// Get the dimension of the underlying simplicial complex.
    pub fn dimension(&self) -> usize {
        self.facets
            .iter()
            .map(|f| if f.is_empty() { 0 } else { f.len() - 1 })
            .max()
            .unwrap_or(0)
    }

    /// Compute the dimension of the moment-angle complex as a CW complex.
    ///
    /// dim(Z_K) = n + dim(K)
    pub fn topological_dimension(&self) -> usize {
        self.n_vertices + self.dimension()
    }

    /// Check if the underlying simplicial complex is a simplex.
    pub fn is_simplex(&self) -> bool {
        self.facets.len() == 1 && self.facets[0].len() == self.n_vertices
    }

    /// Get the h-vector of the underlying simplicial complex.
    ///
    /// This is used in computing topological invariants.
    pub fn h_vector(&self) -> Vec<usize> {
        // Simplified implementation - returns a placeholder
        vec![1] // Would need full simplicial complex implementation
    }

    /// Get the name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl fmt::Display for MomentAngleComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "Moment-angle complex {}", name)?;
        } else {
            write!(f, "Moment-angle complex")?;
        }
        write!(
            f,
            " (n={}, dim={})",
            self.n_vertices,
            self.topological_dimension()
        )
    }
}

/// Examples of moment-angle complexes.
pub struct MomentAngleComplexExamples;

impl MomentAngleComplexExamples {
    /// Moment-angle complex of a simplex.
    ///
    /// Z_Δ^n ≅ (D^2)^n (contractible)
    pub fn simplex(n: usize) -> MomentAngleComplex {
        let vertices: HashSet<usize> = (0..n).collect();
        MomentAngleComplex::with_name(n, vec![vertices], &format!("Z_Δ^{}", n))
    }

    /// Moment-angle complex of a boundary of simplex.
    ///
    /// Z_{∂Δ^n} ≅ S^{2n-1}
    pub fn sphere(n: usize) -> MomentAngleComplex {
        // Boundary of n-simplex has all (n-1)-faces
        let mut facets = Vec::new();
        for i in 0..=n {
            let mut face: HashSet<usize> = (0..=n).collect();
            face.remove(&i);
            facets.push(face);
        }
        MomentAngleComplex::with_name(n + 1, facets, &format!("Z_{{∂Δ^{}}}", n))
    }

    /// Moment-angle complex of a polygon.
    pub fn polygon(n: usize) -> MomentAngleComplex {
        let mut facets = Vec::new();
        for i in 0..n {
            let mut face = HashSet::new();
            face.insert(i);
            face.insert((i + 1) % n);
            facets.push(face);
        }
        MomentAngleComplex::with_name(n, facets, &format!("Z_{{C_{}}}", n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moment_angle_complex_creation() {
        let facets = vec![{
            let mut f = HashSet::new();
            f.insert(0);
            f.insert(1);
            f.insert(2);
            f
        }];
        let mac = MomentAngleComplex::new(3, facets);

        assert_eq!(mac.n_vertices(), 3);
        assert_eq!(mac.dimension(), 2);
        assert_eq!(mac.topological_dimension(), 5); // 3 + 2
    }

    #[test]
    fn test_moment_angle_complex_is_simplex() {
        let facets = vec![{
            let mut f = HashSet::new();
            f.insert(0);
            f.insert(1);
            f.insert(2);
            f
        }];
        let mac = MomentAngleComplex::new(3, facets);
        assert!(mac.is_simplex());

        let facets2 = vec![
            {
                let mut f = HashSet::new();
                f.insert(0);
                f.insert(1);
                f
            },
            {
                let mut f = HashSet::new();
                f.insert(1);
                f.insert(2);
                f
            },
        ];
        let mac2 = MomentAngleComplex::new(3, facets2);
        assert!(!mac2.is_simplex());
    }

    #[test]
    fn test_moment_angle_complex_examples_simplex() {
        let mac = MomentAngleComplexExamples::simplex(3);
        assert_eq!(mac.n_vertices(), 3);
        assert!(mac.is_simplex());
        assert_eq!(mac.name(), Some("Z_Δ^3"));
    }

    #[test]
    fn test_moment_angle_complex_examples_sphere() {
        let mac = MomentAngleComplexExamples::sphere(2);
        assert_eq!(mac.n_vertices(), 3);
        assert_eq!(mac.facets().len(), 3); // Boundary of 2-simplex has 3 faces
    }

    #[test]
    fn test_moment_angle_complex_examples_polygon() {
        let mac = MomentAngleComplexExamples::polygon(5);
        assert_eq!(mac.n_vertices(), 5);
        assert_eq!(mac.facets().len(), 5); // Pentagon has 5 edges
        assert_eq!(mac.dimension(), 1); // Edges are 1-dimensional
    }

    #[test]
    fn test_topological_dimension() {
        let mac = MomentAngleComplexExamples::simplex(3);
        assert_eq!(mac.topological_dimension(), 5); // n=3, dim=2, total=5
    }

    #[test]
    fn test_with_name() {
        let facets = vec![{
            let mut f = HashSet::new();
            f.insert(0);
            f.insert(1);
            f
        }];
        let mac = MomentAngleComplex::with_name(2, facets, "Test MAC");
        assert_eq!(mac.name(), Some("Test MAC"));
    }
}
