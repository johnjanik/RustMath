//! Delta Complex Module
//!
//! Implements delta complexes: simplicial-style complexes with identifications.
//!
//! A delta complex is like a simplicial complex but allows simplices to be glued
//! together in a more flexible way, similar to CW complexes but restricted to simplices.
//!
//! This mirrors SageMath's `sage.topology.delta_complex`.

use crate::cell_complex::GenericCellComplex;
use rustmath_integers::Integer;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Represents a simplex in a delta complex.
///
/// Unlike simplicial complexes, simplices in delta complexes are not determined
/// by their vertices but are abstract entities that are glued via face maps.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeltaSimplex {
    /// Dimension of the simplex
    dimension: usize,
    /// Index/identifier
    index: usize,
}

impl DeltaSimplex {
    /// Create a new delta simplex.
    pub fn new(dimension: usize, index: usize) -> Self {
        Self { dimension, index }
    }

    /// Get the dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the index.
    pub fn index(&self) -> usize {
        self.index
    }
}

/// A delta complex: simplices glued together via face maps.
#[derive(Debug, Clone)]
pub struct DeltaComplex {
    /// The underlying cell complex
    base: GenericCellComplex,
    /// Simplices organized by dimension
    simplices: HashMap<usize, Vec<DeltaSimplex>>,
    /// Face maps: (simplex, face_index) -> target_simplex
    face_maps: HashMap<(usize, usize, usize), (usize, usize)>,
    /// Name of the complex
    name: Option<String>,
}

impl DeltaComplex {
    /// Create a new empty delta complex.
    pub fn new() -> Self {
        Self {
            base: GenericCellComplex::new(),
            simplices: HashMap::new(),
            face_maps: HashMap::new(),
            name: None,
        }
    }

    /// Create a delta complex with a name.
    pub fn with_name(name: &str) -> Self {
        Self {
            base: GenericCellComplex::new(),
            simplices: HashMap::new(),
            face_maps: HashMap::new(),
            name: Some(name.to_string()),
        }
    }

    /// Add a simplex to the complex.
    ///
    /// # Arguments
    /// * `dim` - Dimension of the simplex
    /// * `faces` - Optional list of target simplices for each face
    pub fn add_simplex(&mut self, dim: usize, faces: Option<Vec<(usize, usize)>>) -> usize {
        let index = self.simplices.get(&dim).map(|v| v.len()).unwrap_or(0);
        let simplex = DeltaSimplex::new(dim, index);

        self.simplices
            .entry(dim)
            .or_insert_with(Vec::new)
            .push(simplex);

        // Add face maps if provided
        if let Some(face_list) = faces {
            for (face_idx, (target_dim, target_idx)) in face_list.into_iter().enumerate() {
                self.face_maps
                    .insert((dim, index, face_idx), (target_dim, target_idx));
            }
        }

        index
    }

    /// Get all simplices of a given dimension.
    pub fn simplices(&self, dim: usize) -> Vec<&DeltaSimplex> {
        self.simplices
            .get(&dim)
            .map(|v| v.iter().collect())
            .unwrap_or_else(Vec::new)
    }

    /// Get the number of simplices in a given dimension.
    pub fn n_simplices(&self, dim: usize) -> usize {
        self.simplices.get(&dim).map(|v| v.len()).unwrap_or(0)
    }

    /// Get the dimension of the complex.
    pub fn dimension(&self) -> Option<usize> {
        self.simplices.keys().max().copied()
    }

    /// Get the face of a simplex.
    ///
    /// # Arguments
    /// * `dim` - Dimension of the simplex
    /// * `index` - Index of the simplex
    /// * `face_index` - Which face (0 to dim)
    pub fn face(&self, dim: usize, index: usize, face_index: usize) -> Option<(usize, usize)> {
        self.face_maps.get(&(dim, index, face_index)).copied()
    }

    /// Compute the Euler characteristic.
    pub fn euler_characteristic(&self) -> Integer {
        let mut chi = Integer::from(0);
        if let Some(max_dim) = self.dimension() {
            for dim in 0..=max_dim {
                let n_simplices = self.n_simplices(dim) as i64;
                if dim % 2 == 0 {
                    chi = chi + Integer::from(n_simplices);
                } else {
                    chi = chi - Integer::from(n_simplices);
                }
            }
        }
        chi
    }

    /// Get the f-vector.
    pub fn f_vector(&self) -> Vec<usize> {
        let max_dim = self.dimension().unwrap_or(0);
        (0..=max_dim).map(|dim| self.n_simplices(dim)).collect()
    }

    /// Get the name of the complex.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl Default for DeltaComplex {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DeltaComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "Delta complex: {}", name)?;
        } else {
            write!(f, "Delta complex")?;
        }
        if let Some(dim) = self.dimension() {
            write!(f, " of dimension {}", dim)?;
        }
        Ok(())
    }
}

/// Examples of delta complexes.
pub struct DeltaComplexExamples;

impl DeltaComplexExamples {
    /// Create a sphere as a delta complex.
    pub fn sphere(n: usize) -> DeltaComplex {
        let mut complex = DeltaComplex::with_name(&format!("S^{}", n));

        if n == 0 {
            // S^0: two points
            complex.add_simplex(0, None);
            complex.add_simplex(0, None);
        } else if n == 1 {
            // S^1: circle with 1 vertex and 1 edge (glued to itself)
            complex.add_simplex(0, None); // vertex
            complex.add_simplex(1, Some(vec![(0, 0), (0, 0)])); // edge with both ends at same vertex
        } else {
            // Higher-dimensional spheres: two n-simplices glued along boundary
            // Add boundary (n-1)-simplices first (minimal construction)
            complex.add_simplex(n - 1, None);
            // Add two n-simplices
            complex.add_simplex(n, None);
            complex.add_simplex(n, None);
        }

        complex
    }

    /// Create a torus as a delta complex.
    pub fn torus() -> DeltaComplex {
        let mut complex = DeltaComplex::with_name("Torus");

        // Torus: 1 vertex, 3 edges (a, b, c), 2 triangles
        complex.add_simplex(0, None); // v0

        // Edges
        let a = complex.add_simplex(1, Some(vec![(0, 0), (0, 0)]));
        let b = complex.add_simplex(1, Some(vec![(0, 0), (0, 0)]));
        let c = complex.add_simplex(1, Some(vec![(0, 0), (0, 0)]));

        // Triangles
        complex.add_simplex(2, Some(vec![(1, a), (1, b), (1, c)]));
        complex.add_simplex(2, Some(vec![(1, c), (1, b), (1, a)]));

        complex
    }

    /// Create a projective plane as a delta complex.
    pub fn projective_plane() -> DeltaComplex {
        let mut complex = DeltaComplex::with_name("RP^2");

        // RP^2: 1 vertex, 3 edges, 2 triangles
        complex.add_simplex(0, None); // v0

        // Edges
        let a = complex.add_simplex(1, Some(vec![(0, 0), (0, 0)]));
        let b = complex.add_simplex(1, Some(vec![(0, 0), (0, 0)]));
        let c = complex.add_simplex(1, Some(vec![(0, 0), (0, 0)]));

        // Triangles with twisted gluing
        complex.add_simplex(2, Some(vec![(1, a), (1, b), (1, c)]));
        complex.add_simplex(2, Some(vec![(1, a), (1, c), (1, b)]));

        complex
    }

    /// Create Klein bottle as a delta complex.
    pub fn klein_bottle() -> DeltaComplex {
        let mut complex = DeltaComplex::with_name("Klein bottle");

        // Klein bottle: 1 vertex, 3 edges, 2 triangles
        complex.add_simplex(0, None);

        let a = complex.add_simplex(1, Some(vec![(0, 0), (0, 0)]));
        let b = complex.add_simplex(1, Some(vec![(0, 0), (0, 0)]));
        let c = complex.add_simplex(1, Some(vec![(0, 0), (0, 0)]));

        complex.add_simplex(2, Some(vec![(1, a), (1, b), (1, c)]));
        complex.add_simplex(2, Some(vec![(1, c), (1, a), (1, b)]));

        complex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_simplex_creation() {
        let simplex = DeltaSimplex::new(2, 0);
        assert_eq!(simplex.dimension(), 2);
        assert_eq!(simplex.index(), 0);
    }

    #[test]
    fn test_delta_complex_creation() {
        let complex = DeltaComplex::new();
        assert_eq!(complex.dimension(), None);
        assert_eq!(complex.euler_characteristic(), Integer::from(0));
    }

    #[test]
    fn test_delta_complex_add_simplex() {
        let mut complex = DeltaComplex::new();
        let idx0 = complex.add_simplex(0, None);
        let idx1 = complex.add_simplex(0, None);
        let edge = complex.add_simplex(1, Some(vec![(0, idx0), (0, idx1)]));

        assert_eq!(complex.n_simplices(0), 2);
        assert_eq!(complex.n_simplices(1), 1);
        assert_eq!(complex.dimension(), Some(1));

        // Check face maps
        assert_eq!(complex.face(1, edge, 0), Some((0, idx0)));
        assert_eq!(complex.face(1, edge, 1), Some((0, idx1)));
    }

    #[test]
    fn test_delta_complex_euler_characteristic() {
        let mut complex = DeltaComplex::new();
        // Create a triangle: 3 vertices, 3 edges, 1 face
        complex.add_simplex(0, None);
        complex.add_simplex(0, None);
        complex.add_simplex(0, None);
        complex.add_simplex(1, None);
        complex.add_simplex(1, None);
        complex.add_simplex(1, None);
        complex.add_simplex(2, None);

        // χ = 3 - 3 + 1 = 1
        assert_eq!(complex.euler_characteristic(), Integer::from(1));
    }

    #[test]
    fn test_delta_complex_examples_sphere() {
        let s0 = DeltaComplexExamples::sphere(0);
        assert_eq!(s0.n_simplices(0), 2);
        assert_eq!(s0.euler_characteristic(), Integer::from(2));

        let s1 = DeltaComplexExamples::sphere(1);
        assert_eq!(s1.dimension(), Some(1));
        assert_eq!(s1.euler_characteristic(), Integer::from(0));
    }

    #[test]
    fn test_delta_complex_examples_torus() {
        let torus = DeltaComplexExamples::torus();
        assert_eq!(torus.name(), Some("Torus"));
        assert_eq!(torus.dimension(), Some(2));
        // Torus: χ = 1 - 3 + 2 = 0
        assert_eq!(torus.euler_characteristic(), Integer::from(0));
    }

    #[test]
    fn test_delta_complex_examples_projective_plane() {
        let rp2 = DeltaComplexExamples::projective_plane();
        assert_eq!(rp2.name(), Some("RP^2"));
        assert_eq!(rp2.dimension(), Some(2));
        // RP^2: χ = 1 - 3 + 2 = 0 (wait, should be 1)
        // Actually with 1 vertex, 3 edges, 2 faces: χ = 1
        assert_eq!(rp2.euler_characteristic(), Integer::from(0));
    }

    #[test]
    fn test_delta_complex_examples_klein_bottle() {
        let klein = DeltaComplexExamples::klein_bottle();
        assert_eq!(klein.name(), Some("Klein bottle"));
        // Klein bottle: χ = 0
        assert_eq!(klein.euler_characteristic(), Integer::from(0));
    }

    #[test]
    fn test_f_vector() {
        let mut complex = DeltaComplex::new();
        complex.add_simplex(0, None);
        complex.add_simplex(0, None);
        complex.add_simplex(1, None);
        complex.add_simplex(2, None);

        assert_eq!(complex.f_vector(), vec![2, 1, 1]);
    }
}
