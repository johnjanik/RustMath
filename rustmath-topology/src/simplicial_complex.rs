//! Simplicial Complex Module
//!
//! Implements simplicial complexes: fundamental structures in algebraic topology.
//!
//! A simplicial complex is a collection of simplices (points, edges, triangles, etc.)
//! glued together in a combinatorial way, where every face of a simplex in the complex
//! is also in the complex.
//!
//! This mirrors SageMath's `sage.topology.simplicial_complex`.

use crate::cell_complex::GenericCellComplex;
use rustmath_integers::Integer;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;

/// A simplex: a generalization of a triangle.
///
/// An n-simplex is the convex hull of n+1 affinely independent points.
/// In this implementation, a simplex is identified by its set of vertices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
    /// Vertices of the simplex (sorted)
    vertices: Vec<usize>,
}

impl Simplex {
    /// Create a new simplex from vertices.
    pub fn new(vertices: Vec<usize>) -> Self {
        let mut sorted_vertices = vertices;
        sorted_vertices.sort_unstable();
        sorted_vertices.dedup(); // Remove duplicates
        Self {
            vertices: sorted_vertices,
        }
    }

    /// Get the dimension of the simplex.
    ///
    /// An n-simplex has dimension n (has n+1 vertices).
    pub fn dimension(&self) -> usize {
        if self.vertices.is_empty() {
            0
        } else {
            self.vertices.len() - 1
        }
    }

    /// Get the vertices.
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    /// Check if this simplex is a face of another simplex.
    pub fn is_face_of(&self, other: &Simplex) -> bool {
        self.vertices.iter().all(|v| other.vertices.contains(v))
    }

    /// Get all faces of this simplex.
    pub fn faces(&self) -> Vec<Simplex> {
        let mut faces = Vec::new();

        // Generate all subsets
        let n = self.vertices.len();
        for mask in 0..(1 << n) {
            if mask == (1 << n) - 1 {
                continue; // Skip the simplex itself
            }
            let mut face_vertices = Vec::new();
            for (i, &v) in self.vertices.iter().enumerate() {
                if (mask & (1 << i)) != 0 {
                    face_vertices.push(v);
                }
            }
            if !face_vertices.is_empty() {
                faces.push(Simplex::new(face_vertices));
            }
        }

        faces
    }

    /// Get the boundary (co-dimension 1 faces).
    pub fn boundary(&self) -> Vec<Simplex> {
        if self.vertices.is_empty() {
            return vec![];
        }

        let mut boundary = Vec::new();
        for i in 0..self.vertices.len() {
            let mut face_vertices = self.vertices.clone();
            face_vertices.remove(i);
            if !face_vertices.is_empty() {
                boundary.push(Simplex::new(face_vertices));
            }
        }
        boundary
    }
}

impl fmt::Display for Simplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, v) in self.vertices.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", v)?;
        }
        write!(f, ")")
    }
}

/// A simplicial complex.
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    /// All simplices in the complex, organized by dimension
    simplices: HashMap<usize, HashSet<Simplex>>,
    /// Maximum dimension
    max_dimension: Option<usize>,
    /// Name of the complex
    name: Option<String>,
}

impl SimplicialComplex {
    /// Create a new empty simplicial complex.
    pub fn new() -> Self {
        Self {
            simplices: HashMap::new(),
            max_dimension: None,
            name: None,
        }
    }

    /// Create a simplicial complex with a name.
    pub fn with_name(name: &str) -> Self {
        Self {
            simplices: HashMap::new(),
            max_dimension: None,
            name: Some(name.to_string()),
        }
    }

    /// Create a simplicial complex from a list of facets (maximal simplices).
    pub fn from_facets(facets: Vec<Simplex>) -> Self {
        let mut complex = Self::new();
        for facet in facets {
            complex.add_simplex(facet);
        }
        complex
    }

    /// Add a simplex to the complex (with all its faces).
    pub fn add_simplex(&mut self, simplex: Simplex) {
        let dim = simplex.dimension();
        self.max_dimension = Some(self.max_dimension.map(|d| d.max(dim)).unwrap_or(dim));

        // Add the simplex
        self.simplices
            .entry(dim)
            .or_insert_with(HashSet::new)
            .insert(simplex.clone());

        // Recursively add all faces
        for face in simplex.faces() {
            let face_dim = face.dimension();
            let entry = self.simplices.entry(face_dim).or_insert_with(HashSet::new);
            if !entry.contains(&face) {
                self.add_simplex(face);
            }
        }
    }

    /// Get all simplices of a given dimension.
    pub fn simplices(&self, dim: usize) -> Vec<Simplex> {
        self.simplices
            .get(&dim)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_else(Vec::new)
    }

    /// Get the number of simplices in a given dimension.
    pub fn n_simplices(&self, dim: usize) -> usize {
        self.simplices.get(&dim).map(|set| set.len()).unwrap_or(0)
    }

    /// Get the dimension of the complex.
    pub fn dimension(&self) -> Option<usize> {
        self.max_dimension
    }

    /// Get all facets (maximal simplices).
    pub fn facets(&self) -> Vec<Simplex> {
        let max_dim = match self.max_dimension {
            Some(d) => d,
            None => return vec![],
        };

        let max_simplices = self.simplices(max_dim);

        // Filter to only maximal simplices
        max_simplices
            .into_iter()
            .filter(|s| {
                // Check if this simplex is a face of any higher-dimensional simplex
                self.simplices
                    .iter()
                    .filter(|(&dim, _)| dim > max_dim)
                    .all(|(_, set)| !set.iter().any(|larger| s.is_face_of(larger)))
            })
            .collect()
    }

    /// Compute the Euler characteristic.
    ///
    /// χ = Σ(-1)^i * f_i where f_i is the number of i-simplices.
    pub fn euler_characteristic(&self) -> Integer {
        let mut chi = Integer::from(0);
        if let Some(max_dim) = self.max_dimension {
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
    ///
    /// The f-vector is (f_0, f_1, ..., f_d) where f_i is the number of i-simplices.
    pub fn f_vector(&self) -> Vec<usize> {
        let max_dim = self.max_dimension.unwrap_or(0);
        (0..=max_dim).map(|dim| self.n_simplices(dim)).collect()
    }

    /// Get the h-vector.
    ///
    /// For a (d-1)-dimensional simplicial complex, h_k = Σ(-1)^{k-i} C(d-i, k-i) f_{i-1}
    pub fn h_vector(&self) -> Vec<i64> {
        let f = self.f_vector();
        let d = f.len();

        if d == 0 {
            return vec![1];
        }

        let mut h = vec![0i64; d + 1];
        h[0] = 1;

        for k in 1..=d {
            for i in 0..k {
                let f_val = if i == 0 { 1 } else { f[i - 1] as i64 };
                let binom = binomial(d - i, k - i);
                if (k - i) % 2 == 0 {
                    h[k] += binom * f_val;
                } else {
                    h[k] -= binom * f_val;
                }
            }
        }

        h
    }

    /// Check if the complex contains a specific simplex.
    pub fn contains(&self, simplex: &Simplex) -> bool {
        let dim = simplex.dimension();
        self.simplices
            .get(&dim)
            .map(|set| set.contains(simplex))
            .unwrap_or(false)
    }

    /// Get all vertices.
    pub fn vertices(&self) -> HashSet<usize> {
        self.simplices(0)
            .into_iter()
            .flat_map(|s| s.vertices().to_vec())
            .collect()
    }

    /// Get the number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.vertices().len()
    }

    /// Get the name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the name.
    pub fn set_name(&mut self, name: &str) {
        self.name = Some(name.to_string());
    }
}

impl Default for SimplicialComplex {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SimplicialComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "Simplicial complex {}", name)?;
        } else {
            write!(f, "Simplicial complex")?;
        }
        if let Some(dim) = self.max_dimension {
            write!(f, " of dimension {}", dim)?;
        }
        write!(f, " with f-vector {:?}", self.f_vector())
    }
}

/// Helper function to compute binomial coefficients.
fn binomial(n: usize, k: usize) -> i64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let mut result = 1i64;
    for i in 0..k.min(n - k) {
        result = result * (n - i) as i64 / (i + 1) as i64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_creation() {
        let simplex = Simplex::new(vec![0, 1, 2]);
        assert_eq!(simplex.dimension(), 2);
        assert_eq!(simplex.vertices(), &[0, 1, 2]);
    }

    #[test]
    fn test_simplex_faces() {
        let simplex = Simplex::new(vec![0, 1, 2]);
        let faces = simplex.faces();

        // A 2-simplex has 3 vertices, 3 edges: 7 faces total (excluding itself)
        assert_eq!(faces.len(), 7);
    }

    #[test]
    fn test_simplex_boundary() {
        let simplex = Simplex::new(vec![0, 1, 2]);
        let boundary = simplex.boundary();

        // Boundary of a triangle has 3 edges
        assert_eq!(boundary.len(), 3);
        for edge in boundary {
            assert_eq!(edge.dimension(), 1);
        }
    }

    #[test]
    fn test_simplex_is_face_of() {
        let triangle = Simplex::new(vec![0, 1, 2]);
        let edge = Simplex::new(vec![0, 1]);
        assert!(edge.is_face_of(&triangle));
        assert!(!triangle.is_face_of(&edge));
    }

    #[test]
    fn test_simplicial_complex_creation() {
        let complex = SimplicialComplex::new();
        assert_eq!(complex.dimension(), None);
        assert_eq!(complex.euler_characteristic(), Integer::from(0));
    }

    #[test]
    fn test_simplicial_complex_add_simplex() {
        let mut complex = SimplicialComplex::new();
        let triangle = Simplex::new(vec![0, 1, 2]);
        complex.add_simplex(triangle);

        // Should have 1 triangle, 3 edges, 3 vertices
        assert_eq!(complex.n_simplices(2), 1);
        assert_eq!(complex.n_simplices(1), 3);
        assert_eq!(complex.n_simplices(0), 3);
    }

    #[test]
    fn test_simplicial_complex_euler_characteristic() {
        let mut complex = SimplicialComplex::new();
        let triangle = Simplex::new(vec![0, 1, 2]);
        complex.add_simplex(triangle);

        // χ = 3 - 3 + 1 = 1
        assert_eq!(complex.euler_characteristic(), Integer::from(1));
    }

    #[test]
    fn test_simplicial_complex_f_vector() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));

        assert_eq!(complex.f_vector(), vec![3, 3, 1]);
    }

    #[test]
    fn test_simplicial_complex_h_vector() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));

        let h = complex.h_vector();
        assert_eq!(h[0], 1);
    }

    #[test]
    fn test_simplicial_complex_contains() {
        let mut complex = SimplicialComplex::new();
        let triangle = Simplex::new(vec![0, 1, 2]);
        complex.add_simplex(triangle.clone());

        assert!(complex.contains(&triangle));
        assert!(!complex.contains(&Simplex::new(vec![3, 4])));
    }

    #[test]
    fn test_simplicial_complex_vertices() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));
        complex.add_simplex(Simplex::new(vec![2, 3]));

        let vertices = complex.vertices();
        assert_eq!(vertices.len(), 4);
        assert!(vertices.contains(&0));
        assert!(vertices.contains(&1));
        assert!(vertices.contains(&2));
        assert!(vertices.contains(&3));
    }

    #[test]
    fn test_simplicial_complex_from_facets() {
        let facets = vec![Simplex::new(vec![0, 1, 2]), Simplex::new(vec![2, 3, 4])];
        let complex = SimplicialComplex::from_facets(facets);

        assert_eq!(complex.dimension(), Some(2));
        assert_eq!(complex.n_simplices(2), 2);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial(4, 2), 6);
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(6, 3), 20);
    }

    #[test]
    fn test_simplex_display() {
        let simplex = Simplex::new(vec![0, 1, 2]);
        let display = format!("{}", simplex);
        assert!(display.contains("0") && display.contains("1") && display.contains("2"));
    }

    #[test]
    fn test_simplicial_complex_with_name() {
        let mut complex = SimplicialComplex::with_name("Test");
        assert_eq!(complex.name(), Some("Test"));

        complex.set_name("New Name");
        assert_eq!(complex.name(), Some("New Name"));
    }
}
