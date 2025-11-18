//! Filtered Simplicial Complex Module
//!
//! Implements filtered simplicial complexes for topological data analysis and persistent homology.
//!
//! A filtered simplicial complex is a sequence of nested simplicial complexes,
//! used in persistent homology computations.
//!
//! This mirrors SageMath's `sage.topology.filtered_simplicial_complex`.

use rustmath_integers::Integer;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// A simplex with associated filtration value.
#[derive(Debug, Clone, PartialEq)]
pub struct FilteredSimplex {
    /// Vertices of the simplex
    vertices: Vec<usize>,
    /// Filtration value (when this simplex enters the complex)
    filtration_value: f64,
}

impl FilteredSimplex {
    /// Create a new filtered simplex.
    pub fn new(vertices: Vec<usize>, filtration_value: f64) -> Self {
        let mut sorted_vertices = vertices;
        sorted_vertices.sort_unstable();
        Self {
            vertices: sorted_vertices,
            filtration_value,
        }
    }

    /// Get the dimension.
    pub fn dimension(&self) -> usize {
        if self.vertices.is_empty() {
            0
        } else {
            self.vertices.len() - 1
        }
    }

    /// Get the filtration value.
    pub fn filtration_value(&self) -> f64 {
        self.filtration_value
    }

    /// Get the vertices.
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    /// Get all faces of this simplex.
    pub fn faces(&self) -> Vec<Vec<usize>> {
        if self.vertices.is_empty() {
            return vec![];
        }

        let mut faces = Vec::new();
        let n = self.vertices.len();
        for i in 0..n {
            let mut face = self.vertices.clone();
            face.remove(i);
            faces.push(face);
        }
        faces
    }
}

/// A filtered simplicial complex for persistent homology.
#[derive(Debug, Clone)]
pub struct FilteredSimplicialComplex {
    /// All simplices with their filtration values
    simplices: Vec<FilteredSimplex>,
    /// Maximum dimension
    max_dimension: Option<usize>,
}

impl FilteredSimplicialComplex {
    /// Create a new empty filtered simplicial complex.
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
            max_dimension: None,
        }
    }

    /// Create from a list of filtered simplices.
    pub fn from_simplices(mut simplices: Vec<FilteredSimplex>) -> Self {
        // Sort by filtration value
        simplices.sort_by(|a, b| a.filtration_value.partial_cmp(&b.filtration_value).unwrap());

        let max_dim = simplices.iter().map(|s| s.dimension()).max();

        Self {
            simplices,
            max_dimension: max_dim,
        }
    }

    /// Add a simplex to the complex.
    pub fn add_simplex(&mut self, simplex: FilteredSimplex) {
        let dim = simplex.dimension();
        self.max_dimension = Some(self.max_dimension.map(|d| d.max(dim)).unwrap_or(dim));
        self.simplices.push(simplex);
        // Re-sort by filtration value
        self.simplices
            .sort_by(|a, b| a.filtration_value.partial_cmp(&b.filtration_value).unwrap());
    }

    /// Get all simplices.
    pub fn simplices(&self) -> &[FilteredSimplex] {
        &self.simplices
    }

    /// Get simplices at a specific filtration value.
    pub fn simplices_at_filtration(&self, value: f64) -> Vec<&FilteredSimplex> {
        self.simplices
            .iter()
            .filter(|s| s.filtration_value <= value)
            .collect()
    }

    /// Get the dimension of the complex.
    pub fn dimension(&self) -> Option<usize> {
        self.max_dimension
    }

    /// Get the number of simplices.
    pub fn n_simplices(&self) -> usize {
        self.simplices.len()
    }

    /// Get the range of filtration values.
    pub fn filtration_range(&self) -> Option<(f64, f64)> {
        if self.simplices.is_empty() {
            return None;
        }

        let min_val = self
            .simplices
            .iter()
            .map(|s| s.filtration_value)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_val = self
            .simplices
            .iter()
            .map(|s| s.filtration_value)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        Some((min_val, max_val))
    }

    /// Get all unique filtration values.
    pub fn filtration_values(&self) -> Vec<f64> {
        let mut values: Vec<f64> = self
            .simplices
            .iter()
            .map(|s| s.filtration_value)
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();
        values
    }
}

impl Default for FilteredSimplicialComplex {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for FilteredSimplicialComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Filtered simplicial complex with {} simplices",
            self.n_simplices()
        )?;
        if let Some(dim) = self.max_dimension {
            write!(f, " (max dimension {})", dim)?;
        }
        if let Some((min, max)) = self.filtration_range() {
            write!(f, ", filtration range [{:.3}, {:.3}]", min, max)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filtered_simplex_creation() {
        let simplex = FilteredSimplex::new(vec![0, 1, 2], 1.5);
        assert_eq!(simplex.dimension(), 2);
        assert_eq!(simplex.filtration_value(), 1.5);
        assert_eq!(simplex.vertices(), &[0, 1, 2]);
    }

    #[test]
    fn test_filtered_simplex_faces() {
        let simplex = FilteredSimplex::new(vec![0, 1, 2], 1.0);
        let faces = simplex.faces();
        assert_eq!(faces.len(), 3);
        assert!(faces.contains(&vec![1, 2]));
        assert!(faces.contains(&vec![0, 2]));
        assert!(faces.contains(&vec![0, 1]));
    }

    #[test]
    fn test_filtered_complex_creation() {
        let complex = FilteredSimplicialComplex::new();
        assert_eq!(complex.dimension(), None);
        assert_eq!(complex.n_simplices(), 0);
    }

    #[test]
    fn test_filtered_complex_from_simplices() {
        let simplices = vec![
            FilteredSimplex::new(vec![0], 0.0),
            FilteredSimplex::new(vec![1], 0.0),
            FilteredSimplex::new(vec![0, 1], 1.0),
        ];

        let complex = FilteredSimplicialComplex::from_simplices(simplices);
        assert_eq!(complex.dimension(), Some(1));
        assert_eq!(complex.n_simplices(), 3);
    }

    #[test]
    fn test_filtered_complex_add_simplex() {
        let mut complex = FilteredSimplicialComplex::new();
        complex.add_simplex(FilteredSimplex::new(vec![0], 0.0));
        complex.add_simplex(FilteredSimplex::new(vec![1], 0.0));
        complex.add_simplex(FilteredSimplex::new(vec![0, 1], 1.0));

        assert_eq!(complex.n_simplices(), 3);
        assert_eq!(complex.dimension(), Some(1));
    }

    #[test]
    fn test_filtration_range() {
        let mut complex = FilteredSimplicialComplex::new();
        complex.add_simplex(FilteredSimplex::new(vec![0], 0.5));
        complex.add_simplex(FilteredSimplex::new(vec![1], 1.0));
        complex.add_simplex(FilteredSimplex::new(vec![0, 1], 2.0));

        let range = complex.filtration_range();
        assert_eq!(range, Some((0.5, 2.0)));
    }

    #[test]
    fn test_simplices_at_filtration() {
        let mut complex = FilteredSimplicialComplex::new();
        complex.add_simplex(FilteredSimplex::new(vec![0], 0.0));
        complex.add_simplex(FilteredSimplex::new(vec![1], 0.0));
        complex.add_simplex(FilteredSimplex::new(vec![0, 1], 1.0));
        complex.add_simplex(FilteredSimplex::new(vec![2], 2.0));

        let at_1_0 = complex.simplices_at_filtration(1.0);
        assert_eq!(at_1_0.len(), 3); // Should include simplices with filtration <= 1.0

        let at_0_5 = complex.simplices_at_filtration(0.5);
        assert_eq!(at_0_5.len(), 2); // Only the two 0-simplices
    }

    #[test]
    fn test_filtration_values() {
        let mut complex = FilteredSimplicialComplex::new();
        complex.add_simplex(FilteredSimplex::new(vec![0], 0.0));
        complex.add_simplex(FilteredSimplex::new(vec![1], 0.0));
        complex.add_simplex(FilteredSimplex::new(vec![0, 1], 1.0));
        complex.add_simplex(FilteredSimplex::new(vec![2], 2.0));

        let values = complex.filtration_values();
        assert_eq!(values, vec![0.0, 1.0, 2.0]);
    }
}
