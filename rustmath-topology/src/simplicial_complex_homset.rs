//! Simplicial Complex Homset Module
//!
//! Implements homomorphism sets between simplicial complexes.
//!
//! This mirrors SageMath's `sage.topology.simplicial_complex_homset`.

use crate::simplicial_complex::SimplicialComplex;
use crate::simplicial_complex_morphism::SimplicialComplexMorphism;
use std::collections::HashMap;

/// A homset between two simplicial complexes.
///
/// HomSet(X, Y) is the set of all simplicial maps from X to Y.
#[derive(Debug, Clone)]
pub struct SimplicialComplexHomset {
    /// Source complex
    source: SimplicialComplex,
    /// Target complex
    target: SimplicialComplex,
}

impl SimplicialComplexHomset {
    /// Create a new homset.
    pub fn new(source: SimplicialComplex, target: SimplicialComplex) -> Self {
        Self { source, target }
    }

    /// Get the source complex.
    pub fn source(&self) -> &SimplicialComplex {
        &self.source
    }

    /// Get the target complex.
    pub fn target(&self) -> &SimplicialComplex {
        &self.target
    }

    /// Create an identity morphism (only if source == target).
    pub fn identity(&self) -> Option<SimplicialComplexMorphism> {
        // Check if source and target are the same
        let source_vertices = self.source.vertices();
        let target_vertices = self.target.vertices();

        if source_vertices != target_vertices {
            return None;
        }

        let vertex_map: HashMap<usize, usize> =
            source_vertices.iter().map(|&v| (v, v)).collect();

        Some(SimplicialComplexMorphism::new(
            self.source.clone(),
            self.target.clone(),
            vertex_map,
        ))
    }

    /// Create a constant morphism (all vertices map to a single vertex).
    pub fn constant_morphism(&self, target_vertex: usize) -> Option<SimplicialComplexMorphism> {
        if !self.target.vertices().contains(&target_vertex) {
            return None;
        }

        let vertex_map: HashMap<usize, usize> = self
            .source
            .vertices()
            .into_iter()
            .map(|v| (v, target_vertex))
            .collect();

        Some(SimplicialComplexMorphism::new(
            self.source.clone(),
            self.target.clone(),
            vertex_map,
        ))
    }
}

/// Check if an object is a simplicial complex homset.
pub fn is_simplicial_complex_homset(obj: &SimplicialComplexHomset) -> bool {
    true // Type system ensures this
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplicial_complex::Simplex;

    #[test]
    fn test_homset_creation() {
        let mut source = SimplicialComplex::new();
        source.add_simplex(Simplex::new(vec![0, 1]));

        let mut target = SimplicialComplex::new();
        target.add_simplex(Simplex::new(vec![0, 1, 2]));

        let homset = SimplicialComplexHomset::new(source, target);
        assert_eq!(homset.source().dimension(), Some(1));
        assert_eq!(homset.target().dimension(), Some(2));
    }

    #[test]
    fn test_identity_morphism() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1]));

        let homset = SimplicialComplexHomset::new(complex.clone(), complex.clone());
        let identity = homset.identity();
        assert!(identity.is_some());
    }

    #[test]
    fn test_constant_morphism() {
        let mut source = SimplicialComplex::new();
        source.add_simplex(Simplex::new(vec![0, 1]));

        let mut target = SimplicialComplex::new();
        target.add_simplex(Simplex::new(vec![2, 3, 4]));

        let homset = SimplicialComplexHomset::new(source, target);
        let constant = homset.constant_morphism(2);
        assert!(constant.is_some());
    }

    #[test]
    fn test_is_simplicial_complex_homset() {
        let source = SimplicialComplex::new();
        let target = SimplicialComplex::new();
        let homset = SimplicialComplexHomset::new(source, target);
        assert!(is_simplicial_complex_homset(&homset));
    }
}
