//! Simplicial Complex Morphism Module
//!
//! Implements morphisms (maps) between simplicial complexes.
//!
//! This mirrors SageMath's `sage.topology.simplicial_complex_morphism`.

use crate::simplicial_complex::{SimplicialComplex, Simplex};
use std::collections::HashMap;
use std::fmt;

/// A morphism between simplicial complexes.
///
/// A simplicial map is determined by where it sends the vertices.
/// It must send simplices to simplices.
#[derive(Debug, Clone)]
pub struct SimplicialComplexMorphism {
    /// Source complex
    source: SimplicialComplex,
    /// Target complex
    target: SimplicialComplex,
    /// Vertex map: source vertex -> target vertex
    vertex_map: HashMap<usize, usize>,
}

impl SimplicialComplexMorphism {
    /// Create a new simplicial complex morphism.
    ///
    /// The vertex_map must send vertices of the source to vertices of the target,
    /// and must send simplices to simplices.
    pub fn new(
        source: SimplicialComplex,
        target: SimplicialComplex,
        vertex_map: HashMap<usize, usize>,
    ) -> Self {
        Self {
            source,
            target,
            vertex_map,
        }
    }

    /// Get the source complex.
    pub fn source(&self) -> &SimplicialComplex {
        &self.source
    }

    /// Get the target complex.
    pub fn target(&self) -> &SimplicialComplex {
        &self.target
    }

    /// Get the vertex map.
    pub fn vertex_map(&self) -> &HashMap<usize, usize> {
        &self.vertex_map
    }

    /// Apply the morphism to a vertex.
    pub fn apply_to_vertex(&self, vertex: usize) -> Option<usize> {
        self.vertex_map.get(&vertex).copied()
    }

    /// Apply the morphism to a simplex.
    pub fn apply_to_simplex(&self, simplex: &Simplex) -> Option<Simplex> {
        let mut image_vertices = Vec::new();

        for &v in simplex.vertices() {
            match self.vertex_map.get(&v) {
                Some(&target_v) => image_vertices.push(target_v),
                None => return None,
            }
        }

        // Remove duplicates and sort
        image_vertices.sort_unstable();
        image_vertices.dedup();

        let image_simplex = Simplex::new(image_vertices);

        // Check if the image is in the target complex
        if self.target.contains(&image_simplex) {
            Some(image_simplex)
        } else {
            None
        }
    }

    /// Check if this is a valid simplicial map.
    ///
    /// A map is valid if every simplex in the source maps to a simplex in the target.
    pub fn is_valid(&self) -> bool {
        // Check all vertices are mapped
        for v in self.source.vertices() {
            if !self.vertex_map.contains_key(&v) {
                return false;
            }
        }

        // Check that simplices map to simplices
        if let Some(max_dim) = self.source.dimension() {
            for dim in 0..=max_dim {
                for simplex in self.source.simplices(dim) {
                    if self.apply_to_simplex(&simplex).is_none() {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if this is an isomorphism.
    pub fn is_isomorphism(&self) -> bool {
        if !self.is_valid() {
            return false;
        }

        // Check if the map is bijective on vertices
        let source_n = self.source.n_vertices();
        let target_n = self.target.n_vertices();

        if source_n != target_n {
            return false;
        }

        // Check injectivity
        let image_vertices: std::collections::HashSet<_> = self.vertex_map.values().collect();
        if image_vertices.len() != source_n {
            return false;
        }

        // Check surjectivity
        image_vertices.len() == target_n
    }

    /// Compose with another morphism.
    pub fn compose(&self, other: &SimplicialComplexMorphism) -> Option<SimplicialComplexMorphism> {
        // Check that the target of self equals the source of other
        // (This is simplified; in practice, would need structural equality)

        let mut composed_map = HashMap::new();

        for (&source_v, &intermediate_v) in &self.vertex_map {
            if let Some(&target_v) = other.vertex_map.get(&intermediate_v) {
                composed_map.insert(source_v, target_v);
            } else {
                return None; // Composition not defined
            }
        }

        Some(SimplicialComplexMorphism::new(
            self.source.clone(),
            other.target.clone(),
            composed_map,
        ))
    }
}

impl fmt::Display for SimplicialComplexMorphism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Simplicial map: {} -> {}",
            self.source
                .name()
                .unwrap_or("?"),
            self.target.name().unwrap_or("?")
        )
    }
}

/// Check if an object is a simplicial complex morphism.
pub fn is_simplicial_complex_morphism(obj: &SimplicialComplexMorphism) -> bool {
    true // Type system ensures this
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morphism_creation() {
        let mut source = SimplicialComplex::new();
        source.add_simplex(Simplex::new(vec![0, 1]));

        let mut target = SimplicialComplex::new();
        target.add_simplex(Simplex::new(vec![2, 3]));

        let mut vertex_map = HashMap::new();
        vertex_map.insert(0, 2);
        vertex_map.insert(1, 3);

        let morphism = SimplicialComplexMorphism::new(source, target, vertex_map);
        assert!(morphism.is_valid());
    }

    #[test]
    fn test_apply_to_vertex() {
        let source = SimplicialComplex::new();
        let target = SimplicialComplex::new();

        let mut vertex_map = HashMap::new();
        vertex_map.insert(0, 5);
        vertex_map.insert(1, 6);

        let morphism = SimplicialComplexMorphism::new(source, target, vertex_map);
        assert_eq!(morphism.apply_to_vertex(0), Some(5));
        assert_eq!(morphism.apply_to_vertex(1), Some(6));
        assert_eq!(morphism.apply_to_vertex(2), None);
    }

    #[test]
    fn test_apply_to_simplex() {
        let mut source = SimplicialComplex::new();
        source.add_simplex(Simplex::new(vec![0, 1]));

        let mut target = SimplicialComplex::new();
        target.add_simplex(Simplex::new(vec![2, 3]));

        let mut vertex_map = HashMap::new();
        vertex_map.insert(0, 2);
        vertex_map.insert(1, 3);

        let morphism = SimplicialComplexMorphism::new(source.clone(), target, vertex_map);

        let simplex = Simplex::new(vec![0, 1]);
        let image = morphism.apply_to_simplex(&simplex);
        assert!(image.is_some());
        assert_eq!(image.unwrap().vertices(), &[2, 3]);
    }

    #[test]
    fn test_is_valid() {
        let mut source = SimplicialComplex::new();
        source.add_simplex(Simplex::new(vec![0, 1]));

        let mut target = SimplicialComplex::new();
        target.add_simplex(Simplex::new(vec![2, 3]));

        let mut vertex_map = HashMap::new();
        vertex_map.insert(0, 2);
        vertex_map.insert(1, 3);

        let morphism = SimplicialComplexMorphism::new(source, target, vertex_map);
        assert!(morphism.is_valid());
    }

    #[test]
    fn test_is_isomorphism() {
        let mut source = SimplicialComplex::new();
        source.add_simplex(Simplex::new(vec![0, 1]));

        let mut target = SimplicialComplex::new();
        target.add_simplex(Simplex::new(vec![2, 3]));

        let mut vertex_map = HashMap::new();
        vertex_map.insert(0, 2);
        vertex_map.insert(1, 3);

        let morphism = SimplicialComplexMorphism::new(source, target, vertex_map);
        assert!(morphism.is_isomorphism());
    }

    #[test]
    fn test_is_simplicial_complex_morphism() {
        let source = SimplicialComplex::new();
        let target = SimplicialComplex::new();
        let morphism = SimplicialComplexMorphism::new(source, target, HashMap::new());
        assert!(is_simplicial_complex_morphism(&morphism));
    }
}
