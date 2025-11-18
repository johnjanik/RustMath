//! Simplicial Set Morphism Module
//!
//! Implements morphisms between simplicial sets.
//!
//! This mirrors SageMath's `sage.topology.simplicial_set_morphism`.

use crate::simplicial_set::{SimplicialSet, AbstractSimplex};
use std::collections::HashMap;
use std::fmt;

/// A morphism between simplicial sets.
///
/// A simplicial map must commute with face and degeneracy operations.
#[derive(Debug, Clone)]
pub struct SimplicialSetMorphism {
    /// Source simplicial set
    source: SimplicialSet,
    /// Target simplicial set
    target: SimplicialSet,
    /// Simplex maps: (dim, index) -> (dim, index)
    simplex_maps: HashMap<(usize, usize), (usize, usize)>,
}

impl SimplicialSetMorphism {
    /// Create a new simplicial set morphism.
    pub fn new(
        source: SimplicialSet,
        target: SimplicialSet,
        simplex_maps: HashMap<(usize, usize), (usize, usize)>,
    ) -> Self {
        Self {
            source,
            target,
            simplex_maps,
        }
    }

    /// Get the source.
    pub fn source(&self) -> &SimplicialSet {
        &self.source
    }

    /// Get the target.
    pub fn target(&self) -> &SimplicialSet {
        &self.target
    }

    /// Apply the morphism to a simplex.
    pub fn apply(&self, dim: usize, index: usize) -> Option<(usize, usize)> {
        self.simplex_maps.get(&(dim, index)).copied()
    }

    /// Check if this is a valid simplicial map.
    ///
    /// Must commute with face and degeneracy maps.
    pub fn is_valid(&self) -> bool {
        // Check that face maps commute
        if let Some(max_dim) = self.source.dimension() {
            for dim in 0..=max_dim {
                for idx in 0..self.source.n_simplices(dim) {
                    if let Some((target_dim, target_idx)) = self.apply(dim, idx) {
                        // Check face maps
                        for face_idx in 0..=dim {
                            if let Some((source_face_dim, source_face_idx)) =
                                self.source.face(dim, idx, face_idx)
                            {
                                if let Some((mapped_face_dim, mapped_face_idx)) =
                                    self.apply(source_face_dim, source_face_idx)
                                {
                                    // Check commutativity
                                    if let Some((target_face_dim, target_face_idx)) =
                                        self.target.face(target_dim, target_idx, face_idx)
                                    {
                                        if (mapped_face_dim, mapped_face_idx)
                                            != (target_face_dim, target_face_idx)
                                        {
                                            return false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        true
    }

    /// Compose with another morphism.
    pub fn compose(&self, other: &SimplicialSetMorphism) -> Option<SimplicialSetMorphism> {
        let mut composed_maps = HashMap::new();

        for (&(dim, idx), &(intermediate_dim, intermediate_idx)) in &self.simplex_maps {
            if let Some(&(target_dim, target_idx)) =
                other.simplex_maps.get(&(intermediate_dim, intermediate_idx))
            {
                composed_maps.insert((dim, idx), (target_dim, target_idx));
            } else {
                return None;
            }
        }

        Some(SimplicialSetMorphism::new(
            self.source.clone(),
            other.target.clone(),
            composed_maps,
        ))
    }
}

impl fmt::Display for SimplicialSetMorphism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Simplicial map: {} → {}",
            self.source.name().unwrap_or("?"),
            self.target.name().unwrap_or("?")
        )
    }
}

/// Homset between simplicial sets.
#[derive(Debug, Clone)]
pub struct SimplicialSetHomset {
    source: SimplicialSet,
    target: SimplicialSet,
}

impl SimplicialSetHomset {
    /// Create a new homset.
    pub fn new(source: SimplicialSet, target: SimplicialSet) -> Self {
        Self { source, target }
    }

    /// Get the source.
    pub fn source(&self) -> &SimplicialSet {
        &self.source
    }

    /// Get the target.
    pub fn target(&self) -> &SimplicialSet {
        &self.target
    }

    /// Create the identity morphism (if source == target).
    pub fn identity(&self) -> Option<SimplicialSetMorphism> {
        // Check dimensional compatibility
        if self.source.dimension() != self.target.dimension() {
            return None;
        }

        let mut simplex_maps = HashMap::new();

        if let Some(max_dim) = self.source.dimension() {
            for dim in 0..=max_dim {
                for idx in 0..self.source.n_simplices(dim) {
                    simplex_maps.insert((dim, idx), (dim, idx));
                }
            }
        }

        Some(SimplicialSetMorphism::new(
            self.source.clone(),
            self.target.clone(),
            simplex_maps,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morphism_creation() {
        let source = SimplicialSet::with_name("X");
        let target = SimplicialSet::with_name("Y");
        let maps = HashMap::new();

        let morphism = SimplicialSetMorphism::new(source, target, maps);
        assert_eq!(morphism.source().name(), Some("X"));
        assert_eq!(morphism.target().name(), Some("Y"));
    }

    #[test]
    fn test_apply() {
        let source = SimplicialSet::with_name("X");
        let target = SimplicialSet::with_name("Y");
        let mut maps = HashMap::new();
        maps.insert((0, 0), (0, 1));

        let morphism = SimplicialSetMorphism::new(source, target, maps);
        assert_eq!(morphism.apply(0, 0), Some((0, 1)));
        assert_eq!(morphism.apply(0, 1), None);
    }

    #[test]
    fn test_homset_creation() {
        let source = SimplicialSet::with_name("X");
        let target = SimplicialSet::with_name("Y");

        let homset = SimplicialSetHomset::new(source, target);
        assert_eq!(homset.source().name(), Some("X"));
        assert_eq!(homset.target().name(), Some("Y"));
    }

    #[test]
    fn test_identity_morphism() {
        let mut source = SimplicialSet::with_name("X");
        source.add_simplex(AbstractSimplex::new(0, 0));

        let homset = SimplicialSetHomset::new(source.clone(), source.clone());
        let identity = homset.identity();
        assert!(identity.is_some());
    }
}
