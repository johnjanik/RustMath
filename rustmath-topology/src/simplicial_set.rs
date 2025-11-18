//! Simplicial Set Module
//!
//! Implements simplicial sets: the categorical approach to homotopy theory.
//!
//! A simplicial set is a contravariant functor from the simplex category Δ
//! to the category of sets. It consists of sets of n-simplices with face and
//! degeneracy maps.
//!
//! This mirrors SageMath's `sage.topology.simplicial_set`.

use rustmath_integers::Integer;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// An abstract simplex in a simplicial set.
///
/// Unlike simplices in simplicial complexes, these are abstract entities
/// with face and degeneracy operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AbstractSimplex {
    /// Dimension
    dimension: usize,
    /// Index/identifier
    index: usize,
    /// Name
    name: Option<String>,
}

impl AbstractSimplex {
    /// Create a new abstract simplex.
    pub fn new(dimension: usize, index: usize) -> Self {
        Self {
            dimension,
            index,
            name: None,
        }
    }

    /// Create a named abstract simplex.
    pub fn with_name(dimension: usize, index: usize, name: &str) -> Self {
        Self {
            dimension,
            index,
            name: Some(name.to_string()),
        }
    }

    /// Get the dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the index.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl fmt::Display for AbstractSimplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{}", name)
        } else {
            write!(f, "σ_{}^{}", self.index, self.dimension)
        }
    }
}

/// A simplicial set.
#[derive(Debug, Clone)]
pub struct SimplicialSet {
    /// Simplices organized by dimension
    simplices: HashMap<usize, Vec<AbstractSimplex>>,
    /// Face maps: (n, simplex_idx, face_idx) -> (n-1, target_simplex_idx)
    face_maps: HashMap<(usize, usize, usize), (usize, usize)>,
    /// Degeneracy maps: (n, simplex_idx, deg_idx) -> (n+1, target_simplex_idx)
    degeneracy_maps: HashMap<(usize, usize, usize), (usize, usize)>,
    /// Name
    name: Option<String>,
}

impl SimplicialSet {
    /// Create a new empty simplicial set.
    pub fn new() -> Self {
        Self {
            simplices: HashMap::new(),
            face_maps: HashMap::new(),
            degeneracy_maps: HashMap::new(),
            name: None,
        }
    }

    /// Create a simplicial set with a name.
    pub fn with_name(name: &str) -> Self {
        Self {
            simplices: HashMap::new(),
            face_maps: HashMap::new(),
            degeneracy_maps: HashMap::new(),
            name: Some(name.to_string()),
        }
    }

    /// Add a simplex.
    pub fn add_simplex(&mut self, simplex: AbstractSimplex) -> usize {
        let dim = simplex.dimension();
        let simplices_at_dim = self.simplices.entry(dim).or_insert_with(Vec::new);
        let index = simplices_at_dim.len();
        simplices_at_dim.push(simplex);
        index
    }

    /// Add a face map.
    pub fn add_face_map(
        &mut self,
        source_dim: usize,
        source_idx: usize,
        face_idx: usize,
        target_dim: usize,
        target_idx: usize,
    ) {
        self.face_maps
            .insert((source_dim, source_idx, face_idx), (target_dim, target_idx));
    }

    /// Add a degeneracy map.
    pub fn add_degeneracy_map(
        &mut self,
        source_dim: usize,
        source_idx: usize,
        deg_idx: usize,
        target_dim: usize,
        target_idx: usize,
    ) {
        self.degeneracy_maps
            .insert((source_dim, source_idx, deg_idx), (target_dim, target_idx));
    }

    /// Get simplices of a given dimension.
    pub fn simplices(&self, dim: usize) -> Vec<&AbstractSimplex> {
        self.simplices
            .get(&dim)
            .map(|v| v.iter().collect())
            .unwrap_or_else(Vec::new)
    }

    /// Get the number of simplices in a given dimension.
    pub fn n_simplices(&self, dim: usize) -> usize {
        self.simplices.get(&dim).map(|v| v.len()).unwrap_or(0)
    }

    /// Get the dimension.
    pub fn dimension(&self) -> Option<usize> {
        self.simplices.keys().max().copied()
    }

    /// Get a face of a simplex.
    pub fn face(
        &self,
        dim: usize,
        simplex_idx: usize,
        face_idx: usize,
    ) -> Option<(usize, usize)> {
        self.face_maps.get(&(dim, simplex_idx, face_idx)).copied()
    }

    /// Get a degeneracy of a simplex.
    pub fn degeneracy(
        &self,
        dim: usize,
        simplex_idx: usize,
        deg_idx: usize,
    ) -> Option<(usize, usize)> {
        self.degeneracy_maps
            .get(&(dim, simplex_idx, deg_idx))
            .copied()
    }

    /// Check if a simplex is degenerate.
    pub fn is_degenerate(&self, dim: usize, simplex_idx: usize) -> bool {
        // A simplex is degenerate if it's in the image of a degeneracy map
        if dim == 0 {
            return false;
        }

        self.degeneracy_maps
            .values()
            .any(|&(target_dim, target_idx)| target_dim == dim && target_idx == simplex_idx)
    }

    /// Get all non-degenerate simplices of a given dimension.
    pub fn non_degenerate_simplices(&self, dim: usize) -> Vec<&AbstractSimplex> {
        self.simplices(dim)
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !self.is_degenerate(dim, *idx))
            .map(|(_, simplex)| simplex)
            .collect()
    }

    /// Compute Euler characteristic (using non-degenerate simplices).
    pub fn euler_characteristic(&self) -> Integer {
        let mut chi = Integer::from(0);
        if let Some(max_dim) = self.dimension() {
            for dim in 0..=max_dim {
                let n_non_deg = self.non_degenerate_simplices(dim).len() as i64;
                if dim % 2 == 0 {
                    chi = chi + Integer::from(n_non_deg);
                } else {
                    chi = chi - Integer::from(n_non_deg);
                }
            }
        }
        chi
    }

    /// Get the name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl Default for SimplicialSet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SimplicialSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "Simplicial set: {}", name)?;
        } else {
            write!(f, "Simplicial set")?;
        }
        if let Some(dim) = self.dimension() {
            write!(f, " (dimension {})", dim)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abstract_simplex_creation() {
        let simplex = AbstractSimplex::new(2, 0);
        assert_eq!(simplex.dimension(), 2);
        assert_eq!(simplex.index(), 0);
        assert_eq!(simplex.name(), None);
    }

    #[test]
    fn test_abstract_simplex_with_name() {
        let simplex = AbstractSimplex::with_name(1, 0, "edge");
        assert_eq!(simplex.name(), Some("edge"));
    }

    #[test]
    fn test_simplicial_set_creation() {
        let ss = SimplicialSet::new();
        assert_eq!(ss.dimension(), None);
    }

    #[test]
    fn test_simplicial_set_add_simplex() {
        let mut ss = SimplicialSet::new();
        let simplex = AbstractSimplex::new(0, 0);
        let idx = ss.add_simplex(simplex);
        assert_eq!(idx, 0);
        assert_eq!(ss.n_simplices(0), 1);
    }

    #[test]
    fn test_simplicial_set_face_maps() {
        let mut ss = SimplicialSet::new();
        let v0 = ss.add_simplex(AbstractSimplex::new(0, 0));
        let v1 = ss.add_simplex(AbstractSimplex::new(0, 1));
        let edge = ss.add_simplex(AbstractSimplex::new(1, 0));

        ss.add_face_map(1, edge, 0, 0, v0);
        ss.add_face_map(1, edge, 1, 0, v1);

        assert_eq!(ss.face(1, edge, 0), Some((0, v0)));
        assert_eq!(ss.face(1, edge, 1), Some((0, v1)));
    }

    #[test]
    fn test_simplicial_set_degeneracy_maps() {
        let mut ss = SimplicialSet::new();
        let v0 = ss.add_simplex(AbstractSimplex::new(0, 0));
        let deg = ss.add_simplex(AbstractSimplex::new(1, 0));

        ss.add_degeneracy_map(0, v0, 0, 1, deg);

        assert_eq!(ss.degeneracy(0, v0, 0), Some((1, deg)));
        assert!(ss.is_degenerate(1, deg));
    }

    #[test]
    fn test_non_degenerate_simplices() {
        let mut ss = SimplicialSet::new();
        let v0 = ss.add_simplex(AbstractSimplex::new(0, 0));
        let e0 = ss.add_simplex(AbstractSimplex::new(1, 0));
        let e1 = ss.add_simplex(AbstractSimplex::new(1, 1));

        // Mark e1 as degenerate
        ss.add_degeneracy_map(0, v0, 0, 1, e1);

        let non_deg = ss.non_degenerate_simplices(1);
        assert_eq!(non_deg.len(), 1);
    }

    #[test]
    fn test_simplicial_set_with_name() {
        let ss = SimplicialSet::with_name("Test");
        assert_eq!(ss.name(), Some("Test"));
    }
}
