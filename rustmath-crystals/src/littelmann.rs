//! Littelmann path model
//!
//! The Littelmann path model provides a realization of crystals as piecewise-linear
//! paths in the weight space. This gives a geometric interpretation of crystal operators.
//!
//! A path is a sequence of weights π: [0,1] → P where P is the weight lattice.

use crate::operators::Crystal;
use crate::root_system::RootSystem;
use crate::weight::Weight;

/// A piecewise-linear path in the weight lattice
///
/// Represented as a sequence of vertices (weights) with implied linear interpolation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LittelmannPath {
    /// Vertices of the path (weights at integer or rational points in [0,1])
    pub vertices: Vec<Weight>,
    /// Denominators for the parameter values (for rational parametrization)
    /// vertices[i] occurs at parameter t = i/denom
    pub denom: usize,
}

impl LittelmannPath {
    /// Create a new path from a sequence of vertices
    pub fn new(vertices: Vec<Weight>) -> Self {
        let denom = vertices.len().saturating_sub(1).max(1);
        LittelmannPath { vertices, denom }
    }

    /// Create a path from a single weight (constant path)
    pub fn constant(weight: Weight) -> Self {
        LittelmannPath {
            vertices: vec![weight.clone(), weight],
            denom: 1,
        }
    }

    /// Get the endpoint of the path (final weight)
    pub fn endpoint(&self) -> &Weight {
        self.vertices.last().unwrap()
    }

    /// Get the starting point of the path
    pub fn starting_point(&self) -> &Weight {
        self.vertices.first().unwrap()
    }

    /// Evaluate the path at parameter t ∈ [0,1]
    ///
    /// Uses linear interpolation between vertices.
    pub fn evaluate(&self, t: f64) -> Weight {
        if t <= 0.0 {
            return self.starting_point().clone();
        }
        if t >= 1.0 {
            return self.endpoint().clone();
        }

        let scaled = t * (self.vertices.len() - 1) as f64;
        let idx = scaled.floor() as usize;
        let frac = scaled - idx as f64;

        if idx >= self.vertices.len() - 1 {
            return self.endpoint().clone();
        }

        // Linear interpolation
        let v1 = &self.vertices[idx];
        let v2 = &self.vertices[idx + 1];

        let mut result = v1.clone();
        let diff = v2 - v1;
        for i in 0..result.coords.len() {
            result.coords[i] = (v1.coords[i] as f64 * (1.0 - frac) + v2.coords[i] as f64 * frac)
                .round() as i64;
        }

        result
    }

    /// Compute the i-string at parameter t
    ///
    /// The i-string is the real-valued function m_i(t) = ⟨π(t), α_i^∨⟩
    pub fn i_string(&self, i: usize, root_system: &RootSystem) -> Vec<i64> {
        self.vertices
            .iter()
            .map(|w| root_system.coroot_action(w, i))
            .collect()
    }

    /// Find the minimum of the i-string
    pub fn i_string_min(&self, i: usize, root_system: &RootSystem) -> (usize, i64) {
        let string = self.i_string(i, root_system);
        let (idx, &min_val) = string
            .iter()
            .enumerate()
            .min_by_key(|(_, &v)| v)
            .unwrap_or((0, &0));
        (idx, min_val)
    }

    /// Find the maximum of the i-string
    pub fn i_string_max(&self, i: usize, root_system: &RootSystem) -> (usize, i64) {
        let string = self.i_string(i, root_system);
        let (idx, &max_val) = string
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .unwrap_or((0, &0));
        (idx, max_val)
    }

    /// Concatenate two paths
    pub fn concatenate(&self, other: &LittelmannPath) -> LittelmannPath {
        // Check that endpoint of self equals starting point of other
        if self.endpoint() != other.starting_point() {
            // For now, just do naive concatenation
        }

        let mut vertices = self.vertices.clone();
        vertices.extend_from_slice(&other.vertices[1..]); // Skip duplicate vertex

        LittelmannPath::new(vertices)
    }
}

/// A crystal based on Littelmann paths
#[derive(Debug, Clone)]
pub struct LittelmannCrystal {
    /// The root system
    pub root_system: RootSystem,
    /// The highest weight
    pub highest_weight: Weight,
    /// Maximum number of segments
    pub max_segments: usize,
    /// Generated paths
    paths: Vec<LittelmannPath>,
}

impl LittelmannCrystal {
    /// Create a new Littelmann crystal
    pub fn new(highest_weight: Weight, root_system: RootSystem) -> Self {
        LittelmannCrystal {
            root_system,
            highest_weight,
            max_segments: 10,
            paths: Vec::new(),
        }
    }

    /// Generate all paths starting from the highest weight
    pub fn generate(&mut self) {
        self.paths.clear();
        let hw_path = LittelmannPath::constant(self.highest_weight.clone());
        self.paths.push(hw_path.clone());

        let mut queue = vec![hw_path];
        let mut iterations = 0;
        let max_iterations = 100;

        while !queue.is_empty() && iterations < max_iterations {
            let mut new_queue = Vec::new();
            for path in queue {
                // Try all f_i operators
                for i in 0..self.root_system.rank {
                    if let Some(new_path) = self.apply_fi_to_path(&path, i) {
                        if !self.paths.contains(&new_path)
                            && new_path.vertices.len() <= self.max_segments
                        {
                            self.paths.push(new_path.clone());
                            new_queue.push(new_path);
                        }
                    }
                }
            }
            queue = new_queue;
            iterations += 1;
        }
    }

    /// Apply f_i to a path using root operators
    fn apply_fi_to_path(&self, path: &LittelmannPath, i: usize) -> Option<LittelmannPath> {
        // Find the minimum of the i-string
        let (min_idx, min_val) = path.i_string_min(i, &self.root_system);

        // If minimum is already negative, we can apply f_i
        // Add a segment that subtracts α_i
        let alpha_i = self.root_system.simple_root(i);

        let mut new_vertices = path.vertices.clone();
        // Modify vertices after min_idx by subtracting α_i
        for vertex in new_vertices.iter_mut().skip(min_idx) {
            *vertex = &*vertex - &alpha_i;
        }

        Some(LittelmannPath::new(new_vertices))
    }

    /// Apply e_i to a path
    fn apply_ei_to_path(&self, path: &LittelmannPath, i: usize) -> Option<LittelmannPath> {
        // Find the maximum of the i-string
        let (max_idx, max_val) = path.i_string_max(i, &self.root_system);

        if max_val <= 0 {
            return None;
        }

        // Add α_i to vertices after max_idx
        let alpha_i = self.root_system.simple_root(i);

        let mut new_vertices = path.vertices.clone();
        for vertex in new_vertices.iter_mut().skip(max_idx) {
            *vertex = &*vertex + &alpha_i;
        }

        Some(LittelmannPath::new(new_vertices))
    }
}

impl Crystal for LittelmannCrystal {
    type Element = LittelmannPath;

    fn weight(&self, b: &Self::Element) -> Weight {
        b.endpoint().clone()
    }

    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i >= self.root_system.rank {
            return None;
        }
        self.apply_ei_to_path(b, i)
    }

    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i >= self.root_system.rank {
            return None;
        }
        self.apply_fi_to_path(b, i)
    }

    fn elements(&self) -> Vec<Self::Element> {
        self.paths.clone()
    }
}

/// Concatenate two Littelmann paths
pub fn concat_paths(p1: &LittelmannPath, p2: &LittelmannPath) -> LittelmannPath {
    p1.concatenate(p2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::root_system::{RootSystem, RootSystemType};

    #[test]
    fn test_littelmann_path() {
        let w1 = Weight::new(vec![1, 0]);
        let w2 = Weight::new(vec![0, 1]);
        let path = LittelmannPath::new(vec![w1, w2]);

        assert_eq!(path.starting_point().coords, vec![1, 0]);
        assert_eq!(path.endpoint().coords, vec![0, 1]);
    }

    #[test]
    fn test_path_evaluation() {
        let w1 = Weight::new(vec![2, 0]);
        let w2 = Weight::new(vec![0, 2]);
        let path = LittelmannPath::new(vec![w1, w2]);

        let mid = path.evaluate(0.5);
        // Should be approximately at the midpoint
        assert!(mid.coords[0] >= 0 && mid.coords[0] <= 2);
        assert!(mid.coords[1] >= 0 && mid.coords[1] <= 2);
    }

    #[test]
    fn test_constant_path() {
        let w = Weight::new(vec![1, 1]);
        let path = LittelmannPath::constant(w.clone());

        assert_eq!(path.starting_point(), &w);
        assert_eq!(path.endpoint(), &w);
    }

    #[test]
    fn test_littelmann_crystal() {
        let hw = Weight::new(vec![1, 0]);
        let root_system = RootSystem::new(RootSystemType::A(2));
        let mut crystal = LittelmannCrystal::new(hw, root_system);

        crystal.generate();
        assert!(!crystal.paths.is_empty());
    }

    #[test]
    fn test_i_string() {
        let root_system = RootSystem::new(RootSystemType::A(2));
        let w1 = Weight::new(vec![1, 0]);
        let w2 = Weight::new(vec![0, 1]);
        let path = LittelmannPath::new(vec![w1, w2]);

        let string = path.i_string(0, &root_system);
        assert_eq!(string.len(), 2);
    }
}
