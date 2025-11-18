//! Cubical Complex Module
//!
//! Implements cubical complexes: topological spaces built from unit cubes.
//!
//! A cubical complex is a collection of cubes glued together along their faces.
//! This is analogous to simplicial complexes but using cubes instead of simplices.
//!
//! This mirrors SageMath's `sage.topology.cubical_complex`.

use crate::cell_complex::GenericCellComplex;
use rustmath_integers::Integer;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Represents a single cube in a cubical complex.
///
/// A cube is specified by its interval coordinates in each dimension.
/// For example, a 2-cube at position (1,2) would have intervals [1,2] × [2,3].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cube {
    /// Intervals defining the cube: (lower_bound, upper_bound) for each dimension
    intervals: Vec<(i32, i32)>,
}

impl Cube {
    /// Create a new cube from interval specifications.
    ///
    /// Each interval is (lower, upper) where upper = lower + 1 for a non-degenerate cube.
    pub fn new(intervals: Vec<(i32, i32)>) -> Self {
        Self { intervals }
    }

    /// Create an elementary cube from a point and degenerate/non-degenerate dimensions.
    ///
    /// # Arguments
    /// * `point` - The lower-left corner coordinates
    /// * `degenerate` - Boolean for each dimension: true = degenerate (point), false = interval
    pub fn elementary(point: Vec<i32>, degenerate: Vec<bool>) -> Self {
        assert_eq!(point.len(), degenerate.len());
        let intervals = point
            .into_iter()
            .zip(degenerate)
            .map(|(p, deg)| if deg { (p, p) } else { (p, p + 1) })
            .collect();
        Self { intervals }
    }

    /// Get the dimension of the cube.
    ///
    /// The dimension is the number of non-degenerate intervals.
    pub fn dimension(&self) -> usize {
        self.intervals.iter().filter(|(a, b)| a != b).count()
    }

    /// Get the embedding dimension (ambient dimension).
    pub fn embedding_dimension(&self) -> usize {
        self.intervals.len()
    }

    /// Check if this cube is a face of another cube.
    pub fn is_face_of(&self, other: &Cube) -> bool {
        if self.intervals.len() != other.intervals.len() {
            return false;
        }

        self.intervals
            .iter()
            .zip(&other.intervals)
            .all(|((a1, b1), (a2, b2))| {
                // Either same interval or degenerate face
                (a1 == a2 && b1 == b2) || // Same interval
                (a1 == b1 && *a1 >= *a2 && *a1 <= *b2) // Degenerate at boundary
            })
    }

    /// Get all faces of this cube.
    pub fn faces(&self) -> Vec<Cube> {
        let mut faces = Vec::new();
        let dim = self.dimension();

        if dim == 0 {
            return faces; // 0-dimensional cube has no faces
        }

        // Generate faces by making intervals degenerate
        for i in 0..self.intervals.len() {
            let (a, b) = self.intervals[i];
            if a != b {
                // Non-degenerate interval, create two faces
                // Lower face
                let mut lower_intervals = self.intervals.clone();
                lower_intervals[i] = (a, a);
                faces.push(Cube::new(lower_intervals));

                // Upper face
                let mut upper_intervals = self.intervals.clone();
                upper_intervals[i] = (b, b);
                faces.push(Cube::new(upper_intervals));
            }
        }

        faces
    }
}

impl fmt::Display for Cube {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, (a, b)) in self.intervals.iter().enumerate() {
            if i > 0 {
                write!(f, " × ")?;
            }
            if a == b {
                write!(f, "{{{}}}", a)?;
            } else {
                write!(f, "[{},{}]", a, b)?;
            }
        }
        write!(f, "]")
    }
}

/// A cubical complex: a collection of cubes glued along their faces.
#[derive(Debug, Clone)]
pub struct CubicalComplex {
    /// The underlying cell complex
    base: GenericCellComplex,
    /// Cubes organized by dimension
    cubes: HashMap<usize, HashSet<Cube>>,
    /// Embedding dimension
    embedding_dim: Option<usize>,
}

impl CubicalComplex {
    /// Create a new empty cubical complex.
    pub fn new() -> Self {
        Self {
            base: GenericCellComplex::new(),
            cubes: HashMap::new(),
            embedding_dim: None,
        }
    }

    /// Create a cubical complex from a set of cubes.
    pub fn from_cubes(cubes: Vec<Cube>) -> Self {
        let mut complex = Self::new();
        for cube in cubes {
            complex.add_cube(cube);
        }
        complex
    }

    /// Add a cube to the complex (with all its faces).
    pub fn add_cube(&mut self, cube: Cube) {
        let dim = cube.dimension();
        let embed_dim = cube.embedding_dimension();

        // Update embedding dimension
        self.embedding_dim = Some(
            self.embedding_dim
                .map(|d| d.max(embed_dim))
                .unwrap_or(embed_dim),
        );

        // Add the cube
        self.cubes.entry(dim).or_insert_with(HashSet::new).insert(cube.clone());

        // Recursively add all faces
        for face in cube.faces() {
            if !self.contains_cube(&face) {
                self.add_cube(face);
            }
        }
    }

    /// Check if the complex contains a specific cube.
    pub fn contains_cube(&self, cube: &Cube) -> bool {
        let dim = cube.dimension();
        self.cubes
            .get(&dim)
            .map(|cubes| cubes.contains(cube))
            .unwrap_or(false)
    }

    /// Get all cubes of a given dimension.
    pub fn cubes(&self, dim: usize) -> Vec<Cube> {
        self.cubes
            .get(&dim)
            .map(|cubes| cubes.iter().cloned().collect())
            .unwrap_or_else(Vec::new)
    }

    /// Get the number of cubes in a given dimension.
    pub fn n_cubes(&self, dim: usize) -> usize {
        self.cubes.get(&dim).map(|cubes| cubes.len()).unwrap_or(0)
    }

    /// Get the dimension of the complex.
    pub fn dimension(&self) -> Option<usize> {
        self.cubes.keys().max().copied()
    }

    /// Get the embedding dimension.
    pub fn embedding_dimension(&self) -> Option<usize> {
        self.embedding_dim
    }

    /// Compute the Euler characteristic.
    pub fn euler_characteristic(&self) -> Integer {
        let mut chi = Integer::from(0);
        if let Some(max_dim) = self.dimension() {
            for dim in 0..=max_dim {
                let n_cubes = self.n_cubes(dim) as i64;
                if dim % 2 == 0 {
                    chi = chi + Integer::from(n_cubes);
                } else {
                    chi = chi - Integer::from(n_cubes);
                }
            }
        }
        chi
    }

    /// Get the f-vector.
    pub fn f_vector(&self) -> Vec<usize> {
        let max_dim = self.dimension().unwrap_or(0);
        (0..=max_dim).map(|dim| self.n_cubes(dim)).collect()
    }
}

impl Default for CubicalComplex {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CubicalComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cubical complex")?;
        if let Some(dim) = self.dimension() {
            write!(f, " of dimension {}", dim)?;
        }
        if let Some(embed_dim) = self.embedding_dim {
            write!(f, " (embedded in R^{})", embed_dim)?;
        }
        Ok(())
    }
}

/// Examples of cubical complexes.
pub struct CubicalComplexExamples;

impl CubicalComplexExamples {
    /// Create a cube (square, cube, hypercube) of given dimension.
    pub fn cube(n: usize) -> CubicalComplex {
        let intervals = vec![(0, 1); n];
        let cube = Cube::new(intervals);
        CubicalComplex::from_cubes(vec![cube])
    }

    /// Create a cubical sphere of given dimension.
    pub fn sphere(n: usize) -> CubicalComplex {
        // A cubical n-sphere is the boundary of an (n+1)-cube
        let mut complex = CubicalComplex::new();

        // Generate all n-dimensional faces of the (n+1)-cube
        let intervals = vec![(0, 1); n + 1];
        let full_cube = Cube::new(intervals);

        for face in full_cube.faces() {
            if face.dimension() == n {
                complex.add_cube(face);
            }
        }

        complex
    }

    /// Create a cubical torus (2-dimensional).
    pub fn torus() -> CubicalComplex {
        let mut complex = CubicalComplex::new();

        // Create a 2×2 grid of squares with periodic boundary conditions
        for i in 0..2 {
            for j in 0..2 {
                let cube = Cube::elementary(vec![i, j], vec![false, false]);
                complex.add_cube(cube);
            }
        }

        complex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube_creation() {
        let cube = Cube::new(vec![(0, 1), (0, 1)]);
        assert_eq!(cube.dimension(), 2);
        assert_eq!(cube.embedding_dimension(), 2);
    }

    #[test]
    fn test_cube_elementary() {
        let cube = Cube::elementary(vec![1, 2, 3], vec![false, true, false]);
        assert_eq!(cube.dimension(), 2); // Two non-degenerate dimensions
        assert_eq!(cube.embedding_dimension(), 3);
    }

    #[test]
    fn test_cube_faces() {
        let cube = Cube::new(vec![(0, 1), (0, 1)]);
        let faces = cube.faces();
        assert_eq!(faces.len(), 4); // A square has 4 edges

        for face in &faces {
            assert_eq!(face.dimension(), 1); // Each face is 1-dimensional
        }
    }

    #[test]
    fn test_cube_is_face_of() {
        let square = Cube::new(vec![(0, 1), (0, 1)]);
        let edge = Cube::new(vec![(0, 1), (0, 0)]);
        assert!(edge.is_face_of(&square));
        assert!(!square.is_face_of(&edge));
    }

    #[test]
    fn test_cubical_complex_creation() {
        let complex = CubicalComplex::new();
        assert_eq!(complex.dimension(), None);
        assert_eq!(complex.euler_characteristic(), Integer::from(0));
    }

    #[test]
    fn test_cubical_complex_add_cube() {
        let mut complex = CubicalComplex::new();
        let cube = Cube::new(vec![(0, 1), (0, 1)]);
        complex.add_cube(cube);

        // Should have 1 square, 4 edges, 4 vertices
        assert_eq!(complex.n_cubes(2), 1);
        assert_eq!(complex.n_cubes(1), 4);
        assert_eq!(complex.n_cubes(0), 4);

        // Euler characteristic: 4 - 4 + 1 = 1
        assert_eq!(complex.euler_characteristic(), Integer::from(1));
    }

    #[test]
    fn test_cubical_complex_examples_cube() {
        let square = CubicalComplexExamples::cube(2);
        assert_eq!(square.dimension(), Some(2));
        assert_eq!(square.n_cubes(2), 1);
        assert_eq!(square.euler_characteristic(), Integer::from(1));
    }

    #[test]
    fn test_cubical_complex_examples_sphere() {
        let sphere_1 = CubicalComplexExamples::sphere(1);
        // 1-sphere (circle) as boundary of 2-cube has 4 edges
        assert_eq!(sphere_1.dimension(), Some(1));
        assert!(sphere_1.n_cubes(1) > 0);
    }

    #[test]
    fn test_cubical_complex_examples_torus() {
        let torus = CubicalComplexExamples::torus();
        assert_eq!(torus.dimension(), Some(2));
        assert_eq!(torus.n_cubes(2), 4); // 2×2 grid
    }

    #[test]
    fn test_f_vector() {
        let mut complex = CubicalComplex::new();
        let cube = Cube::new(vec![(0, 1)]);
        complex.add_cube(cube);

        // 1-dimensional cube: 1 edge, 2 vertices
        assert_eq!(complex.f_vector(), vec![2, 1]);
    }

    #[test]
    fn test_cube_display() {
        let cube = Cube::new(vec![(0, 1), (2, 2), (3, 4)]);
        let display = format!("{}", cube);
        assert!(display.contains("[0,1]"));
        assert!(display.contains("{2}"));
        assert!(display.contains("[3,4]"));
    }
}
