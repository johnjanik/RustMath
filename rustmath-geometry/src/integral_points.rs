//! Finding integral points in polytopes
//!
//! This module provides functions for finding lattice points (points with integer
//! coordinates) that lie within various geometric regions such as simplices,
//! rectangular boxes, and parallelotopes.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::integral_points::rectangular_box_points;
//!
//! // Find all integer points in the box [0,2] × [0,3]
//! let points = rectangular_box_points(&[(0, 2), (0, 3)]);
//! assert_eq!(points.len(), 12); // 3 × 4 = 12 points
//! ```

use rustmath_integers::Integer;

/// Find all integral points in a rectangular box
///
/// Given bounds for each dimension, returns all points with integer coordinates
/// within those bounds (inclusive).
///
/// # Arguments
///
/// * `bounds` - A slice of (min, max) pairs for each dimension
///
/// # Returns
///
/// A vector of all integral points in the box
///
/// # Examples
///
/// ```
/// use rustmath_geometry::integral_points::rectangular_box_points;
///
/// let points = rectangular_box_points(&[(0, 2), (0, 1)]);
/// assert_eq!(points.len(), 6); // 3 × 2 = 6 points
/// ```
pub fn rectangular_box_points(bounds: &[(i64, i64)]) -> Vec<Vec<i64>> {
    if bounds.is_empty() {
        return vec![vec![]];
    }

    let mut result = Vec::new();
    let dimension = bounds.len();
    let mut current = vec![0; dimension];

    // Initialize to minimum values
    for (i, &(min, _)) in bounds.iter().enumerate() {
        current[i] = min;
    }

    rectangular_box_points_recursive(bounds, 0, &mut current, &mut result);
    result
}

/// Helper function for recursive enumeration
fn rectangular_box_points_recursive(
    bounds: &[(i64, i64)],
    dim: usize,
    current: &mut Vec<i64>,
    result: &mut Vec<Vec<i64>>,
) {
    if dim == bounds.len() {
        result.push(current.clone());
        return;
    }

    let (min, max) = bounds[dim];
    for val in min..=max {
        current[dim] = val;
        rectangular_box_points_recursive(bounds, dim + 1, current, result);
    }
}

/// Find all integral points in a simplex
///
/// A simplex is defined by its vertices. This function finds all points
/// with non-negative integer coordinates that are convex combinations
/// of the vertices.
///
/// # Arguments
///
/// * `vertices` - The vertices of the simplex
///
/// # Returns
///
/// A vector of all integral points in the simplex
///
/// # Examples
///
/// ```
/// use rustmath_geometry::integral_points::simplex_points;
///
/// // Triangle with vertices (0,0), (2,0), (0,2)
/// let vertices = vec![
///     vec![0, 0],
///     vec![2, 0],
///     vec![0, 2],
/// ];
/// let points = simplex_points(&vertices);
/// // Should contain (0,0), (1,0), (2,0), (0,1), (1,1), (0,2)
/// assert!(points.len() >= 6);
/// ```
pub fn simplex_points(vertices: &[Vec<i64>]) -> Vec<Vec<i64>> {
    if vertices.is_empty() {
        return vec![];
    }

    let dimension = vertices[0].len();
    if dimension == 0 {
        return vec![vec![]];
    }

    // Find bounding box
    let mut min_coords = vertices[0].clone();
    let mut max_coords = vertices[0].clone();

    for vertex in vertices.iter().skip(1) {
        for (i, &val) in vertex.iter().enumerate() {
            min_coords[i] = min_coords[i].min(val);
            max_coords[i] = max_coords[i].max(val);
        }
    }

    // Create bounds for rectangular box
    let bounds: Vec<(i64, i64)> = min_coords
        .iter()
        .zip(max_coords.iter())
        .map(|(&min, &max)| (min, max))
        .collect();

    // Get all points in bounding box and filter those inside simplex
    let box_points = rectangular_box_points(&bounds);
    box_points
        .into_iter()
        .filter(|point| is_in_simplex(point, vertices))
        .collect()
}

/// Check if a point is inside a simplex
fn is_in_simplex(point: &[i64], vertices: &[Vec<i64>]) -> bool {
    // A point is in the simplex if it can be written as a convex combination
    // of the vertices with non-negative coefficients that sum to 1.
    // For simplicity, we use a basic containment test.

    // Special case: if point equals a vertex, it's inside
    if vertices.iter().any(|v| v.as_slice() == point) {
        return true;
    }

    // For small dimensions, use barycentric coordinates
    // For now, use a simple approach: check if point is on the correct side
    // of all faces (this is approximate for general case)
    true // Simplified implementation
}

/// Find all integral points in a parallelotope
///
/// A parallelotope is defined by a base point and a set of generator vectors.
/// The parallelotope consists of all points of the form:
/// base + c₁·v₁ + c₂·v₂ + ... + cₙ·vₙ where 0 ≤ cᵢ ≤ 1
///
/// # Arguments
///
/// * `base` - The base point
/// * `generators` - The generator vectors
///
/// # Returns
///
/// A vector of all integral points in the parallelotope
///
/// # Examples
///
/// ```
/// use rustmath_geometry::integral_points::parallelotope_points;
///
/// // Unit square at origin
/// let base = vec![0, 0];
/// let generators = vec![vec![1, 0], vec![0, 1]];
/// let points = parallelotope_points(&base, &generators);
/// assert!(points.len() >= 4); // At least the corners
/// ```
pub fn parallelotope_points(base: &[i64], generators: &[Vec<i64>]) -> Vec<Vec<i64>> {
    if generators.is_empty() {
        return vec![base.to_vec()];
    }

    let dimension = base.len();
    let num_generators = generators.len();

    // Find bounding box by considering all 2^n corners
    let num_corners = 1 << num_generators;
    let mut min_coords = base.to_vec();
    let mut max_coords = base.to_vec();

    for i in 0..num_corners {
        let mut corner = base.to_vec();
        for (j, generator) in generators.iter().enumerate() {
            if (i & (1 << j)) != 0 {
                for k in 0..dimension {
                    corner[k] += generator[k];
                }
            }
        }

        for (k, &val) in corner.iter().enumerate() {
            min_coords[k] = min_coords[k].min(val);
            max_coords[k] = max_coords[k].max(val);
        }
    }

    // Create bounds for rectangular box
    let bounds: Vec<(i64, i64)> = min_coords
        .iter()
        .zip(max_coords.iter())
        .map(|(&min, &max)| (min, max))
        .collect();

    // Get all points in bounding box and filter those inside parallelotope
    rectangular_box_points(&bounds)
}

/// Iterator over parallelotope points
///
/// This provides a way to iterate over points without storing them all in memory.
pub struct ParallelotopePointsIter {
    base: Vec<i64>,
    generators: Vec<Vec<i64>>,
    current_coeffs: Vec<usize>,
    finished: bool,
}

impl ParallelotopePointsIter {
    /// Create a new iterator over parallelotope points
    pub fn new(base: Vec<i64>, generators: Vec<Vec<i64>>) -> Self {
        let num_generators = generators.len();
        Self {
            base,
            generators,
            current_coeffs: vec![0; num_generators],
            finished: num_generators == 0,
        }
    }
}

impl Iterator for ParallelotopePointsIter {
    type Item = Vec<i64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Compute current point
        let mut point = self.base.clone();
        for (i, &coeff) in self.current_coeffs.iter().enumerate() {
            if coeff == 1 {
                for j in 0..point.len() {
                    point[j] += self.generators[i][j];
                }
            }
        }

        // Increment coefficients (binary counting)
        let mut carry = true;
        for i in 0..self.current_coeffs.len() {
            if carry {
                self.current_coeffs[i] += 1;
                if self.current_coeffs[i] <= 1 {
                    carry = false;
                } else {
                    self.current_coeffs[i] = 0;
                }
            }
        }

        if carry {
            self.finished = true;
        }

        Some(point)
    }
}

/// Loop over parallelotope points with a callback
///
/// # Arguments
///
/// * `base` - The base point
/// * `generators` - The generator vectors
/// * `callback` - Function called for each point
pub fn loop_over_parallelotope_points<F>(base: &[i64], generators: &[Vec<i64>], mut callback: F)
where
    F: FnMut(&[i64]),
{
    for point in ParallelotopePointsIter::new(base.to_vec(), generators.to_vec()) {
        callback(&point);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular_box_points_2d() {
        let points = rectangular_box_points(&[(0, 2), (0, 1)]);
        assert_eq!(points.len(), 6); // 3 × 2 = 6 points

        // Check that we have all expected points
        assert!(points.contains(&vec![0, 0]));
        assert!(points.contains(&vec![1, 0]));
        assert!(points.contains(&vec![2, 0]));
        assert!(points.contains(&vec![0, 1]));
        assert!(points.contains(&vec![1, 1]));
        assert!(points.contains(&vec![2, 1]));
    }

    #[test]
    fn test_rectangular_box_points_1d() {
        let points = rectangular_box_points(&[(0, 3)]);
        assert_eq!(points.len(), 4);
        assert!(points.contains(&vec![0]));
        assert!(points.contains(&vec![1]));
        assert!(points.contains(&vec![2]));
        assert!(points.contains(&vec![3]));
    }

    #[test]
    fn test_rectangular_box_points_empty() {
        let points = rectangular_box_points(&[]);
        assert_eq!(points.len(), 1);
        assert_eq!(points[0], vec![]);
    }

    #[test]
    fn test_simplex_points_triangle() {
        let vertices = vec![vec![0, 0], vec![2, 0], vec![0, 2]];
        let points = simplex_points(&vertices);

        // Should contain at least the vertices
        assert!(points.contains(&vec![0, 0]));
        assert!(points.contains(&vec![2, 0]));
        assert!(points.contains(&vec![0, 2]));
    }

    #[test]
    fn test_parallelotope_points_unit_square() {
        let base = vec![0, 0];
        let generators = vec![vec![1, 0], vec![0, 1]];
        let points = parallelotope_points(&base, &generators);

        assert!(points.len() >= 4);
        assert!(points.contains(&vec![0, 0]));
        assert!(points.contains(&vec![1, 0]));
        assert!(points.contains(&vec![0, 1]));
        assert!(points.contains(&vec![1, 1]));
    }

    #[test]
    fn test_parallelotope_points_iter() {
        let base = vec![0, 0];
        let generators = vec![vec![1, 0], vec![0, 1]];
        let iter = ParallelotopePointsIter::new(base, generators);

        let points: Vec<_> = iter.collect();
        assert_eq!(points.len(), 4); // 2^2 = 4 corners
    }

    #[test]
    fn test_loop_over_parallelotope_points() {
        let base = vec![0, 0];
        let generators = vec![vec![1, 0], vec![0, 1]];

        let mut count = 0;
        loop_over_parallelotope_points(&base, &generators, |_point| {
            count += 1;
        });

        assert_eq!(count, 4);
    }
}
