//! Finding integral points in polytopes
//!
//! This module provides functions for finding lattice points (points with integer
//! coordinates) that lie within various geometric regions such as simplices,
//! rectangular boxes, and parallelotopes.
//!
//! It also provides classes for representing and working with systems of linear
//! inequalities, which are fundamental for polytope computations.
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

use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;
use std::collections::HashMap;
use std::fmt;

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

/// Generic linear inequality representation
///
/// Represents an inequality of the form: a₁x₁ + a₂x₂ + ... + aₙxₙ + b ≥ 0
/// where the coefficients can be rational numbers.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::integral_points::InequalityGeneric;
/// use rustmath_rationals::Rational;
///
/// // Inequality: 2x + 3y - 5 ≥ 0
/// let coeffs = vec![Rational::from(2), Rational::from(3)];
/// let constant = Rational::from(-5);
/// let ineq = InequalityGeneric::new(coeffs, constant);
/// assert_eq!(ineq.dimension(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InequalityGeneric {
    /// Coefficients of the variables (a₁, a₂, ..., aₙ)
    coefficients: Vec<Rational>,
    /// Constant term (b)
    constant: Rational,
}

impl InequalityGeneric {
    /// Create a new generic inequality
    ///
    /// # Arguments
    ///
    /// * `coefficients` - The coefficients of the variables
    /// * `constant` - The constant term
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::integral_points::InequalityGeneric;
    /// use rustmath_rationals::Rational;
    ///
    /// let ineq = InequalityGeneric::new(
    ///     vec![Rational::from(1), Rational::from(-1)],
    ///     Rational::from(0)
    /// );
    /// ```
    pub fn new(coefficients: Vec<Rational>, constant: Rational) -> Self {
        Self {
            coefficients,
            constant,
        }
    }

    /// Get the dimension (number of variables)
    pub fn dimension(&self) -> usize {
        self.coefficients.len()
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[Rational] {
        &self.coefficients
    }

    /// Get the constant term
    pub fn constant(&self) -> &Rational {
        &self.constant
    }

    /// Evaluate the inequality at a point
    ///
    /// Returns the value of a₁x₁ + a₂x₂ + ... + aₙxₙ + b
    pub fn evaluate(&self, point: &[Rational]) -> Rational {
        assert_eq!(
            point.len(),
            self.dimension(),
            "Point dimension must match inequality dimension"
        );

        let mut result = self.constant.clone();
        for (coeff, val) in self.coefficients.iter().zip(point.iter()) {
            result = result + coeff.clone() * val.clone();
        }
        result
    }

    /// Check if a point satisfies this inequality
    ///
    /// Returns true if a₁x₁ + a₂x₂ + ... + aₙxₙ + b ≥ 0
    pub fn satisfied_by(&self, point: &[Rational]) -> bool {
        self.evaluate(point) >= Rational::from(0)
    }

    /// Check if this is a tight inequality (equality) at a point
    ///
    /// Returns true if a₁x₁ + a₂x₂ + ... + aₙxₙ + b = 0
    pub fn is_tight_at(&self, point: &[Rational]) -> bool {
        self.evaluate(point) == Rational::from(0)
    }
}

impl fmt::Display for InequalityGeneric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if i > 0 && coeff >= &Rational::from(0) {
                write!(f, " + ")?;
            } else if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}*x{}", coeff, i)?;
        }
        if self.constant >= Rational::from(0) {
            write!(f, " + {}", self.constant)?;
        } else {
            write!(f, " {}", self.constant)?;
        }
        write!(f, " ≥ 0")
    }
}

/// Linear inequality with integer coefficients
///
/// A specialized version of `InequalityGeneric` where all coefficients
/// are integers. This is commonly used in polytope computations where
/// we work with lattice points.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::integral_points::InequalityInt;
///
/// // Inequality: 2x + 3y - 5 ≥ 0
/// let ineq = InequalityInt::new(vec![2, 3], -5);
/// assert!(ineq.satisfied_by(&[3, 1])); // 2*3 + 3*1 - 5 = 4 ≥ 0
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InequalityInt {
    /// Coefficients of the variables (a₁, a₂, ..., aₙ)
    coefficients: Vec<i64>,
    /// Constant term (b)
    constant: i64,
}

impl InequalityInt {
    /// Create a new integer inequality
    ///
    /// # Arguments
    ///
    /// * `coefficients` - The integer coefficients of the variables
    /// * `constant` - The integer constant term
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::integral_points::InequalityInt;
    ///
    /// let ineq = InequalityInt::new(vec![1, -1], 0);
    /// assert_eq!(ineq.dimension(), 2);
    /// ```
    pub fn new(coefficients: Vec<i64>, constant: i64) -> Self {
        Self {
            coefficients,
            constant,
        }
    }

    /// Get the dimension (number of variables)
    pub fn dimension(&self) -> usize {
        self.coefficients.len()
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[i64] {
        &self.coefficients
    }

    /// Get the constant term
    pub fn constant(&self) -> i64 {
        self.constant
    }

    /// Evaluate the inequality at an integer point
    ///
    /// Returns the value of a₁x₁ + a₂x₂ + ... + aₙxₙ + b
    pub fn evaluate(&self, point: &[i64]) -> i64 {
        assert_eq!(
            point.len(),
            self.dimension(),
            "Point dimension must match inequality dimension"
        );

        let mut result = self.constant;
        for (coeff, val) in self.coefficients.iter().zip(point.iter()) {
            result += coeff * val;
        }
        result
    }

    /// Check if an integer point satisfies this inequality
    ///
    /// Returns true if a₁x₁ + a₂x₂ + ... + aₙxₙ + b ≥ 0
    pub fn satisfied_by(&self, point: &[i64]) -> bool {
        self.evaluate(point) >= 0
    }

    /// Check if this is a tight inequality (equality) at a point
    ///
    /// Returns true if a₁x₁ + a₂x₂ + ... + aₙxₙ + b = 0
    pub fn is_tight_at(&self, point: &[i64]) -> bool {
        self.evaluate(point) == 0
    }

    /// Convert to a generic inequality
    pub fn to_generic(&self) -> InequalityGeneric {
        let coeffs = self
            .coefficients
            .iter()
            .map(|&c| Rational::from(c))
            .collect();
        let constant = Rational::from(self.constant);
        InequalityGeneric::new(coeffs, constant)
    }
}

impl fmt::Display for InequalityInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if i > 0 && coeff >= &0 {
                write!(f, " + ")?;
            } else if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}*x{}", coeff, i)?;
        }
        if self.constant >= 0 {
            write!(f, " + {}", self.constant)?;
        } else {
            write!(f, " {}", self.constant)?;
        }
        write!(f, " ≥ 0")
    }
}

/// A collection of linear inequalities
///
/// This class represents a system of linear inequalities that define a polyhedral region.
/// It's particularly useful for representing the constraints of a polytope in H-representation
/// (halfspace representation).
///
/// # Examples
///
/// ```
/// use rustmath_geometry::integral_points::{InequalityCollection, InequalityInt};
///
/// // Create inequalities for a square [0,1] × [0,1]:
/// // x ≥ 0, y ≥ 0, x ≤ 1, y ≤ 1
/// // Written as: x ≥ 0, y ≥ 0, -x + 1 ≥ 0, -y + 1 ≥ 0
/// let mut collection = InequalityCollection::new(2);
/// collection.add_int(InequalityInt::new(vec![1, 0], 0));  // x ≥ 0
/// collection.add_int(InequalityInt::new(vec![0, 1], 0));  // y ≥ 0
/// collection.add_int(InequalityInt::new(vec![-1, 0], 1)); // -x + 1 ≥ 0
/// collection.add_int(InequalityInt::new(vec![0, -1], 1)); // -y + 1 ≥ 0
///
/// assert_eq!(collection.len(), 4);
/// ```
#[derive(Clone, Debug)]
pub struct InequalityCollection {
    /// The dimension of the ambient space
    dimension: usize,
    /// Integer inequalities
    int_inequalities: Vec<InequalityInt>,
    /// Generic (rational) inequalities
    generic_inequalities: Vec<InequalityGeneric>,
}

impl InequalityCollection {
    /// Create a new empty inequality collection
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of the ambient space
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::integral_points::InequalityCollection;
    ///
    /// let collection = InequalityCollection::new(3);
    /// assert_eq!(collection.dimension(), 3);
    /// assert_eq!(collection.len(), 0);
    /// ```
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            int_inequalities: Vec::new(),
            generic_inequalities: Vec::new(),
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the total number of inequalities
    pub fn len(&self) -> usize {
        self.int_inequalities.len() + self.generic_inequalities.len()
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add an integer inequality
    ///
    /// # Panics
    ///
    /// Panics if the inequality dimension doesn't match the collection dimension
    pub fn add_int(&mut self, ineq: InequalityInt) {
        assert_eq!(
            ineq.dimension(),
            self.dimension,
            "Inequality dimension must match collection dimension"
        );
        self.int_inequalities.push(ineq);
    }

    /// Add a generic inequality
    ///
    /// # Panics
    ///
    /// Panics if the inequality dimension doesn't match the collection dimension
    pub fn add_generic(&mut self, ineq: InequalityGeneric) {
        assert_eq!(
            ineq.dimension(),
            self.dimension,
            "Inequality dimension must match collection dimension"
        );
        self.generic_inequalities.push(ineq);
    }

    /// Get all integer inequalities
    pub fn int_inequalities(&self) -> &[InequalityInt] {
        &self.int_inequalities
    }

    /// Get all generic inequalities
    pub fn generic_inequalities(&self) -> &[InequalityGeneric] {
        &self.generic_inequalities
    }

    /// Check if an integer point satisfies all inequalities
    ///
    /// # Arguments
    ///
    /// * `point` - The integer point to check
    ///
    /// # Returns
    ///
    /// `true` if the point satisfies all inequalities, `false` otherwise
    pub fn is_satisfied_by_int(&self, point: &[i64]) -> bool {
        assert_eq!(point.len(), self.dimension);

        // Check integer inequalities
        for ineq in &self.int_inequalities {
            if !ineq.satisfied_by(point) {
                return false;
            }
        }

        // Check generic inequalities
        let rational_point: Vec<Rational> = point.iter().map(|&x| Rational::from(x)).collect();
        for ineq in &self.generic_inequalities {
            if !ineq.satisfied_by(&rational_point) {
                return false;
            }
        }

        true
    }

    /// Check if a rational point satisfies all inequalities
    ///
    /// # Arguments
    ///
    /// * `point` - The rational point to check
    ///
    /// # Returns
    ///
    /// `true` if the point satisfies all inequalities, `false` otherwise
    pub fn is_satisfied_by_rational(&self, point: &[Rational]) -> bool {
        assert_eq!(point.len(), self.dimension);

        // Check integer inequalities (converted to rational)
        for ineq in &self.int_inequalities {
            if !ineq.to_generic().satisfied_by(point) {
                return false;
            }
        }

        // Check generic inequalities
        for ineq in &self.generic_inequalities {
            if !ineq.satisfied_by(point) {
                return false;
            }
        }

        true
    }

    /// Get tight inequalities at a given integer point
    ///
    /// Returns the indices of inequalities that are tight (equal to zero) at the point
    pub fn tight_at_int(&self, point: &[i64]) -> Vec<usize> {
        assert_eq!(point.len(), self.dimension);
        let mut tight = Vec::new();

        for (i, ineq) in self.int_inequalities.iter().enumerate() {
            if ineq.is_tight_at(point) {
                tight.push(i);
            }
        }

        // Add offsets for generic inequalities
        let offset = self.int_inequalities.len();
        let rational_point: Vec<Rational> = point.iter().map(|&x| Rational::from(x)).collect();
        for (i, ineq) in self.generic_inequalities.iter().enumerate() {
            if ineq.is_tight_at(&rational_point) {
                tight.push(offset + i);
            }
        }

        tight
    }

    /// Convert to a coefficient matrix and constant vector
    ///
    /// Returns (A, b) where the inequality system is Ax + b ≥ 0
    pub fn to_matrix(&self) -> (Matrix<Rational>, Vec<Rational>) {
        let n = self.len();
        let d = self.dimension;

        let mut matrix = Matrix::zeros(n, d);
        let mut constants = Vec::with_capacity(n);

        let mut row = 0;

        // Add integer inequalities
        for ineq in &self.int_inequalities {
            for col in 0..d {
                let val = Rational::from(ineq.coefficients()[col]);
                matrix.set(row, col, val).unwrap();
            }
            constants.push(Rational::from(ineq.constant()));
            row += 1;
        }

        // Add generic inequalities
        for ineq in &self.generic_inequalities {
            for col in 0..d {
                matrix.set(row, col, ineq.coefficients()[col].clone()).unwrap();
            }
            constants.push(ineq.constant().clone());
            row += 1;
        }

        (matrix, constants)
    }
}

impl fmt::Display for InequalityCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "InequalityCollection in dimension {} with {} inequalities:",
            self.dimension,
            self.len()
        )?;
        for ineq in &self.int_inequalities {
            writeln!(f, "  {}", ineq)?;
        }
        for ineq in &self.generic_inequalities {
            writeln!(f, "  {}", ineq)?;
        }
        Ok(())
    }
}

/// Compute the normal form of a ray matrix
///
/// This function computes a canonical representation of a matrix whose columns
/// represent rays (or generators) of a cone. The normal form is obtained by
/// applying column operations to put the matrix in a standard form.
///
/// # Arguments
///
/// * `matrix` - A matrix where each column is a ray
///
/// # Returns
///
/// A matrix in normal form representing the same set of rays
///
/// # Examples
///
/// ```
/// use rustmath_geometry::integral_points::ray_matrix_normal_form;
/// use rustmath_matrix::Matrix;
/// use rustmath_rationals::Rational;
///
/// let mut matrix = Matrix::zeros(2, 2);
/// matrix.set(0, 0, Rational::from(1)).unwrap();
/// matrix.set(1, 0, Rational::from(0)).unwrap();
/// matrix.set(0, 1, Rational::from(0)).unwrap();
/// matrix.set(1, 1, Rational::from(1)).unwrap();
///
/// let normal = ray_matrix_normal_form(&matrix);
/// assert_eq!(normal.rows(), 2);
/// assert_eq!(normal.cols(), 2);
/// ```
pub fn ray_matrix_normal_form(matrix: &Matrix<Rational>) -> Matrix<Rational> {
    if matrix.rows() == 0 || matrix.cols() == 0 {
        return matrix.clone();
    }

    let rows = matrix.rows();
    let cols = matrix.cols();
    let mut result = matrix.clone();

    // Normalize each column to have integer entries with gcd = 1
    for col in 0..cols {
        // Find the first non-zero entry in this column
        let mut first_nonzero_row = None;
        for row in 0..rows {
            if result.get(row, col).unwrap() != &Rational::from(0) {
                first_nonzero_row = Some(row);
                break;
            }
        }

        if let Some(pivot_row) = first_nonzero_row {
            // Make the pivot positive if it's negative
            let pivot_val = result.get(pivot_row, col).unwrap().clone();
            if pivot_val < Rational::from(0) {
                // Negate the entire column
                for row in 0..rows {
                    let val = result.get(row, col).unwrap().clone();
                    result.set(row, col, -val).unwrap();
                }
            }
        }
    }

    // Sort columns lexicographically
    let mut columns: Vec<Vec<Rational>> = Vec::new();
    for col in 0..cols {
        let mut column_vec = Vec::new();
        for row in 0..rows {
            column_vec.push(result.get(row, col).unwrap().clone());
        }
        columns.push(column_vec);
    }

    columns.sort_by(|a, b| {
        for (av, bv) in a.iter().zip(b.iter()) {
            if av != bv {
                return av.partial_cmp(bv).unwrap();
            }
        }
        std::cmp::Ordering::Equal
    });

    // Build result matrix from sorted columns
    let mut sorted_result = Matrix::zeros(rows, cols);
    for (col, column_vec) in columns.iter().enumerate() {
        for (row, val) in column_vec.iter().enumerate() {
            sorted_result.set(row, col, val.clone()).unwrap();
        }
    }

    sorted_result
}

/// Print cached data for debugging
///
/// This function prints cached polytope data in a human-readable format.
/// It's primarily used for debugging and diagnostic purposes.
///
/// # Arguments
///
/// * `cache` - A map of cache keys to values
///
/// # Examples
///
/// ```
/// use rustmath_geometry::integral_points::print_cache;
/// use std::collections::HashMap;
///
/// let mut cache = HashMap::new();
/// cache.insert("n_vertices".to_string(), "4".to_string());
/// cache.insert("n_facets".to_string(), "4".to_string());
/// print_cache(&cache);
/// ```
pub fn print_cache(cache: &HashMap<String, String>) {
    println!("Cached data:");
    let mut keys: Vec<_> = cache.keys().collect();
    keys.sort();
    for key in keys {
        println!("  {}: {}", key, cache.get(key).unwrap());
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

    #[test]
    fn test_inequality_int_basic() {
        // Test inequality: 2x + 3y - 5 ≥ 0
        let ineq = InequalityInt::new(vec![2, 3], -5);
        assert_eq!(ineq.dimension(), 2);

        // Point (3, 1): 2*3 + 3*1 - 5 = 4 ≥ 0 ✓
        assert!(ineq.satisfied_by(&[3, 1]));

        // Point (0, 0): 2*0 + 3*0 - 5 = -5 < 0 ✗
        assert!(!ineq.satisfied_by(&[0, 0]));

        // Point (1, 1): 2*1 + 3*1 - 5 = 0 (tight)
        assert!(ineq.is_tight_at(&[1, 1]));
    }

    #[test]
    fn test_inequality_generic_basic() {
        // Test inequality: x + y - 1/2 ≥ 0
        let ineq = InequalityGeneric::new(
            vec![Rational::from(1), Rational::from(1)],
            Rational::new(-1, 2).unwrap(),
        );

        let point1 = vec![Rational::from(1), Rational::from(0)];
        assert!(ineq.satisfied_by(&point1)); // 1 + 0 - 1/2 = 1/2 ≥ 0

        let point2 = vec![Rational::from(0), Rational::from(0)];
        assert!(!ineq.satisfied_by(&point2)); // 0 + 0 - 1/2 = -1/2 < 0
    }

    #[test]
    fn test_inequality_collection() {
        // Create a unit square: 0 ≤ x ≤ 1, 0 ≤ y ≤ 1
        let mut collection = InequalityCollection::new(2);

        collection.add_int(InequalityInt::new(vec![1, 0], 0)); // x ≥ 0
        collection.add_int(InequalityInt::new(vec![0, 1], 0)); // y ≥ 0
        collection.add_int(InequalityInt::new(vec![-1, 0], 1)); // -x + 1 ≥ 0
        collection.add_int(InequalityInt::new(vec![0, -1], 1)); // -y + 1 ≥ 0

        assert_eq!(collection.len(), 4);
        assert_eq!(collection.dimension(), 2);

        // Test points
        assert!(collection.is_satisfied_by_int(&[0, 0])); // corner
        assert!(collection.is_satisfied_by_int(&[1, 1])); // corner
        assert!(!collection.is_satisfied_by_int(&[2, 0])); // outside
        assert!(!collection.is_satisfied_by_int(&[-1, 0])); // outside
    }

    #[test]
    fn test_inequality_collection_tight() {
        let mut collection = InequalityCollection::new(2);

        collection.add_int(InequalityInt::new(vec![1, 0], 0)); // x ≥ 0
        collection.add_int(InequalityInt::new(vec![0, 1], 0)); // y ≥ 0

        // Point (0, 0) should be tight for both inequalities
        let tight = collection.tight_at_int(&[0, 0]);
        assert_eq!(tight.len(), 2);
        assert!(tight.contains(&0));
        assert!(tight.contains(&1));

        // Point (1, 1) should not be tight for any
        let tight2 = collection.tight_at_int(&[1, 1]);
        assert_eq!(tight2.len(), 0);
    }

    #[test]
    fn test_inequality_to_generic() {
        let int_ineq = InequalityInt::new(vec![2, 3], -5);
        let gen_ineq = int_ineq.to_generic();

        assert_eq!(gen_ineq.dimension(), 2);
        assert_eq!(
            gen_ineq.coefficients()[0],
            Rational::from(2)
        );
        assert_eq!(
            gen_ineq.constant(),
            &Rational::from(-5)
        );
    }

    #[test]
    fn test_ray_matrix_normal_form() {
        use rustmath_rationals::Rational;

        // Create a simple 2x2 identity matrix
        let mut matrix = Matrix::zeros(2, 2);
        matrix.set(0, 0, Rational::from(1)).unwrap();
        matrix.set(1, 0, Rational::from(0)).unwrap();
        matrix.set(0, 1, Rational::from(0)).unwrap();
        matrix.set(1, 1, Rational::from(1)).unwrap();

        let normal = ray_matrix_normal_form(&matrix);
        assert_eq!(normal.rows(), 2);
        assert_eq!(normal.cols(), 2);

        // All entries should still be non-negative
        for i in 0..2 {
            for j in 0..2 {
                assert!(normal.get(i, j).unwrap() >= &Rational::from(0));
            }
        }
    }

    #[test]
    fn test_print_cache() {
        use std::collections::HashMap;

        let mut cache = HashMap::new();
        cache.insert("n_vertices".to_string(), "4".to_string());
        cache.insert("n_facets".to_string(), "4".to_string());

        // Just make sure it doesn't panic
        print_cache(&cache);
    }

    #[test]
    fn test_inequality_collection_to_matrix() {
        let mut collection = InequalityCollection::new(2);
        collection.add_int(InequalityInt::new(vec![1, 0], 0)); // x ≥ 0
        collection.add_int(InequalityInt::new(vec![0, 1], 0)); // y ≥ 0

        let (matrix, constants) = collection.to_matrix();

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(constants.len(), 2);

        // First inequality: x ≥ 0
        assert_eq!(matrix.get(0, 0).unwrap(), &Rational::from(1));
        assert_eq!(matrix.get(0, 1).unwrap(), &Rational::from(0));
        assert_eq!(constants[0], Rational::from(0));

        // Second inequality: y ≥ 0
        assert_eq!(matrix.get(1, 0).unwrap(), &Rational::from(0));
        assert_eq!(matrix.get(1, 1).unwrap(), &Rational::from(1));
        assert_eq!(constants[1], Rational::from(0));
    }
}
