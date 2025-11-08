//! Quadratic forms over the integers
//!
//! A quadratic form is a homogeneous polynomial of degree 2 in n variables.
//! For example, Q(x, y) = x² + xy + y² is a binary quadratic form.
//!
//! Every quadratic form can be represented by a symmetric matrix A such that
//! Q(x) = x^T A x, where x is a column vector.

use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;

/// A quadratic form over the integers
///
/// Represented by a symmetric matrix A such that Q(x) = x^T A x.
/// The matrix is stored in upper-triangular form for efficiency.
///
/// # Examples
///
/// ```
/// use rustmath_numbertheory::quadratic_forms::QuadraticForm;
/// use rustmath_integers::Integer;
///
/// // Q(x, y) = x² + y² (sum of two squares)
/// let matrix = vec![
///     vec![Integer::from(1), Integer::from(0)],
///     vec![Integer::from(0), Integer::from(1)],
/// ];
/// let form = QuadraticForm::new(matrix).unwrap();
///
/// // Evaluate at (3, 4): 3² + 4² = 25
/// let result = form.evaluate(&[Integer::from(3), Integer::from(4)]).unwrap();
/// assert_eq!(result, Integer::from(25));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuadraticForm {
    /// The symmetric matrix representing the quadratic form
    /// Stored as a Vec<Vec<Integer>> where matrix[i][j] is the (i,j) entry
    matrix: Vec<Vec<Integer>>,
    /// Number of variables (dimension)
    dimension: usize,
}

impl QuadraticForm {
    /// Create a new quadratic form from a symmetric matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - A symmetric n×n matrix of integers
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The matrix is not square
    /// - The matrix is not symmetric
    /// - The matrix is empty
    pub fn new(matrix: Vec<Vec<Integer>>) -> Result<Self> {
        if matrix.is_empty() {
            return Err(MathError::InvalidArgument(
                "Matrix cannot be empty".to_string(),
            ));
        }

        let n = matrix.len();

        // Check that all rows have the same length
        for row in &matrix {
            if row.len() != n {
                return Err(MathError::InvalidArgument(
                    "Matrix must be square".to_string(),
                ));
            }
        }

        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                if matrix[i][j] != matrix[j][i] {
                    return Err(MathError::InvalidArgument(
                        "Matrix must be symmetric".to_string(),
                    ));
                }
            }
        }

        Ok(QuadraticForm {
            matrix,
            dimension: n,
        })
    }

    /// Create a diagonal quadratic form with given coefficients
    ///
    /// For coefficients [a₁, a₂, ..., aₙ], creates the form
    /// Q(x₁, ..., xₙ) = a₁x₁² + a₂x₂² + ... + aₙxₙ²
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_numbertheory::quadratic_forms::QuadraticForm;
    /// use rustmath_integers::Integer;
    ///
    /// // Q(x, y, z) = x² + 2y² + 3z²
    /// let form = QuadraticForm::diagonal(vec![
    ///     Integer::from(1),
    ///     Integer::from(2),
    ///     Integer::from(3),
    /// ]).unwrap();
    /// ```
    pub fn diagonal(coefficients: Vec<Integer>) -> Result<Self> {
        if coefficients.is_empty() {
            return Err(MathError::InvalidArgument(
                "Coefficients cannot be empty".to_string(),
            ));
        }

        let n = coefficients.len();
        let mut matrix = vec![vec![Integer::zero(); n]; n];

        for i in 0..n {
            matrix[i][i] = coefficients[i].clone();
        }

        Ok(QuadraticForm {
            matrix,
            dimension: n,
        })
    }

    /// Get the dimension (number of variables) of this form
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the matrix representation
    pub fn matrix(&self) -> &[Vec<Integer>] {
        &self.matrix
    }

    /// Evaluate the quadratic form at a given point
    ///
    /// Computes Q(x) = x^T A x
    ///
    /// # Arguments
    ///
    /// * `point` - A vector of integers [x₁, x₂, ..., xₙ]
    ///
    /// # Errors
    ///
    /// Returns an error if the point dimension doesn't match the form dimension
    pub fn evaluate(&self, point: &[Integer]) -> Result<Integer> {
        if point.len() != self.dimension {
            return Err(MathError::InvalidArgument(format!(
                "Point dimension {} doesn't match form dimension {}",
                point.len(),
                self.dimension
            )));
        }

        let mut result = Integer::zero();

        // Compute x^T A x
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                result = result + point[i].clone() * self.matrix[i][j].clone() * point[j].clone();
            }
        }

        Ok(result)
    }

    /// Check if this quadratic form represents a given integer n
    ///
    /// Returns true if there exist integers x₁, ..., xₙ such that Q(x₁, ..., xₙ) = n
    ///
    /// Note: This performs an exhaustive search up to a bound, so it may not find
    /// all representations for large n or high-dimensional forms.
    ///
    /// # Arguments
    ///
    /// * `n` - The integer to check
    /// * `bound` - Maximum absolute value to search for each variable
    pub fn represents(&self, n: &Integer, bound: usize) -> bool {
        self.find_representation(n, bound).is_some()
    }

    /// Find a representation of n by this quadratic form
    ///
    /// Returns a vector x such that Q(x) = n, or None if no representation
    /// is found within the search bound.
    ///
    /// # Arguments
    ///
    /// * `n` - The integer to represent
    /// * `bound` - Maximum absolute value to search for each variable
    pub fn find_representation(&self, n: &Integer, bound: usize) -> Option<Vec<Integer>> {
        // For dimension 1
        if self.dimension == 1 {
            let a = &self.matrix[0][0];
            if a.is_zero() {
                return if n.is_zero() {
                    Some(vec![Integer::zero()])
                } else {
                    None
                };
            }

            // Check if n/a is a perfect square
            if n % a != Integer::zero() {
                return None;
            }

            let quotient = n.clone() / a.clone();
            if let Ok(sqrt) = quotient.sqrt() {
                if &(sqrt.clone() * sqrt.clone()) == &quotient {
                    return Some(vec![sqrt]);
                }
            }
            return None;
        }

        // For higher dimensions, use recursive search
        self.search_representation(n, bound, 0, &mut vec![Integer::zero(); self.dimension])
    }

    /// Recursive helper for find_representation
    fn search_representation(
        &self,
        target: &Integer,
        bound: usize,
        var_index: usize,
        current: &mut Vec<Integer>,
    ) -> Option<Vec<Integer>> {
        if var_index == self.dimension {
            // Base case: evaluate and check
            if let Ok(value) = self.evaluate(current) {
                if &value == target {
                    return Some(current.clone());
                }
            }
            return None;
        }

        // Try values from -bound to bound for current variable
        let bound_int = bound as i64;
        for val in -bound_int..=bound_int {
            current[var_index] = Integer::from(val);
            if let Some(result) = self.search_representation(target, bound, var_index + 1, current)
            {
                return Some(result);
            }
        }

        None
    }

    /// Count the number of representations of n by this quadratic form
    ///
    /// Returns the number of distinct integer vectors x such that Q(x) = n,
    /// searching within the given bound.
    ///
    /// # Arguments
    ///
    /// * `n` - The integer to represent
    /// * `bound` - Maximum absolute value to search for each variable
    pub fn count_representations(&self, n: &Integer, bound: usize) -> usize {
        let mut count = 0;
        self.count_representations_recursive(
            n,
            bound,
            0,
            &mut vec![Integer::zero(); self.dimension],
            &mut count,
        );
        count
    }

    /// Recursive helper for count_representations
    fn count_representations_recursive(
        &self,
        target: &Integer,
        bound: usize,
        var_index: usize,
        current: &mut Vec<Integer>,
        count: &mut usize,
    ) {
        if var_index == self.dimension {
            // Base case: evaluate and check
            if let Ok(value) = self.evaluate(current) {
                if &value == target {
                    *count += 1;
                }
            }
            return;
        }

        // Try values from -bound to bound for current variable
        let bound_int = bound as i64;
        for val in -bound_int..=bound_int {
            current[var_index] = Integer::from(val);
            self.count_representations_recursive(target, bound, var_index + 1, current, count);
        }
    }

    /// Compute the discriminant of the quadratic form
    ///
    /// The discriminant is the determinant of the matrix representation.
    /// For binary forms ax² + bxy + cy², the discriminant is b² - 4ac.
    pub fn discriminant(&self) -> Integer {
        // For now, implement determinant for small dimensions
        match self.dimension {
            1 => self.matrix[0][0].clone(),
            2 => {
                // det = ad - bc for [[a,b],[c,d]]
                let a = &self.matrix[0][0];
                let b = &self.matrix[0][1];
                let c = &self.matrix[1][0];
                let d = &self.matrix[1][1];
                a.clone() * d.clone() - b.clone() * c.clone()
            }
            3 => {
                // Use rule of Sarrus for 3x3
                let m = &self.matrix;
                let pos = m[0][0].clone() * m[1][1].clone() * m[2][2].clone()
                    + m[0][1].clone() * m[1][2].clone() * m[2][0].clone()
                    + m[0][2].clone() * m[1][0].clone() * m[2][1].clone();
                let neg = m[0][2].clone() * m[1][1].clone() * m[2][0].clone()
                    + m[0][1].clone() * m[1][0].clone() * m[2][2].clone()
                    + m[0][0].clone() * m[1][2].clone() * m[2][1].clone();
                pos - neg
            }
            _ => {
                // For larger dimensions, would need full determinant algorithm
                // For now, return zero as placeholder
                Integer::zero()
            }
        }
    }
}

impl std::fmt::Display for QuadraticForm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuadraticForm({}×{}, disc={})", self.dimension, self.dimension, self.discriminant())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_of_two_squares() {
        // Q(x, y) = x² + y²
        let matrix = vec![
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let form = QuadraticForm::new(matrix).unwrap();

        // 5 = 1² + 2² = 2² + 1²
        assert!(form.represents(&Integer::from(5), 10));

        // 3² + 4² = 25
        let result = form.evaluate(&[Integer::from(3), Integer::from(4)]).unwrap();
        assert_eq!(result, Integer::from(25));

        // 3 cannot be written as sum of two squares
        assert!(!form.represents(&Integer::from(3), 10));
    }

    #[test]
    fn test_diagonal_form() {
        // Q(x, y, z) = x² + 2y² + 3z²
        let form = QuadraticForm::diagonal(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
        ])
        .unwrap();

        assert_eq!(form.dimension(), 3);

        // 1² + 2(1²) + 3(1²) = 1 + 2 + 3 = 6
        let result = form
            .evaluate(&[Integer::from(1), Integer::from(1), Integer::from(1)])
            .unwrap();
        assert_eq!(result, Integer::from(6));
    }

    #[test]
    fn test_binary_form_with_cross_term() {
        // Q(x, y) = x² + xy + y²
        let matrix = vec![
            vec![Integer::from(1), Integer::from(1)],
            vec![Integer::from(1), Integer::from(1)],
        ];

        // This should fail - not symmetric with diagonal form
        // The correct symmetric matrix for x² + xy + y² is:
        // [[1, 1/2], [1/2, 1]] but we use integers, so we'd scale to:
        // [[2, 1], [1, 2]] and divide result by 2

        // Let's use the proper integer form: 2x² + 2xy + 2y² = 2(x² + xy + y²)
        let matrix = vec![
            vec![Integer::from(2), Integer::from(1)],
            vec![Integer::from(1), Integer::from(2)],
        ];
        let form = QuadraticForm::new(matrix).unwrap();

        // Q(1, 0) = 2(1² + 0 + 0) = 2
        let result = form.evaluate(&[Integer::from(1), Integer::from(0)]).unwrap();
        assert_eq!(result, Integer::from(2));

        // Q(1, 1) = 2(1² + 1·1 + 1²) = 2(3) = 6
        let result = form.evaluate(&[Integer::from(1), Integer::from(1)]).unwrap();
        assert_eq!(result, Integer::from(6));
    }

    #[test]
    fn test_find_representation() {
        // Q(x, y) = x² + y²
        let form = QuadraticForm::diagonal(vec![Integer::from(1), Integer::from(1)]).unwrap();

        // Find representation of 5
        let repr = form.find_representation(&Integer::from(5), 5);
        assert!(repr.is_some());
        let r = repr.unwrap();
        let value = form.evaluate(&r).unwrap();
        assert_eq!(value, Integer::from(5));
    }

    #[test]
    fn test_count_representations() {
        // Q(x, y) = x² + y²
        let form = QuadraticForm::diagonal(vec![Integer::from(1), Integer::from(1)]).unwrap();

        // Count representations of 5 = 1² + 2² (with signs and order)
        // (1,2), (-1,2), (1,-2), (-1,-2), (2,1), (-2,1), (2,-1), (-2,-1)
        let count = form.count_representations(&Integer::from(5), 5);
        assert_eq!(count, 8);

        // 0 has only one representation: (0, 0)
        let count = form.count_representations(&Integer::from(0), 5);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_discriminant() {
        // Q(x, y) = x² + y²
        let form = QuadraticForm::diagonal(vec![Integer::from(1), Integer::from(1)]).unwrap();
        assert_eq!(form.discriminant(), Integer::from(1));

        // Q(x, y) = 2x² + xy + 3y²
        let matrix = vec![
            vec![Integer::from(4), Integer::from(1)],
            vec![Integer::from(1), Integer::from(6)],
        ];
        let form = QuadraticForm::new(matrix).unwrap();
        // det = 4*6 - 1*1 = 24 - 1 = 23
        assert_eq!(form.discriminant(), Integer::from(23));
    }

    #[test]
    fn test_invalid_matrix() {
        // Non-square matrix should fail
        let matrix = vec![
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0)],
        ];
        assert!(QuadraticForm::new(matrix).is_err());

        // Non-symmetric matrix should fail
        let matrix = vec![
            vec![Integer::from(1), Integer::from(2)],
            vec![Integer::from(3), Integer::from(4)],
        ];
        assert!(QuadraticForm::new(matrix).is_err());
    }
}
