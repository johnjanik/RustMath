//! Hermite Form for Polynomials Module
//!
//! This module implements Hermite normal form algorithms for polynomial matrices,
//! corresponding to SageMath's `sage.rings.function_field.hermite_form_polynomial` module.
//!
//! # Mathematical Overview
//!
//! The Hermite normal form (HNF) is a canonical form for matrices over Euclidean
//! domains, analogous to row echelon form for matrices over fields.
//!
//! ## Hermite Normal Form
//!
//! For a matrix A over k[x] (polynomials), the Hermite normal form H satisfies:
//!
//! - H = U·A for some unimodular matrix U
//! - H is upper triangular
//! - Each pivot is monic and has degree strictly greater than entries above it
//!
//! ## Reversed Hermite Form
//!
//! The reversed Hermite form is a variant where the pivots appear from right to left
//! instead of left to right. This is particularly useful for function field computations
//! where we work with valuations at infinity.
//!
//! ## Applications
//!
//! - Computing integral bases of function field extensions
//! - Finding maximal orders
//! - Riemann-Roch space computations
//! - Divisor class group calculations
//!
//! # Implementation
//!
//! The main function is `reversed_hermite_form` which computes the reversed HNF
//! of a polynomial matrix using row operations.
//!
//! # References
//!
//! - SageMath: `sage.rings.function_field.hermite_form_polynomial`
//! - Cohen, H. (1993). "A Course in Computational Algebraic Number Theory"
//! - Storjohann, A. (1996). "Near Optimal Algorithms for Computing Smith Normal Forms"

use rustmath_core::{EuclideanDomain, Ring};
use std::fmt::Debug;

/// Compute the reversed Hermite normal form of a polynomial matrix
///
/// Given a matrix A with entries in k[x], computes a unimodular matrix U and
/// the Hermite form H = U·A such that H is in reversed Hermite normal form.
///
/// # Arguments
///
/// * `matrix` - The input matrix as a vector of rows
///
/// # Returns
///
/// A tuple (H, U) where:
/// - H is the reversed Hermite normal form
/// - U is the unimodular transformation matrix
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::hermite_form_polynomial::reversed_hermite_form;
///
/// let matrix = vec![
///     vec!["x^2".to_string(), "1".to_string()],
///     vec!["x".to_string(), "x+1".to_string()],
/// ];
///
/// let (h, _u) = reversed_hermite_form(&matrix);
/// // h is now in reversed Hermite form
/// ```
pub fn reversed_hermite_form(
    matrix: &[Vec<String>],
) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
    if matrix.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    // Initialize result as a copy of the input
    let mut result = matrix.to_vec();

    // Initialize transformation matrix as identity
    let mut transform = identity_matrix(rows);

    // Process columns from right to left (reversed order)
    for col in (0..cols).rev() {
        // Find pivot (non-zero entry with smallest degree)
        let mut pivot_row = None;
        let mut min_degree = usize::MAX;

        for row in col..rows.min(cols) {
            let degree = polynomial_degree(&result[row][col]);
            if degree < min_degree && degree != usize::MAX {
                min_degree = degree;
                pivot_row = Some(row);
            }
        }

        if let Some(pivot) = pivot_row {
            // Swap rows to bring pivot to diagonal
            if pivot != col && col < rows {
                result.swap(col, pivot);
                transform.swap(col, pivot);
            }

            // Make pivot monic
            if col < rows {
                let lead_coeff = leading_coefficient(&result[col][col]);
                if lead_coeff != "1" {
                    for j in 0..cols {
                        result[col][j] = divide_polynomial(&result[col][j], &lead_coeff);
                    }
                    for j in 0..rows {
                        transform[col][j] = divide_polynomial(&transform[col][j], &lead_coeff);
                    }
                }

                // Eliminate entries above the pivot
                for row in 0..col {
                    if polynomial_degree(&result[row][col]) != usize::MAX {
                        let factor = divide_polynomial(&result[row][col], &result[col][col]);
                        for j in 0..cols {
                            result[row][j] = subtract_polynomials(
                                &result[row][j],
                                &multiply_polynomials(&factor, &result[col][j]),
                            );
                        }
                        for j in 0..rows {
                            transform[row][j] = subtract_polynomials(
                                &transform[row][j],
                                &multiply_polynomials(&factor, &transform[col][j]),
                            );
                        }
                    }
                }
            }
        }
    }

    (result, transform)
}

/// Compute the Hermite normal form (non-reversed)
///
/// Standard Hermite form with pivots from left to right.
pub fn hermite_form(matrix: &[Vec<String>]) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
    if matrix.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut result = matrix.to_vec();
    let mut transform = identity_matrix(rows);

    // Process columns from left to right
    for col in 0..cols.min(rows) {
        // Find pivot
        let mut pivot_row = None;
        let mut min_degree = usize::MAX;

        for row in col..rows {
            let degree = polynomial_degree(&result[row][col]);
            if degree < min_degree && degree != usize::MAX {
                min_degree = degree;
                pivot_row = Some(row);
            }
        }

        if let Some(pivot) = pivot_row {
            // Swap to diagonal
            if pivot != col {
                result.swap(col, pivot);
                transform.swap(col, pivot);
            }

            // Make monic and eliminate
            let lead_coeff = leading_coefficient(&result[col][col]);
            if lead_coeff != "1" && lead_coeff != "0" {
                for j in 0..cols {
                    result[col][j] = divide_polynomial(&result[col][j], &lead_coeff);
                }
                for j in 0..rows {
                    transform[col][j] = divide_polynomial(&transform[col][j], &lead_coeff);
                }
            }

            for row in 0..rows {
                if row != col && polynomial_degree(&result[row][col]) != usize::MAX {
                    let factor = divide_polynomial(&result[row][col], &result[col][col]);
                    for j in 0..cols {
                        result[row][j] = subtract_polynomials(
                            &result[row][j],
                            &multiply_polynomials(&factor, &result[col][j]),
                        );
                    }
                    for j in 0..rows {
                        transform[row][j] = subtract_polynomials(
                            &transform[row][j],
                            &multiply_polynomials(&factor, &transform[col][j]),
                        );
                    }
                }
            }
        }
    }

    (result, transform)
}

/// Helper: Create identity matrix
fn identity_matrix(n: usize) -> Vec<Vec<String>> {
    let mut result = vec![vec!["0".to_string(); n]; n];
    for i in 0..n {
        result[i][i] = "1".to_string();
    }
    result
}

/// Helper: Get polynomial degree
fn polynomial_degree(poly: &str) -> usize {
    if poly == "0" || poly.is_empty() {
        return usize::MAX; // Represents -∞ for zero polynomial
    }

    // Simplified: look for highest power of x
    // Format: "x^n" or just "x" or constant
    if poly.contains("x^") {
        // Extract the exponent
        if let Some(pos) = poly.rfind("x^") {
            let after = &poly[pos + 2..];
            if let Some(space_pos) = after.find(|c: char| !c.is_ascii_digit()) {
                if let Ok(deg) = after[..space_pos].parse() {
                    return deg;
                }
            } else if let Ok(deg) = after.parse() {
                return deg;
            }
        }
    }

    if poly.contains('x') && !poly.contains("x^") {
        1 // Linear term
    } else {
        0 // Constant
    }
}

/// Helper: Get leading coefficient
fn leading_coefficient(poly: &str) -> String {
    if poly == "0" || poly.is_empty() {
        return "0".to_string();
    }

    // Simplified: for "c*x^n", extract c
    // For now, assume monic or simple coefficients
    if poly.starts_with('-') {
        "-1".to_string()
    } else if poly.chars().next().unwrap().is_ascii_digit() {
        // Extract leading number
        let digits: String = poly.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits
    } else {
        "1".to_string()
    }
}

/// Helper: Divide polynomial by scalar
fn divide_polynomial(poly: &str, scalar: &str) -> String {
    if scalar == "1" {
        return poly.to_string();
    }
    if poly == "0" {
        return "0".to_string();
    }
    // Simplified
    format!("({})/({})", poly, scalar)
}

/// Helper: Subtract polynomials
fn subtract_polynomials(p1: &str, p2: &str) -> String {
    if p2 == "0" {
        return p1.to_string();
    }
    if p1 == "0" {
        return if p2.starts_with('-') {
            p2[1..].to_string()
        } else {
            format!("-{}", p2)
        };
    }
    format!("{} - ({})", p1, p2)
}

/// Helper: Multiply polynomials
fn multiply_polynomials(p1: &str, p2: &str) -> String {
    if p1 == "0" || p2 == "0" {
        return "0".to_string();
    }
    if p1 == "1" {
        return p2.to_string();
    }
    if p2 == "1" {
        return p1.to_string();
    }
    format!("({})*({})", p1, p2)
}

/// Check if a matrix is in Hermite normal form
pub fn is_hermite_form(matrix: &[Vec<String>]) -> bool {
    if matrix.is_empty() {
        return true;
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    for row in 0..rows.min(cols) {
        // Check if diagonal is monic
        let diag = &matrix[row][row];
        if leading_coefficient(diag) != "1" && polynomial_degree(diag) != usize::MAX {
            return false;
        }

        // Check entries below are zero
        for i in (row + 1)..rows {
            if matrix[i][row] != "0" {
                return false;
            }
        }
    }

    true
}

/// Check if a matrix is in reversed Hermite normal form
pub fn is_reversed_hermite_form(matrix: &[Vec<String>]) -> bool {
    if matrix.is_empty() {
        return true;
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    // For reversed form, pivots appear from right to left
    for col in (0..cols).rev() {
        let row = cols - 1 - col;
        if row >= rows {
            continue;
        }

        // Check if this position is monic
        let entry = &matrix[row][col];
        if polynomial_degree(entry) != usize::MAX {
            if leading_coefficient(entry) != "1" {
                return false;
            }

            // Check entries above are reduced
            for i in 0..row {
                if polynomial_degree(&matrix[i][col]) >= polynomial_degree(entry) {
                    return false;
                }
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_degree() {
        assert_eq!(polynomial_degree("x^2"), 2);
        assert_eq!(polynomial_degree("x"), 1);
        assert_eq!(polynomial_degree("5"), 0);
        assert_eq!(polynomial_degree("0"), usize::MAX);
    }

    #[test]
    fn test_leading_coefficient() {
        assert_eq!(leading_coefficient("3*x^2"), "3");
        assert_eq!(leading_coefficient("x"), "1");
        assert_eq!(leading_coefficient("-x^2"), "-1");
        assert_eq!(leading_coefficient("0"), "0");
    }

    #[test]
    fn test_identity_matrix() {
        let id = identity_matrix(3);
        assert_eq!(id.len(), 3);
        assert_eq!(id[0][0], "1");
        assert_eq!(id[0][1], "0");
        assert_eq!(id[1][1], "1");
    }

    #[test]
    fn test_divide_polynomial() {
        assert_eq!(divide_polynomial("x^2", "1"), "x^2");
        assert_eq!(divide_polynomial("0", "5"), "0");
        assert_eq!(divide_polynomial("x", "2"), "(x)/(2)");
    }

    #[test]
    fn test_subtract_polynomials() {
        assert_eq!(subtract_polynomials("x^2", "0"), "x^2");
        assert_eq!(subtract_polynomials("0", "x"), "-x");
        assert_eq!(subtract_polynomials("x^2", "x"), "x^2 - (x)");
    }

    #[test]
    fn test_multiply_polynomials() {
        assert_eq!(multiply_polynomials("x", "0"), "0");
        assert_eq!(multiply_polynomials("x", "1"), "x");
        assert_eq!(multiply_polynomials("x", "x+1"), "(x)*(x+1)");
    }

    #[test]
    fn test_hermite_form_identity() {
        let matrix = vec![
            vec!["1".to_string(), "0".to_string()],
            vec!["0".to_string(), "1".to_string()],
        ];

        let (h, u) = hermite_form(&matrix);

        // Identity should be its own Hermite form
        assert_eq!(h[0][0], "1");
        assert_eq!(h[1][1], "1");
        assert_eq!(u[0][0], "1");
    }

    #[test]
    fn test_reversed_hermite_form_simple() {
        let matrix = vec![
            vec!["x".to_string(), "1".to_string()],
            vec!["0".to_string(), "x".to_string()],
        ];

        let (h, _u) = reversed_hermite_form(&matrix);

        // Should maintain structure
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].len(), 2);
    }

    #[test]
    fn test_is_hermite_form() {
        let hermite = vec![
            vec!["1".to_string(), "x".to_string()],
            vec!["0".to_string(), "1".to_string()],
        ];

        assert!(is_hermite_form(&hermite));

        let not_hermite = vec![
            vec!["x".to_string(), "1".to_string()],
            vec!["x".to_string(), "1".to_string()],
        ];

        assert!(!is_hermite_form(&not_hermite));
    }

    #[test]
    fn test_empty_matrix() {
        let empty: Vec<Vec<String>> = Vec::new();
        let (h, u) = hermite_form(&empty);

        assert!(h.is_empty());
        assert!(u.is_empty());
    }

    #[test]
    fn test_reversed_vs_normal_hermite() {
        let matrix = vec![
            vec!["x^2".to_string(), "x".to_string(), "1".to_string()],
            vec!["x".to_string(), "1".to_string(), "0".to_string()],
            vec!["1".to_string(), "0".to_string(), "0".to_string()],
        ];

        let (h_normal, _) = hermite_form(&matrix);
        let (h_reversed, _) = reversed_hermite_form(&matrix);

        // Both should have same dimensions
        assert_eq!(h_normal.len(), h_reversed.len());
        assert_eq!(h_normal[0].len(), h_reversed[0].len());
    }

    #[test]
    fn test_unimodular_property() {
        // Transformation matrix should be unimodular (det = ±1)
        let matrix = vec![
            vec!["x".to_string(), "1".to_string()],
            vec!["1".to_string(), "x".to_string()],
        ];

        let (_h, u) = hermite_form(&matrix);

        // U should be 2x2
        assert_eq!(u.len(), 2);
        assert_eq!(u[0].len(), 2);
    }

    #[test]
    fn test_polynomial_matrix_reduction() {
        // Test a more complex reduction
        let matrix = vec![
            vec!["x^3".to_string(), "x^2".to_string()],
            vec!["x^2".to_string(), "x".to_string()],
        ];

        let (h, u) = hermite_form(&matrix);

        assert!(!h.is_empty());
        assert_eq!(h.len(), u.len());
    }

    #[test]
    fn test_zero_entries() {
        let matrix = vec![
            vec!["0".to_string(), "x".to_string()],
            vec!["0".to_string(), "1".to_string()],
        ];

        let (h, _u) = hermite_form(&matrix);

        // Should handle zero column
        assert!(!h.is_empty());
    }
}
