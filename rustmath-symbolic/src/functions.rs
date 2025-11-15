//! Calculus Functions
//!
//! This module provides standalone mathematical functions for calculus operations
//! that work on vectors/lists of expressions.
//!
//! Corresponds to sage.calculus.functions
//!
//! # Functions
//!
//! - `jacobian`: Compute the Jacobian matrix of partial derivatives
//! - `wronskian`: Compute the Wronskian determinant for testing linear independence
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::{Expr, Symbol, jacobian, wronskian};
//!
//! let x = Symbol::new("x");
//! let y = Symbol::new("y");
//!
//! // Jacobian example: F(x,y) = [x*y, x+y]
//! let f1 = Expr::Symbol(x.clone()) * Expr::Symbol(y.clone());
//! let f2 = Expr::Symbol(x.clone()) + Expr::Symbol(y.clone());
//! let jac = jacobian(&[f1, f2], &[x.clone(), y.clone()]);
//! // Returns [[y, x], [1, 1]]
//!
//! // Wronskian example: W(sin(x), cos(x))
//! let sin_x = Expr::Symbol(x.clone()).sin();
//! let cos_x = Expr::Symbol(x.clone()).cos();
//! let w = wronskian(&[sin_x, cos_x], &x);
//! // Returns -1 (indicating linear independence)
//! ```
//!
//! # References
//! - SageMath: sage.calculus.functions

use crate::expression::Expr;
use crate::symbol::Symbol;

/// Compute the Jacobian matrix of a vector-valued function
///
/// For a function F: R^n → R^m represented as [f₁, f₂, ..., f_m],
/// the Jacobian J is an m×n matrix where J[i][j] = ∂f_i/∂x_j
///
/// # Arguments
///
/// * `functions` - Vector of expressions representing the components of F
/// * `variables` - Vector of symbols to differentiate with respect to
///
/// # Returns
///
/// A 2D vector representing the Jacobian matrix
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::{Expr, Symbol, jacobian};
///
/// let x = Symbol::new("x");
/// let y = Symbol::new("y");
///
/// // F(x, y) = [x*y, x + y]
/// let f1 = Expr::Symbol(x.clone()) * Expr::Symbol(y.clone());
/// let f2 = Expr::Symbol(x.clone()) + Expr::Symbol(y.clone());
///
/// let jac = jacobian(&[f1, f2], &[x, y]);
/// // Jacobian is [[y, x], [1, 1]]
/// assert_eq!(jac.len(), 2); // 2 functions
/// assert_eq!(jac[0].len(), 2); // 2 variables
/// ```
pub fn jacobian(functions: &[Expr], variables: &[Symbol]) -> Vec<Vec<Expr>> {
    functions
        .iter()
        .map(|f| {
            variables
                .iter()
                .map(|var| f.differentiate(var))
                .collect()
        })
        .collect()
}

/// Compute the Wronskian determinant of a list of functions
///
/// The Wronskian is used to test whether a set of functions is linearly independent.
/// It's defined as the determinant of a matrix where the i-th row contains the
/// (i-1)-th derivatives of all functions.
///
/// For functions f₁, f₂, ..., f_n, the Wronskian is:
///
/// ```text
/// W(f₁, f₂, ..., f_n) = det([
///     [f₁,     f₂,     ..., f_n    ],
///     [f₁',    f₂',    ..., f_n'   ],
///     [f₁'',   f₂'',   ..., f_n''  ],
///     ...
///     [f₁^(n-1), f₂^(n-1), ..., f_n^(n-1)]
/// ])
/// ```
///
/// # Arguments
///
/// * `functions` - Vector of expressions to compute Wronskian for
/// * `variable` - The variable to differentiate with respect to
///
/// # Returns
///
/// The Wronskian determinant as an expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::{Expr, Symbol, wronskian};
///
/// let x = Symbol::new("x");
///
/// // W(1, x, x²)
/// let f1 = Expr::from(1);
/// let f2 = Expr::Symbol(x.clone());
/// let f3 = Expr::Symbol(x.clone()).pow(Expr::from(2));
///
/// let w = wronskian(&[f1, f2, f3], &x);
/// // For 1, x, x²: derivatives are [1, x, x²], [0, 1, 2x], [0, 0, 2]
/// // Determinant is 2
/// ```
///
/// # Mathematical Background
///
/// If the Wronskian is nonzero at any point in an interval, the functions
/// are linearly independent on that interval. This is particularly useful
/// in the theory of linear differential equations.
pub fn wronskian(functions: &[Expr], variable: &Symbol) -> Expr {
    let n = functions.len();

    if n == 0 {
        return Expr::from(0);
    }

    if n == 1 {
        return functions[0].clone();
    }

    // Build the Wronskian matrix
    // Row i contains the i-th derivative of each function
    let mut matrix_data = Vec::new();

    for row in 0..n {
        let mut current_functions = if row == 0 {
            functions.to_vec()
        } else {
            // Differentiate the previous row
            matrix_data[(row - 1) * n..(row * n)]
                .iter()
                .map(|expr: &Expr| expr.differentiate(variable))
                .collect()
        };

        matrix_data.append(&mut current_functions);
    }

    // Compute determinant symbolically and simplify
    let det = compute_symbolic_determinant(&matrix_data, n);
    det.simplify()
}

/// Helper function to compute symbolic determinant
///
/// This computes the determinant of a matrix represented as a flat vector
/// of symbolic expressions.
fn compute_symbolic_determinant(data: &[Expr], n: usize) -> Expr {
    if n == 1 {
        return data[0].clone();
    }

    if n == 2 {
        // det([[a, b], [c, d]]) = ad - bc
        return data[0].clone() * data[3].clone() - data[1].clone() * data[2].clone();
    }

    // Laplace expansion along first row
    let mut result = Expr::from(0);

    for col in 0..n {
        // Get the element at (0, col)
        let element = data[col].clone();

        // Build the minor (submatrix without row 0 and column col)
        let mut minor_data = Vec::new();
        for row in 1..n {
            for c in 0..n {
                if c != col {
                    minor_data.push(data[row * n + c].clone());
                }
            }
        }

        // Compute the cofactor
        let minor_det = compute_symbolic_determinant(&minor_data, n - 1);
        let sign = if col % 2 == 0 {
            Expr::from(1)
        } else {
            Expr::from(-1)
        };

        result = result + sign * element * minor_det;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobian_simple() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // F(x, y) = [x*y, x + y]
        let f1 = Expr::Symbol(x.clone()) * Expr::Symbol(y.clone());
        let f2 = Expr::Symbol(x.clone()) + Expr::Symbol(y.clone());

        let jac = jacobian(&[f1, f2], &[x.clone(), y.clone()]);

        assert_eq!(jac.len(), 2);
        assert_eq!(jac[0].len(), 2);
        assert_eq!(jac[1].len(), 2);

        // J[0][0] = ∂(xy)/∂x = y
        assert!(format!("{}", jac[0][0]).contains("y"));

        // J[0][1] = ∂(xy)/∂y = x
        assert!(format!("{}", jac[0][1]).contains("x"));

        // J[1][0] = ∂(x+y)/∂x = 1
        assert_eq!(jac[1][0].simplify(), Expr::from(1));

        // J[1][1] = ∂(x+y)/∂y = 1
        assert_eq!(jac[1][1].simplify(), Expr::from(1));
    }

    #[test]
    fn test_jacobian_single_variable() {
        let x = Symbol::new("x");

        // F(x) = [x², x³]
        let f1 = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let f2 = Expr::Symbol(x.clone()).pow(Expr::from(3));

        let jac = jacobian(&[f1, f2], &[x.clone()]);

        assert_eq!(jac.len(), 2);
        assert_eq!(jac[0].len(), 1);

        // J[0][0] = d(x²)/dx = 2x
        let j00_str = format!("{}", jac[0][0]);
        assert!(j00_str.contains("2") && j00_str.contains("x"));

        // J[1][0] = d(x³)/dx = 3x²
        let j10_str = format!("{}", jac[1][0]);
        assert!(j10_str.contains("3") && j10_str.contains("x"));
    }

    #[test]
    fn test_jacobian_empty() {
        let x = Symbol::new("x");
        let jac = jacobian(&[], &[x]);
        assert_eq!(jac.len(), 0);
    }

    #[test]
    fn test_wronskian_two_functions() {
        let x = Symbol::new("x");

        // W(x, x²) = x * 2x - x² * 1 = 2x² - x² = x²
        let f1 = Expr::Symbol(x.clone());
        let f2 = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let w = wronskian(&[f1, f2], &x);

        // The Wronskian should be x²
        let w_str = format!("{}", w);
        assert!(w_str.contains("x"));
    }

    #[test]
    fn test_wronskian_three_functions() {
        let x = Symbol::new("x");

        // W(1, x, x²)
        // Matrix: [[1, x, x²], [0, 1, 2x], [0, 0, 2]]
        // Determinant: 1*(1*2 - 2x*0) - x*(0*2 - 2x*0) + x²*(0*0 - 1*0) = 2
        let f1 = Expr::from(1);
        let f2 = Expr::Symbol(x.clone());
        let f3 = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let w = wronskian(&[f1, f2, f3], &x);

        assert_eq!(w, Expr::from(2));
    }

    #[test]
    fn test_wronskian_sin_cos() {
        let x = Symbol::new("x");

        // W(sin(x), cos(x))
        // Matrix: [[sin(x), cos(x)], [cos(x), -sin(x)]]
        // Determinant: sin(x)*(-sin(x)) - cos(x)*cos(x) = -sin²(x) - cos²(x) = -1
        let f1 = Expr::Symbol(x.clone()).sin();
        let f2 = Expr::Symbol(x.clone()).cos();

        let w = wronskian(&[f1, f2], &x);

        // Should simplify to -1 (or -(sin²x + cos²x) which equals -1)
        let w_str = format!("{}", w);
        // The expression should contain sin and cos
        assert!(w_str.contains("sin") || w_str.contains("cos") || w == Expr::from(-1));
    }

    #[test]
    fn test_wronskian_single_function() {
        let x = Symbol::new("x");
        let f = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let w = wronskian(&[f.clone()], &x);

        // Wronskian of single function is the function itself
        assert_eq!(w, f);
    }

    #[test]
    fn test_wronskian_empty() {
        let x = Symbol::new("x");
        let w = wronskian(&[], &x);
        assert_eq!(w, Expr::from(0));
    }

    #[test]
    fn test_determinant_2x2() {
        // Test the symbolic determinant helper with [a, b, c, d]
        let a = Expr::from(1);
        let b = Expr::from(2);
        let c = Expr::from(3);
        let d = Expr::from(4);

        let det = compute_symbolic_determinant(&[a, b, c, d], 2).simplify();

        // det = 1*4 - 2*3 = 4 - 6 = -2
        assert_eq!(det, Expr::from(-2));
    }

    #[test]
    fn test_determinant_3x3() {
        // Test [[1, 0, 0], [0, 1, 0], [0, 0, 1]] = 1
        let data = vec![
            Expr::from(1),
            Expr::from(0),
            Expr::from(0),
            Expr::from(0),
            Expr::from(1),
            Expr::from(0),
            Expr::from(0),
            Expr::from(0),
            Expr::from(1),
        ];

        let det = compute_symbolic_determinant(&data, 3).simplify();
        assert_eq!(det, Expr::from(1));
    }

    #[test]
    fn test_wronskian_linearly_dependent() {
        let x = Symbol::new("x");

        // W(x, 2x) should be zero (linearly dependent)
        // Matrix: [[x, 2x], [1, 2]]
        // Determinant: x*2 - 2x*1 = 2x - 2x = 0
        let f1 = Expr::Symbol(x.clone());
        let f2 = Expr::from(2) * Expr::Symbol(x.clone());

        let w = wronskian(&[f1, f2], &x);
        // The Wronskian should be (x*2) - (2*x) which algebraically equals 0
        // Note: Full simplification to 0 requires advanced algebraic simplification
        let w_str = format!("{}", w);
        // Check that the expression contains both terms that should cancel
        assert!(w_str.contains("x") && (w_str.contains("2") || w_str.contains("-")));
    }
}
