//! Taylor series expansion
//!
//! This module provides Taylor and Maclaurin series expansions for symbolic expressions.

use crate::differentiation::differentiate;
use crate::limits::substitute;
use rustmath_symbolic::Expr;

/// Compute the Taylor series expansion of an expression
///
/// # Arguments
///
/// * `expr` - The expression to expand
/// * `var` - The variable to expand around
/// * `point` - The point to expand around (use 0 for Maclaurin series)
/// * `order` - The order of the expansion (number of terms)
///
/// # Returns
///
/// The Taylor series expansion up to the specified order
///
/// # Examples
///
/// ```
/// use rustmath_calculus::taylor::taylor;
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// // Taylor series of exp(x) around 0: 1 + x + x^2/2! + x^3/3! + ...
/// let expr = x.clone().exp();
/// let series = taylor(&expr, "x", &Expr::from(0), 4);
/// // Result: 1 + x + x^2/2 + x^3/6 + x^4/24
/// ```
pub fn taylor(expr: &Expr, var: &str, point: &Expr, order: usize) -> Expr {
    let mut result = Expr::from(0);
    let mut current_derivative = expr.clone();
    let mut factorial = 1i64;

    for n in 0..=order {
        // Evaluate the n-th derivative at the point
        let deriv_at_point = substitute(&current_derivative, var, point);

        // Create the term: f^(n)(a) * (x - a)^n / n!
        let x_minus_a = if is_zero(point) {
            Expr::symbol(var)
        } else {
            Expr::symbol(var) - point.clone()
        };

        let term = if n == 0 {
            deriv_at_point
        } else {
            deriv_at_point * x_minus_a.pow(Expr::from(n as i64)) / Expr::from(factorial)
        };

        result = result + term;

        // Compute next derivative and factorial for next iteration
        if n < order {
            current_derivative = differentiate(&current_derivative, var);
            factorial *= (n + 1) as i64;
        }
    }

    result
}

/// Compute the Maclaurin series expansion (Taylor series around 0)
///
/// # Arguments
///
/// * `expr` - The expression to expand
/// * `var` - The variable to expand
/// * `order` - The order of the expansion
///
/// # Returns
///
/// The Maclaurin series expansion up to the specified order
///
/// # Examples
///
/// ```
/// use rustmath_calculus::taylor::maclaurin;
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// // Maclaurin series of sin(x): x - x^3/3! + x^5/5! - ...
/// let expr = x.clone().sin();
/// let series = maclaurin(&expr, "x", 5);
/// ```
pub fn maclaurin(expr: &Expr, var: &str, order: usize) -> Expr {
    taylor(expr, var, &Expr::from(0), order)
}

/// Helper function to check if an expression is zero
fn is_zero(expr: &Expr) -> bool {
    match expr {
        Expr::Integer(n) => *n == rustmath_integers::Integer::from(0),
        Expr::Rational(r) => *r.numerator() == rustmath_integers::Integer::from(0),
        _ => false,
    }
}

/// Compute the Laurent series expansion (allows negative powers)
///
/// # Arguments
///
/// * `expr` - The expression to expand
/// * `var` - The variable to expand around
/// * `point` - The point to expand around
/// * `min_order` - The minimum order (can be negative for poles)
/// * `max_order` - The maximum order
///
/// # Returns
///
/// The Laurent series expansion
///
/// # Examples
///
/// ```
/// use rustmath_calculus::taylor::laurent;
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// // Laurent series of 1/x around 0 includes negative powers
/// let expr = Expr::from(1) / x.clone();
/// // This is a simplified example - full implementation would handle poles
/// ```
pub fn laurent(
    expr: &Expr,
    var: &str,
    point: &Expr,
    min_order: i64,
    max_order: i64,
) -> Expr {
    // For now, if min_order >= 0, just use Taylor series
    if min_order >= 0 {
        return taylor(expr, var, point, max_order as usize);
    }

    // Full Laurent series would require residue computation
    // This is a placeholder for the general case
    taylor(expr, var, point, max_order as usize)
}

/// Compute the power series coefficients
///
/// # Arguments
///
/// * `expr` - The expression to expand
/// * `var` - The variable
/// * `point` - The expansion point
/// * `order` - The order of expansion
///
/// # Returns
///
/// A vector of coefficients [c0, c1, c2, ...] where the series is
/// c0 + c1*(x-a) + c2*(x-a)^2 + ...
///
/// # Examples
///
/// ```
/// use rustmath_calculus::taylor::series_coefficients;
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// let expr = x.clone().exp();
/// let coeffs = series_coefficients(&expr, "x", &Expr::from(0), 4);
/// // For exp(x): [1, 1, 1/2, 1/6, 1/24]
/// ```
pub fn series_coefficients(
    expr: &Expr,
    var: &str,
    point: &Expr,
    order: usize,
) -> Vec<Expr> {
    let mut coefficients = Vec::new();
    let mut current_derivative = expr.clone();
    let mut factorial = 1i64;

    for n in 0..=order {
        // Evaluate the n-th derivative at the point
        let deriv_at_point = substitute(&current_derivative, var, point);

        // The coefficient is f^(n)(a) / n!
        let coeff = if n == 0 {
            deriv_at_point
        } else {
            deriv_at_point / Expr::from(factorial)
        };

        coefficients.push(coeff);

        // Compute next derivative and factorial
        if n < order {
            current_derivative = differentiate(&current_derivative, var);
            factorial *= (n + 1) as i64;
        }
    }

    coefficients
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_taylor_constant() {
        // Taylor series of a constant is just the constant
        let c = Expr::from(42);
        let series = taylor(&c, "x", &Expr::from(0), 3);
        assert_eq!(series, Expr::from(42));
    }

    #[test]
    fn test_taylor_linear() {
        // Taylor series of x around 0 is just x
        let x = Expr::symbol("x");
        let _series = taylor(&x, "x", &Expr::from(0), 3);
        // Result should be 0 + x (after simplification)
        // The actual result might be: 0 + 1*x + 0*x^2 + 0*x^3
    }

    #[test]
    fn test_taylor_polynomial() {
        // Taylor series of x^2 around 0
        let x = Expr::symbol("x");
        let expr = x.clone() * x.clone();
        let _series = taylor(&expr, "x", &Expr::from(0), 3);
        // Result should be: 0 + 0*x + 1*x^2
    }

    #[test]
    fn test_maclaurin_identity() {
        let x = Expr::symbol("x");
        let expr = x.clone();

        let taylor_result = taylor(&expr, "x", &Expr::from(0), 2);
        let maclaurin_result = maclaurin(&expr, "x", 2);

        // Both should give the same result
        assert_eq!(taylor_result, maclaurin_result);
    }

    #[test]
    fn test_series_coefficients_constant() {
        let c = Expr::from(5);
        let coeffs = series_coefficients(&c, "x", &Expr::from(0), 3);

        assert_eq!(coeffs[0], Expr::from(5));
        assert_eq!(coeffs[1], Expr::from(0));
        assert_eq!(coeffs[2], Expr::from(0));
        assert_eq!(coeffs[3], Expr::from(0));
    }

    #[test]
    fn test_series_coefficients_linear() {
        let x = Expr::symbol("x");
        let coeffs = series_coefficients(&x, "x", &Expr::from(0), 3);

        // For f(x) = x: f(0) = 0, f'(0) = 1, f''(0) = 0, ...
        assert_eq!(coeffs[0], Expr::from(0));
        assert_eq!(coeffs[1], Expr::from(1));
        assert_eq!(coeffs[2], Expr::from(0));
        assert_eq!(coeffs[3], Expr::from(0));
    }

    #[test]
    fn test_laurent_reduces_to_taylor() {
        let x = Expr::symbol("x");
        let expr = x.clone() * x.clone();

        let taylor_result = taylor(&expr, "x", &Expr::from(0), 3);
        let laurent_result = laurent(&expr, "x", &Expr::from(0), 0, 3);

        assert_eq!(taylor_result, laurent_result);
    }

    #[test]
    fn test_taylor_different_point() {
        let x = Expr::symbol("x");
        // Taylor series of x around point 2
        let _series = taylor(&x, "x", &Expr::from(2), 2);
        // Should give: 2 + 1*(x-2) = x
    }
}
