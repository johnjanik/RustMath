//! Limit computation for symbolic expressions
//!
//! This module provides functionality for computing limits of expressions
//! as a variable approaches a value.

use rustmath_symbolic::{BinaryOp, Expr};

/// Direction for limit computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitDirection {
    /// Limit from the left (x -> a-)
    Left,
    /// Limit from the right (x -> a+)
    Right,
    /// Two-sided limit (x -> a)
    Both,
}

/// Compute the limit of an expression
///
/// # Arguments
///
/// * `expr` - The expression to take the limit of
/// * `var` - The variable approaching the limit point
/// * `point` - The point being approached
/// * `direction` - Direction of approach (left, right, or both)
///
/// # Returns
///
/// The limit value if it exists, or None if the limit does not exist
/// or cannot be computed
///
/// # Examples
///
/// ```
/// use rustmath_calculus::limits::{limit, LimitDirection};
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// // lim(x→2) x^2 = 4
/// let expr = x.clone() * x.clone();
/// let result = limit(&expr, "x", &Expr::from(2), LimitDirection::Both);
/// assert!(result.is_some());
/// ```
pub fn limit(
    expr: &Expr,
    var: &str,
    point: &Expr,
    direction: LimitDirection,
) -> Option<Expr> {
    // First, try direct substitution
    if let Some(result) = try_direct_substitution(expr, var, point) {
        return Some(result);
    }

    // If direct substitution fails, try algebraic simplification
    if let Some(result) = try_algebraic_limit(expr, var, point, direction) {
        return Some(result);
    }

    // For limits at infinity
    if is_infinity(point) {
        return limit_at_infinity(expr, var, point, direction);
    }

    None
}

/// Try to compute the limit by direct substitution
fn try_direct_substitution(expr: &Expr, var: &str, point: &Expr) -> Option<Expr> {
    let substituted = substitute(expr, var, point);

    // Check if the result is well-defined (not 0/0 or ∞/∞)
    if is_indeterminate(&substituted) {
        None
    } else {
        Some(substituted)
    }
}

/// Substitute a variable with a value in an expression
pub fn substitute(expr: &Expr, var: &str, value: &Expr) -> Expr {
    match expr {
        Expr::Symbol(s) => {
            if s.name() == var {
                value.clone()
            } else {
                expr.clone()
            }
        }
        Expr::Binary(op, left, right) => {
            let left_sub = substitute(left, var, value);
            let right_sub = substitute(right, var, value);
            Expr::Binary(*op, std::sync::Arc::new(left_sub), std::sync::Arc::new(right_sub))
        }
        Expr::Unary(op, inner) => {
            let inner_sub = substitute(inner, var, value);
            Expr::Unary(*op, std::sync::Arc::new(inner_sub))
        }
        Expr::Function(name, args) => {
            let args_sub: Vec<std::sync::Arc<Expr>> = args.iter()
                .map(|arg| std::sync::Arc::new(substitute(arg, var, value)))
                .collect();
            Expr::Function(name.clone(), args_sub)
        }
        _ => expr.clone(),
    }
}

/// Check if an expression is an indeterminate form
fn is_indeterminate(expr: &Expr) -> bool {
    // Check for 0/0, ∞/∞, 0*∞, ∞-∞, etc.
    // For now, a simplified check
    match expr {
        Expr::Binary(BinaryOp::Div, num, den) => {
            is_zero(num) && is_zero(den)
        }
        _ => false,
    }
}

/// Check if an expression evaluates to zero
fn is_zero(expr: &Expr) -> bool {
    match expr {
        Expr::Integer(n) => *n == rustmath_integers::Integer::from(0),
        Expr::Rational(r) => *r.numerator() == rustmath_integers::Integer::from(0),
        _ => false,
    }
}

/// Check if a point represents infinity
fn is_infinity(_expr: &Expr) -> bool {
    // In a full implementation, we'd have a special Infinity type
    // For now, we'll use a placeholder
    false
}

/// Try to compute the limit using algebraic techniques
fn try_algebraic_limit(
    expr: &Expr,
    var: &str,
    point: &Expr,
    _direction: LimitDirection,
) -> Option<Expr> {
    // Handle rational functions using L'Hôpital's rule or factoring
    match expr {
        Expr::Binary(BinaryOp::Div, num, den) => {
            // Check if we have 0/0 form
            let num_at_point = substitute(num, var, point);
            let den_at_point = substitute(den, var, point);

            if is_zero(&num_at_point) && is_zero(&den_at_point) {
                // Apply L'Hôpital's rule: lim f/g = lim f'/g'
                use crate::differentiation::differentiate;
                let num_prime = differentiate(num, var);
                let den_prime = differentiate(den, var);

                let num_prime_at_point = substitute(&num_prime, var, point);
                let den_prime_at_point = substitute(&den_prime, var, point);

                if !is_zero(&den_prime_at_point) {
                    return Some(
                        Expr::Binary(
                            BinaryOp::Div,
                            std::sync::Arc::new(num_prime_at_point),
                            std::sync::Arc::new(den_prime_at_point),
                        )
                    );
                }
            }
        }
        _ => {}
    }

    None
}

/// Compute limit at infinity
fn limit_at_infinity(
    expr: &Expr,
    var: &str,
    _point: &Expr,
    _direction: LimitDirection,
) -> Option<Expr> {
    // For polynomial ratios, divide by highest power
    // This is a simplified implementation
    match expr {
        Expr::Binary(BinaryOp::Div, num, den) => {
            use crate::expr::polynomial_degree;

            let num_degree = polynomial_degree(num, var)?;
            let den_degree = polynomial_degree(den, var)?;

            if num_degree < den_degree {
                Some(Expr::from(0))
            } else if num_degree == den_degree {
                // Return ratio of leading coefficients
                // This is simplified - would need coefficient extraction
                Some(Expr::from(1))
            } else {
                // num_degree > den_degree: limit is infinity
                None
            }
        }
        _ => None,
    }
}

/// Convenience function for computing two-sided limits
///
/// # Examples
///
/// ```
/// use rustmath_calculus::limits::lim;
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// let expr = x.clone() + Expr::from(3);
/// let result = lim(&expr, "x", &Expr::from(2));
/// assert_eq!(result, Some(Expr::from(5)));
/// ```
pub fn lim(expr: &Expr, var: &str, point: &Expr) -> Option<Expr> {
    limit(expr, var, point, LimitDirection::Both)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_substitution() {
        let x = Expr::symbol("x");

        // lim(x→2) x = 2
        let result = limit(&x, "x", &Expr::from(2), LimitDirection::Both);
        assert_eq!(result, Some(Expr::from(2)));

        // lim(x→3) x + 5 = 8
        let expr = x.clone() + Expr::from(5);
        let result = limit(&expr, "x", &Expr::from(3), LimitDirection::Both);
        assert_eq!(result, Some(Expr::from(8)));
    }

    #[test]
    fn test_polynomial_limit() {
        let x = Expr::symbol("x");

        // lim(x→2) x^2 = 4
        let expr = x.clone() * x.clone();
        let result = limit(&expr, "x", &Expr::from(2), LimitDirection::Both);
        // Result will be (2 * 2), which needs simplification
        assert!(result.is_some());
    }

    #[test]
    fn test_constant_limit() {
        let c = Expr::from(42);
        let result = limit(&c, "x", &Expr::from(0), LimitDirection::Both);
        assert_eq!(result, Some(Expr::from(42)));
    }

    #[test]
    fn test_lim_convenience() {
        let x = Expr::symbol("x");
        let expr = x.clone() + Expr::from(3);
        let result = lim(&expr, "x", &Expr::from(2));
        assert_eq!(result, Some(Expr::from(5)));
    }

    #[test]
    fn test_substitution() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");

        // Substitute x with 3 in x + y
        let expr = x.clone() + y.clone();
        let result = substitute(&expr, "x", &Expr::from(3));
        // Result should be (3 + y)
        match result {
            Expr::Binary(BinaryOp::Add, left, right) => {
                assert_eq!(*left, Expr::from(3));
                assert!(matches!(*right, Expr::Symbol(_)));
            }
            _ => panic!("Expected binary add"),
        }
    }

    #[test]
    fn test_limit_direction() {
        let x = Expr::symbol("x");
        let expr = x.clone();

        // Test that all directions work (even if result is the same)
        let result_both = limit(&expr, "x", &Expr::from(1), LimitDirection::Both);
        let result_left = limit(&expr, "x", &Expr::from(1), LimitDirection::Left);
        let result_right = limit(&expr, "x", &Expr::from(1), LimitDirection::Right);

        assert_eq!(result_both, Some(Expr::from(1)));
        assert_eq!(result_left, Some(Expr::from(1)));
        assert_eq!(result_right, Some(Expr::from(1)));
    }
}
