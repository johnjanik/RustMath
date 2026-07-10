//! Limit computation for symbolic expressions
//!
//! This module provides functionality for computing limits of expressions
//! as a variable approaches a value.

use rustmath_symbolic::simplify::simplify;
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

    // Fold constants so that e.g. (3 + 5) becomes 8, and so that a form like
    // ((4 - 4) / (2 - 2)) becomes the literal 0/0 that the zero-division
    // check below can see. `simplify` never folds a division by zero, so
    // indeterminate forms survive simplification.
    let simplified = simplify(&substituted);

    // Direct substitution is invalid whenever it produced a division by
    // zero anywhere in the expression (0/0, 1/0, ...). The limit may still
    // exist; later strategies (L'Hôpital, ...) get their chance.
    if contains_division_by_zero(&simplified) {
        None
    } else {
        Some(simplified)
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

/// Check if an expression contains a division whose denominator is
/// literally zero (after simplification), e.g. `0/0` or `1/0`.
fn contains_division_by_zero(expr: &Expr) -> bool {
    match expr {
        Expr::Binary(BinaryOp::Div, _, den) if is_zero(den) => true,
        Expr::Binary(_, left, right) => {
            contains_division_by_zero(left) || contains_division_by_zero(right)
        }
        Expr::Unary(_, inner) => contains_division_by_zero(inner),
        Expr::Function(_, args) => args.iter().any(|a| contains_division_by_zero(a)),
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
            let num_at_point = simplify(&substitute(num, var, point));
            let den_at_point = simplify(&substitute(den, var, point));

            if is_zero(&num_at_point) && is_zero(&den_at_point) {
                // Apply L'Hôpital's rule: lim f/g = lim f'/g'
                use crate::differentiation::differentiate;
                let num_prime = differentiate(num, var);
                let den_prime = differentiate(den, var);

                let num_prime_at_point = simplify(&substitute(&num_prime, var, point));
                let den_prime_at_point = simplify(&substitute(&den_prime, var, point));

                if !is_zero(&den_prime_at_point) {
                    return Some(simplify(&Expr::Binary(
                        BinaryOp::Div,
                        std::sync::Arc::new(num_prime_at_point),
                        std::sync::Arc::new(den_prime_at_point),
                    )));
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
        assert_eq!(result, Some(Expr::from(4)));
    }

    #[test]
    fn test_lhopital() {
        let x = Expr::symbol("x");

        // lim(x→2) (x^2 - 4)/(x - 2) = 4 (0/0 form, L'Hôpital / factoring)
        let num = x.clone() * x.clone() - Expr::from(4);
        let den = x.clone() - Expr::from(2);
        let expr = num / den;
        let result = limit(&expr, "x", &Expr::from(2), LimitDirection::Both);
        assert_eq!(result, Some(Expr::from(4)));
    }

    #[test]
    fn test_division_by_zero_is_not_a_value() {
        let x = Expr::symbol("x");

        // lim(x→0) 1/x does not exist (two-sided); we must not return "1/0"
        let expr = Expr::from(1) / x.clone();
        let result = limit(&expr, "x", &Expr::from(0), LimitDirection::Both);
        assert_eq!(result, None);
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
