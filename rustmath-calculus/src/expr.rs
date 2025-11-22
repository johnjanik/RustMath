//! Symbolic expression utilities
//!
//! This module provides utility functions for working with symbolic expressions,
//! including expression validation, variable extraction, and expression properties.

use rustmath_symbolic::Expr;
use std::collections::HashSet;

/// Convert a string representation to a symbolic expression
///
/// This is a placeholder for expression parsing functionality.
/// Full parsing would require a lexer and parser implementation.
///
/// # Arguments
///
/// * `s` - String representation of the expression
///
/// # Returns
///
/// Returns the parsed expression, or an error if parsing fails
///
/// # Examples
///
/// ```
/// use rustmath_calculus::expr::symbolic_expression;
/// // For now, this is a placeholder
/// // let expr = symbolic_expression("x^2 + 3*x + 2");
/// ```
pub fn symbolic_expression(s: &str) -> Result<Expr, String> {
    // Placeholder: In a full implementation, this would parse the string
    // and construct an Expr tree
    Err(format!("Expression parsing not yet implemented for: {}", s))
}

/// Extract all variables from a symbolic expression
///
/// # Arguments
///
/// * `expr` - The expression to analyze
///
/// # Returns
///
/// A HashSet containing the names of all variables in the expression
///
/// # Examples
///
/// ```
/// use rustmath_calculus::expr::variables;
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// let y = Expr::symbol("y");
/// let expr = x * y + Expr::from(5);
/// let vars = variables(&expr);
/// assert_eq!(vars.len(), 2);
/// assert!(vars.contains("x"));
/// assert!(vars.contains("y"));
/// ```
pub fn variables(expr: &Expr) -> HashSet<String> {
    let mut vars = HashSet::new();
    collect_variables(expr, &mut vars);
    vars
}

/// Recursively collect variables from an expression
fn collect_variables(expr: &Expr, vars: &mut HashSet<String>) {
    match expr {
        Expr::Symbol(s) => {
            vars.insert(s.name().to_string());
        }
        Expr::Binary(_, left, right) => {
            collect_variables(left, vars);
            collect_variables(right, vars);
        }
        Expr::Unary(_, inner) => {
            collect_variables(inner, vars);
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_variables(arg, vars);
            }
        }
        _ => {}
    }
}

/// Check if an expression is constant (contains no variables)
///
/// # Arguments
///
/// * `expr` - The expression to check
///
/// # Returns
///
/// True if the expression contains no variables, false otherwise
///
/// # Examples
///
/// ```
/// use rustmath_calculus::expr::is_constant;
/// use rustmath_symbolic::Expr;
///
/// assert!(is_constant(&Expr::from(42)));
/// assert!(is_constant(&(Expr::from(3) + Expr::from(7))));
/// assert!(!is_constant(&Expr::symbol("x")));
/// ```
pub fn is_constant(expr: &Expr) -> bool {
    variables(expr).is_empty()
}

/// Check if an expression is polynomial in the given variable
///
/// # Arguments
///
/// * `expr` - The expression to check
/// * `var` - The variable name
///
/// # Returns
///
/// True if the expression is polynomial in the variable, false otherwise
///
/// # Examples
///
/// ```
/// use rustmath_calculus::expr::is_polynomial;
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// let poly = x.clone() * x.clone() + Expr::from(3) * x.clone() + Expr::from(2);
/// assert!(is_polynomial(&poly, "x"));
///
/// let non_poly = x.clone().sin();
/// assert!(!is_polynomial(&non_poly, "x"));
/// ```
pub fn is_polynomial(expr: &Expr, var: &str) -> bool {
    match expr {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => true,
        Expr::Symbol(_s) => true, // A variable is a polynomial of degree 1
        Expr::Binary(op, left, right) => {
            use rustmath_symbolic::BinaryOp;
            match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                    is_polynomial(left, var) && is_polynomial(right, var)
                }
                BinaryOp::Pow => {
                    // f^n is polynomial if f is polynomial and n is a non-negative integer
                    if is_polynomial(left, var) {
                        if let Expr::Integer(n) = &**right {
                            use rustmath_integers::Integer;
                            n >= &Integer::from(0)
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                BinaryOp::Div => {
                    // f/g is polynomial only if g is constant with respect to var
                    is_polynomial(left, var) && !variables(right).contains(var)
                }
                BinaryOp::Mod => {
                    // Modulo is not a polynomial operation
                    false
                }
            }
        }
        Expr::Unary(op, _) => {
            use rustmath_symbolic::UnaryOp;
            match op {
                UnaryOp::Neg => is_polynomial(expr, var),
                _ => false, // sin, cos, exp, log, sqrt are not polynomial
            }
        }
        Expr::Function(_, _) => false, // Functions are not polynomial
    }
}

/// Get the degree of a polynomial expression in the given variable
///
/// # Arguments
///
/// * `expr` - The polynomial expression
/// * `var` - The variable name
///
/// # Returns
///
/// The degree of the polynomial, or None if the expression is not polynomial
///
/// # Examples
///
/// ```
/// use rustmath_calculus::expr::polynomial_degree;
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// let poly = x.clone() * x.clone() + Expr::from(3) * x.clone() + Expr::from(2);
/// assert_eq!(polynomial_degree(&poly, "x"), Some(2));
/// ```
pub fn polynomial_degree(expr: &Expr, var: &str) -> Option<i64> {
    if !is_polynomial(expr, var) {
        return None;
    }

    match expr {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => Some(0),
        Expr::Symbol(s) => {
            if s.name() == var {
                Some(1)
            } else {
                Some(0)
            }
        }
        Expr::Binary(op, left, right) => {
            use rustmath_symbolic::BinaryOp;
            match op {
                BinaryOp::Add | BinaryOp::Sub => {
                    let left_deg = polynomial_degree(left, var)?;
                    let right_deg = polynomial_degree(right, var)?;
                    Some(left_deg.max(right_deg))
                }
                BinaryOp::Mul => {
                    let left_deg = polynomial_degree(left, var)?;
                    let right_deg = polynomial_degree(right, var)?;
                    Some(left_deg + right_deg)
                }
                BinaryOp::Pow => {
                    if let Expr::Integer(n) = &**right {
                        let left_deg = polynomial_degree(left, var)?;
                        // Convert Integer to i64 for multiplication
                        // This is a simplification - would need better conversion in production
                        
                        let n_str = format!("{}", n);
                        let n_val: i64 = n_str.parse().ok()?;
                        Some(left_deg * n_val)
                    } else {
                        None
                    }
                }
                BinaryOp::Div => {
                    // Only valid if denominator doesn't contain var
                    if variables(right).contains(var) {
                        None
                    } else {
                        polynomial_degree(left, var)
                    }
                }
                BinaryOp::Mod => {
                    // Modulo is not a polynomial operation
                    None
                }
            }
        }
        Expr::Unary(op, inner) => {
            use rustmath_symbolic::UnaryOp;
            match op {
                UnaryOp::Neg => polynomial_degree(inner, var),
                _ => None,
            }
        }
        Expr::Function(_, _) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variables_extraction() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let z = Expr::symbol("z");

        // Single variable
        let vars = variables(&x);
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));

        // Multiple variables
        let expr = x.clone() * y.clone() + z.clone();
        let vars = variables(&expr);
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));

        // No variables (constant)
        let c = Expr::from(42);
        let vars = variables(&c);
        assert_eq!(vars.len(), 0);
    }

    #[test]
    fn test_is_constant() {
        assert!(is_constant(&Expr::from(42)));
        assert!(is_constant(&(Expr::from(3) + Expr::from(7))));
        assert!(!is_constant(&Expr::symbol("x")));

        let x = Expr::symbol("x");
        assert!(!is_constant(&(x.clone() + Expr::from(5))));
    }

    #[test]
    fn test_is_polynomial() {
        let x = Expr::symbol("x");

        // Constants are polynomials
        assert!(is_polynomial(&Expr::from(42), "x"));

        // x is polynomial
        assert!(is_polynomial(&x, "x"));

        // x^2 + 3*x + 2 is polynomial
        let poly = x.clone() * x.clone() + Expr::from(3) * x.clone() + Expr::from(2);
        assert!(is_polynomial(&poly, "x"));

        // sin(x) is not polynomial
        assert!(!is_polynomial(&x.clone().sin(), "x"));

        // exp(x) is not polynomial
        assert!(!is_polynomial(&x.clone().exp(), "x"));
    }

    #[test]
    fn test_polynomial_degree() {
        let x = Expr::symbol("x");

        // Constant has degree 0
        assert_eq!(polynomial_degree(&Expr::from(42), "x"), Some(0));

        // x has degree 1
        assert_eq!(polynomial_degree(&x, "x"), Some(1));

        // x^2 has degree 2
        let x_squared = x.clone().pow(Expr::from(2));
        assert_eq!(polynomial_degree(&x_squared, "x"), Some(2));

        // x^2 + 3*x + 2 has degree 2
        let poly = x.clone() * x.clone() + Expr::from(3) * x.clone() + Expr::from(2);
        assert_eq!(polynomial_degree(&poly, "x"), Some(2));

        // x^3 * x^2 = x^5 has degree 5
        let high_poly = x.clone().pow(Expr::from(3)) * x.clone().pow(Expr::from(2));
        assert_eq!(polynomial_degree(&high_poly, "x"), Some(5));
    }

    #[test]
    fn test_variables_in_complex_expression() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");

        // (x + y)^2 contains x and y
        let expr = (x.clone() + y.clone()).pow(Expr::from(2));
        let vars = variables(&expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));

        // sin(x) + cos(y) contains x and y
        let expr = x.sin() + y.cos();
        let vars = variables(&expr);
        assert_eq!(vars.len(), 2);
    }
}
