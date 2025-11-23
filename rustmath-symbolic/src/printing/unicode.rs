//! Unicode pretty printing for symbolic expressions
//!
//! Converts expressions to Unicode mathematical notation for terminal display.

use crate::expression::{BinaryOp, Expr, UnaryOp};
use std::fmt;

/// Convert a digit to its Unicode superscript equivalent
fn to_superscript(c: char) -> char {
    match c {
        '0' => '⁰',
        '1' => '¹',
        '2' => '²',
        '3' => '³',
        '4' => '⁴',
        '5' => '⁵',
        '6' => '⁶',
        '7' => '⁷',
        '8' => '⁸',
        '9' => '⁹',
        '+' => '⁺',
        '-' => '⁻',
        '=' => '⁼',
        '(' => '⁽',
        ')' => '⁾',
        _ => c,
    }
}

/// Convert a digit to its Unicode subscript equivalent
fn to_subscript(c: char) -> char {
    match c {
        '0' => '₀',
        '1' => '₁',
        '2' => '₂',
        '3' => '₃',
        '4' => '₄',
        '5' => '₅',
        '6' => '₆',
        '7' => '₇',
        '8' => '₈',
        '9' => '₉',
        '+' => '₊',
        '-' => '₋',
        '=' => '₌',
        '(' => '₍',
        ')' => '₎',
        _ => c,
    }
}

/// Convert a string to superscript
fn superscript(s: &str) -> String {
    s.chars().map(to_superscript).collect()
}

/// Check if an expression needs parentheses when used as a child
fn needs_parens(expr: &Expr, parent_op: BinaryOp) -> bool {
    match expr {
        Expr::Binary(op, _, _) => {
            let expr_prec = operator_precedence(*op);
            let parent_prec = operator_precedence(parent_op);
            expr_prec < parent_prec
        }
        Expr::Unary(UnaryOp::Neg, _) => {
            // Negative numbers need parens in certain contexts
            matches!(parent_op, BinaryOp::Pow | BinaryOp::Div)
        }
        _ => false,
    }
}

/// Get operator precedence (higher = binds tighter)
fn operator_precedence(op: BinaryOp) -> i32 {
    match op {
        BinaryOp::Add | BinaryOp::Sub => 1,
        BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 2,
        BinaryOp::Pow => 3,
    }
}

/// Pretty print an expression to Unicode
pub fn to_unicode(expr: &Expr) -> String {
    to_unicode_impl(expr, None)
}

fn to_unicode_impl(expr: &Expr, parent_op: Option<BinaryOp>) -> String {
    match expr {
        Expr::Integer(n) => format!("{}", n),
        Expr::Rational(r) => {
            if r.denominator() == &rustmath_integers::Integer::from(1) {
                format!("{}", r.numerator())
            } else {
                format!("{}/{}", r.numerator(), r.denominator())
            }
        }
        Expr::Real(f) => format!("{}", f),
        Expr::Symbol(s) => {
            let name = s.name();
            // Replace common symbols with Unicode equivalents
            match name {
                "pi" => "π".to_string(),
                "theta" => "θ".to_string(),
                "alpha" => "α".to_string(),
                "beta" => "β".to_string(),
                "gamma" => "γ".to_string(),
                "delta" => "δ".to_string(),
                "epsilon" => "ε".to_string(),
                "lambda" => "λ".to_string(),
                "mu" => "μ".to_string(),
                "sigma" => "σ".to_string(),
                "phi" => "φ".to_string(),
                "omega" => "ω".to_string(),
                "oo" | "infinity" => "∞".to_string(),
                _ => name.to_string(),
            }
        }
        Expr::Binary(op, left, right) => {
            let left_str = if needs_parens(left.as_ref(), *op) {
                format!("({})", to_unicode_impl(left.as_ref(), Some(*op)))
            } else {
                to_unicode_impl(left.as_ref(), Some(*op))
            };

            let right_str = if needs_parens(right.as_ref(), *op) {
                format!("({})", to_unicode_impl(right.as_ref(), Some(*op)))
            } else {
                to_unicode_impl(right.as_ref(), Some(*op))
            };

            let result = match op {
                BinaryOp::Add => format!("{} + {}", left_str, right_str),
                BinaryOp::Sub => format!("{} − {}", left_str, right_str),
                BinaryOp::Mul => {
                    // Use invisible multiplication or · for cleaner output
                    format!("{}·{}", left_str, right_str)
                }
                BinaryOp::Div => format!("{}/{}", left_str, right_str),
                BinaryOp::Pow => {
                    // Try to use superscript for simple exponents
                    if is_simple_number(right.as_ref()) {
                        format!("{}{}", left_str, superscript(&right_str))
                    } else {
                        format!("{}^({})", left_str, right_str)
                    }
                }
                BinaryOp::Mod => format!("{} mod {}", left_str, right_str),
            };

            // Add outer parentheses if needed
            if let Some(parent) = parent_op {
                if operator_precedence(*op) < operator_precedence(parent) {
                    format!("({})", result)
                } else {
                    result
                }
            } else {
                result
            }
        }
        Expr::Unary(op, arg) => {
            let arg_str = to_unicode_impl(arg.as_ref(), None);
            match op {
                UnaryOp::Neg => format!("−{}", arg_str),
                UnaryOp::Sin => format!("sin({})", arg_str),
                UnaryOp::Cos => format!("cos({})", arg_str),
                UnaryOp::Tan => format!("tan({})", arg_str),
                UnaryOp::Sinh => format!("sinh({})", arg_str),
                UnaryOp::Cosh => format!("cosh({})", arg_str),
                UnaryOp::Tanh => format!("tanh({})", arg_str),
                UnaryOp::Arcsin => format!("arcsin({})", arg_str),
                UnaryOp::Arccos => format!("arccos({})", arg_str),
                UnaryOp::Arctan => format!("arctan({})", arg_str),
                UnaryOp::Arcsinh => format!("arcsinh({})", arg_str),
                UnaryOp::Arccosh => format!("arccosh({})", arg_str),
                UnaryOp::Arctanh => format!("arctanh({})", arg_str),
                UnaryOp::Exp => format!("e^({})", arg_str),
                UnaryOp::Log => format!("ln({})", arg_str),
                UnaryOp::Sqrt => format!("√({})", arg_str),
                UnaryOp::Abs => format!("|{}|", arg_str),
                UnaryOp::Sign => format!("sgn({})", arg_str),
                UnaryOp::Gamma => format!("Γ({})", arg_str),
                UnaryOp::Factorial => format!("{}!", arg_str),
                UnaryOp::Erf => format!("erf({})", arg_str),
                UnaryOp::Zeta => format!("ζ({})", arg_str),
            }
        }
        Expr::Function(name, args) => {
            let args_str: Vec<String> = args
                .iter()
                .map(|arg| to_unicode_impl(arg.as_ref(), None))
                .collect();
            format!("{}({})", name, args_str.join(", "))
        }
    }
}

/// Check if an expression is a simple number (for superscript rendering)
fn is_simple_number(expr: &Expr) -> bool {
    match expr {
        Expr::Integer(_) => true,
        Expr::Unary(UnaryOp::Neg, inner) => matches!(inner.as_ref(), Expr::Integer(_)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_unicode_integer() {
        let expr = Expr::from(42);
        assert_eq!(to_unicode(&expr), "42");
    }

    #[test]
    fn test_unicode_symbol() {
        let x = Expr::symbol("x");
        assert_eq!(to_unicode(&x), "x");
    }

    #[test]
    fn test_unicode_pi() {
        let pi = Expr::symbol("pi");
        assert_eq!(to_unicode(&pi), "π");
    }

    #[test]
    fn test_unicode_addition() {
        let x = Expr::symbol("x");
        let expr = x + Expr::from(2);
        assert_eq!(to_unicode(&expr), "x + 2");
    }

    #[test]
    fn test_unicode_power_simple() {
        let x = Expr::symbol("x");
        let expr = x.pow(Expr::from(2));
        assert_eq!(to_unicode(&expr), "x²");
    }

    #[test]
    fn test_unicode_sqrt() {
        let x = Expr::symbol("x");
        let expr = x.sqrt();
        assert_eq!(to_unicode(&expr), "√(x)");
    }

    #[test]
    fn test_unicode_fraction() {
        let expr = Expr::from(1) / Expr::from(2);
        assert!(to_unicode(&expr).contains("/"));
    }

    #[test]
    fn test_unicode_complex_expression() {
        let x = Expr::symbol("x");
        let expr = x.clone().pow(Expr::from(2)) + Expr::from(2) * x + Expr::from(1);
        let result = to_unicode(&expr);
        // Should contain x² and unicode symbols
        assert!(result.contains("²") || result.contains("^"));
    }
}
