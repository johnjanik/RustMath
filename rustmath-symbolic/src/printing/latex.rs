//! LaTeX pretty printing for symbolic expressions
//!
//! Converts expressions to LaTeX notation for use in Jupyter notebooks and documents.

use crate::expression::{BinaryOp, Expr, UnaryOp};

/// Convert an expression to LaTeX format
pub fn to_latex(expr: &Expr) -> String {
    to_latex_impl(expr, None)
}

fn to_latex_impl(expr: &Expr, parent_op: Option<BinaryOp>) -> String {
    match expr {
        Expr::Integer(n) => format!("{}", n),
        Expr::Rational(r) => {
            if r.denominator() == &rustmath_integers::Integer::from(1) {
                format!("{}", r.numerator())
            } else {
                format!("\\frac{{{}}}{{{}}}", r.numerator(), r.denominator())
            }
        }
        Expr::Real(f) => format!("{}", f),
        Expr::Symbol(s) => {
            let name = s.name();
            // Handle Greek letters and special symbols
            match name {
                "pi" => "\\pi".to_string(),
                "theta" => "\\theta".to_string(),
                "alpha" => "\\alpha".to_string(),
                "beta" => "\\beta".to_string(),
                "gamma" => "\\gamma".to_string(),
                "Gamma" => "\\Gamma".to_string(),
                "delta" => "\\delta".to_string(),
                "Delta" => "\\Delta".to_string(),
                "epsilon" => "\\epsilon".to_string(),
                "lambda" => "\\lambda".to_string(),
                "mu" => "\\mu".to_string(),
                "sigma" => "\\sigma".to_string(),
                "Sigma" => "\\Sigma".to_string(),
                "phi" => "\\phi".to_string(),
                "Phi" => "\\Phi".to_string(),
                "omega" => "\\omega".to_string(),
                "Omega" => "\\Omega".to_string(),
                "oo" | "infinity" => "\\infty".to_string(),
                "I" => "i".to_string(), // Imaginary unit
                // Multi-character symbols should be in \mathrm
                name if name.len() > 1 => format!("\\mathrm{{{}}}", name),
                _ => name.to_string(),
            }
        }
        Expr::Binary(op, left, right) => {
            let left_str = to_latex_impl(left.as_ref(), Some(*op));
            let right_str = to_latex_impl(right.as_ref(), Some(*op));

            let result = match op {
                BinaryOp::Add => format!("{} + {}", left_str, right_str),
                BinaryOp::Sub => format!("{} - {}", left_str, right_str),
                BinaryOp::Mul => {
                    // Use \cdot for multiplication
                    format!("{} \\cdot {}", left_str, right_str)
                }
                BinaryOp::Div => {
                    // Use \frac for division
                    format!("\\frac{{{}}}{{{}}}", left_str, right_str)
                }
                BinaryOp::Pow => {
                    // Use ^{} for exponentiation
                    format!("{{{}}}^{{{}}}", left_str, right_str)
                }
                BinaryOp::Mod => {
                    format!("{} \\bmod {}", left_str, right_str)
                }
            };

            // Add parentheses if needed based on precedence
            if let Some(parent) = parent_op {
                if operator_precedence(*op) < operator_precedence(parent) && !matches!(op, BinaryOp::Div) {
                    format!("\\left({}\\right)", result)
                } else {
                    result
                }
            } else {
                result
            }
        }
        Expr::Unary(op, arg) => {
            let arg_str = to_latex_impl(arg.as_ref(), None);
            match op {
                UnaryOp::Neg => format!("-{}", arg_str),
                UnaryOp::Sin => format!("\\sin\\left({}\\right)", arg_str),
                UnaryOp::Cos => format!("\\cos\\left({}\\right)", arg_str),
                UnaryOp::Tan => format!("\\tan\\left({}\\right)", arg_str),
                UnaryOp::Sinh => format!("\\sinh\\left({}\\right)", arg_str),
                UnaryOp::Cosh => format!("\\cosh\\left({}\\right)", arg_str),
                UnaryOp::Tanh => format!("\\tanh\\left({}\\right)", arg_str),
                UnaryOp::Arcsin => format!("\\arcsin\\left({}\\right)", arg_str),
                UnaryOp::Arccos => format!("\\arccos\\left({}\\right)", arg_str),
                UnaryOp::Arctan => format!("\\arctan\\left({}\\right)", arg_str),
                UnaryOp::Arcsinh => format!("\\text{{arcsinh}}\\left({}\\right)", arg_str),
                UnaryOp::Arccosh => format!("\\text{{arccosh}}\\left({}\\right)", arg_str),
                UnaryOp::Arctanh => format!("\\text{{arctanh}}\\left({}\\right)", arg_str),
                UnaryOp::Exp => format!("e^{{{}}}", arg_str),
                UnaryOp::Log => format!("\\ln\\left({}\\right)", arg_str),
                UnaryOp::Sqrt => format!("\\sqrt{{{}}}", arg_str),
                UnaryOp::Abs => format!("\\left|{}\\right|", arg_str),
                UnaryOp::Sign => format!("\\text{{sgn}}\\left({}\\right)", arg_str),
                UnaryOp::Gamma => format!("\\Gamma\\left({}\\right)", arg_str),
                UnaryOp::Factorial => format!("{}!", arg_str),
                UnaryOp::Erf => format!("\\text{{erf}}\\left({}\\right)", arg_str),
                UnaryOp::Zeta => format!("\\zeta\\left({}\\right)", arg_str),
            }
        }
        Expr::Function(name, args) => {
            let args_str: Vec<String> = args
                .iter()
                .map(|arg| to_latex_impl(arg.as_ref(), None))
                .collect();
            format!("\\text{{{}}}\\left({}\\right)", name, args_str.join(", "))
        }
    }
}

fn operator_precedence(op: BinaryOp) -> i32 {
    match op {
        BinaryOp::Add | BinaryOp::Sub => 1,
        BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 2,
        BinaryOp::Pow => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_latex_integer() {
        let expr = Expr::from(42);
        assert_eq!(to_latex(&expr), "42");
    }

    #[test]
    fn test_latex_symbol() {
        let x = Expr::symbol("x");
        assert_eq!(to_latex(&x), "x");
    }

    #[test]
    fn test_latex_pi() {
        let pi = Expr::symbol("pi");
        assert_eq!(to_latex(&pi), "\\pi");
    }

    #[test]
    fn test_latex_fraction() {
        let expr = Expr::from(1) / Expr::from(2);
        assert_eq!(to_latex(&expr), "\\frac{1}{2}");
    }

    #[test]
    fn test_latex_power() {
        let x = Expr::symbol("x");
        let expr = x.pow(Expr::from(2));
        assert_eq!(to_latex(&expr), "{x}^{2}");
    }

    #[test]
    fn test_latex_sqrt() {
        let x = Expr::symbol("x");
        let expr = x.sqrt();
        assert_eq!(to_latex(&expr), "\\sqrt{x}");
    }

    #[test]
    fn test_latex_sin() {
        let x = Expr::symbol("x");
        let expr = x.sin();
        assert_eq!(to_latex(&expr), "\\sin\\left(x\\right)");
    }

    #[test]
    fn test_latex_complex_expression() {
        let x = Expr::symbol("x");
        let expr = x.clone().pow(Expr::from(2)) + Expr::from(2) * x + Expr::from(1);
        let latex = to_latex(&expr);
        // Should contain LaTeX power notation
        assert!(latex.contains("^") || latex.contains("\\cdot"));
    }

    #[test]
    fn test_latex_greek() {
        let theta = Expr::symbol("theta");
        assert_eq!(to_latex(&theta), "\\theta");

        let alpha = Expr::symbol("alpha");
        assert_eq!(to_latex(&alpha), "\\alpha");
    }
}
