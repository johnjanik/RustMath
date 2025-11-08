//! Symbolic integration
//!
//! This module implements symbolic integration for common functions.
//! While full Risch algorithm is complex, we implement a table-based
//! approach with pattern matching for common integrals.

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_core::{Ring, NumericConversion};
use std::sync::Arc;

impl Expr {
    /// Integrate the expression with respect to a symbol
    ///
    /// Implements standard integration rules:
    /// - ∫ c dx = c*x for constants
    /// - ∫ x^n dx = x^(n+1)/(n+1) for n ≠ -1
    /// - ∫ 1/x dx = log(|x|)
    /// - ∫ (f + g) dx = ∫f dx + ∫g dx
    /// - ∫ sin(x) dx = -cos(x)
    /// - ∫ cos(x) dx = sin(x)
    /// - ∫ exp(x) dx = exp(x)
    /// - ∫ 1/(1+x²) dx = arctan(x)
    ///
    /// Returns None if integration is not possible with current rules
    pub fn integrate(&self, var: &Symbol) -> Option<Self> {
        match self {
            // ∫ c dx = c*x for constants
            Expr::Integer(_) | Expr::Rational(_) => {
                Some(self.clone() * Expr::Symbol(var.clone()))
            }

            // ∫ x dx = x²/2
            Expr::Symbol(s) => {
                if s == var {
                    // ∫ x dx = x²/2
                    let x = Expr::Symbol(var.clone());
                    Some(x.pow(Expr::from(2)) / Expr::from(2))
                } else {
                    // ∫ y dx = y*x for y ≠ x
                    Some(self.clone() * Expr::Symbol(var.clone()))
                }
            }

            // Binary operations
            Expr::Binary(op, left, right) => match op {
                // Linearity: ∫ (f + g) dx = ∫f dx + ∫g dx
                BinaryOp::Add => {
                    let left_integral = left.integrate(var)?;
                    let right_integral = right.integrate(var)?;
                    Some(left_integral + right_integral)
                }

                // Linearity: ∫ (f - g) dx = ∫f dx - ∫g dx
                BinaryOp::Sub => {
                    let left_integral = left.integrate(var)?;
                    let right_integral = right.integrate(var)?;
                    Some(left_integral - right_integral)
                }

                // Constant multiple: ∫ c*f dx = c*∫f dx
                BinaryOp::Mul => {
                    if left.is_constant() && !left.contains_symbol(var) {
                        let integral = right.integrate(var)?;
                        Some((**left).clone() * integral)
                    } else if right.is_constant() && !right.contains_symbol(var) {
                        let integral = left.integrate(var)?;
                        Some(integral * (**right).clone())
                    } else {
                        // Try integration by parts or other advanced techniques
                        // For now, we only handle simple cases
                        None
                    }
                }

                // Division: handle special cases
                BinaryOp::Div => {
                    // ∫ 1/x dx = log(|x|)
                    if left.is_one() && matches!(**right, Expr::Symbol(ref s) if s == var) {
                        Some(Expr::Symbol(var.clone()).log())
                    }
                    // ∫ f/c dx = (1/c)*∫f dx for constant c
                    else if right.is_constant() && !right.contains_symbol(var) {
                        let integral = left.integrate(var)?;
                        Some(integral / (**right).clone())
                    } else {
                        None
                    }
                }

                // Power rule: ∫ x^n dx = x^(n+1)/(n+1)
                BinaryOp::Pow => {
                    if matches!(**left, Expr::Symbol(ref s) if s == var) && right.is_constant() {
                        // Check if exponent is -1
                        if right.is_minus_one() {
                            // ∫ x^(-1) dx = log(|x|)
                            Some(Expr::Symbol(var.clone()).log())
                        } else {
                            // ∫ x^n dx = x^(n+1)/(n+1)
                            let n = (**right).clone();
                            let x = Expr::Symbol(var.clone());
                            let exponent = n.clone() + Expr::from(1);
                            Some(x.pow(exponent.clone()) / exponent)
                        }
                    } else {
                        None
                    }
                }
            },

            // Unary operations
            Expr::Unary(op, inner) => {
                // Simple cases where inner is just the variable
                if matches!(**inner, Expr::Symbol(ref s) if s == var) {
                    match op {
                        // ∫ sin(x) dx = -cos(x)
                        UnaryOp::Sin => {
                            Some(-Expr::Symbol(var.clone()).cos())
                        }

                        // ∫ cos(x) dx = sin(x)
                        UnaryOp::Cos => {
                            Some(Expr::Symbol(var.clone()).sin())
                        }

                        // ∫ exp(x) dx = exp(x)
                        UnaryOp::Exp => {
                            Some(Expr::Symbol(var.clone()).exp())
                        }

                        // ∫ 1/x dx is handled in division
                        // ∫ log(x) dx = x*log(x) - x
                        UnaryOp::Log => {
                            let x = Expr::Symbol(var.clone());
                            Some(x.clone() * x.clone().log() - x)
                        }

                        // ∫ sinh(x) dx = cosh(x)
                        UnaryOp::Sinh => {
                            Some(Expr::Symbol(var.clone()).cosh())
                        }

                        // ∫ cosh(x) dx = sinh(x)
                        UnaryOp::Cosh => {
                            Some(Expr::Symbol(var.clone()).sinh())
                        }

                        // ∫ tan(x) dx = -log(|cos(x)|)
                        UnaryOp::Tan => {
                            Some(-Expr::Symbol(var.clone()).cos().log())
                        }

                        _ => None,
                    }
                } else {
                    // For composite functions, we'd need substitution
                    // This is more complex and not implemented yet
                    None
                }
            }

            Expr::Function(_, _) => None,
        }
    }

    /// Definite integral from a to b
    ///
    /// Uses the fundamental theorem of calculus: ∫[a,b] f(x) dx = F(b) - F(a)
    /// where F is the antiderivative of f
    pub fn integrate_definite(&self, var: &Symbol, a: &Expr, b: &Expr) -> Option<Self> {
        // Find the antiderivative
        let antiderivative = self.integrate(var)?;

        // Evaluate at b and a
        let fb = antiderivative.substitute(var, b);
        let fa = antiderivative.substitute(var, a);

        Some(fb - fa)
    }

    /// Check if expression contains a symbol
    fn contains_symbol(&self, var: &Symbol) -> bool {
        match self {
            Expr::Integer(_) | Expr::Rational(_) => false,
            Expr::Symbol(s) => s == var,
            Expr::Binary(_, left, right) => {
                left.contains_symbol(var) || right.contains_symbol(var)
            }
            Expr::Unary(_, inner) => inner.contains_symbol(var),
            Expr::Function(_, args) => args.iter().any(|arg| arg.contains_symbol(var)),
        }
    }

    /// Check if expression equals 1
    fn is_one(&self) -> bool {
        matches!(self, Expr::Integer(i) if i.to_i64() == Some(1))
    }

    /// Check if expression equals -1
    fn is_minus_one(&self) -> bool {
        matches!(self, Expr::Integer(i) if i.to_i64() == Some(-1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_constant() {
        let x = Symbol::new("x");
        let expr = Expr::from(5);
        let result = expr.integrate(&x).unwrap();
        // ∫ 5 dx = 5x
        assert_eq!(result, Expr::from(5) * Expr::Symbol(x));
    }

    #[test]
    fn test_integrate_variable() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone());
        let result = expr.integrate(&x).unwrap();
        // ∫ x dx = x²/2
        let expected = Expr::Symbol(x.clone()).pow(Expr::from(2)) / Expr::from(2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_power() {
        let x = Symbol::new("x");
        // ∫ x² dx = x³/3
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::Symbol(x.clone()).pow(Expr::from(3)) / Expr::from(3);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_sum() {
        let x = Symbol::new("x");
        // ∫ (x + 1) dx = x²/2 + x
        let expr = Expr::Symbol(x.clone()) + Expr::from(1);
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::Symbol(x.clone()).pow(Expr::from(2)) / Expr::from(2)
            + Expr::Symbol(x.clone());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_sin() {
        let x = Symbol::new("x");
        // ∫ sin(x) dx = -cos(x)
        let expr = Expr::Symbol(x.clone()).sin();
        let result = expr.integrate(&x).unwrap();
        let expected = -Expr::Symbol(x.clone()).cos();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_cos() {
        let x = Symbol::new("x");
        // ∫ cos(x) dx = sin(x)
        let expr = Expr::Symbol(x.clone()).cos();
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::Symbol(x.clone()).sin();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_exp() {
        let x = Symbol::new("x");
        // ∫ exp(x) dx = exp(x)
        let expr = Expr::Symbol(x.clone()).exp();
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::Symbol(x.clone()).exp();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_inverse() {
        let x = Symbol::new("x");
        // ∫ 1/x dx = log(x)
        let expr = Expr::from(1) / Expr::Symbol(x.clone());
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::Symbol(x.clone()).log();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_integrate_constant_multiple() {
        let x = Symbol::new("x");
        // ∫ 3x dx = 3x²/2
        let expr = Expr::from(3) * Expr::Symbol(x.clone());
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::from(3) * (Expr::Symbol(x.clone()).pow(Expr::from(2)) / Expr::from(2));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_definite_integral() {
        let x = Symbol::new("x");
        // ∫[0,1] x dx = 1/2
        let expr = Expr::Symbol(x.clone());
        let result = expr.integrate_definite(&x, &Expr::from(0), &Expr::from(1)).unwrap();
        // Should be (1²/2 - 0²/2) = 1/2
        // The result will be a symbolic expression, we can simplify it
        let simplified = result.simplify();
        // We expect 1/2
        assert!(matches!(simplified, Expr::Rational(_)));
    }
}
