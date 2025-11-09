//! Algebraic equation solving
//!
//! This module provides functionality for solving algebraic equations symbolically.

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::{solve_cubic, solve_quadratic, solve_quartic, QuadraticRoots};
use rustmath_rationals::Rational;
use std::sync::Arc;

/// Solution to an equation
#[derive(Debug, Clone, PartialEq)]
pub enum Solution {
    /// A single expression solution
    Expr(Expr),
    /// No solutions
    None,
    /// Infinitely many solutions
    All,
    /// Multiple solutions
    Multiple(Vec<Expr>),
}

impl Expr {
    /// Solve an equation for a variable
    ///
    /// This attempts to solve `self = 0` for the given variable.
    /// For equations of the form `lhs = rhs`, first convert to `lhs - rhs = 0`.
    ///
    /// Supports:
    /// - Linear equations
    /// - Quadratic equations
    /// - Cubic equations
    /// - Quartic equations
    /// - Some simple transcendental equations
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    /// use rustmath_symbolic::Symbol;
    ///
    /// let x = Expr::symbol("x");
    ///
    /// // Solve x + 1 = 0
    /// let expr = x.clone() + Expr::from(1);
    /// let solutions = expr.solve(&Symbol::new("x"));
    /// ```
    pub fn solve(&self, var: &Symbol) -> Solution {
        solve_equation(self, var)
    }

    /// Solve an equation of the form lhs = rhs for a variable
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    /// use rustmath_symbolic::Symbol;
    ///
    /// let x = Expr::symbol("x");
    ///
    /// // Solve 2x + 1 = 5
    /// let lhs = Expr::from(2) * x.clone() + Expr::from(1);
    /// let rhs = Expr::from(5);
    /// let solutions = Expr::solve_equation(lhs, rhs, &Symbol::new("x"));
    /// ```
    pub fn solve_equation(lhs: Expr, rhs: Expr, var: &Symbol) -> Solution {
        // Convert to lhs - rhs = 0
        let equation = Expr::Binary(BinaryOp::Sub, Arc::new(lhs), Arc::new(rhs));
        solve_equation(&equation.simplify(), var)
    }
}

/// Main equation solving function
fn solve_equation(expr: &Expr, var: &Symbol) -> Solution {
    // First simplify the expression
    let simplified = expr.simplify();

    // Check if it's a polynomial
    if simplified.is_polynomial(var) {
        return solve_polynomial(&simplified, var);
    }

    // Try some special cases for transcendental equations
    solve_transcendental(&simplified, var)
}

/// Solve a polynomial equation
fn solve_polynomial(expr: &Expr, var: &Symbol) -> Solution {
    let degree = match expr.degree(var) {
        Some(d) => d,
        None => return Solution::None,
    };

    match degree {
        0 => solve_constant(expr),
        1 => solve_linear(expr, var),
        2 => solve_quadratic_symbolic(expr, var),
        3 => solve_cubic_symbolic(expr, var),
        4 => solve_quartic_symbolic(expr, var),
        _ => Solution::None, // Higher degree polynomials not supported yet
    }
}

/// Solve a constant equation (no variable)
fn solve_constant(expr: &Expr) -> Solution {
    match expr {
        Expr::Integer(n) if n.is_zero() => Solution::All,
        Expr::Rational(r) if r.is_zero() => Solution::All,
        _ => Solution::None,
    }
}

/// Solve a linear equation: ax + b = 0
fn solve_linear(expr: &Expr, var: &Symbol) -> Solution {
    let a = match expr.coefficient(var, 1) {
        Some(coeff) => coeff,
        None => return Solution::None,
    };

    let b = match expr.coefficient(var, 0) {
        Some(coeff) => coeff,
        None => return Solution::None,
    };

    // Check if a is zero
    if is_zero(&a) {
        return if is_zero(&b) {
            Solution::All
        } else {
            Solution::None
        };
    }

    // x = -b/a
    let solution = Expr::Binary(
        BinaryOp::Div,
        Arc::new(Expr::Unary(UnaryOp::Neg, Arc::new(b))),
        Arc::new(a),
    );

    Solution::Expr(solution.simplify())
}

/// Solve a quadratic equation: ax^2 + bx + c = 0
fn solve_quadratic_symbolic(expr: &Expr, var: &Symbol) -> Solution {
    let a = match expr.coefficient(var, 2) {
        Some(coeff) => coeff,
        None => return Solution::None,
    };

    let b = match expr.coefficient(var, 1) {
        Some(coeff) => coeff,
        None => return Solution::None,
    };

    let c = match expr.coefficient(var, 0) {
        Some(coeff) => coeff,
        None => return Solution::None,
    };

    // Try to solve numerically if all coefficients are integers
    if let (Some(a_i), Some(b_i), Some(c_i)) = (
        expr_to_integer_big(&a),
        expr_to_integer_big(&b),
        expr_to_integer_big(&c),
    ) {
        return solve_quadratic_numeric(a_i, b_i, c_i);
    }

    // Symbolic quadratic formula: x = (-b ± sqrt(b^2 - 4ac)) / 2a
    let discriminant = Expr::Binary(
        BinaryOp::Sub,
        Arc::new(Expr::Binary(
            BinaryOp::Pow,
            Arc::new(b.clone()),
            Arc::new(Expr::from(2)),
        )),
        Arc::new(Expr::Binary(
            BinaryOp::Mul,
            Arc::new(Expr::from(4)),
            Arc::new(Expr::Binary(BinaryOp::Mul, Arc::new(a.clone()), Arc::new(c))),
        )),
    )
    .simplify();

    let sqrt_disc = Expr::Unary(UnaryOp::Sqrt, Arc::new(discriminant));
    let two_a = Expr::Binary(BinaryOp::Mul, Arc::new(Expr::from(2)), Arc::new(a));

    // x₁ = (-b + sqrt(disc)) / 2a
    let sol1 = Expr::Binary(
        BinaryOp::Div,
        Arc::new(Expr::Binary(
            BinaryOp::Add,
            Arc::new(Expr::Unary(UnaryOp::Neg, Arc::new(b.clone()))),
            Arc::new(sqrt_disc.clone()),
        )),
        Arc::new(two_a.clone()),
    )
    .simplify();

    // x₂ = (-b - sqrt(disc)) / 2a
    let sol2 = Expr::Binary(
        BinaryOp::Div,
        Arc::new(Expr::Binary(
            BinaryOp::Sub,
            Arc::new(Expr::Unary(UnaryOp::Neg, Arc::new(b))),
            Arc::new(sqrt_disc),
        )),
        Arc::new(two_a),
    )
    .simplify();

    Solution::Multiple(vec![sol1, sol2])
}

/// Solve quadratic equation numerically
fn solve_quadratic_numeric(a: Integer, b: Integer, c: Integer) -> Solution {
    let roots = match solve_quadratic(a, b, c) {
        Ok(r) => r,
        Err(_) => return Solution::None,
    };

    match roots {
        QuadraticRoots::OneRational(r) => Solution::Expr(Expr::Rational(r)),
        QuadraticRoots::TwoRational(r1, r2) => {
            Solution::Multiple(vec![Expr::Rational(r1), Expr::Rational(r2)])
        }
        QuadraticRoots::TwoIrrational { p, d, q } => {
            // Return symbolic form: (-p ± sqrt(d)) / q
            let sqrt_d = Expr::Unary(UnaryOp::Sqrt, Arc::new(Expr::Integer(d)));
            let neg_p = Expr::Integer(-p.clone());
            let q_expr = Expr::Integer(q.clone());

            let sol1 = Expr::Binary(
                BinaryOp::Div,
                Arc::new(Expr::Binary(
                    BinaryOp::Add,
                    Arc::new(neg_p.clone()),
                    Arc::new(sqrt_d.clone()),
                )),
                Arc::new(q_expr.clone()),
            );

            let sol2 = Expr::Binary(
                BinaryOp::Div,
                Arc::new(Expr::Binary(
                    BinaryOp::Sub,
                    Arc::new(neg_p),
                    Arc::new(sqrt_d),
                )),
                Arc::new(q_expr),
            );

            Solution::Multiple(vec![sol1, sol2])
        }
        QuadraticRoots::Complex { .. } => Solution::None, // Complex roots not supported yet
    }
}

/// Solve a cubic equation
fn solve_cubic_symbolic(expr: &Expr, var: &Symbol) -> Solution {
    // Extract coefficients
    let a = match expr.coefficient(var, 3) {
        Some(coeff) => coeff,
        None => return Solution::None,
    };

    let b = match expr.coefficient(var, 2) {
        Some(coeff) => coeff,
        None => return Solution::None,
    };

    let c = match expr.coefficient(var, 1) {
        Some(coeff) => coeff,
        None => return Solution::None,
    };

    let d = match expr.coefficient(var, 0) {
        Some(coeff) => coeff,
        None => return Solution::None,
    };

    // Try numeric solving if all coefficients are integers
    if let (Some(a_i), Some(b_i), Some(c_i), Some(d_i)) = (
        expr_to_integer_big(&a),
        expr_to_integer_big(&b),
        expr_to_integer_big(&c),
        expr_to_integer_big(&d),
    ) {
        let roots = match solve_cubic(a_i, b_i, c_i, d_i) {
            Ok(r) => r,
            Err(_) => return Solution::None,
        };

        return match roots {
            rustmath_polynomials::CubicRoots::OneRational(r) => {
                Solution::Multiple(vec![Expr::Rational(r)])
            }
            rustmath_polynomials::CubicRoots::TwoRational(r1, r2) => {
                Solution::Multiple(vec![Expr::Rational(r1), Expr::Rational(r2)])
            }
            rustmath_polynomials::CubicRoots::ThreeRational(r1, r2, r3) => {
                Solution::Multiple(vec![
                    Expr::Rational(r1),
                    Expr::Rational(r2),
                    Expr::Rational(r3),
                ])
            }
            _ => Solution::None, // Other cases not supported yet
        };
    }

    // Symbolic cubic formula would be very complex, not implemented yet
    Solution::None
}

/// Solve a quartic equation
fn solve_quartic_symbolic(expr: &Expr, var: &Symbol) -> Solution {
    // Extract coefficients
    let coeffs: Vec<Option<Expr>> = (0..=4).map(|i| expr.coefficient(var, i)).collect();

    // Check if all coefficients were successfully extracted
    if coeffs.iter().any(|c| c.is_none()) {
        return Solution::None;
    }

    let coeffs: Vec<Expr> = coeffs.into_iter().map(|c| c.unwrap()).collect();

    // Try numeric solving if all coefficients are integers
    let integers: Vec<Option<Integer>> =
        coeffs.iter().map(expr_to_integer_big).collect();

    if integers.iter().all(|i| i.is_some()) {
        let integers: Vec<Integer> = integers.into_iter().map(|i| i.unwrap()).collect();

        let roots = match solve_quartic(
            integers[4].clone(),
            integers[3].clone(),
            integers[2].clone(),
            integers[1].clone(),
            integers[0].clone(),
        ) {
            Ok(r) => r,
            Err(_) => return Solution::None,
        };

        return match roots {
            rustmath_polynomials::QuarticRoots::Rational(rationals) => {
                let solutions: Vec<Expr> = rationals.into_iter().map(Expr::Rational).collect();
                if solutions.is_empty() {
                    Solution::None
                } else {
                    Solution::Multiple(solutions)
                }
            }
            rustmath_polynomials::QuarticRoots::Symbolic { .. } => {
                Solution::None // Symbolic roots not supported yet
            }
        };
    }

    Solution::None
}

/// Solve transcendental equations (limited support)
fn solve_transcendental(expr: &Expr, var: &Symbol) -> Solution {
    match expr {
        // exp(x) - c = 0 => x = log(c) OR log(x) - c = 0 => x = exp(c)
        Expr::Binary(BinaryOp::Sub, left, right) => {
            // exp(x) = c => x = log(c)
            if let Expr::Unary(UnaryOp::Exp, arg) = left.as_ref() {
                if let Expr::Symbol(s) = arg.as_ref() {
                    if s == var {
                        return Solution::Expr(Expr::Unary(UnaryOp::Log, right.clone()));
                    }
                }
            }

            // log(x) = c => x = exp(c)
            if let Expr::Unary(UnaryOp::Log, arg) = left.as_ref() {
                if let Expr::Symbol(s) = arg.as_ref() {
                    if s == var {
                        return Solution::Expr(Expr::Unary(UnaryOp::Exp, right.clone()));
                    }
                }
            }

            Solution::None
        }

        _ => Solution::None,
    }
}

/// Check if an expression is zero
fn is_zero(expr: &Expr) -> bool {
    match expr {
        Expr::Integer(n) => n.is_zero(),
        Expr::Rational(r) => r.is_zero(),
        _ => false,
    }
}

/// Convert expression to rational if possible
fn expr_to_rational(expr: &Expr) -> Option<Rational> {
    match expr {
        Expr::Integer(n) => n.to_i64().and_then(|i| Rational::new(i, 1).ok()),
        Expr::Rational(r) => Some(r.clone()),
        Expr::Binary(BinaryOp::Div, num, den) => {
            let num_i = expr_to_integer(num)?;
            let den_i = expr_to_integer(den)?;
            Rational::new(num_i, den_i).ok()
        }
        Expr::Unary(UnaryOp::Neg, inner) => {
            let r = expr_to_rational(inner)?;
            Some(-r)
        }
        _ => None,
    }
}

/// Convert expression to Integer (big integer) if possible
fn expr_to_integer_big(expr: &Expr) -> Option<Integer> {
    match expr {
        Expr::Integer(n) => Some(n.clone()),
        Expr::Unary(UnaryOp::Neg, inner) => expr_to_integer_big(inner).map(|x| -x),
        _ => None,
    }
}

/// Convert expression to integer if possible
fn expr_to_integer(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Integer(n) => n.to_i64(),
        Expr::Unary(UnaryOp::Neg, inner) => expr_to_integer(inner).map(|x| -x),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_linear() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0]; // Extract the actual Symbol

        // x + 1 = 0 => x = -1
        let expr = x.clone() + Expr::from(1);
        let solution = expr.solve(var_x);

        match solution {
            Solution::Expr(sol) => {
                assert_eq!(sol.simplify(), Expr::from(-1));
            }
            _ => panic!("Expected single solution"),
        }
    }

    #[test]
    fn test_solve_linear_general() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0]; // Extract the actual Symbol

        // 2x + 4 = 0 => x = -2
        let expr = Expr::from(2) * x.clone() + Expr::from(4);
        let solution = expr.solve(var_x);

        match solution {
            Solution::Expr(sol) => {
                assert_eq!(sol.simplify(), Expr::from(-2));
            }
            _ => panic!("Expected single solution"),
        }
    }

    #[test]
    fn test_solve_quadratic_simple() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0]; // Extract the actual Symbol

        // x^2 - 1 = 0 => x = ±1
        let expr = x.clone().pow(Expr::from(2)) - Expr::from(1);
        let solution = expr.solve(var_x);

        match solution {
            Solution::Multiple(sols) => {
                assert_eq!(sols.len(), 2);
            }
            _ => panic!("Expected multiple solutions"),
        }
    }

    #[test]
    fn test_solve_constant_zero() {
        // 0 = 0 => all x
        let expr = Expr::from(0);
        let solution = expr.solve(&Symbol::new("x"));
        assert_eq!(solution, Solution::All);
    }

    #[test]
    fn test_solve_constant_nonzero() {
        // 5 = 0 => no solution
        let expr = Expr::from(5);
        let solution = expr.solve(&Symbol::new("x"));
        assert_eq!(solution, Solution::None);
    }

    #[test]
    fn test_solve_equation_form() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0]; // Extract the actual Symbol

        // 2x + 1 = 5 => x = 2
        let lhs = Expr::from(2) * x.clone() + Expr::from(1);
        let rhs = Expr::from(5);
        let solution = Expr::solve_equation(lhs, rhs, var_x);

        match solution {
            Solution::Expr(sol) => {
                assert_eq!(sol.simplify(), Expr::from(2));
            }
            _ => panic!("Expected single solution"),
        }
    }
}
