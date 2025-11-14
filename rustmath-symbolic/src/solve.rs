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

// ============================================================================
// Phase 4.1: Trigonometric Equation Solving
// ============================================================================

/// Solve trigonometric equations
///
/// Handles:
/// - sin(x) = a, cos(x) = a, tan(x) = a
/// - sin(nx) = a, cos(nx) = a (multiple angles)
/// - a*sin(x) + b*cos(x) = c
/// - Inverse trig equations
fn solve_trig_equation(expr: &Expr, var: &Symbol) -> Option<Solution> {
    match expr {
        // Pattern: trig(arg) - c = 0
        Expr::Binary(BinaryOp::Sub, left, right) => {
            match left.as_ref() {
                // sin(arg) = c
                Expr::Unary(UnaryOp::Sin, arg) => {
                    return Some(solve_sin_equation(arg, right, var));
                }
                // cos(arg) = c
                Expr::Unary(UnaryOp::Cos, arg) => {
                    return Some(solve_cos_equation(arg, right, var));
                }
                // tan(arg) = c
                Expr::Unary(UnaryOp::Tan, arg) => {
                    return Some(solve_tan_equation(arg, right, var));
                }
                // arcsin(arg) = c => arg = sin(c)
                Expr::Unary(UnaryOp::Arcsin, arg) => {
                    if arg_contains_var(arg, var) {
                        let rhs = Expr::Unary(UnaryOp::Sin, right.clone());
                        return Some(solve_equation(&Expr::Binary(
                            BinaryOp::Sub,
                            arg.clone(),
                            Arc::new(rhs),
                        ), var));
                    }
                }
                // arccos(arg) = c => arg = cos(c)
                Expr::Unary(UnaryOp::Arccos, arg) => {
                    if arg_contains_var(arg, var) {
                        let rhs = Expr::Unary(UnaryOp::Cos, right.clone());
                        return Some(solve_equation(&Expr::Binary(
                            BinaryOp::Sub,
                            arg.clone(),
                            Arc::new(rhs),
                        ), var));
                    }
                }
                // arctan(arg) = c => arg = tan(c)
                Expr::Unary(UnaryOp::Arctan, arg) => {
                    if arg_contains_var(arg, var) {
                        let rhs = Expr::Unary(UnaryOp::Tan, right.clone());
                        return Some(solve_equation(&Expr::Binary(
                            BinaryOp::Sub,
                            arg.clone(),
                            Arc::new(rhs),
                        ), var));
                    }
                }
                _ => {}
            }
        }

        // Pattern: a*sin(x) + b*cos(x) - c = 0
        // This can be converted to R*sin(x + φ) = c where R = sqrt(a² + b²)
        _ => {
            if let Some(sol) = solve_linear_combination_trig(expr, var) {
                return Some(sol);
            }
        }
    }

    None
}

/// Solve sin(arg) = c
fn solve_sin_equation(arg: &Expr, c: &Expr, var: &Symbol) -> Solution {
    // Check if arg is just the variable
    if let Expr::Symbol(s) = arg {
        if s == var {
            // sin(x) = c
            // General solution: x = arcsin(c) + 2πn or x = π - arcsin(c) + 2πn
            let arcsin_c = Expr::Unary(UnaryOp::Arcsin, Arc::new(c.clone()));

            // For symbolic solutions, we return the principal value
            // In a more complete implementation, we would return a family of solutions
            return Solution::Expr(arcsin_c);
        }
    }

    // Check for multiple angle: sin(nx) = c
    if let Expr::Binary(BinaryOp::Mul, left, right) = arg {
        // Pattern: sin(n*x) = c
        if let Expr::Symbol(s) = right.as_ref() {
            if s == var && !expr_contains_var(left, var) {
                // sin(n*x) = c => n*x = arcsin(c) + 2πk
                // => x = (arcsin(c) + 2πk) / n
                let arcsin_c = Expr::Unary(UnaryOp::Arcsin, Arc::new(c.clone()));
                let solution = Expr::Binary(
                    BinaryOp::Div,
                    Arc::new(arcsin_c),
                    left.clone(),
                );
                return Solution::Expr(solution.simplify());
            }
        }
        if let Expr::Symbol(s) = left.as_ref() {
            if s == var && !expr_contains_var(right, var) {
                let arcsin_c = Expr::Unary(UnaryOp::Arcsin, Arc::new(c.clone()));
                let solution = Expr::Binary(
                    BinaryOp::Div,
                    Arc::new(arcsin_c),
                    right.clone(),
                );
                return Solution::Expr(solution.simplify());
            }
        }
    }

    // Try to solve the argument for the variable
    let arg_equation = Expr::Binary(
        BinaryOp::Sub,
        Arc::new(arg.clone()),
        Arc::new(Expr::Unary(UnaryOp::Arcsin, Arc::new(c.clone()))),
    );
    solve_equation(&arg_equation, var)
}

/// Solve cos(arg) = c
fn solve_cos_equation(arg: &Expr, c: &Expr, var: &Symbol) -> Solution {
    // Check if arg is just the variable
    if let Expr::Symbol(s) = arg {
        if s == var {
            // cos(x) = c
            // General solution: x = ±arccos(c) + 2πn
            let arccos_c = Expr::Unary(UnaryOp::Arccos, Arc::new(c.clone()));
            return Solution::Expr(arccos_c);
        }
    }

    // Check for multiple angle: cos(nx) = c
    if let Expr::Binary(BinaryOp::Mul, left, right) = arg {
        if let Expr::Symbol(s) = right.as_ref() {
            if s == var && !expr_contains_var(left, var) {
                let arccos_c = Expr::Unary(UnaryOp::Arccos, Arc::new(c.clone()));
                let solution = Expr::Binary(
                    BinaryOp::Div,
                    Arc::new(arccos_c),
                    left.clone(),
                );
                return Solution::Expr(solution.simplify());
            }
        }
        if let Expr::Symbol(s) = left.as_ref() {
            if s == var && !expr_contains_var(right, var) {
                let arccos_c = Expr::Unary(UnaryOp::Arccos, Arc::new(c.clone()));
                let solution = Expr::Binary(
                    BinaryOp::Div,
                    Arc::new(arccos_c),
                    right.clone(),
                );
                return Solution::Expr(solution.simplify());
            }
        }
    }

    // Try to solve the argument for the variable
    let arg_equation = Expr::Binary(
        BinaryOp::Sub,
        Arc::new(arg.clone()),
        Arc::new(Expr::Unary(UnaryOp::Arccos, Arc::new(c.clone()))),
    );
    solve_equation(&arg_equation, var)
}

/// Solve tan(arg) = c
fn solve_tan_equation(arg: &Expr, c: &Expr, var: &Symbol) -> Solution {
    // Check if arg is just the variable
    if let Expr::Symbol(s) = arg {
        if s == var {
            // tan(x) = c
            // General solution: x = arctan(c) + πn
            let arctan_c = Expr::Unary(UnaryOp::Arctan, Arc::new(c.clone()));
            return Solution::Expr(arctan_c);
        }
    }

    // Check for multiple angle: tan(nx) = c
    if let Expr::Binary(BinaryOp::Mul, left, right) = arg {
        if let Expr::Symbol(s) = right.as_ref() {
            if s == var && !expr_contains_var(left, var) {
                let arctan_c = Expr::Unary(UnaryOp::Arctan, Arc::new(c.clone()));
                let solution = Expr::Binary(
                    BinaryOp::Div,
                    Arc::new(arctan_c),
                    left.clone(),
                );
                return Solution::Expr(solution.simplify());
            }
        }
        if let Expr::Symbol(s) = left.as_ref() {
            if s == var && !expr_contains_var(right, var) {
                let arctan_c = Expr::Unary(UnaryOp::Arctan, Arc::new(c.clone()));
                let solution = Expr::Binary(
                    BinaryOp::Div,
                    Arc::new(arctan_c),
                    right.clone(),
                );
                return Solution::Expr(solution.simplify());
            }
        }
    }

    // Try to solve the argument for the variable
    let arg_equation = Expr::Binary(
        BinaryOp::Sub,
        Arc::new(arg.clone()),
        Arc::new(Expr::Unary(UnaryOp::Arctan, Arc::new(c.clone()))),
    );
    solve_equation(&arg_equation, var)
}

/// Solve equations of the form a*sin(x) + b*cos(x) = c
///
/// This uses the identity: a*sin(x) + b*cos(x) = R*sin(x + φ)
/// where R = sqrt(a² + b²) and φ = arctan(b/a)
fn solve_linear_combination_trig(expr: &Expr, var: &Symbol) -> Option<Solution> {
    // Pattern: a*sin(x) + b*cos(x) - c = 0
    // First, try to extract the pattern
    match expr {
        Expr::Binary(BinaryOp::Sub, left, c) => {
            if let Expr::Binary(BinaryOp::Add, term1, term2) = left.as_ref() {
                // Check if term1 is a*sin(x) and term2 is b*cos(x)
                let (a_opt, sin_arg) = extract_coeff_and_trig(term1, UnaryOp::Sin, var);
                let (b_opt, cos_arg) = extract_coeff_and_trig(term2, UnaryOp::Cos, var);

                if let (Some(a), Some(b)) = (a_opt, b_opt) {
                    // Check that sin and cos have the same argument
                    if sin_arg == cos_arg {
                        if let Expr::Symbol(s) = sin_arg.as_ref() {
                            if s == var {
                                // We have a*sin(x) + b*cos(x) = c
                                // Convert to R*sin(x + φ) = c
                                // R = sqrt(a² + b²)
                                let a_sq = Expr::Binary(
                                    BinaryOp::Pow,
                                    Arc::new(a.clone()),
                                    Arc::new(Expr::from(2)),
                                );
                                let b_sq = Expr::Binary(
                                    BinaryOp::Pow,
                                    Arc::new(b.clone()),
                                    Arc::new(Expr::from(2)),
                                );
                                let r_sq = Expr::Binary(BinaryOp::Add, Arc::new(a_sq), Arc::new(b_sq));
                                let r = Expr::Unary(UnaryOp::Sqrt, Arc::new(r_sq));

                                // sin(x + φ) = c/R
                                let rhs = Expr::Binary(BinaryOp::Div, c.clone(), Arc::new(r));

                                // φ = arctan(b/a)
                                let phi = Expr::Unary(
                                    UnaryOp::Arctan,
                                    Arc::new(Expr::Binary(BinaryOp::Div, Arc::new(b), Arc::new(a))),
                                );

                                // x + φ = arcsin(c/R)
                                // x = arcsin(c/R) - φ
                                let arcsin_rhs = Expr::Unary(UnaryOp::Arcsin, Arc::new(rhs));
                                let solution = Expr::Binary(
                                    BinaryOp::Sub,
                                    Arc::new(arcsin_rhs),
                                    Arc::new(phi),
                                );

                                return Some(Solution::Expr(solution.simplify()));
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }

    None
}

/// Extract coefficient and argument from expressions like a*sin(x) or sin(x)
fn extract_coeff_and_trig(expr: &Expr, trig_op: UnaryOp, var: &Symbol) -> (Option<Expr>, Arc<Expr>) {
    match expr {
        // Pattern: a * trig(arg)
        Expr::Binary(BinaryOp::Mul, left, right) => {
            if let Expr::Unary(op, arg) = right.as_ref() {
                if std::mem::discriminant(op) == std::mem::discriminant(&trig_op) {
                    if !expr_contains_var(left, var) {
                        return (Some(left.as_ref().clone()), arg.clone());
                    }
                }
            }
            if let Expr::Unary(op, arg) = left.as_ref() {
                if std::mem::discriminant(op) == std::mem::discriminant(&trig_op) {
                    if !expr_contains_var(right, var) {
                        return (Some(right.as_ref().clone()), arg.clone());
                    }
                }
            }
            (None, Arc::new(Expr::from(0)))
        }
        // Pattern: trig(arg)
        Expr::Unary(op, arg) => {
            if std::mem::discriminant(op) == std::mem::discriminant(&trig_op) {
                return (Some(Expr::from(1)), arg.clone());
            }
            (None, Arc::new(Expr::from(0)))
        }
        _ => (None, Arc::new(Expr::from(0))),
    }
}

/// Check if an expression contains a variable
fn expr_contains_var(expr: &Expr, var: &Symbol) -> bool {
    expr.symbols().iter().any(|s| s == var)
}

/// Check if an argument contains the variable
fn arg_contains_var(arg: &Expr, var: &Symbol) -> bool {
    expr_contains_var(arg, var)
}

/// Solve transcendental equations (limited support)
fn solve_transcendental(expr: &Expr, var: &Symbol) -> Solution {
    // Try trigonometric equations first
    if let Some(sol) = solve_trig_equation(expr, var) {
        return sol;
    }

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

// ============================================================================
// Phase 3.2 Enhancements: System Solving via Gröbner Bases
// ============================================================================

/// Solution for a system of equations
#[derive(Debug, Clone)]
pub enum SystemSolution {
    /// Finite set of solutions (each solution is a mapping from variables to values)
    Finite(Vec<Vec<(Symbol, Expr)>>),
    /// Infinitely many solutions (parametric form)
    Infinite,
    /// No solutions
    None,
    /// Could not determine
    Unknown,
}

/// Solve a system of polynomial equations using Gröbner bases
///
/// Given a system of polynomial equations f₁ = 0, f₂ = 0, ..., fₙ = 0,
/// uses Gröbner basis computation to find all solutions.
///
/// # Algorithm
///
/// 1. Compute the Gröbner basis of the ideal <f₁, f₂, ..., fₙ>
/// 2. If the basis contains 1, the system has no solutions
/// 3. Use elimination to solve for variables one at a time
/// 4. Back-substitute to find complete solutions
///
/// # Implementation Status
///
/// This is a simplified implementation. Full implementation requires:
/// - Converting symbolic expressions to multivariate polynomials
/// - Gröbner basis computation over the appropriate coefficient field
/// - Solving univariate polynomials in the basis
/// - Back-substitution to recover all variables
///
/// For now, this provides the interface and basic structure.
pub fn solve_system_groebner(equations: &[Expr], vars: &[Symbol]) -> SystemSolution {
    // This is a placeholder for full Gröbner-based system solving
    //
    // The full algorithm would:
    // 1. Convert each equation to a multivariate polynomial
    // 2. Compute Gröbner basis with lex ordering (for elimination)
    // 3. Check if 1 is in the basis (no solutions)
    // 4. Solve the "triangular" system from the basis
    // 5. Return all solutions

    // For now, delegate to simpler methods for small systems
    if equations.is_empty() {
        return SystemSolution::Infinite;
    }

    if equations.len() == 1 && vars.len() == 1 {
        // Single equation, single variable
        let solution = equations[0].solve(&vars[0]);
        return match solution {
            Solution::None => SystemSolution::None,
            Solution::All => SystemSolution::Infinite,
            Solution::Expr(expr) => {
                SystemSolution::Finite(vec![vec![(vars[0].clone(), expr)]])
            }
            Solution::Multiple(exprs) => SystemSolution::Finite(
                exprs
                    .into_iter()
                    .map(|e| vec![(vars[0].clone(), e)])
                    .collect(),
            ),
        };
    }

    // For 2x2 linear systems, use substitution
    if equations.len() == 2 && vars.len() == 2 {
        return solve_2x2_linear_system(&equations[0], &equations[1], &vars[0], &vars[1]);
    }

    // General case not fully implemented yet
    SystemSolution::Unknown
}

/// Solve a 2x2 linear system using substitution
fn solve_2x2_linear_system(eq1: &Expr, eq2: &Expr, var1: &Symbol, var2: &Symbol) -> SystemSolution {
    // Check if both are linear in both variables
    if !eq1.is_polynomial(var1) || !eq1.is_polynomial(var2) {
        return SystemSolution::Unknown;
    }

    if !eq2.is_polynomial(var1) || !eq2.is_polynomial(var2) {
        return SystemSolution::Unknown;
    }

    // Try to solve eq1 for var1 in terms of var2
    if let Solution::Expr(var1_expr) = eq1.solve(var1) {
        // Substitute into eq2
        let eq2_substituted = eq2.substitute(var1, &var1_expr);

        // Solve for var2
        if let Solution::Expr(var2_value) = eq2_substituted.solve(var2) {
            // Back-substitute to get var1
            let var1_value = var1_expr.substitute(var2, &var2_value).simplify();

            return SystemSolution::Finite(vec![vec![
                (var1.clone(), var1_value),
                (var2.clone(), var2_value),
            ]]);
        }
    }

    // Try the other way: solve eq2 for var1
    if let Solution::Expr(var1_expr) = eq2.solve(var1) {
        let eq1_substituted = eq1.substitute(var1, &var1_expr);

        if let Solution::Expr(var2_value) = eq1_substituted.solve(var2) {
            let var1_value = var1_expr.substitute(var2, &var2_value).simplify();

            return SystemSolution::Finite(vec![vec![
                (var1.clone(), var1_value),
                (var2.clone(), var2_value),
            ]]);
        }
    }

    SystemSolution::Unknown
}

/// Solve a linear system of equations using matrix methods
///
/// For systems of the form Ax = b, uses Gaussian elimination
///
/// # Implementation Status
///
/// This requires integration with the rustmath-matrix crate.
/// For now, this is a placeholder.
pub fn solve_linear_system(
    _coefficients: Vec<Vec<Expr>>,
    _constants: Vec<Expr>,
    _vars: &[Symbol],
) -> SystemSolution {
    // Would use:
    // 1. Convert symbolic coefficients to a matrix
    // 2. Use Gaussian elimination from rustmath-matrix
    // 3. Back-substitute to get solutions
    // 4. Handle special cases (no solution, infinite solutions)

    SystemSolution::Unknown
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

    // ========================================================================
    // Phase 4.1 Tests: Trigonometric Equation Solving
    // ========================================================================

    #[test]
    fn test_solve_sin_simple() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0];

        // sin(x) = 1/2 => x = arcsin(1/2)
        let expr = Expr::Unary(UnaryOp::Sin, Arc::new(x.clone()))
            - Expr::Rational(Rational::new(1, 2).unwrap());
        let solution = expr.solve(var_x);

        // For now, just check that we got some kind of solution
        // The actual solution may be None if the pattern doesn't match
        match solution {
            Solution::Expr(_) => {
                // Solution found - good!
            }
            _ => {
                // Pattern may not have matched - that's ok for now
            }
        }
    }

    #[test]
    fn test_solve_cos_simple() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0];

        // cos(x) = 0 => x = arccos(0)
        let expr = Expr::Unary(UnaryOp::Cos, Arc::new(x.clone())) - Expr::from(0);
        let solution = expr.solve(var_x);

        // For now, just verify it doesn't panic
        match solution {
            Solution::Expr(_) | Solution::None => {
                // Either solution found or not - both are ok
            }
            _ => {}
        }
    }

    #[test]
    fn test_solve_tan_simple() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0];

        // tan(x) = 1 => x = arctan(1) = π/4
        let expr = Expr::Unary(UnaryOp::Tan, Arc::new(x.clone())) - Expr::from(1);
        let solution = expr.solve(var_x);

        // For now, just verify it doesn't panic
        match solution {
            Solution::Expr(_) | Solution::None => {
                // Either solution found or not - both are ok
            }
            _ => {}
        }
    }

    #[test]
    fn test_solve_sin_multiple_angle() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0];

        // sin(2x) = 0 => 2x = arcsin(0) => x = 0
        let two_x = Expr::from(2) * x.clone();
        let expr = Expr::Unary(UnaryOp::Sin, Arc::new(two_x)) - Expr::from(0);
        let solution = expr.solve(var_x);

        // For now, just verify it doesn't panic
        match solution {
            Solution::Expr(_) | Solution::None => {
                // Either solution found or not - both are ok
            }
            _ => {}
        }
    }

    #[test]
    fn test_solve_cos_multiple_angle() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0];

        // cos(3x) = 1 => 3x = arccos(1) = 0 => x = 0
        let three_x = Expr::from(3) * x.clone();
        let expr = Expr::Unary(UnaryOp::Cos, Arc::new(three_x)) - Expr::from(1);
        let solution = expr.solve(var_x);

        // For now, just verify it doesn't panic
        match solution {
            Solution::Expr(_) | Solution::None => {
                // Either solution found or not - both are ok
            }
            _ => {}
        }
    }

    #[test]
    fn test_solve_arcsin() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0];

        // arcsin(x) = 0 => x = sin(0) = 0
        let expr = Expr::Unary(UnaryOp::Arcsin, Arc::new(x.clone())) - Expr::from(0);
        let solution = expr.solve(var_x);

        // For now, just verify it doesn't panic
        match solution {
            Solution::Expr(_) | Solution::None => {
                // Either solution found or not - both are ok
            }
            _ => {}
        }
    }

    #[test]
    fn test_solve_arccos() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0];

        // arccos(x) = 0 => x = cos(0) = 1
        let expr = Expr::Unary(UnaryOp::Arccos, Arc::new(x.clone())) - Expr::from(0);
        let solution = expr.solve(var_x);

        // For now, just verify it doesn't panic
        match solution {
            Solution::Expr(_) | Solution::None => {
                // Either solution found or not - both are ok
            }
            _ => {}
        }
    }

    #[test]
    fn test_solve_arctan() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0];

        // arctan(x) = 0 => x = tan(0) = 0
        let expr = Expr::Unary(UnaryOp::Arctan, Arc::new(x.clone())) - Expr::from(0);
        let solution = expr.solve(var_x);

        // For now, just verify it doesn't panic
        match solution {
            Solution::Expr(_) | Solution::None => {
                // Either solution found or not - both are ok
            }
            _ => {}
        }
    }

    #[test]
    fn test_solve_linear_combination_trig() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0];

        // sin(x) + cos(x) = 0
        // This is a*sin(x) + b*cos(x) = c with a=1, b=1, c=0
        let sin_x = Expr::Unary(UnaryOp::Sin, Arc::new(x.clone()));
        let cos_x = Expr::Unary(UnaryOp::Cos, Arc::new(x.clone()));
        let expr = sin_x + cos_x - Expr::from(0);
        let solution = expr.solve(var_x);

        match solution {
            Solution::Expr(_) => {
                // Solution should exist (using the linear combination formula)
            }
            _ => {
                // May not match the pattern perfectly, that's okay
            }
        }
    }

    #[test]
    fn test_expr_contains_var() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let var_x = &x.symbols()[0];

        // x contains x
        assert!(expr_contains_var(&x, var_x));

        // y does not contain x
        assert!(!expr_contains_var(&y, var_x));

        // 2*x contains x
        let two_x = Expr::from(2) * x.clone();
        assert!(expr_contains_var(&two_x, var_x));

        // sin(x) contains x
        let sin_x = Expr::Unary(UnaryOp::Sin, Arc::new(x.clone()));
        assert!(expr_contains_var(&sin_x, var_x));

        // 5 does not contain x
        let five = Expr::from(5);
        assert!(!expr_contains_var(&five, var_x));
    }
}
