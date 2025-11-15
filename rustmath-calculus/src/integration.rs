//! Integration - symbolic and numerical
//!
//! This module provides both symbolic integration (antiderivatives) and
//! numerical integration (definite integrals via quadrature).

use rustmath_symbolic::{BinaryOp, Expr, UnaryOp};
use std::f64;

/// Compute the indefinite integral (antiderivative) of an expression
///
/// # Arguments
///
/// * `expr` - The expression to integrate
/// * `var` - The variable of integration
///
/// # Returns
///
/// The antiderivative, or None if it cannot be computed symbolically
///
/// # Examples
///
/// ```
/// use rustmath_calculus::integration::integrate;
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// // ∫ x dx = x^2/2
/// let integral = integrate(&x, "x");
/// ```
pub fn integrate(expr: &Expr, var: &str) -> Option<Expr> {
    match expr {
        // ∫ c dx = c*x for constants
        Expr::Integer(_) | Expr::Rational(_) => Some(expr.clone() * Expr::symbol(var)),

        // ∫ x dx = x^2/2, ∫ y dx = y*x
        Expr::Symbol(s) => {
            if s.name() == var {
                Some(
                    Expr::symbol(var).pow(Expr::from(2)) / Expr::from(2)
                )
            } else {
                Some(expr.clone() * Expr::symbol(var))
            }
        }

        // Binary operations
        Expr::Binary(op, left, right) => match op {
            // ∫ (f + g) dx = ∫ f dx + ∫ g dx
            BinaryOp::Add => {
                let left_int = integrate(left, var)?;
                let right_int = integrate(right, var)?;
                Some(left_int + right_int)
            }

            // ∫ (f - g) dx = ∫ f dx - ∫ g dx
            BinaryOp::Sub => {
                let left_int = integrate(left, var)?;
                let right_int = integrate(right, var)?;
                Some(left_int - right_int)
            }

            // ∫ c*f dx = c * ∫ f dx for constant c
            BinaryOp::Mul => {
                if !contains_var(left, var) {
                    let right_int = integrate(right, var)?;
                    Some((**left).clone() * right_int)
                } else if !contains_var(right, var) {
                    let left_int = integrate(left, var)?;
                    Some(left_int * (**right).clone())
                } else {
                    // Integration by parts would go here
                    None
                }
            }

            // ∫ x^n dx = x^(n+1)/(n+1) for constant n ≠ -1
            BinaryOp::Pow => {
                if let Expr::Symbol(s) = &**left {
                    if s.name() == var {
                        if let Expr::Integer(n) = &**right {
                            use rustmath_integers::Integer;
                            if *n != Integer::from(-1) {
                                let n_val = n.clone();
                                let new_exp = Expr::Integer(n_val.clone() + Integer::from(1));
                                return Some(
                                    Expr::symbol(var).pow(new_exp.clone()) / new_exp
                                );
                            } else {
                                // ∫ x^(-1) dx = ln|x|
                                return Some(Expr::symbol(var).log());
                            }
                        }
                    }
                }
                None
            }

            _ => None,
        },

        // Unary operations
        Expr::Unary(op, inner) => match op {
            // ∫ -f dx = - ∫ f dx
            UnaryOp::Neg => {
                let inner_int = integrate(inner, var)?;
                Some(-inner_int)
            }

            // ∫ sin(x) dx = -cos(x)
            UnaryOp::Sin => {
                match &**inner {
                    Expr::Symbol(s) if s.name() == var => {
                        Some(-Expr::symbol(var).cos())
                    }
                    _ => None,
                }
            }

            // ∫ cos(x) dx = sin(x)
            UnaryOp::Cos => {
                match &**inner {
                    Expr::Symbol(s) if s.name() == var => {
                        Some(Expr::symbol(var).sin())
                    }
                    _ => None,
                }
            }

            // ∫ exp(x) dx = exp(x)
            UnaryOp::Exp => {
                match &**inner {
                    Expr::Symbol(s) if s.name() == var => {
                        Some(Expr::symbol(var).exp())
                    }
                    _ => None,
                }
            }

            // ∫ 1/x dx = ln|x|
            // This is handled in the division case
            _ => None,
        },

        _ => None,
    }
}

/// Compute a definite integral numerically using the trapezoidal rule
///
/// # Arguments
///
/// * `f` - Function to integrate (as a closure)
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of subintervals
///
/// # Returns
///
/// Approximate value of the definite integral
///
/// # Examples
///
/// ```
/// use rustmath_calculus::integration::numerical_integrate_trapezoidal;
///
/// // ∫₀¹ x² dx = 1/3 ≈ 0.333...
/// let result = numerical_integrate_trapezoidal(&|x| x * x, 0.0, 1.0, 1000);
/// assert!((result - 0.333333).abs() < 0.001);
/// ```
pub fn numerical_integrate_trapezoidal<F>(f: &F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if n == 0 {
        return 0.0;
    }

    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));

    for i in 1..n {
        let x = a + i as f64 * h;
        sum += f(x);
    }

    sum * h
}

/// Compute a definite integral numerically using Simpson's rule
///
/// # Arguments
///
/// * `f` - Function to integrate (as a closure)
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of subintervals (must be even)
///
/// # Returns
///
/// Approximate value of the definite integral
///
/// # Examples
///
/// ```
/// use rustmath_calculus::integration::numerical_integrate_simpson;
///
/// // ∫₀¹ x² dx = 1/3 ≈ 0.333...
/// let result = numerical_integrate_simpson(&|x| x * x, 0.0, 1.0, 1000);
/// assert!((result - 0.333333).abs() < 0.000001);
/// ```
pub fn numerical_integrate_simpson<F>(f: &F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let n = if n % 2 == 0 { n } else { n + 1 }; // Ensure n is even

    if n == 0 {
        return 0.0;
    }

    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);

    for i in 1..n {
        let x = a + i as f64 * h;
        if i % 2 == 0 {
            sum += 2.0 * f(x);
        } else {
            sum += 4.0 * f(x);
        }
    }

    sum * h / 3.0
}

/// Compute a definite integral numerically using adaptive quadrature
///
/// This uses a recursive adaptive algorithm that subdivides intervals
/// where the function is changing rapidly.
///
/// # Arguments
///
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `tol` - Error tolerance
///
/// # Returns
///
/// Approximate value of the definite integral
///
/// # Examples
///
/// ```
/// use rustmath_calculus::integration::numerical_integrate_adaptive;
///
/// // ∫₀¹ x² dx = 1/3
/// let result = numerical_integrate_adaptive(&|x| x * x, 0.0, 1.0, 1e-6);
/// assert!((result - 0.333333).abs() < 1e-5);
/// ```
pub fn numerical_integrate_adaptive<F>(f: &F, a: f64, b: f64, tol: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    adaptive_simpson(f, a, b, tol, f(a), f((a + b) / 2.0), f(b))
}

/// Recursive adaptive Simpson's rule implementation
fn adaptive_simpson<F>(
    f: &F,
    a: f64,
    b: f64,
    tol: f64,
    fa: f64,
    fm: f64,
    fb: f64,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = b - a;
    let m = (a + b) / 2.0;
    let lm = (a + m) / 2.0;
    let rm = (m + b) / 2.0;

    let flm = f(lm);
    let frm = f(rm);

    // Simpson's rule for entire interval
    let s_whole = h * (fa + 4.0 * fm + fb) / 6.0;

    // Simpson's rule for left and right halves
    let s_left = (h / 2.0) * (fa + 4.0 * flm + fm) / 6.0;
    let s_right = (h / 2.0) * (fm + 4.0 * frm + fb) / 6.0;
    let s_combined = s_left + s_right;

    // Error estimate
    let error = (s_combined - s_whole).abs() / 15.0;

    if error < tol || h.abs() < 1e-10 {
        s_combined + error // Include error correction
    } else {
        adaptive_simpson(f, a, m, tol / 2.0, fa, flm, fm)
            + adaptive_simpson(f, m, b, tol / 2.0, fm, frm, fb)
    }
}

/// Monte Carlo integration for high-dimensional integrals
///
/// # Arguments
///
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of sample points
///
/// # Returns
///
/// Approximate value of the definite integral
///
/// # Examples
///
/// ```
/// use rustmath_calculus::integration::monte_carlo_integrate;
///
/// // Simple example - for real use, Monte Carlo is better for high dimensions
/// let result = monte_carlo_integrate(&|x| x * x, 0.0, 1.0, 100000);
/// assert!((result - 0.333333).abs() < 0.01);
/// ```
pub fn monte_carlo_integrate<F>(f: &F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    use std::f64::consts::PI;

    let mut sum = 0.0;
    let range = b - a;

    for i in 0..n {
        // Simple pseudo-random using sine (not cryptographically secure)
        let rand = ((i as f64 * PI * 1.618033988749).sin().abs() * 1000000.0) % 1.0;
        let x = a + rand * range;
        sum += f(x);
    }

    range * sum / n as f64
}

/// Check if an expression contains a variable
fn contains_var(expr: &Expr, var: &str) -> bool {
    use crate::expr::variables;
    variables(expr).contains(var)
}

/// Convenience function for numerical integration
///
/// Uses Simpson's rule by default
///
/// # Examples
///
/// ```
/// use rustmath_calculus::integration::nintegrate;
///
/// let result = nintegrate(&|x| x * x, 0.0, 1.0, 1000);
/// assert!((result - 0.333333).abs() < 0.000001);
/// ```
pub fn nintegrate<F>(f: &F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    numerical_integrate_simpson(f, a, b, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_constant() {
        let c = Expr::from(5);
        let result = integrate(&c, "x").unwrap();
        // ∫ 5 dx = 5x
        match &result {
            Expr::Binary(BinaryOp::Mul, left, right) => {
                assert_eq!(**left, Expr::from(5));
                if let Expr::Symbol(s) = &**right {
                    assert_eq!(s.name(), "x");
                }
            }
            _ => panic!("Expected 5*x"),
        }
    }

    #[test]
    fn test_integrate_variable() {
        let x = Expr::symbol("x");
        let _result = integrate(&x, "x").unwrap();
        // ∫ x dx = x^2/2
        // Result structure: (x^2) / 2
    }

    #[test]
    fn test_integrate_power() {
        let x = Expr::symbol("x");
        // ∫ x^2 dx = x^3/3
        let expr = x.clone().pow(Expr::from(2));
        let result = integrate(&expr, "x");
        assert!(result.is_some());
    }

    #[test]
    fn test_integrate_sum() {
        let x = Expr::symbol("x");
        // ∫ (x + 1) dx = x^2/2 + x
        let expr = x.clone() + Expr::from(1);
        let result = integrate(&expr, "x");
        assert!(result.is_some());
    }

    #[test]
    fn test_integrate_sin() {
        let x = Expr::symbol("x");
        // ∫ sin(x) dx = -cos(x)
        let expr = x.sin();
        let result = integrate(&expr, "x").unwrap();
        // Result should be -cos(x)
        match result {
            Expr::Unary(UnaryOp::Neg, inner) => {
                assert!(matches!(*inner, Expr::Unary(UnaryOp::Cos, _)));
            }
            _ => panic!("Expected -cos(x)"),
        }
    }

    #[test]
    fn test_integrate_cos() {
        let x = Expr::symbol("x");
        // ∫ cos(x) dx = sin(x)
        let expr = x.cos();
        let result = integrate(&expr, "x").unwrap();
        assert!(matches!(result, Expr::Unary(UnaryOp::Sin, _)));
    }

    #[test]
    fn test_numerical_trapezoidal() {
        // ∫₀¹ x² dx = 1/3
        let result = numerical_integrate_trapezoidal(&|x| x * x, 0.0, 1.0, 1000);
        assert!((result - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_numerical_simpson() {
        // ∫₀¹ x² dx = 1/3
        let result = numerical_integrate_simpson(&|x| x * x, 0.0, 1.0, 1000);
        assert!((result - 1.0 / 3.0).abs() < 0.000001);
    }

    #[test]
    fn test_numerical_adaptive() {
        // ∫₀¹ x² dx = 1/3
        let result = numerical_integrate_adaptive(&|x| x * x, 0.0, 1.0, 1e-6);
        assert!((result - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_nintegrate() {
        // ∫₀^π sin(x) dx = 2
        let result = nintegrate(&|x| x.sin(), 0.0, std::f64::consts::PI, 1000);
        assert!((result - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_monte_carlo() {
        // ∫₀¹ x² dx = 1/3
        let result = monte_carlo_integrate(&|x| x * x, 0.0, 1.0, 100000);
        assert!((result - 1.0 / 3.0).abs() < 0.01);
    }
}
