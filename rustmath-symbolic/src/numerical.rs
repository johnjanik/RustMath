//! Numerical methods for calculus
//!
//! This module implements numerical integration and other numerical calculus methods.

use crate::expression::Expr;
use crate::symbol::Symbol;

/// Numerical integration result
#[derive(Debug, Clone, PartialEq)]
pub struct IntegrationResult {
    /// The computed integral value
    pub value: f64,
    /// Estimated error (if available)
    pub error: Option<f64>,
    /// Number of function evaluations
    pub evaluations: usize,
}

/// Numerical integration using the Trapezoidal rule
///
/// Approximates ∫[a,b] f(x) dx using trapezoids
///
/// # Arguments
///
/// * `expr` - The expression to integrate
/// * `var` - The variable of integration
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of subdivisions (more = more accurate)
pub fn trapezoidal(expr: &Expr, var: &Symbol, a: f64, b: f64, n: usize) -> IntegrationResult {
    let h = (b - a) / n as f64;
    let mut sum = 0.0;
    let mut evaluations = 0;

    // Evaluate at endpoints
    let fa = eval_at(expr, var, a);
    let fb = eval_at(expr, var, b);
    sum += (fa + fb) / 2.0;
    evaluations += 2;

    // Evaluate at interior points
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += eval_at(expr, var, x);
        evaluations += 1;
    }

    IntegrationResult {
        value: h * sum,
        error: None,
        evaluations,
    }
}

/// Numerical integration using Simpson's rule
///
/// Approximates ∫[a,b] f(x) dx using parabolic arcs
/// More accurate than trapezoidal for smooth functions
///
/// # Arguments
///
/// * `expr` - The expression to integrate
/// * `var` - The variable of integration
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of subdivisions (must be even)
pub fn simpson(expr: &Expr, var: &Symbol, a: f64, b: f64, n: usize) -> IntegrationResult {
    assert!(n % 2 == 0, "Simpson's rule requires even number of subdivisions");

    let h = (b - a) / n as f64;
    let mut sum = 0.0;
    let mut evaluations = 0;

    // First and last points
    sum += eval_at(expr, var, a);
    sum += eval_at(expr, var, b);
    evaluations += 2;

    // Odd points (coefficient 4)
    for i in (1..n).step_by(2) {
        let x = a + i as f64 * h;
        sum += 4.0 * eval_at(expr, var, x);
        evaluations += 1;
    }

    // Even points (coefficient 2)
    for i in (2..n).step_by(2) {
        let x = a + i as f64 * h;
        sum += 2.0 * eval_at(expr, var, x);
        evaluations += 1;
    }

    IntegrationResult {
        value: (h / 3.0) * sum,
        error: None,
        evaluations,
    }
}

/// Adaptive Simpson's rule with error estimation
///
/// Recursively subdivides intervals where error is large
///
/// # Arguments
///
/// * `expr` - The expression to integrate
/// * `var` - The variable of integration
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `tolerance` - Desired error tolerance
/// * `max_depth` - Maximum recursion depth
pub fn adaptive_simpson(
    expr: &Expr,
    var: &Symbol,
    a: f64,
    b: f64,
    tolerance: f64,
    max_depth: usize,
) -> IntegrationResult {
    let mut total_evaluations = 0;

    fn adaptive_helper(
        expr: &Expr,
        var: &Symbol,
        a: f64,
        b: f64,
        fa: f64,
        fb: f64,
        fc: f64,
        tolerance: f64,
        depth: usize,
        max_depth: usize,
        evaluations: &mut usize,
    ) -> f64 {
        let c = (a + b) / 2.0;
        let h = b - a;

        let fd = eval_at(expr, var, (a + c) / 2.0);
        let fe = eval_at(expr, var, (c + b) / 2.0);
        *evaluations += 2;

        let s_ab = (h / 6.0) * (fa + 4.0 * fc + fb);
        let s_ac = (h / 12.0) * (fa + 4.0 * fd + fc);
        let s_cb = (h / 12.0) * (fc + 4.0 * fe + fb);
        let s_total = s_ac + s_cb;

        let error = (s_total - s_ab).abs() / 15.0;

        if depth >= max_depth || error < tolerance {
            s_total + error // Include error correction
        } else {
            let left = adaptive_helper(
                expr,
                var,
                a,
                c,
                fa,
                fc,
                fd,
                tolerance / 2.0,
                depth + 1,
                max_depth,
                evaluations,
            );
            let right = adaptive_helper(
                expr,
                var,
                c,
                b,
                fc,
                fb,
                fe,
                tolerance / 2.0,
                depth + 1,
                max_depth,
                evaluations,
            );
            left + right
        }
    }

    let fa = eval_at(expr, var, a);
    let fb = eval_at(expr, var, b);
    let fc = eval_at(expr, var, (a + b) / 2.0);
    total_evaluations += 3;

    let value = adaptive_helper(
        expr,
        var,
        a,
        b,
        fa,
        fb,
        fc,
        tolerance,
        0,
        max_depth,
        &mut total_evaluations,
    );

    IntegrationResult {
        value,
        error: Some(tolerance),
        evaluations: total_evaluations,
    }
}

/// Gaussian quadrature (Gauss-Legendre with n=2)
///
/// Very accurate for polynomial integrands
///
/// # Arguments
///
/// * `expr` - The expression to integrate
/// * `var` - The variable of integration
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n_intervals` - Number of intervals to subdivide
pub fn gauss_legendre(
    expr: &Expr,
    var: &Symbol,
    a: f64,
    b: f64,
    n_intervals: usize,
) -> IntegrationResult {
    // Gauss-Legendre nodes and weights for n=2
    let nodes = [-1.0 / f64::sqrt(3.0), 1.0 / f64::sqrt(3.0)];
    let weights = [1.0, 1.0];

    let interval_width = (b - a) / n_intervals as f64;
    let mut sum = 0.0;
    let mut evaluations = 0;

    for i in 0..n_intervals {
        let left = a + i as f64 * interval_width;
        let right = left + interval_width;
        let mid = (left + right) / 2.0;
        let half_width = (right - left) / 2.0;

        for j in 0..2 {
            let x = mid + half_width * nodes[j];
            sum += weights[j] * eval_at(expr, var, x);
            evaluations += 1;
        }

        sum *= half_width;
    }

    IntegrationResult {
        value: sum,
        error: None,
        evaluations,
    }
}

/// Monte Carlo integration
///
/// Useful for high-dimensional integrals or irregular domains
///
/// # Arguments
///
/// * `expr` - The expression to integrate
/// * `var` - The variable of integration
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n_samples` - Number of random samples
pub fn monte_carlo(
    expr: &Expr,
    var: &Symbol,
    a: f64,
    b: f64,
    n_samples: usize,
) -> IntegrationResult {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_samples {
        let x = rng.gen_range(a..b);
        let fx = eval_at(expr, var, x);
        sum += fx;
        sum_sq += fx * fx;
    }

    let mean = sum / n_samples as f64;
    let variance = (sum_sq / n_samples as f64) - (mean * mean);
    let std_error = (variance / n_samples as f64).sqrt();

    let value = (b - a) * mean;
    let error = (b - a) * std_error;

    IntegrationResult {
        value,
        error: Some(error),
        evaluations: n_samples,
    }
}

/// Romberg integration
///
/// Extrapolation of trapezoidal rule for high accuracy
///
/// # Arguments
///
/// * `expr` - The expression to integrate
/// * `var` - The variable of integration
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `max_iterations` - Maximum number of iterations
/// * `tolerance` - Desired error tolerance
pub fn romberg(
    expr: &Expr,
    var: &Symbol,
    a: f64,
    b: f64,
    max_iterations: usize,
    tolerance: f64,
) -> IntegrationResult {
    let mut r = vec![vec![0.0; max_iterations]; max_iterations];
    let mut evaluations = 0;

    // First trapezoidal estimate
    let fa = eval_at(expr, var, a);
    let fb = eval_at(expr, var, b);
    r[0][0] = 0.5 * (b - a) * (fa + fb);
    evaluations += 2;

    for i in 1..max_iterations {
        // Trapezoidal rule with 2^i intervals
        let n = 1 << i; // 2^i
        let h = (b - a) / n as f64;
        let mut sum = 0.0;

        // Only evaluate at new points
        for j in (1..n).step_by(2) {
            let x = a + j as f64 * h;
            sum += eval_at(expr, var, x);
            evaluations += 1;
        }

        r[i][0] = 0.5 * r[i - 1][0] + h * sum;

        // Richardson extrapolation
        for j in 1..=i {
            let power = 4_f64.powi(j as i32);
            r[i][j] = (power * r[i][j - 1] - r[i - 1][j - 1]) / (power - 1.0);
        }

        // Check convergence
        if i > 0 && (r[i][i] - r[i - 1][i - 1]).abs() < tolerance {
            return IntegrationResult {
                value: r[i][i],
                error: Some((r[i][i] - r[i - 1][i - 1]).abs()),
                evaluations,
            };
        }
    }

    IntegrationResult {
        value: r[max_iterations - 1][max_iterations - 1],
        error: None,
        evaluations,
    }
}

/// Helper function to evaluate expression at a point
fn eval_at(expr: &Expr, var: &Symbol, x: f64) -> f64 {
    // Create a rational approximation of x
    // For simplicity, we multiply by 1000000 to handle up to 6 decimal places
    let scale = 1_000_000;
    let numerator = (x * scale as f64).round() as i64;

    use rustmath_rationals::Rational;
    let value = Expr::Rational(Rational::new(numerator, scale).unwrap_or(Rational::new(0, 1).unwrap()));
    let result = expr.substitute(var, &value);
    result.eval_float().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::Expr;

    #[test]
    fn test_trapezoidal_constant() {
        let x = Symbol::new("x");
        let expr = Expr::from(2); // ∫2 dx from 0 to 1 = 2

        let result = trapezoidal(&expr, &x, 0.0, 1.0, 10);
        assert!((result.value - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_trapezoidal_linear() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()); // ∫x dx from 0 to 1 = 0.5

        let result = trapezoidal(&expr, &x, 0.0, 1.0, 100);
        assert!((result.value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_simpson_quadratic() {
        let x = Symbol::new("x");
        // ∫x² dx from 0 to 1 = 1/3
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let result = simpson(&expr, &x, 0.0, 1.0, 100);
        assert!((result.value - (1.0 / 3.0)).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_simpson() {
        let x = Symbol::new("x");
        // ∫x² dx from 0 to 1 = 1/3
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let result = adaptive_simpson(&expr, &x, 0.0, 1.0, 1e-6, 10);
        assert!((result.value - (1.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_gauss_legendre() {
        let x = Symbol::new("x");
        // ∫x² dx from 0 to 1 = 1/3
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let result = gauss_legendre(&expr, &x, 0.0, 1.0, 10);
        assert!((result.value - (1.0 / 3.0)).abs() < 0.01);
    }

    #[test]
    fn test_monte_carlo() {
        let x = Symbol::new("x");
        let expr = Expr::from(1); // ∫1 dx from 0 to 2 = 2

        let result = monte_carlo(&expr, &x, 0.0, 2.0, 10000);
        // Monte Carlo has higher error
        assert!((result.value - 2.0).abs() < 0.1);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_romberg() {
        let x = Symbol::new("x");
        // ∫x² dx from 0 to 1 = 1/3
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let result = romberg(&expr, &x, 0.0, 1.0, 10, 1e-6);
        assert!((result.value - (1.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_trig_integration() {
        let x = Symbol::new("x");
        // ∫sin(x) dx from 0 to π ≈ 2
        let expr = Expr::Symbol(x.clone()).sin();

        let result = simpson(&expr, &x, 0.0, std::f64::consts::PI, 100);
        assert!((result.value - 2.0).abs() < 0.01);
    }
}
