//! Numerical optimization and root finding
//!
//! This module provides implementations of numerical optimization and root-finding algorithms
//! compatible with SageMath's `sage.numerical.optimize` interface.
//!
//! # Algorithms
//!
//! ## Root Finding
//! - **Bisection**: Simple, robust, guaranteed convergence when f(a) and f(b) have opposite signs
//! - **Brent's Method**: Combines bisection, secant, and inverse quadratic interpolation for fast convergence
//!
//! ## Optimization
//! - **Golden Section Search**: Derivative-free method for unimodal functions
//! - **Least Squares**: Linear and nonlinear curve fitting
//!
//! # Convergence Criteria
//!
//! All algorithms use multiple convergence criteria:
//! - **Absolute tolerance** (`tol`): |x_new - x_old| < tol
//! - **Function tolerance**: |f(x)| < ftol (for root finding)
//! - **Relative tolerance**: |x_new - x_old| / |x_old| < rtol (optional)
//! - **Maximum iterations**: Prevents infinite loops
//!
//! Default tolerances:
//! - Root finding: tol = 1e-10, max_iter = 100
//! - Optimization: tol = 1e-8, max_iter = 500

use std::f64;

/// Result of a root-finding operation
#[derive(Clone, Debug)]
pub struct RootResult {
    /// The root value
    pub root: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Estimated error
    pub error: f64,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// Result of an optimization operation
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    /// The optimal x value
    pub x: f64,
    /// The function value at the optimum
    pub fx: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Estimated error in x
    pub error: f64,
}

/// Result of a curve fitting operation
#[derive(Clone, Debug)]
pub struct FitResult {
    /// The fitted parameters
    pub params: Vec<f64>,
    /// Residual sum of squares
    pub residual: f64,
    /// Whether the fit converged
    pub converged: bool,
    /// Number of iterations (for nonlinear fits)
    pub iterations: usize,
}

/// Find a root of f in the interval [a, b] using Brent's method
///
/// Brent's method is a robust root-finding algorithm that combines the reliability
/// of bisection with the speed of more advanced methods (secant and inverse quadratic
/// interpolation). It is guaranteed to converge for continuous functions where
/// f(a) and f(b) have opposite signs.
///
/// # Arguments
///
/// * `f` - The function to find the root of
/// * `a` - Left endpoint of the interval
/// * `b` - Right endpoint of the interval
///
/// # Convergence
///
/// The algorithm converges when:
/// - |b - a| < 2 * (2 * |x| * EPSILON + tol/2), where EPSILON is machine epsilon
/// - OR max_iter iterations are reached
///
/// # Returns
///
/// `Some(RootResult)` if the function values at `a` and `b` have opposite signs,
/// `None` otherwise.
///
/// # Examples
///
/// ```
/// use rustmath_numerical::optimize::find_root;
///
/// // Find root of f(x) = x^2 - 2 (i.e., sqrt(2))
/// let f = |x: f64| x * x - 2.0;
/// let result = find_root(f, 0.0, 2.0).unwrap();
/// assert!((result.root - 2.0_f64.sqrt()).abs() < 1e-9);
/// ```
pub fn find_root<F>(f: F, a: f64, b: f64) -> Option<RootResult>
where
    F: Fn(f64) -> f64,
{
    brent_root(f, a, b, 1e-10, 100)
}

/// Brent's method for root finding
///
/// A hybrid algorithm that combines bisection, secant method, and inverse
/// quadratic interpolation. It maintains a bracketing interval and chooses
/// the best method at each step.
///
/// # Algorithm
///
/// 1. Start with interval [a, b] where f(a) and f(b) have opposite signs
/// 2. At each iteration, try inverse quadratic interpolation
/// 3. If that fails or doesn't improve enough, try secant method
/// 4. If that fails, fall back to bisection
/// 5. Continue until convergence or max iterations
///
/// # Convergence Rate
///
/// Superlinear convergence (approximately 1.6) near the root, making it much
/// faster than bisection while maintaining robustness.
pub fn brent_root<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Option<RootResult>
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    let mut fb = f(b);

    // Check if root is bracketed
    if fa * fb > 0.0 {
        return None;
    }

    // Ensure |f(b)| <= |f(a)|
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    let mut iterations = 0;

    while iterations < max_iter {
        // Check convergence
        let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * tol;
        let m = 0.5 * (c - b);

        if fb.abs() < tol || m.abs() <= tol1 {
            return Some(RootResult {
                root: b,
                iterations,
                error: fb.abs(),
                converged: true,
            });
        }

        // Try inverse quadratic interpolation or secant method
        let mut p;
        let mut q;
        let s;

        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            // Inverse quadratic interpolation
            s = fb / fa;

            if (a - c).abs() < f64::EPSILON {
                // Secant method
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                // Inverse quadratic interpolation
                q = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }

            // Ensure p is positive
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }

            // Check if interpolation is acceptable
            let min1 = 3.0 * m * q - (tol1 * q).abs();
            let min2 = (e * q).abs();

            if 2.0 * p < min1.min(min2) {
                // Accept interpolation
                e = d;
                d = p / q;
            } else {
                // Interpolation failed, use bisection
                d = m;
                e = d;
            }
        } else {
            // Bisection
            d = m;
            e = d;
        }

        // Update a
        a = b;
        fa = fb;

        // Update b
        if d.abs() > tol1 {
            b += d;
        } else if m > 0.0 {
            b += tol1;
        } else {
            b -= tol1;
        }

        fb = f(b);

        // Ensure root remains bracketed
        if (fb > 0.0 && fc > 0.0) || (fb <= 0.0 && fc <= 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }

        // Ensure |f(b)| <= |f(a)|
        if fc.abs() < fb.abs() {
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = a;
            fc = fa;
        }

        iterations += 1;
    }

    Some(RootResult {
        root: b,
        iterations,
        error: fb.abs(),
        converged: false,
    })
}

/// Find a local minimum of f in the interval [a, b] using golden section search
///
/// Golden section search is a derivative-free optimization method for unimodal
/// functions (functions with a single minimum in the interval). It uses the
/// golden ratio to efficiently narrow the search interval.
///
/// # Arguments
///
/// * `f` - The function to minimize (must be unimodal in [a, b])
/// * `a` - Left endpoint of the interval
/// * `b` - Right endpoint of the interval
///
/// # Convergence
///
/// The algorithm converges when the interval width |b - a| < tol.
/// The golden ratio ensures that each iteration reduces the interval by
/// approximately 38.2%, giving linear convergence.
///
/// # Examples
///
/// ```
/// use rustmath_numerical::optimize::find_local_minimum;
///
/// // Minimize f(x) = (x - 3)^2, minimum at x = 3
/// let f = |x: f64| (x - 3.0) * (x - 3.0);
/// let result = find_local_minimum(f, 0.0, 5.0);
/// assert!((result.x - 3.0).abs() < 1e-7);
/// ```
pub fn find_local_minimum<F>(f: F, a: f64, b: f64) -> OptimizationResult
where
    F: Fn(f64) -> f64,
{
    golden_section_search(f, a, b, 1e-8, 500)
}

/// Golden section search for function minimization
///
/// Uses the golden ratio φ = (1 + √5) / 2 ≈ 1.618 to optimally partition
/// the search interval. At each iteration, evaluates the function at two
/// interior points and eliminates the portion of the interval that cannot
/// contain the minimum.
///
/// # Algorithm
///
/// 1. Compute interior points: x1 = a + (1-φ⁻¹)(b-a), x2 = a + φ⁻¹(b-a)
/// 2. Evaluate f(x1) and f(x2)
/// 3. If f(x1) < f(x2), minimum is in [a, x2]; otherwise in [x1, b]
/// 4. Repeat until |b - a| < tol
///
/// # Convergence
///
/// Linear convergence with rate φ⁻¹ ≈ 0.618. Requires no derivatives.
pub fn golden_section_search<F>(
    f: F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> OptimizationResult
where
    F: Fn(f64) -> f64,
{
    #[allow(dead_code)]
    const PHI: f64 = 1.618033988749895; // Golden ratio
    const RESPHI: f64 = 0.6180339887498949; // 1/φ = φ - 1

    let mut x1 = a + RESPHI * (b - a);
    let mut x2 = b - RESPHI * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    let mut iterations = 0;

    while (b - a).abs() > tol && iterations < max_iter {
        if f1 < f2 {
            // Minimum is in [a, x2]
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + RESPHI * (b - a);
            f1 = f(x1);
        } else {
            // Minimum is in [x1, b]
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - RESPHI * (b - a);
            f2 = f(x2);
        }

        iterations += 1;
    }

    let x_min = if f1 < f2 { x1 } else { x2 };
    let fx_min = if f1 < f2 { f1 } else { f2 };

    OptimizationResult {
        x: x_min,
        fx: fx_min,
        iterations,
        converged: (b - a).abs() <= tol,
        error: (b - a).abs(),
    }
}

/// Find a local maximum of f in the interval [a, b]
///
/// This is a convenience wrapper around `find_local_minimum` that negates
/// the function to convert maximization to minimization.
///
/// # Examples
///
/// ```
/// use rustmath_numerical::optimize::find_local_maximum;
///
/// // Maximize f(x) = -(x - 2)^2 + 5, maximum at x = 2
/// let f = |x: f64| -(x - 2.0) * (x - 2.0) + 5.0;
/// let result = find_local_maximum(f, 0.0, 4.0);
/// assert!((result.x - 2.0).abs() < 1e-7);
/// ```
pub fn find_local_maximum<F>(f: F, a: f64, b: f64) -> OptimizationResult
where
    F: Fn(f64) -> f64,
{
    let neg_f = |x: f64| -f(x);
    let mut result = find_local_minimum(neg_f, a, b);
    result.fx = -result.fx; // Restore original function value
    result
}

/// Fit a model to data using linear least squares
///
/// Solves the linear least squares problem: min ||y - X*params||²
/// where X is the design matrix constructed by evaluating basis functions
/// on the input data points.
///
/// # Arguments
///
/// * `data` - Slice of (x, y) data points
/// * `model` - Vector of basis functions (e.g., [|x| 1.0, |x| x, |x| x*x] for quadratic)
///
/// # Algorithm
///
/// 1. Construct design matrix X where X[i][j] = model[j](x[i])
/// 2. Solve normal equations: X^T X params = X^T y
/// 3. Uses simple Gaussian elimination (consider using QR decomposition for numerical stability)
///
/// # Returns
///
/// `FitResult` containing the fitted parameters and residual sum of squares.
///
/// # Examples
///
/// ```
/// use rustmath_numerical::optimize::find_fit;
///
/// // Fit a line y = a + b*x to data
/// let data = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)];
/// let model = vec![Box::new(|_: f64| 1.0) as Box<dyn Fn(f64) -> f64>,
///                  Box::new(|x: f64| x) as Box<dyn Fn(f64) -> f64>];
/// let result = find_fit(&data, &model);
/// // Should fit y = 1 + 2x
/// assert!((result.params[0] - 1.0).abs() < 1e-10);
/// assert!((result.params[1] - 2.0).abs() < 1e-10);
/// ```
pub fn find_fit<F>(data: &[(f64, f64)], model: &[F]) -> FitResult
where
    F: Fn(f64) -> f64,
{
    let n = data.len();
    let m = model.len();

    if n < m {
        return FitResult {
            params: vec![0.0; m],
            residual: f64::INFINITY,
            converged: false,
            iterations: 0,
        };
    }

    // Construct design matrix X (n x m)
    let mut x = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            x[i][j] = model[j](data[i].0);
        }
    }

    // Construct y vector
    let y: Vec<f64> = data.iter().map(|&(_, yi)| yi).collect();

    // Solve normal equations: X^T X params = X^T y
    // Compute X^T X (m x m)
    let mut xtx = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x[k][i] * x[k][j];
            }
            xtx[i][j] = sum;
        }
    }

    // Compute X^T y (m x 1)
    let mut xty = vec![0.0; m];
    for i in 0..m {
        let mut sum = 0.0;
        for k in 0..n {
            sum += x[k][i] * y[k];
        }
        xty[i] = sum;
    }

    // Solve xtx * params = xty using Gaussian elimination
    let params = match solve_linear_system(&xtx, &xty) {
        Some(p) => p,
        None => {
            return FitResult {
                params: vec![0.0; m],
                residual: f64::INFINITY,
                converged: false,
                iterations: 0,
            }
        }
    };

    // Compute residual sum of squares
    let mut residual = 0.0;
    for i in 0..n {
        let mut pred = 0.0;
        for j in 0..m {
            pred += params[j] * x[i][j];
        }
        let err = y[i] - pred;
        residual += err * err;
    }

    FitResult {
        params,
        residual,
        converged: true,
        iterations: 1,
    }
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return None;
    }

    // Create augmented matrix [A|b]
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = aug[k][k].abs();
        for i in (k + 1)..n {
            if aug[i][k].abs() > max_val {
                max_val = aug[i][k].abs();
                max_row = i;
            }
        }

        // Check for singular matrix
        if max_val < 1e-14 {
            return None;
        }

        // Swap rows
        if max_row != k {
            aug.swap(k, max_row);
        }

        // Eliminate below pivot
        for i in (k + 1)..n {
            let factor = aug[i][k] / aug[k][k];
            for j in k..=n {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brent_root_sqrt2() {
        // f(x) = x^2 - 2, root at x = sqrt(2)
        let f = |x: f64| x * x - 2.0;
        let result = find_root(f, 0.0, 2.0).unwrap();

        assert!((result.root - 2.0_f64.sqrt()).abs() < 1e-9);
        assert!(result.converged);
        println!("sqrt(2) found in {} iterations", result.iterations);
    }

    #[test]
    fn test_brent_root_cubic() {
        // f(x) = x^3 - x - 2, root near x = 1.52
        let f = |x: f64| x * x * x - x - 2.0;
        let result = find_root(f, 1.0, 2.0).unwrap();

        assert!(f(result.root).abs() < 1e-9);
        assert!(result.converged);
    }

    #[test]
    fn test_brent_root_transcendental() {
        // f(x) = cos(x) - x, root near x = 0.739
        let f = |x: f64| x.cos() - x;
        let result = find_root(f, 0.0, 1.0).unwrap();

        assert!(f(result.root).abs() < 1e-9);
        assert!(result.converged);
    }

    #[test]
    fn test_brent_root_no_bracket() {
        // f(x) = x^2 + 1, no real roots
        let f = |x: f64| x * x + 1.0;
        let result = find_root(f, 0.0, 2.0);

        assert!(result.is_none());
    }

    #[test]
    fn test_golden_section_quadratic() {
        // f(x) = (x - 3)^2, minimum at x = 3
        let f = |x: f64| (x - 3.0) * (x - 3.0);
        let result = find_local_minimum(f, 0.0, 5.0);

        assert!((result.x - 3.0).abs() < 1e-7);
        assert!(result.fx.abs() < 1e-10);
        assert!(result.converged);
        println!(
            "Quadratic minimum found in {} iterations",
            result.iterations
        );
    }

    #[test]
    fn test_golden_section_quartic() {
        // f(x) = x^4 - 14x^3 + 60x^2 - 70x, minimum near x = 0.6
        let f = |x: f64| x * x * x * x - 14.0 * x * x * x + 60.0 * x * x - 70.0 * x;
        let result = find_local_minimum(f, 0.0, 2.0);

        // Check that we found a local minimum
        let h = 1e-6;
        assert!(f(result.x - h) >= result.fx);
        assert!(f(result.x + h) >= result.fx);
        assert!(result.converged);
    }

    #[test]
    fn test_find_local_maximum() {
        // f(x) = -(x - 2)^2 + 5, maximum at x = 2, value = 5
        let f = |x: f64| -(x - 2.0) * (x - 2.0) + 5.0;
        let result = find_local_maximum(f, 0.0, 4.0);

        assert!((result.x - 2.0).abs() < 1e-7);
        assert!((result.fx - 5.0).abs() < 1e-10);
        assert!(result.converged);
    }

    #[test]
    fn test_find_maximum_sin() {
        // f(x) = sin(x), maximum at x = π/2 in [0, π]
        let f = |x: f64| x.sin();
        let result = find_local_maximum(f, 0.0, std::f64::consts::PI);

        assert!((result.x - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((result.fx - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_fit_linear() {
        // Fit y = 1 + 2x to exact data
        let data = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0)];
        let model: Vec<Box<dyn Fn(f64) -> f64>> = vec![
            Box::new(|_: f64| 1.0),  // Constant term
            Box::new(|x: f64| x),    // Linear term
        ];

        let result = find_fit(&data, &model);

        assert!(result.converged);
        assert!((result.params[0] - 1.0).abs() < 1e-10);
        assert!((result.params[1] - 2.0).abs() < 1e-10);
        assert!(result.residual < 1e-20);
        println!("Linear fit: y = {} + {}x", result.params[0], result.params[1]);
    }

    #[test]
    fn test_find_fit_quadratic() {
        // Fit y = 2 - 3x + x^2 to exact data
        let data = vec![
            (0.0, 2.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 2.0),
            (4.0, 6.0),
        ];
        let model: Vec<Box<dyn Fn(f64) -> f64>> = vec![
            Box::new(|_: f64| 1.0),       // Constant
            Box::new(|x: f64| x),         // Linear
            Box::new(|x: f64| x * x),     // Quadratic
        ];

        let result = find_fit(&data, &model);

        assert!(result.converged);
        assert!((result.params[0] - 2.0).abs() < 1e-10);
        assert!((result.params[1] - (-3.0)).abs() < 1e-10);
        assert!((result.params[2] - 1.0).abs() < 1e-10);
        assert!(result.residual < 1e-20);
    }

    #[test]
    fn test_find_fit_exponential_linearized() {
        // Fit exponential y = a * exp(b*x) via linearization: ln(y) = ln(a) + b*x
        // Using points from y = 2 * exp(0.5*x)
        let data = vec![
            (0.0, 2.0_f64.ln()),
            (1.0, (2.0 * (0.5_f64).exp()).ln()),
            (2.0, (2.0 * (1.0_f64).exp()).ln()),
            (3.0, (2.0 * (1.5_f64).exp()).ln()),
        ];

        let model: Vec<Box<dyn Fn(f64) -> f64>> = vec![
            Box::new(|_: f64| 1.0),
            Box::new(|x: f64| x),
        ];

        let result = find_fit(&data, &model);

        assert!(result.converged);
        // ln(2) ≈ 0.693
        assert!((result.params[0] - 2.0_f64.ln()).abs() < 1e-10);
        assert!((result.params[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_find_fit_noisy_data() {
        // Fit to noisy linear data
        let data = vec![
            (0.0, 1.1),
            (1.0, 2.9),
            (2.0, 5.2),
            (3.0, 6.8),
            (4.0, 9.1),
        ];
        let model: Vec<Box<dyn Fn(f64) -> f64>> = vec![
            Box::new(|_: f64| 1.0),
            Box::new(|x: f64| x),
        ];

        let result = find_fit(&data, &model);

        assert!(result.converged);
        // Should be approximately y = 1 + 2x with some error
        assert!((result.params[0] - 1.0).abs() < 0.5);
        assert!((result.params[1] - 2.0).abs() < 0.5);
        assert!(result.residual > 0.0); // Non-zero residual for noisy data
        println!("Noisy fit: y = {} + {}x, residual = {}",
                 result.params[0], result.params[1], result.residual);
    }

    #[test]
    fn test_solve_linear_system() {
        // Test solving 2x2 system:
        // 2x + y = 5
        // x + 3y = 6
        // Solution: x = 2.076923..., y = 0.846153...
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 6.0];

        let x = solve_linear_system(&a, &b).unwrap();

        // Verify solution
        assert!((2.0 * x[0] + x[1] - 5.0).abs() < 1e-10);
        assert!((x[0] + 3.0 * x[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_3x3() {
        // Test 3x3 system
        let a = vec![
            vec![3.0, 2.0, -1.0],
            vec![2.0, -2.0, 4.0],
            vec![-1.0, 0.5, -1.0],
        ];
        let b = vec![1.0, -2.0, 0.0];

        let x = solve_linear_system(&a, &b).unwrap();

        // Verify solution
        for i in 0..3 {
            let mut sum = 0.0;
            for j in 0..3 {
                sum += a[i][j] * x[j];
            }
            assert!((sum - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_brent_vs_exact() {
        // Compare Brent's method accuracy with known mathematical constants

        // sqrt(3)
        let f = |x: f64| x * x - 3.0;
        let result = find_root(f, 1.0, 2.0).unwrap();
        assert!((result.root - 3.0_f64.sqrt()).abs() < 1e-14);

        // cbrt(7)
        let f = |x: f64| x * x * x - 7.0;
        let result = find_root(f, 1.0, 3.0).unwrap();
        assert!((result.root - 7.0_f64.cbrt()).abs() < 1e-14);
    }

    #[test]
    fn test_golden_section_convergence() {
        // Test that golden section achieves expected accuracy
        let f = |x: f64| (x - std::f64::consts::PI) * (x - std::f64::consts::PI);

        let result = golden_section_search(f, 0.0, 5.0, 1e-10, 1000);

        assert!((result.x - std::f64::consts::PI).abs() < 1e-10);
        assert!(result.converged);
    }
}
