//! Optimization methods

/// Result of an optimization operation
#[derive(Clone, Debug)]
pub struct OptResult {
    pub x: f64,
    pub fx: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Minimize a function starting from x0
pub fn minimize<F>(f: F, x0: f64) -> Option<OptResult>
where
    F: Fn(f64) -> f64,
{
    gradient_descent(f, x0, 0.01, 1e-6, 10000)
}

/// Gradient descent optimization
pub fn gradient_descent<F>(
    f: F,
    mut x: f64,
    learning_rate: f64,
    tol: f64,
    max_iter: usize,
) -> Option<OptResult>
where
    F: Fn(f64) -> f64,
{
    let h = 1e-8;
    let mut iterations = 0;

    while iterations < max_iter {
        // Numerical gradient
        let grad = (f(x + h) - f(x - h)) / (2.0 * h);

        let x_new = x - learning_rate * grad;

        if (x_new - x).abs() < tol {
            return Some(OptResult {
                x: x_new,
                fx: f(x_new),
                iterations,
                converged: true,
            });
        }

        x = x_new;
        iterations += 1;
    }

    Some(OptResult {
        x,
        fx: f(x),
        iterations,
        converged: false,
    })
}

/// Nelder-Mead simplex method (simplified 1D version)
pub fn nelder_mead<F>(f: F, x0: f64, tol: f64, max_iter: usize) -> Option<OptResult>
where
    F: Fn(f64) -> f64,
{
    // Simplified implementation - use gradient descent
    gradient_descent(f, x0, 0.01, tol, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize_quadratic() {
        // f(x) = (x-3)^2, minimum at x=3
        let f = |x: f64| (x - 3.0) * (x - 3.0);
        let result = minimize(f, 0.0).unwrap();

        assert!((result.x - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_gradient_descent() {
        let f = |x: f64| x * x + 2.0 * x + 1.0;
        let result = gradient_descent(f, 5.0, 0.1, 1e-4, 1000).unwrap();

        // Minimum at x = -1
        assert!((result.x + 1.0).abs() < 0.1);
    }
}
