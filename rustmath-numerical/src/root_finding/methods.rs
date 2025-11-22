//! Root finding algorithms
//!
//! Find zeros of functions using various numerical methods

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

/// Find a root of f in the interval [a, b] using the best available method
pub fn find_root<F>(f: F, a: f64, b: f64) -> Option<RootResult>
where
    F: Fn(f64) -> f64,
{
    // Use bisection as the default robust method
    bisection(f, a, b, 1e-10, 1000)
}

/// Bisection method for root finding
///
/// Finds a root of f in [a, b] where f(a) and f(b) have opposite signs
pub fn bisection<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Option<RootResult>
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    let fb = f(b);

    // Check if f(a) and f(b) have opposite signs
    if fa * fb > 0.0 {
        return None;
    }

    let mut iterations = 0;
    let mut c = a;

    while (b - a).abs() > tol && iterations < max_iter {
        c = (a + b) / 2.0;
        let fc = f(c);

        if fc.abs() < tol {
            return Some(RootResult {
                root: c,
                iterations,
                error: fc.abs(),
                converged: true,
            });
        }

        if fa * fc < 0.0 {
            b = c;
            // fb = fc; // Unused assignment removed
        } else {
            a = c;
            fa = fc;
        }

        iterations += 1;
    }

    Some(RootResult {
        root: c,
        iterations,
        error: (b - a).abs(),
        converged: (b - a).abs() <= tol,
    })
}

/// Newton-Raphson method for root finding
///
/// Requires the function and its derivative
pub fn newton_raphson<F, DF>(
    f: F,
    df: DF,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Option<RootResult>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    let mut x = x0;
    let mut iterations = 0;

    while iterations < max_iter {
        let fx = f(x);
        let dfx = df(x);

        if dfx.abs() < 1e-14 {
            return None; // Derivative too small
        }

        let x_new = x - fx / dfx;

        if (x_new - x).abs() < tol {
            return Some(RootResult {
                root: x_new,
                iterations,
                error: fx.abs(),
                converged: true,
            });
        }

        x = x_new;
        iterations += 1;
    }

    Some(RootResult {
        root: x,
        iterations,
        error: f(x).abs(),
        converged: false,
    })
}

/// Secant method for root finding
///
/// Like Newton-Raphson but approximates the derivative
pub fn secant<F>(f: F, x0: f64, x1: f64, tol: f64, max_iter: usize) -> Option<RootResult>
where
    F: Fn(f64) -> f64,
{
    let mut x_prev = x0;
    let mut x = x1;
    let mut iterations = 0;

    while iterations < max_iter {
        let fx = f(x);
        let fx_prev = f(x_prev);

        if (fx - fx_prev).abs() < 1e-14 {
            return None; // Division by zero
        }

        let x_new = x - fx * (x - x_prev) / (fx - fx_prev);

        if (x_new - x).abs() < tol {
            return Some(RootResult {
                root: x_new,
                iterations,
                error: f(x_new).abs(),
                converged: true,
            });
        }

        x_prev = x;
        x = x_new;
        iterations += 1;
    }

    Some(RootResult {
        root: x,
        iterations,
        error: f(x).abs(),
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bisection_simple() {
        // f(x) = x^2 - 4, root at x = 2
        let f = |x: f64| x * x - 4.0;
        let result = bisection(f, 0.0, 5.0, 1e-6, 100).unwrap();

        assert!((result.root - 2.0).abs() < 1e-5);
        assert!(result.converged);
    }

    #[test]
    fn test_bisection_no_root() {
        // f(x) = x^2 + 1, no real roots
        let f = |x: f64| x * x + 1.0;
        let result = bisection(f, 0.0, 5.0, 1e-6, 100);

        assert!(result.is_none());
    }

    #[test]
    fn test_newton_raphson() {
        // f(x) = x^2 - 4, f'(x) = 2x, root at x = 2
        let f = |x: f64| x * x - 4.0;
        let df = |x: f64| 2.0 * x;

        let result = newton_raphson(f, df, 1.0, 1e-6, 100).unwrap();

        assert!((result.root - 2.0).abs() < 1e-5);
        assert!(result.converged);
    }

    #[test]
    fn test_secant() {
        // f(x) = x^3 - 2, root at x = ∛2 ≈ 1.2599
        let f = |x: f64| x * x * x - 2.0;

        let result = secant(f, 1.0, 2.0, 1e-6, 100).unwrap();

        assert!((result.root - 2.0_f64.cbrt()).abs() < 1e-5);
        assert!(result.converged);
    }

    #[test]
    fn test_find_root() {
        let f = |x: f64| x * x - 9.0;
        let result = find_root(f, 0.0, 5.0).unwrap();

        assert!((result.root - 3.0).abs() < 1e-8);
    }
}
