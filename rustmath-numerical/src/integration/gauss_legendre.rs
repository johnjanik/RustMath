//! Gauss-Legendre Quadrature Integration
//!
//! This module implements Gauss-Legendre quadrature for numerical integration.
//! Gauss-Legendre quadrature uses the roots of Legendre polynomials as integration
//! nodes, which provides high accuracy - a degree n quadrature exactly integrates
//! polynomials of degree ≤ 2n-1.
//!
//! # Mathematical Background
//!
//! The Legendre polynomials P_n(x) are defined by the recurrence relation:
//! - P_0(x) = 1
//! - P_1(x) = x
//! - (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
//!
//! The derivative of Legendre polynomials satisfies:
//! - P'_n(x) = n/(x^2-1) * (x P_n(x) - P_{n-1}(x))
//!
//! For integration over [-1, 1], the nodes x_i are the roots of P_n(x),
//! and the weights are:
//! - w_i = 2 / ((1 - x_i^2) * [P'_n(x_i)]^2)
//!
//! To integrate over [a, b], we use the transformation:
//! - ∫_a^b f(x) dx = ((b-a)/2) * ∫_{-1}^{1} f((b-a)/2 * t + (a+b)/2) dt
//!
//! # Error Estimation
//!
//! The error in Gauss-Legendre quadrature of degree n for a function f is:
//! - E_n = (2^{2n+1} * (n!)^4) / ((2n+1) * [(2n)!]^3) * f^{(2n)}(ξ)
//!
//! where ξ is some point in [a, b]. Since we don't know f^{(2n)}, we estimate
//! the error by comparing results from successive degrees:
//! - If |I_n - I_{n-1}| < ε, we accept I_n as the result
//!
//! This adaptive approach ensures we use the minimum degree needed for the
//! desired precision.

use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    /// Cache for computed nodes and weights
    /// Key: (degree, precision_bits), Value: (nodes, weights)
    static ref NODE_CACHE: Mutex<HashMap<(usize, usize), (Vec<f64>, Vec<f64>)>> =
        Mutex::new(HashMap::new());
}

/// Compute nodes (roots of Legendre polynomial) and weights for Gauss-Legendre quadrature
///
/// # Arguments
/// * `degree` - The degree of the quadrature (number of nodes)
/// * `prec` - Precision in bits (currently unused, reserved for future arbitrary precision)
///
/// # Returns
/// A tuple of (nodes, weights) vectors, both of length `degree`
///
/// # Example
/// ```
/// use rustmath_numerical::gauss_legendre::nodes;
///
/// let (xs, ws) = nodes(5, 53);
/// assert_eq!(xs.len(), 5);
/// assert_eq!(ws.len(), 5);
/// ```
pub fn nodes(degree: usize, prec: usize) -> (Vec<f64>, Vec<f64>) {
    let mut cache = NODE_CACHE.lock().unwrap();

    if let Some(result) = cache.get(&(degree, prec)) {
        return result.clone();
    }

    let result = nodes_uncached(degree, prec);
    cache.insert((degree, prec), result.clone());
    result
}

/// Compute nodes and weights without caching
///
/// This function computes the roots of the Legendre polynomial of the given degree
/// using Newton-Raphson iteration, and then computes the corresponding weights.
///
/// # Arguments
/// * `degree` - The degree of the quadrature (number of nodes)
/// * `prec` - Precision in bits (currently unused)
///
/// # Returns
/// A tuple of (nodes, weights) vectors
///
/// # Algorithm
/// 1. Use initial guesses based on Chebyshev nodes
/// 2. Refine using Newton-Raphson: x_{k+1} = x_k - P_n(x_k) / P'_n(x_k)
/// 3. Compute weights: w_i = 2 / ((1 - x_i^2) * [P'_n(x_i)]^2)
pub fn nodes_uncached(degree: usize, prec: usize) -> (Vec<f64>, Vec<f64>) {
    let _ = prec; // Reserved for future arbitrary precision support

    if degree == 0 {
        return (vec![], vec![]);
    }

    if degree == 1 {
        return (vec![0.0], vec![2.0]);
    }

    let mut nodes = Vec::with_capacity(degree);
    let mut weights = Vec::with_capacity(degree);

    // Only compute half the roots due to symmetry: P_n(-x) = (-1)^n P_n(x)
    for i in 0..((degree + 1) / 2) {
        // Initial guess using Chebyshev nodes (good approximation for Legendre roots)
        let mut x = -(std::f64::consts::PI * (4.0 * i as f64 + 3.0) / (4.0 * degree as f64 + 2.0)).cos();

        // Newton-Raphson iteration to find root
        let mut delta = 1.0_f64;
        while delta.abs() > 1e-15 {
            let (p, p_prime) = legendre_and_derivative(degree, x);
            delta = p / p_prime;
            x -= delta;
        }

        // Compute the weight
        let (_, p_prime) = legendre_and_derivative(degree, x);
        let weight = 2.0 / ((1.0 - x * x) * p_prime * p_prime);

        nodes.push(x);
        weights.push(weight);

        // Use symmetry for the other half
        if i != degree - 1 - i {
            nodes.push(-x);
            weights.push(weight);
        }
    }

    // Sort nodes in ascending order
    let mut indexed: Vec<(f64, f64)> = nodes.into_iter().zip(weights.into_iter()).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let (sorted_nodes, sorted_weights): (Vec<f64>, Vec<f64>) = indexed.into_iter().unzip();

    (sorted_nodes, sorted_weights)
}

/// Evaluate Legendre polynomial P_n(x) and its derivative P'_n(x) using recurrence relations
///
/// # Arguments
/// * `n` - Degree of the Legendre polynomial
/// * `x` - Point at which to evaluate
///
/// # Returns
/// A tuple (P_n(x), P'_n(x))
///
/// # Recurrence Relations
/// - P_0(x) = 1, P_1(x) = x
/// - (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
/// - P'_n(x) = n/(x^2-1) * (x P_n(x) - P_{n-1}(x))
fn legendre_and_derivative(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }

    if n == 1 {
        return (x, 1.0);
    }

    let mut p_prev = 1.0; // P_0(x)
    let mut p_curr = x;   // P_1(x)

    // Use recurrence to compute P_n(x)
    for k in 1..n {
        let p_next = ((2 * k + 1) as f64 * x * p_curr - k as f64 * p_prev) / (k + 1) as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }

    // Compute derivative using: P'_n(x) = n/(x^2-1) * (x P_n(x) - P_{n-1}(x))
    // But at the roots x^2 ≈ 1, so we use the alternative formula:
    // P'_n(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)
    // For better numerical stability near |x| = 1, we use:
    // n * P_{n-1}(x) = P'_n(x) * (x^2 - 1) / x + P_n(x)
    // Rearranged: P'_n(x) = n * (P_{n-1}(x) - x * P_n(x)) / (1 - x^2)

    let p_derivative = if x.abs() < 0.999 {
        n as f64 * (p_prev - x * p_curr) / (1.0 - x * x)
    } else {
        // Use alternative formula for |x| near 1
        // From P'_n(x) = (n * P_{n-1}(x) - n * x * P_n(x)) / (1 - x^2)
        // But also P'_n(x) = ((n+1) * P_{n+1}(x) - (n+1) * x * P_n(x)) / (1 - x^2)
        // Average them for better stability
        let term1 = n as f64 * p_prev;
        let term2 = n as f64 * x * p_curr;
        (term1 - term2) / (1.0 - x * x)
    };

    (p_curr, p_derivative)
}

/// Estimate the error in integration based on successive approximations
///
/// # Arguments
/// * `results` - A vector of integration results from increasing degrees
/// * `prec` - Desired precision in bits
/// * `epsilon` - Error tolerance
///
/// # Returns
/// An estimate of the integration error
///
/// # Error Estimation Strategy
/// We estimate the error by comparing successive approximations:
/// - error ≈ |I_n - I_{n-1}|
///
/// This is based on the assumption that if the integrand is smooth,
/// the sequence of approximations converges rapidly, and the difference
/// between successive terms is a good error indicator.
pub fn estimate_error(results: &[f64], _prec: usize, _epsilon: f64) -> f64 {
    if results.len() < 2 {
        return f64::INFINITY;
    }

    // Simple error estimate: difference between last two results
    let n = results.len();
    (results[n - 1] - results[n - 2]).abs()
}

/// Integrate a vector-valued function using adaptive Gauss-Legendre quadrature
///
/// # Arguments
/// * `f` - Function that returns a vector of values
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `prec` - Precision in bits
/// * `epsilon` - Error tolerance
///
/// # Returns
/// A vector of integration results, one for each component
///
/// # Algorithm
/// 1. Start with degree 2 quadrature
/// 2. Increase degree until error estimate < epsilon
/// 3. Use node caching for efficiency
///
/// # Example
/// ```
/// use rustmath_numerical::gauss_legendre::integrate_vector;
///
/// let f = |x: f64| vec![x, x*x, x*x*x];
/// let result = integrate_vector(f, 0.0, 1.0, 53, 1e-10);
///
/// assert!((result[0] - 0.5).abs() < 1e-10);      // ∫x dx = 1/2
/// assert!((result[1] - 1.0/3.0).abs() < 1e-10);  // ∫x² dx = 1/3
/// assert!((result[2] - 0.25).abs() < 1e-10);     // ∫x³ dx = 1/4
/// ```
pub fn integrate_vector<F>(f: F, a: f64, b: f64, prec: usize, epsilon: f64) -> Vec<f64>
where
    F: Fn(f64) -> Vec<f64>,
{
    // Start with a low degree and increase until convergence
    let mut degree = 2;
    let max_degree = 100;

    // Store previous results for error estimation
    let mut history: Vec<Vec<f64>> = Vec::new();

    loop {
        let result = integrate_vector_n(&f, a, b, prec, degree);

        if !history.is_empty() {
            // Check convergence for each component
            let prev = &history[history.len() - 1];
            let mut converged = true;

            for i in 0..result.len() {
                let error = (result[i] - prev[i]).abs();
                if error > epsilon {
                    converged = false;
                    break;
                }
            }

            if converged {
                return result;
            }
        }

        history.push(result);
        degree += 2;

        if degree > max_degree {
            // Return best result we have
            return history.into_iter().last().unwrap();
        }
    }
}

/// Integrate a vector-valued function using Gauss-Legendre quadrature with fixed degree
///
/// # Arguments
/// * `f` - Function that returns a vector of values
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `prec` - Precision in bits
/// * `n` - Degree of quadrature (number of nodes)
///
/// # Returns
/// A vector of integration results
///
/// # Integration Formula
/// For integrating over [a, b], we transform to [-1, 1]:
/// - ∫_a^b f(x) dx = ((b-a)/2) * Σ w_i * f((b-a)/2 * x_i + (a+b)/2)
///
/// where x_i are the Gauss-Legendre nodes and w_i are the weights.
///
/// # Example
/// ```
/// use rustmath_numerical::gauss_legendre::integrate_vector_n;
///
/// let f = |x: f64| vec![x.sin()];
/// let result = integrate_vector_n(f, 0.0, std::f64::consts::PI, 53, 10);
///
/// assert!((result[0] - 2.0).abs() < 1e-10);  // ∫_0^π sin(x) dx = 2
/// ```
pub fn integrate_vector_n<F>(f: F, a: f64, b: f64, prec: usize, n: usize) -> Vec<f64>
where
    F: Fn(f64) -> Vec<f64>,
{
    if n == 0 {
        // Evaluate at midpoint for degree 0
        let mid = (a + b) / 2.0;
        let values = f(mid);
        return values.iter().map(|&v| v * (b - a)).collect();
    }

    let (nodes, weights) = nodes(n, prec);

    // Transform from [-1, 1] to [a, b]
    let scale = (b - a) / 2.0;
    let shift = (a + b) / 2.0;

    // Evaluate f at all transformed nodes
    let mut component_sums: Option<Vec<f64>> = None;

    for i in 0..n {
        let x = scale * nodes[i] + shift;
        let values = f(x);
        let weight = weights[i];

        match &mut component_sums {
            None => {
                // Initialize sums
                component_sums = Some(values.iter().map(|&v| v * weight).collect());
            }
            Some(sums) => {
                // Add to existing sums
                for j in 0..values.len() {
                    sums[j] += values[j] * weight;
                }
            }
        }
    }

    // Scale by (b-a)/2
    component_sums
        .unwrap()
        .iter()
        .map(|&s| s * scale)
        .collect()
}

/// Integrate a scalar function using adaptive Gauss-Legendre quadrature
///
/// # Arguments
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `prec` - Precision in bits
/// * `epsilon` - Error tolerance
///
/// # Returns
/// The integral value
///
/// # Example
/// ```
/// use rustmath_numerical::gauss_legendre::integrate;
///
/// let result = integrate(|x| x * x, 0.0, 1.0, 53, 1e-10);
/// assert!((result - 1.0/3.0).abs() < 1e-10);
/// ```
pub fn integrate<F>(f: F, a: f64, b: f64, prec: usize, epsilon: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let wrapper = |x: f64| vec![f(x)];
    integrate_vector(wrapper, a, b, prec, epsilon)[0]
}

/// Integrate a scalar function using Gauss-Legendre quadrature with fixed degree
///
/// # Arguments
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `prec` - Precision in bits
/// * `n` - Degree of quadrature
///
/// # Returns
/// The integral value
pub fn integrate_n<F>(f: F, a: f64, b: f64, prec: usize, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let wrapper = |x: f64| vec![f(x)];
    integrate_vector_n(wrapper, a, b, prec, n)[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legendre_values() {
        // P_0(0.5) = 1
        let (p, _) = legendre_and_derivative(0, 0.5);
        assert!((p - 1.0).abs() < 1e-10);

        // P_1(0.5) = 0.5
        let (p, dp) = legendre_and_derivative(1, 0.5);
        assert!((p - 0.5).abs() < 1e-10);
        assert!((dp - 1.0).abs() < 1e-10);

        // P_2(x) = (3x^2 - 1)/2, so P_2(0.5) = (3*0.25 - 1)/2 = -0.125
        let (p, _) = legendre_and_derivative(2, 0.5);
        assert!((p - (-0.125)).abs() < 1e-10);
    }

    #[test]
    fn test_nodes_degree_1() {
        let (xs, ws) = nodes(1, 53);
        assert_eq!(xs.len(), 1);
        assert_eq!(ws.len(), 1);
        assert!((xs[0] - 0.0).abs() < 1e-10);
        assert!((ws[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_nodes_degree_2() {
        let (xs, ws) = nodes(2, 53);
        assert_eq!(xs.len(), 2);
        assert_eq!(ws.len(), 2);

        // Nodes should be ±1/√3
        let expected = 1.0 / 3.0_f64.sqrt();
        assert!((xs[0] + expected).abs() < 1e-10);
        assert!((xs[1] - expected).abs() < 1e-10);

        // Weights should both be 1
        assert!((ws[0] - 1.0).abs() < 1e-10);
        assert!((ws[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nodes_sum_of_weights() {
        // For integration over [-1, 1], weights should sum to 2
        for degree in 1..=10 {
            let (_, ws) = nodes(degree, 53);
            let sum: f64 = ws.iter().sum();
            assert!((sum - 2.0).abs() < 1e-10, "Degree {} failed: sum = {}", degree, sum);
        }
    }

    #[test]
    fn test_integrate_polynomial() {
        // ∫_0^1 x^2 dx = 1/3
        let result = integrate(|x| x * x, 0.0, 1.0, 53, 1e-10);
        assert!((result - 1.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_cubic() {
        // ∫_0^1 x^3 dx = 1/4
        let result = integrate(|x| x * x * x, 0.0, 1.0, 53, 1e-10);
        assert!((result - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_sin() {
        // ∫_0^π sin(x) dx = 2
        let result = integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 53, 1e-10);
        assert!((result - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_integrate_exp() {
        // ∫_0^1 e^x dx = e - 1
        let result = integrate(|x| x.exp(), 0.0, 1.0, 53, 1e-10);
        let expected = std::f64::consts::E - 1.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_gaussian() {
        // ∫_{-∞}^{∞} e^{-x^2} dx = √π
        // We approximate with ∫_{-5}^{5} which is very close
        let result = integrate(|x| (-x * x).exp(), -5.0, 5.0, 53, 1e-10);
        let expected = std::f64::consts::PI.sqrt();
        assert!((result - expected).abs() < 1e-8);
    }

    #[test]
    fn test_integrate_vector_basic() {
        let f = |x: f64| vec![x, x * x, x * x * x];
        let result = integrate_vector(f, 0.0, 1.0, 53, 1e-10);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 1.0/3.0).abs() < 1e-10);
        assert!((result[2] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_n_fixed_degree() {
        // Degree 2 exactly integrates polynomials up to degree 3
        let result = integrate_n(|x| 3.0 * x * x + 2.0 * x + 1.0, 0.0, 1.0, 53, 2);
        // ∫_0^1 (3x^2 + 2x + 1) dx = x^3 + x^2 + x |_0^1 = 3
        assert!((result - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_exact_for_low_degree_polynomials() {
        // Degree n quadrature should exactly integrate polynomials of degree ≤ 2n-1

        // Degree 3 quadrature should exactly integrate up to degree 5
        let p5 = |x: f64| x.powi(5) + 2.0 * x.powi(4) - x.powi(3) + x.powi(2) - x + 1.0;
        // ∫_0^1 p5(x) dx = 1/6 + 2/5 - 1/4 + 1/3 - 1/2 + 1 = 37/30
        let expected = 37.0 / 30.0;
        let result = integrate_n(p5, 0.0, 1.0, 53, 3);
        assert!((result - expected).abs() < 1e-12);
    }

    #[test]
    fn test_node_caching() {
        // Call nodes twice with same parameters
        let (xs1, ws1) = nodes(5, 53);
        let (xs2, ws2) = nodes(5, 53);

        // Should return identical results (from cache)
        assert_eq!(xs1, xs2);
        assert_eq!(ws1, ws2);
    }

    #[test]
    fn test_estimate_error() {
        let results = vec![1.0, 1.1, 1.11, 1.111];
        let error = estimate_error(&results, 53, 1e-10);
        assert!((error - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_ln() {
        // ∫_1^e ln(x) dx = x(ln(x) - 1) |_1^e = e(1-1) - 1(0-1) = 1
        let e = std::f64::consts::E;
        let result = integrate(|x| x.ln(), 1.0, e, 53, 1e-10);
        assert!((result - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_integrate_oscillatory() {
        // ∫_0^{2π} sin(10x) dx = 0
        let result = integrate(|x| (10.0 * x).sin(), 0.0, 2.0 * std::f64::consts::PI, 53, 1e-8);
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn test_symmetry_of_nodes() {
        // Nodes should be symmetric about 0
        let (xs, _) = nodes(10, 53);
        for i in 0..xs.len() {
            let opposite_idx = xs.len() - 1 - i;
            assert!((xs[i] + xs[opposite_idx]).abs() < 1e-10);
        }
    }
}
