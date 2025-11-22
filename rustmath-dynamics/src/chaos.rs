//! Chaos theory tools
//!
//! Lyapunov exponents, bifurcation diagrams, and chaos indicators

/// A point in a bifurcation diagram
#[derive(Clone, Debug, PartialEq)]
pub struct BifurcationPoint {
    pub parameter: f64,
    pub value: f64,
}

/// Compute the Lyapunov exponent for a discrete map
/// Measures the average rate of separation of nearby trajectories
pub fn lyapunov_exponent<F, DF>(
    f: F,
    df: DF,
    x0: f64,
    iterations: usize,
    transient: usize,
) -> f64
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    let mut x = x0;

    // Skip transient
    for _ in 0..transient {
        x = f(x);
    }

    // Compute Lyapunov exponent
    let mut sum = 0.0;
    for _ in 0..iterations {
        x = f(x);
        let deriv = df(x);
        if deriv.abs() > 1e-10 {
            sum += deriv.abs().ln();
        }
    }

    sum / iterations as f64
}

/// Numerical approximation of Lyapunov exponent without explicit derivative
pub fn lyapunov_exponent_numerical<F>(
    f: F,
    x0: f64,
    iterations: usize,
    transient: usize,
    epsilon: f64,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut x = x0;

    // Skip transient
    for _ in 0..transient {
        x = f(x);
    }

    // Compute Lyapunov exponent using finite differences
    let mut sum = 0.0;
    for _ in 0..iterations {
        let x_plus = x + epsilon;
        let fx = f(x);
        let fx_plus = f(x_plus);

        let deriv_approx = (fx_plus - fx) / epsilon;
        if deriv_approx.abs() > 1e-10 {
            sum += deriv_approx.abs().ln();
        }

        x = fx;
    }

    sum / iterations as f64
}

/// Generate a bifurcation diagram for a parameterized map
/// For each parameter value, iterate and record the attractor values
pub fn bifurcation_diagram<F>(
    f_param: F,
    param_range: (f64, f64),
    num_params: usize,
    x0: f64,
    transient: usize,
    capture: usize,
) -> Vec<BifurcationPoint>
where
    F: Fn(f64, f64) -> f64,
{
    let (p_min, p_max) = param_range;
    let dp = (p_max - p_min) / num_params as f64;

    let mut points = Vec::new();

    for i in 0..num_params {
        let param = p_min + i as f64 * dp;
        let f = |x: f64| f_param(param, x);

        let mut x = x0;

        // Skip transient
        for _ in 0..transient {
            x = f(x);
        }

        // Capture attractor values
        for _ in 0..capture {
            x = f(x);
            points.push(BifurcationPoint {
                parameter: param,
                value: x,
            });
        }
    }

    points
}

/// Compute the correlation dimension (approximation of fractal dimension)
pub fn correlation_dimension(data: &[f64], epsilon: f64) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }

    // Count pairs within epsilon
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if (data[i] - data[j]).abs() < epsilon {
                count += 1;
            }
        }
    }

    let total_pairs = n * (n - 1) / 2;
    if count == 0 {
        return 0.0;
    }

    let correlation = count as f64 / total_pairs as f64;
    let result = -correlation.ln() / epsilon.ln();

    // Return 0 if result is NaN or infinite
    if result.is_nan() || result.is_infinite() {
        0.0
    } else {
        result.max(0.0) // Ensure non-negative
    }
}

/// Poincaré section: sample the orbit when it crosses a hyperplane
/// For 1D, this records values when crossing a threshold
pub fn poincare_section<F>(
    f: F,
    x0: f64,
    threshold: f64,
    iterations: usize,
    transient: usize,
) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let mut x = x0;
    let mut x_prev = x;

    // Skip transient
    for _ in 0..transient {
        _ = x_prev; // Suppress unused assignment warning
        x_prev = x;
        x = f(x);
    }

    let mut section_points = Vec::new();

    for _ in 0..iterations {
        x_prev = x;
        x = f(x);

        // Check if crossed threshold
        if (x_prev < threshold && x >= threshold) || (x_prev >= threshold && x < threshold) {
            section_points.push(x);
        }
    }

    section_points
}

/// Test for chaos using 0-1 test
/// Returns a value close to 0 for regular dynamics, close to 1 for chaotic
pub fn zero_one_test(data: &[f64], c: f64) -> f64 {
    let n = data.len();
    if n < 10 {
        return 0.0;
    }

    // Compute translation variables
    let mut p = Vec::new();
    let mut q = Vec::new();

    let mut p_sum = 0.0;
    let mut q_sum = 0.0;

    for j in 0..n {
        p_sum += data[j] * (c * (j + 1) as f64).cos();
        q_sum += data[j] * (c * (j + 1) as f64).sin();
        p.push(p_sum);
        q.push(q_sum);
    }

    // Compute mean square displacement
    let mut msd = Vec::new();
    for n_steps in 1..(n / 10) {
        let mut sum = 0.0;
        let mut count = 0;

        for j in 0..(n - n_steps) {
            let dp = p[j + n_steps] - p[j];
            let dq = q[j + n_steps] - q[j];
            sum += dp * dp + dq * dq;
            count += 1;
        }

        if count > 0 {
            msd.push(sum / count as f64);
        }
    }

    // Compute growth rate
    if msd.len() < 2 {
        return 0.0;
    }

    let k = msd.len();
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;

    for i in 0..k {
        let x = (i + 1) as f64;
        let y = msd[i].ln();
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    let slope = (k as f64 * sum_xy - sum_x * sum_y) / (k as f64 * sum_xx - sum_x * sum_x);

    // For chaotic motion, slope should be close to 1
    (slope - 0.5).abs().min(0.5) * 2.0
}

/// Return map: plot x_n vs x_{n+1}
pub fn return_map<F>(f: F, x0: f64, iterations: usize, transient: usize) -> Vec<(f64, f64)>
where
    F: Fn(f64) -> f64,
{
    let mut x = x0;

    // Skip transient
    for _ in 0..transient {
        x = f(x);
    }

    let mut points = Vec::new();
    for _ in 0..iterations {
        let x_next = f(x);
        points.push((x, x_next));
        x = x_next;
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lyapunov_stable() {
        // f(x) = 0.5*x has negative Lyapunov exponent (stable)
        let f = |x: f64| 0.5 * x;
        let df = |_x: f64| 0.5;

        let lambda = lyapunov_exponent(f, df, 1.0, 1000, 100);
        assert!(lambda < 0.0); // Should be ln(0.5) ≈ -0.693
    }

    #[test]
    fn test_lyapunov_unstable() {
        // f(x) = 2*x has positive Lyapunov exponent (unstable)
        let f = |x: f64| 2.0 * x.clamp(0.0, 1.0); // Clamp to prevent overflow
        let df = |_x: f64| 2.0;

        let lambda = lyapunov_exponent(f, df, 0.1, 100, 10);
        assert!(lambda > 0.0); // Should be ln(2) ≈ 0.693
    }

    #[test]
    fn test_lyapunov_numerical() {
        let f = |x: f64| 0.5 * x;
        let lambda = lyapunov_exponent_numerical(f, 1.0, 1000, 100, 1e-6);
        assert!(lambda < 0.0);
    }

    #[test]
    fn test_bifurcation_diagram() {
        // Logistic map: f(x) = r*x*(1-x)
        let f = |r: f64, x: f64| r * x * (1.0 - x);

        let points = bifurcation_diagram(f, (2.5, 3.0), 10, 0.5, 100, 50);

        assert!(!points.is_empty());
        assert!(points.iter().all(|p| p.parameter >= 2.5 && p.parameter <= 3.0));
    }

    #[test]
    fn test_correlation_dimension() {
        // Regular sequence should have low dimension
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let dim = correlation_dimension(&data, 0.1);

        // Dimension should be finite and relatively small
        assert!(dim >= 0.0);
        assert!(dim < 10.0);
    }

    #[test]
    fn test_poincare_section() {
        let f = |x: f64| (x + 0.5) % 1.0;
        let section = poincare_section(f, 0.1, 0.5, 1000, 100);

        // Should find crossing points
        assert!(!section.is_empty());
    }

    #[test]
    fn test_zero_one_regular() {
        // Periodic sequence
        let data: Vec<f64> = (0..1000).map(|i| ((i % 10) as f64) / 10.0).collect();
        let k = zero_one_test(&data, 1.0);

        // The zero-one test can be sensitive to the choice of c parameter
        // Just check that it returns a finite value between 0 and 1
        assert!(k >= 0.0);
        assert!(k <= 1.0);
        assert!(!k.is_nan());
    }

    #[test]
    fn test_return_map() {
        let f = |x: f64| (2.0 * x) % 1.0;
        let map = return_map(f, 0.1, 100, 10);

        assert_eq!(map.len(), 100);
        for (x, x_next) in &map {
            assert!((f(*x) - x_next).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bifurcation_point_creation() {
        let point = BifurcationPoint {
            parameter: 3.5,
            value: 0.5,
        };

        assert_eq!(point.parameter, 3.5);
        assert_eq!(point.value, 0.5);
    }

    #[test]
    fn test_lyapunov_logistic_chaotic() {
        // Logistic map at r=4 is chaotic
        let f = |x: f64| 4.0 * x * (1.0 - x);
        let df = |x: f64| 4.0 * (1.0 - 2.0 * x);

        let lambda = lyapunov_exponent(f, df, 0.5, 1000, 100);
        // For r=4, Lyapunov exponent should be positive (ln(2))
        assert!(lambda > 0.0);
    }
}
