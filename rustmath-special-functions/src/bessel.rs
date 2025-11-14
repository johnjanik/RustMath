//! Bessel functions
//!
//! Bessel functions are solutions to Bessel's differential equation and appear
//! in many physics and engineering applications.

use std::f64::consts::PI;

/// Bessel function of the first kind
///
/// J_n(x) is a solution to Bessel's differential equation:
/// x^2 y'' + x y' + (x^2 - n^2) y = 0
///
/// # Examples
/// ```
/// use rustmath_special_functions::bessel::bessel_j;
/// // J_0(0) = 1
/// assert!((bessel_j(0, 0.0) - 1.0).abs() < 1e-10);
/// ```
pub fn bessel_j(n: i32, x: f64) -> f64 {
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }

    if n < 0 {
        // J_{-n}(x) = (-1)^n J_n(x)
        let sign = if (-n) % 2 == 0 { 1.0 } else { -1.0 };
        return sign * bessel_j(-n, x);
    }

    if x.abs() < 10.0 {
        bessel_j_series(n, x)
    } else {
        bessel_j_asymptotic(n, x)
    }
}

fn bessel_j_series(n: i32, x: f64) -> f64 {
    use crate::gamma::gamma;

    const MAX_TERMS: usize = 100;
    const EPSILON: f64 = 1e-15;

    let mut sum = 0.0;
    let half_x = x / 2.0;

    for k in 0..MAX_TERMS {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let numerator = sign * half_x.powi(2 * k as i32 + n);
        let denominator = factorial(k) * gamma((k + n as usize + 1) as f64);
        let term = numerator / denominator;

        sum += term;

        if term.abs() < EPSILON {
            break;
        }
    }

    sum
}

fn bessel_j_asymptotic(n: i32, x: f64) -> f64 {
    // Asymptotic expansion for large x
    let mu = 4.0 * (n as f64).powi(2);
    let sqrt_term = (2.0 / (PI * x)).sqrt();
    let cos_term = (x - (n as f64) * PI / 2.0 - PI / 4.0).cos();

    // Leading term of asymptotic expansion
    sqrt_term * cos_term
}

/// Bessel function of the second kind (Neumann function)
///
/// Y_n(x) is the second linearly independent solution to Bessel's equation.
///
/// # Examples
/// ```
/// use rustmath_special_functions::bessel::bessel_y;
/// let result = bessel_y(0, 1.0);
/// // Y_0(1) ≈ 0.088
/// assert!((result - 0.088).abs() < 0.01);
/// ```
pub fn bessel_y(n: i32, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    if x < 2.0 {
        bessel_y_series(n, x)
    } else {
        bessel_y_asymptotic(n, x)
    }
}

fn bessel_y_series(n: i32, x: f64) -> f64 {
    // Y_n(x) = (J_n(x) cos(n*pi) - J_{-n}(x)) / sin(n*pi)
    // For integer n, use limiting form
    if n == 0 {
        bessel_y0_series(x)
    } else {
        // Use recurrence or limiting form
        let jn = bessel_j(n, x);
        let j_minus_n = bessel_j(-n, x);
        let n_pi = (n as f64) * PI;

        (jn * n_pi.cos() - j_minus_n) / n_pi.sin()
    }
}

fn bessel_y0_series(x: f64) -> f64 {
    // Special series for Y_0
    const EULER_GAMMA: f64 = 0.5772156649015329;

    let j0 = bessel_j(0, x);
    let mut sum = 0.0;
    let half_x = x / 2.0;

    const MAX_TERMS: usize = 50;
    for k in 1..MAX_TERMS {
        let k_f = k as f64;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let term = sign * half_x.powi(2 * k as i32) * harmonic(k) / (factorial(k).powi(2));
        sum += term;
    }

    (2.0 / PI) * (j0 * (half_x.ln() + EULER_GAMMA) + sum)
}

fn bessel_y_asymptotic(n: i32, x: f64) -> f64 {
    // Asymptotic expansion for large x
    let sqrt_term = (2.0 / (PI * x)).sqrt();
    let sin_term = (x - (n as f64) * PI / 2.0 - PI / 4.0).sin();

    sqrt_term * sin_term
}

fn factorial(n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    (1..=n).map(|i| i as f64).product()
}

fn harmonic(n: usize) -> f64 {
    (1..=n).map(|i| 1.0 / i as f64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bessel_j0_zero() {
        assert!((bessel_j(0, 0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bessel_j_symmetry() {
        // J_{-n}(x) = (-1)^n J_n(x)
        let x = 2.0;
        assert!((bessel_j(-2, x) - bessel_j(2, x)).abs() < 1e-9);
        assert!((bessel_j(-1, x) + bessel_j(1, x)).abs() < 1e-9);
    }

    #[test]
    fn test_bessel_j_small_x() {
        // J_0(x) ≈ 1 for small x
        assert!((bessel_j(0, 0.1) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bessel_y_positive_domain() {
        // Y functions should work for x > 0
        let result = bessel_y(0, 1.0);
        assert!(result.is_finite());
    }
}
