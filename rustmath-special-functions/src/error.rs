//! Error functions
//!
//! The error function and related functions appear in probability,
//! statistics, and solutions to the heat equation.

use std::f64::consts::PI;

/// Error function
///
/// Defined as: erf(x) = (2/sqrt(pi)) * integral from 0 to x of e^(-t^2) dt
///
/// # Examples
/// ```
/// use rustmath_special_functions::error::erf;
/// assert!(erf(0.0).abs() < 1e-10);
/// assert!((erf(1.0) - 0.8427).abs() < 0.001);
/// ```
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }

    if x < 0.0 {
        return -erf(-x);
    }

    if x > 5.0 {
        return 1.0;
    }

    if x < 2.0 {
        erf_series(x)
    } else {
        1.0 - erfc(x)
    }
}

/// Complementary error function
///
/// Defined as: erfc(x) = 1 - erf(x)
///
/// # Examples
/// ```
/// use rustmath_special_functions::error::erfc;
/// assert!((erfc(0.0) - 1.0).abs() < 1e-10);
/// ```
pub fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }

    if x > 5.0 {
        return 0.0;
    }

    if x < 2.0 {
        1.0 - erf_series(x)
    } else {
        erfc_continued_fraction(x)
    }
}

fn erf_series(x: f64) -> f64 {
    // Series expansion: erf(x) = (2/sqrt(pi)) * sum of (-1)^n * x^(2n+1) / (n! * (2n+1))
    const MAX_TERMS: usize = 100;
    const EPSILON: f64 = 1e-15;

    let mut sum = 0.0;
    let x2 = x * x;

    for n in 0..MAX_TERMS {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let numerator = sign * x.powi(2 * n as i32 + 1);
        let denominator = factorial(n) * (2 * n + 1) as f64;
        let term = numerator / denominator;

        sum += term;

        if term.abs() < EPSILON {
            break;
        }
    }

    (2.0 / PI.sqrt()) * sum
}

fn erfc_continued_fraction(x: f64) -> f64 {
    // Continued fraction expansion for erfc(x)
    const MAX_ITER: usize = 100;
    const EPSILON: f64 = 1e-15;

    let x2 = x * x;
    let mut a = 1.0;
    let mut b = x2 + 0.5;
    let mut c = b;
    let mut d = 1.0 / b;
    let mut h = d;

    for n in 1..MAX_ITER {
        let n_f = n as f64;
        a = -n_f * (n_f - 0.5);
        b += 2.0;
        d = a * d + b;

        if d.abs() < 1e-30 {
            d = 1e-30;
        }

        c = b + a / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }

        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPSILON {
            break;
        }
    }

    ((-x2).exp() / (x * PI.sqrt())) * h
}

fn factorial(n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    (1..=n).map(|i| i as f64).product()
}

/// Inverse error function
///
/// Returns x such that erf(x) = y
pub fn erf_inv(y: f64) -> f64 {
    if y.abs() >= 1.0 {
        if y > 0.0 {
            return f64::INFINITY;
        } else {
            return f64::NEG_INFINITY;
        }
    }

    if y == 0.0 {
        return 0.0;
    }

    // Use Newton-Raphson iteration
    let mut x = if y > 0.0 { 1.0 } else { -1.0 };

    for _ in 0..20 {
        let fx = erf(x) - y;
        let fpx = (2.0 / PI.sqrt()) * (-x * x).exp();
        x -= fx / fpx;
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_zero() {
        assert!(erf(0.0).abs() < 1e-10);
    }

    #[test]
    fn test_erf_symmetry() {
        // erf(-x) = -erf(x)
        assert!((erf(-1.0) + erf(1.0)).abs() < 1e-10);
        assert!((erf(-2.0) + erf(2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_erfc_zero() {
        assert!((erfc(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_erf_erfc_complement() {
        for x in &[0.5, 1.0, 1.5, 2.0, 3.0] {
            assert!((erf(*x) + erfc(*x) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_erf_known_values() {
        // erf(1) ≈ 0.8427
        assert!((erf(1.0) - 0.8427007929).abs() < 1e-6);
    }

    #[test]
    fn test_erf_inv_zero() {
        assert!(erf_inv(0.0).abs() < 1e-10);
    }

    #[test]
    fn test_erf_inv_round_trip() {
        for x in &[0.5, 1.0, 1.5] {
            let y = erf(*x);
            let x_back = erf_inv(y);
            assert!((x - x_back).abs() < 1e-9);
        }
    }
}
