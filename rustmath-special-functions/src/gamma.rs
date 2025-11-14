//! Gamma function and related functions
//!
//! The Gamma function is a continuous extension of the factorial function.
//! For positive integers n: Gamma(n) = (n-1)!

use std::f64::consts::PI;

/// Compute the Gamma function using Lanczos approximation
///
/// The Gamma function is defined as:
/// Gamma(z) = integral from 0 to infinity of t^(z-1) * e^(-t) dt
///
/// For positive integers n: Gamma(n) = (n-1)!
///
/// # Examples
/// ```
/// use rustmath_special_functions::gamma::gamma;
/// assert!((gamma(5.0) - 24.0).abs() < 1e-10); // Gamma(5) = 4! = 24
/// ```
pub fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        // Use reflection formula: Gamma(1-z) * Gamma(z) = pi / sin(pi*z)
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        // Lanczos approximation
        lanczos_gamma(x)
    }
}

/// Natural logarithm of the Gamma function
///
/// Useful for large arguments where Gamma(x) would overflow.
///
/// # Examples
/// ```
/// use rustmath_special_functions::gamma::ln_gamma;
/// let result = ln_gamma(10.0);
/// assert!((result - 12.8018274800814691).abs() < 1e-10);
/// ```
pub fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        let y = 1.0 - x;
        (PI / ((PI * x).sin())).ln() - ln_gamma(y)
    } else {
        lanczos_ln_gamma(x)
    }
}

/// Digamma function (logarithmic derivative of Gamma)
///
/// psi(x) = d/dx ln(Gamma(x)) = Gamma'(x) / Gamma(x)
///
/// # Examples
/// ```
/// use rustmath_special_functions::gamma::digamma;
/// let result = digamma(1.0);
/// assert!((result + 0.5772156649015329).abs() < 1e-10); // -gamma (Euler-Mascheroni constant)
/// ```
pub fn digamma(x: f64) -> f64 {
    if x < 0.0 {
        // Reflection formula
        digamma(1.0 - x) - PI / (PI * x).tan()
    } else if x < 1.0 {
        // Use recurrence relation: psi(x+1) = psi(x) + 1/x
        digamma(1.0 + x) - 1.0 / x
    } else {
        // Series expansion for x >= 1
        let mut result = 0.0;
        let mut y = x;

        // Use recurrence to get to x >= 10 for better series convergence
        while y < 10.0 {
            result -= 1.0 / y;
            y += 1.0;
        }

        // Asymptotic series
        let z = 1.0 / (y * y);
        result += y.ln() - 0.5 / y - z * (1.0/12.0 - z * (1.0/120.0 - z / 252.0));
        result
    }
}

// Lanczos coefficients for g=7, n=9
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEF: [f64; 9] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
];

fn lanczos_gamma(x: f64) -> f64 {
    if x < 0.5 {
        return PI / ((PI * x).sin() * lanczos_gamma(1.0 - x));
    }

    let z = x - 1.0;
    let mut a = LANCZOS_COEF[0];
    for i in 1..9 {
        a += LANCZOS_COEF[i] / (z + i as f64);
    }

    let t = z + LANCZOS_G + 0.5;
    let sqrt_2pi = (2.0 * PI).sqrt();
    sqrt_2pi * t.powf(z + 0.5) * (-t).exp() * a
}

fn lanczos_ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        return (PI / ((PI * x).sin())).ln() - lanczos_ln_gamma(1.0 - x);
    }

    let z = x - 1.0;
    let mut a = LANCZOS_COEF[0];
    for i in 1..9 {
        a += LANCZOS_COEF[i] / (z + i as f64);
    }

    let t = z + LANCZOS_G + 0.5;
    let log_sqrt_2pi = 0.5 * (2.0 * PI).ln();
    log_sqrt_2pi + (z + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_small_integers() {
        assert!((gamma(1.0) - 1.0).abs() < 1e-10);
        assert!((gamma(2.0) - 1.0).abs() < 1e-10);
        assert!((gamma(3.0) - 2.0).abs() < 1e-10);
        assert!((gamma(4.0) - 6.0).abs() < 1e-10);
        assert!((gamma(5.0) - 24.0).abs() < 1e-10);
        assert!((gamma(6.0) - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_half() {
        assert!((gamma(0.5) - PI.sqrt()).abs() < 1e-10);
        assert!((gamma(1.5) - PI.sqrt() * 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ln_gamma() {
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-10);
        assert!((ln_gamma(2.0) - 0.0).abs() < 1e-10);
        assert!((ln_gamma(10.0) - 12.8018274800814691).abs() < 1e-9);
    }

    #[test]
    fn test_digamma_at_one() {
        // psi(1) = -gamma (Euler-Mascheroni constant ≈ -0.5772156649)
        let euler_mascheroni = 0.5772156649015329;
        assert!((digamma(1.0) + euler_mascheroni).abs() < 1e-9);
    }
}
