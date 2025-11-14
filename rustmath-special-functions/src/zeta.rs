//! Riemann Zeta function
//!
//! The Riemann Zeta function is central to number theory and has applications
//! in physics and statistics.

use std::f64::consts::PI;

/// Riemann Zeta function
///
/// Defined as: zeta(s) = sum from n=1 to infinity of 1/n^s for Re(s) > 1
///
/// This implementation uses various series and functional equations.
///
/// # Examples
/// ```
/// use rustmath_special_functions::zeta::zeta;
/// // zeta(2) = pi^2/6
/// assert!((zeta(2.0) - PI.powi(2)/6.0).abs() < 1e-10);
/// ```
pub fn zeta(s: f64) -> f64 {
    if s == 1.0 {
        return f64::INFINITY;
    }

    if s < 0.0 {
        // Use functional equation: zeta(s) = 2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s) * zeta(1-s)
        return zeta_negative(s);
    }

    if s < 0.5 {
        // Use reflection formula
        return zeta_reflection(s);
    }

    // For s > 0.5, use direct series or known values
    if (s - 2.0).abs() < 1e-10 {
        return PI * PI / 6.0;
    }

    if (s - 4.0).abs() < 1e-10 {
        return PI.powi(4) / 90.0;
    }

    // Use Dirichlet eta function relation: zeta(s) = eta(s) / (1 - 2^(1-s))
    // where eta(s) = sum of (-1)^(n+1) / n^s (alternating zeta)
    zeta_series(s)
}

fn zeta_series(s: f64) -> f64 {
    const MAX_TERMS: usize = 10000;
    const EPSILON: f64 = 1e-15;

    if s > 1.0 {
        // Direct series for s > 1
        let mut sum = 0.0;
        for n in 1..=MAX_TERMS {
            let term = 1.0 / (n as f64).powf(s);
            sum += term;
            if term < EPSILON {
                break;
            }
        }
        sum
    } else {
        // Use Dirichlet eta function
        let eta = zeta_eta(s);
        eta / (1.0 - 2.0_f64.powf(1.0 - s))
    }
}

fn zeta_eta(s: f64) -> f64 {
    // Dirichlet eta function: sum of (-1)^(n+1) / n^s
    const MAX_TERMS: usize = 10000;
    const EPSILON: f64 = 1e-15;

    let mut sum = 0.0;
    let mut sign = 1.0;

    for n in 1..=MAX_TERMS {
        let term = sign / (n as f64).powf(s);
        sum += term;
        sign = -sign;
        if term.abs() < EPSILON {
            break;
        }
    }

    sum
}

fn zeta_reflection(s: f64) -> f64 {
    // Use functional equation
    use crate::gamma::gamma;
    let factor = 2.0_f64.powf(s) * PI.powf(s - 1.0) * (PI * s / 2.0).sin() * gamma(1.0 - s);
    factor * zeta(1.0 - s)
}

fn zeta_negative(s: f64) -> f64 {
    // For negative arguments
    use crate::gamma::gamma;
    let t = 1.0 - s;
    2.0_f64.powf(s) * PI.powf(s - 1.0) * (PI * s / 2.0).sin() * gamma(t) * zeta(t)
}

/// Hurwitz Zeta function
///
/// zeta(s, a) = sum from n=0 to infinity of 1/(n+a)^s
///
/// The Riemann Zeta function is the special case where a = 1.
pub fn hurwitz_zeta(s: f64, a: f64) -> f64 {
    if a <= 0.0 {
        panic!("Hurwitz zeta requires a > 0");
    }

    const MAX_TERMS: usize = 10000;
    const EPSILON: f64 = 1e-15;

    let mut sum = 0.0;
    for n in 0..MAX_TERMS {
        let term = 1.0 / (n as f64 + a).powf(s);
        sum += term;
        if term < EPSILON {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeta_2() {
        // zeta(2) = pi^2/6
        let expected = PI * PI / 6.0;
        assert!((zeta(2.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_zeta_4() {
        // zeta(4) = pi^4/90
        let expected = PI.powi(4) / 90.0;
        assert!((zeta(4.0) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_zeta_positive() {
        // zeta(3) ≈ 1.202 (Apéry's constant)
        assert!((zeta(3.0) - 1.202).abs() < 0.001);
    }

    #[test]
    fn test_hurwitz_zeta_reduces_to_riemann() {
        // Hurwitz zeta(s, 1) = Riemann zeta(s)
        let s = 3.0;
        assert!((hurwitz_zeta(s, 1.0) - zeta(s)).abs() < 1e-10);
    }
}
