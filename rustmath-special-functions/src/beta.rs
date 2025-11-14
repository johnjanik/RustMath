//! Beta function
//!
//! The Beta function is related to the Gamma function by:
//! Beta(x, y) = Gamma(x) * Gamma(y) / Gamma(x + y)

use crate::gamma::gamma;

/// Beta function
///
/// Defined as: Beta(x, y) = integral from 0 to 1 of t^(x-1) * (1-t)^(y-1) dt
///
/// Related to Gamma function by: Beta(x, y) = Gamma(x) * Gamma(y) / Gamma(x + y)
///
/// # Examples
/// ```
/// use rustmath_special_functions::beta::beta;
/// assert!((beta(1.0, 1.0) - 1.0).abs() < 1e-10);
/// assert!((beta(2.0, 3.0) - 1.0/12.0).abs() < 1e-10);
/// ```
pub fn beta(x: f64, y: f64) -> f64 {
    gamma(x) * gamma(y) / gamma(x + y)
}

/// Natural logarithm of Beta function
///
/// More numerically stable for large arguments.
///
/// # Examples
/// ```
/// use rustmath_special_functions::beta::ln_beta;
/// let result = ln_beta(5.0, 10.0);
/// assert!(result < 0.0); // Beta(5, 10) < 1
/// ```
pub fn ln_beta(x: f64, y: f64) -> f64 {
    use crate::gamma::ln_gamma;
    ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y)
}

/// Incomplete Beta function
///
/// B(x; a, b) = integral from 0 to x of t^(a-1) * (1-t)^(b-1) dt
///
/// This is a simplified implementation using series expansion.
pub fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return beta(a, b);
    }

    // Use series expansion
    let bt = x.powf(a) * (1.0 - x).powf(b) / beta(a, b);

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_continued_fraction(x, a, b) / a
    } else {
        beta(a, b) - bt * beta_continued_fraction(1.0 - x, b, a) / b
    }
}

fn beta_continued_fraction(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 100;
    const EPSILON: f64 = 1e-15;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;

    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;
        let m2 = 2 * m;
        let m2_f = m2 as f64;

        let aa = m_f * (b - m_f) * x / ((qam + m2_f) * (a + m2_f));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa2 = -(a + m_f) * (qab + m_f) * x / ((a + m2_f) * (qap + m2_f));
        d = 1.0 + aa2 * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa2 / c;
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

    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_symmetric() {
        // Beta(x, y) = Beta(y, x)
        assert!((beta(2.0, 3.0) - beta(3.0, 2.0)).abs() < 1e-10);
        assert!((beta(5.0, 7.0) - beta(7.0, 5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_beta_special_values() {
        assert!((beta(1.0, 1.0) - 1.0).abs() < 1e-10);
        assert!((beta(2.0, 1.0) - 0.5).abs() < 1e-10);
        assert!((beta(1.0, 2.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_incomplete_beta_bounds() {
        let a = 2.0;
        let b = 3.0;
        assert!((incomplete_beta(0.0, a, b) - 0.0).abs() < 1e-10);
        assert!((incomplete_beta(1.0, a, b) - beta(a, b)).abs() < 1e-9);
    }
}
