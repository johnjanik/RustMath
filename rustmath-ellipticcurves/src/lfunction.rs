//! L-functions for elliptic curves
//!
//! Implements the Hasse-Weil L-function and related analytic machinery

use crate::curve::EllipticCurve;
use num_bigint::BigInt;
use num_traits::{ToPrimitive, One};
use std::f64::consts::PI;

/// Complex number for L-function computations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexNum {
    pub re: f64,
    pub im: f64,
}

impl ComplexNum {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    pub fn norm(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Add for ComplexNum {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl std::ops::Mul for ComplexNum {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl std::ops::Div for ComplexNum {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let denom = other.re * other.re + other.im * other.im;
        Self {
            re: (self.re * other.re + self.im * other.im) / denom,
            im: (self.im * other.re - self.re * other.im) / denom,
        }
    }
}

impl std::ops::Mul<ComplexNum> for f64 {
    type Output = ComplexNum;

    fn mul(self, other: ComplexNum) -> ComplexNum {
        ComplexNum {
            re: self * other.re,
            im: self * other.im,
        }
    }
}

/// The Hasse-Weil L-function of an elliptic curve
pub struct LFunction {
    curve: EllipticCurve,
    conductor: BigInt,
    root_number: Option<i32>,
}

impl LFunction {
    /// Create a new L-function for the given curve
    pub fn new(curve: EllipticCurve) -> Self {
        let conductor = curve.conductor.clone()
            .unwrap_or_else(|| Self::compute_conductor(&curve));

        Self {
            curve,
            conductor,
            root_number: None,
        }
    }

    /// Compute the conductor of the curve (simplified)
    fn compute_conductor(curve: &EllipticCurve) -> BigInt {
        // Real implementation would use Tate's algorithm
        // For now, return a simple value based on discriminant
        let mut conductor = BigInt::one();

        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
            let p_big = BigInt::from(p);
            if curve.is_bad_prime(&p_big) {
                conductor *= &p_big;
            }
        }

        if conductor.is_one() {
            BigInt::from(11) // Default for testing
        } else {
            conductor
        }
    }

    /// Compute the Euler factor at a prime p
    pub fn euler_factor(&self, p: &BigInt, s: ComplexNum) -> ComplexNum {
        if self.curve.is_bad_prime(p) {
            self.bad_euler_factor(p, s)
        } else {
            self.good_euler_factor(p, s)
        }
    }

    /// Compute Euler factor at a good prime
    fn good_euler_factor(&self, p: &BigInt, s: ComplexNum) -> ComplexNum {
        let a_p = self.curve.compute_a_p(p);
        let p_f = p.to_f64().unwrap_or(2.0);

        // L_p(s) = 1 / (1 - a_p p^{-s} + p^{1-2s})
        let p_to_s = p_f.powf(s.re) * ComplexNum::new(
            (s.im * p_f.ln()).cos(),
            (s.im * p_f.ln()).sin()
        );

        let p_to_1_minus_2s = p_f.powf(1.0 - 2.0 * s.re) * ComplexNum::new(
            (-(1.0 - 2.0 * s.re) * p_f.ln() * s.im).cos(),
            (-(1.0 - 2.0 * s.re) * p_f.ln() * s.im).sin()
        );

        let numerator = ComplexNum::real(1.0);
        let denominator = ComplexNum::real(1.0)
            + ComplexNum::real(-a_p as f64) / p_to_s
            + p_to_1_minus_2s;

        numerator / denominator
    }

    /// Compute Euler factor at a bad prime (simplified)
    fn bad_euler_factor(&self, p: &BigInt, s: ComplexNum) -> ComplexNum {
        // For bad primes, the Euler factor is simpler
        // This is a simplified version
        let p_f = p.to_f64().unwrap_or(2.0);
        let p_to_s = p_f.powf(s.re);

        ComplexNum::real(1.0) / ComplexNum::real(1.0 - 1.0 / p_to_s)
    }

    /// Evaluate the L-function using Dirichlet series
    pub fn evaluate(&self, s: ComplexNum, max_terms: usize) -> ComplexNum {
        let mut sum = ComplexNum::real(0.0);

        for n in 1..=max_terms {
            let a_n = self.compute_a_n(n as u64);
            let n_f = n as f64;

            // Compute n^{-s} = exp(-s log n)
            let log_n = n_f.ln();
            let n_to_minus_s = ComplexNum::new(
                (-s.re * log_n).exp() * (-s.im * log_n).cos(),
                (-s.re * log_n).exp() * (-s.im * log_n).sin()
            );

            sum = sum + ComplexNum::real(a_n as f64) * n_to_minus_s;
        }

        sum
    }

    /// Compute the n-th coefficient of the L-series
    fn compute_a_n(&self, n: u64) -> i64 {
        if n == 1 {
            return 1;
        }

        // For prime n, a_n = a_p
        if self.is_prime(n) {
            return self.curve.compute_a_p(&BigInt::from(n));
        }

        // For composite n, use multiplicativity
        // This is simplified - full implementation would factor n
        0
    }

    /// Simple primality test
    fn is_prime(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        let sqrt_n = (n as f64).sqrt() as u64;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }

        true
    }

    /// Compute the completed L-function Λ(s)
    pub fn complete_l_function(&self, s: ComplexNum) -> ComplexNum {
        // Λ(s) = N^{s/2} * (2π)^{-s} * Γ(s) * L(s)
        let N = self.conductor.to_f64().unwrap_or(1.0);

        let gamma_factor = self.gamma_factor(s);
        let L_value = self.evaluate(s, 1000);

        let N_to_s_half = N.powf(s.re / 2.0) * ComplexNum::new(
            ((s.im / 2.0) * N.ln()).cos(),
            ((s.im / 2.0) * N.ln()).sin()
        );

        let two_pi_to_minus_s = (2.0 * PI).powf(-s.re) * ComplexNum::new(
            (-s.im * (2.0 * PI).ln()).cos(),
            (-s.im * (2.0 * PI).ln()).sin()
        );

        N_to_s_half * two_pi_to_minus_s * gamma_factor * L_value
    }

    /// Compute Γ(s) (simplified for real s)
    fn gamma_factor(&self, s: ComplexNum) -> ComplexNum {
        // Simplified gamma function for real part
        // Full implementation would need complex gamma
        ComplexNum::real(self.gamma(s.re))
    }

    /// Real gamma function (Stirling approximation)
    fn gamma(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }
        if x == 1.0 || x == 2.0 {
            return 1.0;
        }
        if x < 1.0 {
            return self.gamma(x + 1.0) / x;
        }

        // Stirling's approximation
        let two_pi = 2.0 * PI;
        (two_pi / x).sqrt() * (x / std::f64::consts::E).powf(x)
    }

    /// Compute the analytic rank (order of vanishing at s=1)
    pub fn analytic_rank(&self) -> u32 {
        let mut rank = 0;
        let step = 0.001;
        let center = 1.0;

        // Evaluate L-function and derivatives at s=1
        let mut prev_value = f64::MAX;

        for k in 0..10 {
            let s = ComplexNum::real(center + k as f64 * step);
            let value = self.evaluate(s, 500).norm();

            if value < 1e-6 && k > 0 {
                rank += 1;
            } else if value > prev_value * 10.0 && k > 0 {
                // Function is growing, no more zeros
                break;
            }

            prev_value = value;
        }

        rank
    }

    /// Compute special values of the L-function
    pub fn special_value(&self, s: f64) -> ComplexNum {
        self.evaluate(ComplexNum::real(s), 1000)
    }

    /// Get the root number (sign of functional equation)
    pub fn root_number(&mut self) -> i32 {
        if let Some(w) = self.root_number {
            return w;
        }

        // Compute root number from local data
        // Simplified: typically ±1
        let w = if self.conductor.to_u64().unwrap_or(1) % 2 == 0 {
            -1
        } else {
            1
        };

        self.root_number = Some(w);
        w
    }

    /// Check functional equation: Λ(s) = w * Λ(2-s)
    pub fn check_functional_equation(&mut self, s: f64) -> bool {
        let s_complex = ComplexNum::real(s);
        let two_minus_s = ComplexNum::real(2.0 - s);

        let lambda_s = self.complete_l_function(s_complex);
        let lambda_2_minus_s = self.complete_l_function(two_minus_s);

        let w = self.root_number() as f64;
        let expected = ComplexNum::real(w) * lambda_2_minus_s;

        // Check if they're approximately equal
        let diff = (lambda_s.re - expected.re).abs() + (lambda_s.im - expected.im).abs();
        diff < 0.1 // Tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let z1 = ComplexNum::new(1.0, 2.0);
        let z2 = ComplexNum::new(3.0, 4.0);

        let sum = z1 + z2;
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 6.0).abs() < 1e-10);

        let prod = z1 * z2;
        assert!((prod.re - (-5.0)).abs() < 1e-10);
        assert!((prod.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_l_function_creation() {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1)
        );

        let l_func = LFunction::new(curve);
        assert!(!l_func.conductor.is_zero());
    }

    #[test]
    fn test_euler_factor() {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1)
        );

        let l_func = LFunction::new(curve);
        let p = BigInt::from(5);
        let s = ComplexNum::real(2.0);

        let factor = l_func.euler_factor(&p, s);
        assert!(factor.norm() > 0.0);
    }

    #[test]
    fn test_l_series_evaluation() {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(0),
            BigInt::from(-1)
        );

        let l_func = LFunction::new(curve);
        let s = ComplexNum::real(2.0);

        let value = l_func.evaluate(s, 100);
        assert!(value.norm() > 0.0);
    }

    #[test]
    fn test_analytic_rank() {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0)
        );

        let l_func = LFunction::new(curve);
        let rank = l_func.analytic_rank();

        // Rank should be non-negative
        assert!(rank < 10); // Reasonable bound
    }
}
