//! L-functions for elliptic curves
//!
//! Implements the Hasse-Weil L-function and related analytic machinery

use crate::curve::EllipticCurve;
use rustmath_integers::Integer;
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
    conductor: Integer,
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

    /// The conductor of the curve, N = prod p^{f_p}, with every local
    /// exponent computed by Tate's algorithm (see `crate::tate`). This
    /// replaces the old squarefree "product of bad primes" semistable
    /// approximation, which is kept below as
    /// [`Self::compute_conductor_semistable_approx`] for callers that
    /// explicitly want the cheap approximation.
    ///
    /// Cost note: this factors the discriminant (trial division), which is
    /// fine for moderate discriminants but can be slow when the
    /// discriminant has large prime factors.
    pub(crate) fn compute_conductor(curve: &EllipticCurve) -> Integer {
        curve.compute_conductor()
    }

    /// DOCUMENTED FALLBACK (approximation, not Tate's algorithm):
    /// approximate the conductor of the curve as the product of its bad
    /// primes (checked only for p in 2..=31), each raised to the first
    /// power.
    ///
    /// The product-of-bad-primes value equals the true conductor only when
    /// the curve has multiplicative (semistable) reduction at every bad
    /// prime and the given model is minimal; it silently undercounts for
    /// curves with additive reduction or wild ramification at 2 or 3
    /// (where the true exponent can exceed 1), and overcounts at primes
    /// where a non-minimal model hides good reduction. Prefer
    /// [`Self::compute_conductor`] (exact, via Tate's algorithm); this
    /// remains only as a cheap factoring-free approximation for semistable
    /// small-bad-prime curves.
    #[allow(dead_code)]
    pub(crate) fn compute_conductor_semistable_approx(curve: &EllipticCurve) -> Integer {
        let mut conductor = Integer::one();

        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
            let p_big = Integer::from(p);
            if curve.is_bad_prime(&p_big) {
                conductor = conductor * p_big;
            }
        }

        if conductor.is_one() {
            // No bad prime <= 31 was found. Every elliptic curve over Q has
            // bad reduction somewhere, so this means either the curve's bad
            // primes lie outside the scanned range, or the semistable
            // assumption above does not hold. This approximation refuses to
            // guess in that case; use `compute_conductor` (Tate) instead.
            unimplemented!(
                "the bad-prime-product approximation cannot handle curves with no bad prime \
                 <= 31; use the exact Tate-based compute_conductor instead"
            );
        }

        conductor
    }

    /// Compute the Euler factor at a prime p
    pub fn euler_factor(&self, p: &Integer, s: ComplexNum) -> ComplexNum {
        if self.curve.is_bad_prime(p) {
            self.bad_euler_factor(p, s)
        } else {
            self.good_euler_factor(p, s)
        }
    }

    /// Compute Euler factor at a good prime
    fn good_euler_factor(&self, p: &Integer, s: ComplexNum) -> ComplexNum {
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
    fn bad_euler_factor(&self, p: &Integer, s: ComplexNum) -> ComplexNum {
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
            return self.curve.compute_a_p(&Integer::from(n));
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

    /// Compute the analytic rank (order of vanishing of L(E,s) at s=1)
    ///
    /// NOT YET IMPLEMENTED (facade). The previous body sampled
    /// `evaluate(s, 500)` -- a raw truncated partial sum of the Dirichlet
    /// series -- at ten points near s=1 and called it "rank" if the norm
    /// dipped below 1e-6 or grew by 10x. That has no mathematical
    /// justification here: the Dirichlet series for L(E,s) only converges
    /// absolutely for Re(s) > 3/2, so truncating it at s=1 (inside the
    /// non-convergent region) does not approximate the true value, and a
    /// small sampled norm is just as likely to be truncation noise as a
    /// genuine zero. A trustworthy analytic rank requires evaluating the
    /// analytically-continued L-function (via the functional equation and
    /// genuine modular-form Fourier coefficients, not this partial sum)
    /// with a certified numerical method for detecting the order of
    /// vanishing.
    pub fn analytic_rank(&self) -> u32 {
        unimplemented!(
            "analytic rank not yet implemented (facade): requires a numerically-certified \
             evaluation of the analytically continued L-function near s=1 (e.g. via the \
             functional equation and genuine modular-form coefficients), not naive sampling \
             of truncated Dirichlet-series partial sums, which are not even guaranteed to \
             converge at s=1"
        )
    }

    /// Compute special values of the L-function
    pub fn special_value(&self, s: f64) -> ComplexNum {
        self.evaluate(ComplexNum::real(s), 1000)
    }

    /// Get the root number (sign of functional equation)
    ///
    /// NOT YET IMPLEMENTED (facade). The previous body derived the root
    /// number from conductor parity alone ("even conductor => -1, odd =>
    /// +1"), which is not a real formula: the true root number is a product
    /// of local root numbers at each bad prime (and at the archimedean
    /// place), determined by the reduction type via Tate's algorithm, and
    /// does not reduce to conductor parity in general. Returning that
    /// heuristic would silently give the wrong sign for many curves, so we
    /// refuse to guess.
    pub fn root_number(&mut self) -> i32 {
        if let Some(w) = self.root_number {
            return w;
        }

        unimplemented!(
            "root number not yet implemented (facade): requires local root numbers at each \
             bad prime (from Tate's algorithm / reduction type) and at the archimedean place, \
             not a conductor-parity heuristic"
        )
    }

    /// Check functional equation: Λ(s) = w * Λ(2-s)
    ///
    /// Depends on `root_number`, which is currently `unimplemented!()`; this
    /// will panic whenever `root_number` would.
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
            Integer::from(-1),
            Integer::from(1)
        );

        let l_func = LFunction::new(curve);
        // y² = x³ - x + 1 has conductor 92 = 2²·23 (Tate: type IV with
        // f=2 at 2, I1 at 23; PARI/GP ellglobalred-verified). The old
        // semistable approximation would have given 2·23 = 46 here.
        assert_eq!(l_func.conductor, Integer::from(92));
    }

    #[test]
    fn test_euler_factor() {
        let curve = EllipticCurve::from_short_weierstrass(
            Integer::from(-1),
            Integer::from(1)
        );

        let l_func = LFunction::new(curve);
        let p = Integer::from(5);
        let s = ComplexNum::real(2.0);

        let factor = l_func.euler_factor(&p, s);
        assert!(factor.norm() > 0.0);
    }

    #[test]
    fn test_l_series_evaluation() {
        let curve = EllipticCurve::from_short_weierstrass(
            Integer::from(0),
            Integer::from(-1)
        );

        let l_func = LFunction::new(curve);
        let s = ComplexNum::real(2.0);

        let value = l_func.evaluate(s, 100);
        assert!(value.norm() > 0.0);
    }

    #[test]
    #[ignore = "facade -> unimplemented; needs real descent/L-function (Phase 4)"]
    fn test_analytic_rank() {
        let curve = EllipticCurve::from_short_weierstrass(
            Integer::from(-1),
            Integer::from(0)
        );

        let l_func = LFunction::new(curve);
        let rank = l_func.analytic_rank();

        // Rank should be non-negative
        assert!(rank < 10); // Reasonable bound
    }
}
