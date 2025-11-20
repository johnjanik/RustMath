//! # Dirichlet L-functions
//!
//! This module implements Dirichlet L-functions and related analytic number theory.
//!
//! A Dirichlet L-function associated to a character χ modulo N is defined by:
//! L(s, χ) = Σ_{n=1}^∞ χ(n)/n^s for Re(s) > 1
//!
//! These functions satisfy a functional equation relating L(s, χ) and L(1-s, χ̄).
//!
//! ## Features
//!
//! - Dirichlet series evaluation
//! - Euler product representation
//! - Functional equations
//! - Special values (at s=0, s=1, etc.)
//! - Approximate functional equation
//! - Critical line evaluation (Re(s) = 1/2)
//! - Zero finding and verification

use crate::dirichlet::DirichletCharacter;
use num_bigint::BigInt;
use num_complex::Complex64;
use num_traits::{One, ToPrimitive, Zero};
use std::f64::consts::PI;

/// A Dirichlet L-function L(s, χ)
#[derive(Debug, Clone)]
pub struct DirichletLFunction {
    /// The associated Dirichlet character
    character: DirichletCharacter,
    /// Conductor of the character
    conductor: u64,
    /// Whether the character is primitive
    is_primitive: bool,
}

impl DirichletLFunction {
    /// Create a new Dirichlet L-function for the given character
    ///
    /// # Arguments
    /// * `character` - The Dirichlet character χ
    pub fn new(character: DirichletCharacter) -> Self {
        let conductor = character.conductor().to_u64().unwrap_or(1);
        let is_primitive = character.is_primitive();

        DirichletLFunction {
            character,
            conductor,
            is_primitive,
        }
    }

    /// Evaluate the L-function using the Dirichlet series
    ///
    /// L(s, χ) = Σ_{n=1}^∞ χ(n)/n^s
    ///
    /// # Arguments
    /// * `s` - The complex argument
    /// * `max_terms` - Maximum number of terms to sum
    ///
    /// # Returns
    /// The value L(s, χ) computed via partial sum
    pub fn evaluate_series(&self, s: Complex64, max_terms: usize) -> Complex64 {
        let mut sum = Complex64::zero();

        for n in 1..=max_terms {
            let chi_n = self.character.eval(&BigInt::from(n));
            if chi_n == 0 {
                continue;
            }

            // Compute n^(-s) = exp(-s * log(n))
            let n_f = n as f64;
            let log_n = n_f.ln();
            let n_to_minus_s = Complex64::new(
                (-s.re * log_n).exp() * (-s.im * log_n).cos(),
                (-s.re * log_n).exp() * (-s.im * log_n).sin(),
            );

            // Convert chi_n (integer power of root of unity) to complex
            let chi_n_complex = self.character_to_complex(chi_n);

            sum += chi_n_complex * n_to_minus_s;
        }

        sum
    }

    /// Evaluate using Euler product representation
    ///
    /// L(s, χ) = Π_p (1 - χ(p)p^{-s})^{-1}
    ///
    /// # Arguments
    /// * `s` - The complex argument
    /// * `max_prime` - Consider primes up to this bound
    pub fn evaluate_euler_product(&self, s: Complex64, max_prime: u64) -> Complex64 {
        let mut product = Complex64::new(1.0, 0.0);

        for p in self.primes_up_to(max_prime) {
            let euler_factor = self.euler_factor(&BigInt::from(p), s);
            product *= euler_factor;
        }

        product
    }

    /// Compute the Euler factor at a prime p
    ///
    /// L_p(s, χ) = (1 - χ(p)p^{-s})^{-1}
    pub fn euler_factor(&self, p: &BigInt, s: Complex64) -> Complex64 {
        let chi_p = self.character.eval(p);

        if chi_p == 0 {
            // Bad prime (divides the conductor)
            return Complex64::new(1.0, 0.0);
        }

        let p_f = p.to_f64().unwrap_or(2.0);

        // Compute p^(-s)
        let log_p = p_f.ln();
        let p_to_minus_s = Complex64::new(
            (-s.re * log_p).exp() * (-s.im * log_p).cos(),
            (-s.re * log_p).exp() * (-s.im * log_p).sin(),
        );

        // Convert chi_p to complex
        let chi_p_complex = self.character_to_complex(chi_p);

        // Return (1 - χ(p)p^{-s})^{-1}
        Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) - chi_p_complex * p_to_minus_s)
    }

    /// Compute the functional equation
    ///
    /// Λ(s, χ) = N^{s/2} (π)^{-s/2} Γ((s+a)/2) L(s, χ)
    /// where a = 0 if χ(-1) = 1 (even), a = 1 if χ(-1) = -1 (odd)
    ///
    /// The functional equation is: Λ(s, χ) = W(χ) Λ(1-s, χ̄)
    pub fn completed_l_function(&self, s: Complex64) -> Complex64 {
        let N = self.conductor as f64;
        let a = if self.character.is_even() { 0.0 } else { 1.0 };

        // N^{s/2}
        let N_power = Complex64::new(
            (s.re / 2.0 * N.ln()).exp() * (s.im / 2.0 * N.ln()).cos(),
            (s.re / 2.0 * N.ln()).exp() * (s.im / 2.0 * N.ln()).sin(),
        );

        // π^{-s/2}
        let pi_power = Complex64::new(
            (-s.re / 2.0 * PI.ln()).exp() * (-s.im / 2.0 * PI.ln()).cos(),
            (-s.re / 2.0 * PI.ln()).exp() * (-s.im / 2.0 * PI.ln()).sin(),
        );

        // Γ((s+a)/2)
        let gamma_arg = (s + Complex64::new(a, 0.0)) / 2.0;
        let gamma_value = self.complex_gamma(gamma_arg);

        // L(s, χ)
        let L_value = self.evaluate_series(s, 1000);

        N_power * pi_power * gamma_value * L_value
    }

    /// Compute special values of the L-function
    pub fn special_value(&self, s: f64) -> Complex64 {
        self.evaluate_series(Complex64::new(s, 0.0), 2000)
    }

    /// Compute L(1, χ) - particularly important for class number formulas
    pub fn value_at_one(&self) -> Complex64 {
        if self.character.is_trivial() {
            // L(1, χ₀) diverges (pole)
            return Complex64::new(f64::INFINITY, 0.0);
        }

        self.special_value(1.0)
    }

    /// Compute L(0, χ)
    ///
    /// For primitive characters, L(0, χ) = -B_{1,χ}/1
    /// where B_{1,χ} is the generalized Bernoulli number
    pub fn value_at_zero(&self) -> Complex64 {
        if self.character.is_trivial() {
            // L(0, χ₀) = -1/2 (for the trivial character mod 1)
            return Complex64::new(-0.5, 0.0);
        }

        // Use functional equation to compute from L(1, χ̄)
        // L(0, χ) = -B_{1,χ}
        self.generalized_bernoulli_number(1)
    }

    /// Approximate functional equation for efficient computation
    ///
    /// L(s, χ) ≈ Σ_{n≤X} χ(n)/n^s + W(χ) N^{1/2-s} Σ_{n≤Y} χ̄(n)/n^{1-s}
    ///
    /// where X and Y are chosen optimally
    pub fn approximate_functional_equation(&self, s: Complex64) -> Complex64 {
        // Optimal choice: X ≈ Y ≈ √N / (2π)
        let N = self.conductor as f64;
        let cutoff = (N.sqrt() / (2.0 * PI) * s.norm()).ceil() as usize;

        // First sum: direct series
        let mut sum1 = Complex64::zero();
        for n in 1..=cutoff {
            let chi_n = self.character.eval(&BigInt::from(n));
            if chi_n == 0 {
                continue;
            }
            let n_f = n as f64;
            let n_to_minus_s = (n_f).powc(-s);
            sum1 += self.character_to_complex(chi_n) * n_to_minus_s;
        }

        // Second sum: from functional equation
        let W = self.root_number();
        let N_power = N.powc(0.5 - s);

        let mut sum2 = Complex64::zero();
        for n in 1..=cutoff {
            let chi_n_bar = self.character.eval(&BigInt::from(n)); // Conjugate
            if chi_n_bar == 0 {
                continue;
            }
            let n_f = n as f64;
            let one_minus_s = Complex64::new(1.0, 0.0) - s;
            let n_to_minus_one_plus_s = (n_f).powc(-one_minus_s);
            sum2 += self.character_to_complex(-chi_n_bar) * n_to_minus_one_plus_s;
        }

        sum1 + W * N_power * sum2
    }

    /// Evaluate on the critical line Re(s) = 1/2
    ///
    /// # Arguments
    /// * `t` - The imaginary part (evaluates at s = 1/2 + it)
    pub fn critical_line_value(&self, t: f64) -> Complex64 {
        let s = Complex64::new(0.5, t);
        self.approximate_functional_equation(s)
    }

    /// Find zeros on the critical line in a given range
    ///
    /// # Arguments
    /// * `t_min` - Minimum imaginary part
    /// * `t_max` - Maximum imaginary part
    /// * `step` - Step size for search
    ///
    /// # Returns
    /// Approximate t values where L(1/2 + it, χ) ≈ 0
    pub fn find_critical_zeros(&self, t_min: f64, t_max: f64, step: f64) -> Vec<f64> {
        let mut zeros = Vec::new();
        let mut prev_value = self.critical_line_value(t_min);

        let mut t = t_min + step;
        while t <= t_max {
            let value = self.critical_line_value(t);

            // Check for sign change in real or imaginary part
            if prev_value.re * value.re < 0.0 || prev_value.im * value.im < 0.0 {
                // Refine zero location
                let zero_t = self.refine_zero(t - step, t);
                zeros.push(zero_t);
            }

            prev_value = value;
            t += step;
        }

        zeros
    }

    /// Refine a zero location using bisection
    fn refine_zero(&self, t_low: f64, t_high: f64) -> f64 {
        let mut a = t_low;
        let mut b = t_high;

        for _ in 0..20 {
            // 20 iterations should be enough
            let mid = (a + b) / 2.0;
            let value = self.critical_line_value(mid).norm();

            if value < 1e-10 {
                return mid;
            }

            let value_a = self.critical_line_value(a).norm();
            let value_b = self.critical_line_value(b).norm();

            if value_a < value_b {
                b = mid;
            } else {
                a = mid;
            }
        }

        (a + b) / 2.0
    }

    /// Compute the root number W(χ) in the functional equation
    ///
    /// W(χ) = τ(χ) / (i^a √N)
    /// where τ(χ) is the Gauss sum and a = 0 or 1 depending on parity
    pub fn root_number(&self) -> Complex64 {
        let gauss_sum = self.gauss_sum();
        let N = self.conductor as f64;
        let a = if self.character.is_even() { 0.0 } else { 1.0 };

        let i_power = Complex64::new(0.0, 1.0).powf(a);
        let sqrt_N = N.sqrt();

        gauss_sum / (i_power * sqrt_N)
    }

    /// Compute the Gauss sum τ(χ)
    ///
    /// τ(χ) = Σ_{a mod N} χ(a) e^{2πia/N}
    pub fn gauss_sum(&self) -> Complex64 {
        let N = self.conductor;
        let mut sum = Complex64::zero();

        for a in 0..N {
            let chi_a = self.character.eval(&BigInt::from(a));
            if chi_a == 0 {
                continue;
            }

            // e^{2πia/N}
            let angle = 2.0 * PI * (a as f64) / (N as f64);
            let exponential = Complex64::new(angle.cos(), angle.sin());

            sum += self.character_to_complex(chi_a) * exponential;
        }

        sum
    }

    /// Compute generalized Bernoulli numbers B_{k,χ}
    ///
    /// B_{k,χ} = (N^k / k) Σ_{a=1}^{N-1} χ(a) B_k(a/N)
    fn generalized_bernoulli_number(&self, k: u32) -> Complex64 {
        let N = self.conductor;
        let mut sum = Complex64::zero();

        for a in 1..N {
            let chi_a = self.character.eval(&BigInt::from(a));
            if chi_a == 0 {
                continue;
            }

            let bernoulli_poly = self.bernoulli_polynomial(k, (a as f64) / (N as f64));
            sum += self.character_to_complex(chi_a) * Complex64::new(bernoulli_poly, 0.0);
        }

        sum * Complex64::new((N.pow(k as u32) as f64) / (k as f64), 0.0)
    }

    /// Bernoulli polynomial B_k(x)
    fn bernoulli_polynomial(&self, k: u32, x: f64) -> f64 {
        match k {
            0 => 1.0,
            1 => x - 0.5,
            2 => x * x - x + 1.0 / 6.0,
            3 => x * x * x - 1.5 * x * x + 0.5 * x,
            _ => {
                // Use recurrence or series (simplified)
                x.powi(k as i32) - (k as f64) * x.powi(k as i32 - 1) / 2.0
            }
        }
    }

    /// Convert character value (integer power of root of unity) to complex number
    fn character_to_complex(&self, chi_n: i32) -> Complex64 {
        if chi_n == 0 {
            return Complex64::zero();
        }
        if chi_n == 1 {
            return Complex64::new(1.0, 0.0);
        }
        if chi_n == -1 {
            return Complex64::new(-1.0, 0.0);
        }

        // For general case, we need the order of the character
        // chi_n represents the power k where χ(n) = e^{2πik/order}
        let order = self.character.order() as f64;
        let angle = 2.0 * PI * (chi_n as f64) / order;
        Complex64::new(angle.cos(), angle.sin())
    }

    /// Complex gamma function (Stirling approximation)
    fn complex_gamma(&self, z: Complex64) -> Complex64 {
        // Use Stirling's approximation for large |z|
        // Γ(z) ≈ √(2π/z) * (z/e)^z

        if z.re <= 0.0 {
            // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
            let one_minus_z = Complex64::new(1.0, 0.0) - z;
            let pi_z = Complex64::new(PI, 0.0) * z;
            let sin_pi_z = ((Complex64::new(0.0, 1.0) * pi_z).exp()
                - (Complex64::new(0.0, -1.0) * pi_z).exp())
                / Complex64::new(0.0, 2.0);

            return Complex64::new(PI, 0.0) / (sin_pi_z * self.complex_gamma(one_minus_z));
        }

        // Stirling approximation
        let two_pi = 2.0 * PI;
        let e = std::f64::consts::E;

        let sqrt_term = (two_pi / z).sqrt();
        let power_term = (z / e).powc(z);

        sqrt_term * power_term
    }

    /// Generate primes up to a bound (simple sieve)
    fn primes_up_to(&self, bound: u64) -> Vec<u64> {
        if bound < 2 {
            return Vec::new();
        }

        let mut is_prime = vec![true; (bound + 1) as usize];
        is_prime[0] = false;
        is_prime[1] = false;

        for i in 2..=((bound as f64).sqrt() as u64) {
            if is_prime[i as usize] {
                let mut j = i * i;
                while j <= bound {
                    is_prime[j as usize] = false;
                    j += i;
                }
            }
        }

        (2..=bound).filter(|&n| is_prime[n as usize]).collect()
    }

    /// Get the associated Dirichlet character
    pub fn character(&self) -> &DirichletCharacter {
        &self.character
    }

    /// Get the conductor
    pub fn conductor(&self) -> u64 {
        self.conductor
    }

    /// Check if the character is primitive
    pub fn is_primitive(&self) -> bool {
        self.is_primitive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dirichlet::trivial_character;
    use num_bigint::BigInt;

    #[test]
    fn test_trivial_character_l_function() {
        // L(s, χ₀) for the trivial character is related to the Riemann zeta function
        let chi = trivial_character(BigInt::from(1));
        let L = DirichletLFunction::new(chi);

        // L(2, χ₀) should be close to ζ(2) = π²/6
        let value = L.special_value(2.0);
        let expected = PI * PI / 6.0;
        assert!((value.re - expected).abs() < 0.01);
    }

    #[test]
    fn test_euler_product() {
        let chi = trivial_character(BigInt::from(1));
        let L = DirichletLFunction::new(chi);

        let s = Complex64::new(2.0, 0.0);
        let series_value = L.evaluate_series(s, 100);
        let product_value = L.evaluate_euler_product(s, 20);

        // Should be approximately equal
        assert!((series_value - product_value).norm() < 0.1);
    }

    #[test]
    fn test_critical_line_evaluation() {
        let chi = trivial_character(BigInt::from(1));
        let L = DirichletLFunction::new(chi);

        // Evaluate at s = 1/2 + 14.134725i (known Riemann zeta zero)
        let value = L.critical_line_value(14.134725);

        // Should be close to zero for Riemann zeta
        assert!(value.norm() < 1.0); // Rough check
    }

    #[test]
    fn test_gauss_sum() {
        let chi = trivial_character(BigInt::from(5));
        let L = DirichletLFunction::new(chi);

        let gauss = L.gauss_sum();
        // For trivial character, Gauss sum should equal the modulus
        assert!((gauss.norm() - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_approximate_functional_equation() {
        let chi = trivial_character(BigInt::from(1));
        let L = DirichletLFunction::new(chi);

        let s = Complex64::new(0.5, 10.0);
        let approx_value = L.approximate_functional_equation(s);
        let series_value = L.evaluate_series(s, 500);

        // Should be reasonably close
        assert!((approx_value - series_value).norm() / series_value.norm() < 0.5);
    }

    #[test]
    fn test_value_at_zero() {
        let chi = trivial_character(BigInt::from(1));
        let L = DirichletLFunction::new(chi);

        let value = L.value_at_zero();
        // L(0, χ₀) = -1/2
        assert!((value.re - (-0.5)).abs() < 0.01);
    }
}
