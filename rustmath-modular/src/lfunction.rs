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
use rustmath_complex::Complex;
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::f64::consts::PI;

/// The Bernoulli numbers `B_0, ..., B_n` (EXACT rationals), in the convention
/// `B_1 = -1/2`.
///
/// From the defining recurrence `sum_{j=0}^{m} C(m+1, j) B_j = 0` (`m >= 1`),
/// i.e. `B_m = -1/(m+1) sum_{j<m} C(m+1, j) B_j`, with `B_0 = 1`.
///
/// `1, -1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66, ...`
pub fn bernoulli_numbers(n: u32) -> Vec<Rational> {
    fn int(x: u64) -> Integer {
        Integer::from(x)
    }
    fn rat(num: Integer) -> Rational {
        Rational::new(num, Integer::one()).expect("denominator 1 is nonzero")
    }

    let mut b: Vec<Rational> = Vec::with_capacity(n as usize + 1);
    b.push(Rational::one());

    for m in 1..=n as usize {
        // binomial(m+1, j) for j = 0 ..= m-1, built up multiplicatively
        let mut binom = Integer::one(); // C(m+1, 0)
        let mut sum = Rational::zero();
        for (j, b_j) in b.iter().enumerate().take(m) {
            if j > 0 {
                // C(m+1, j) = C(m+1, j-1) * (m + 2 - j) / j
                binom = binom * int((m + 2 - j) as u64) / int(j as u64);
            }
            sum = sum + rat(binom.clone()) * b_j.clone();
        }
        let scale = Rational::new(Integer::from(-1), int((m + 1) as u64)).expect("m + 1 > 0");
        b.push(scale * sum);
    }

    b
}

/// The Bernoulli polynomial `B_k(x)` as its EXACT rational coefficients, in
/// ASCENDING order (`c[i]` is the coefficient of `x^i`, so `c.len() == k + 1`).
///
/// `B_k(x) = sum_{j=0}^{k} C(k, j) B_j x^{k-j}` with `B_j` the Bernoulli numbers
/// above.  Checked against PARI's `bernpol(k)` for `k = 0..10` in the tests --
/// e.g. `bernpol(4) = x^4 - 2x^3 + x^2 - 1/30`.
///
/// (The old code returned `x^k - k x^{k-1}/2` for `k >= 4`, which is not the
/// Bernoulli polynomial at all: it is only the first two terms.)
pub fn bernoulli_polynomial(k: u32) -> Vec<Rational> {
    let b = bernoulli_numbers(k);
    let mut coeffs = vec![Rational::zero(); k as usize + 1];

    let mut binom = Integer::one(); // C(k, 0)
    for j in 0..=k as usize {
        if j > 0 {
            // C(k, j) = C(k, j-1) * (k + 1 - j) / j
            binom = binom * Integer::from((k as usize + 1 - j) as u64) / Integer::from(j as u64);
        }
        // contributes C(k, j) B_j to the coefficient of x^{k-j}
        coeffs[k as usize - j] = Rational::new(binom.clone(), Integer::one())
            .expect("denominator 1 is nonzero")
            * b[j].clone();
    }

    coeffs
}

/// `B_k(x)` at an exact rational `x`, by Horner on [`bernoulli_polynomial`].
pub fn bernoulli_polynomial_at(k: u32, x: &Rational) -> Rational {
    let coeffs = bernoulli_polynomial(k);
    let mut acc = Rational::zero();
    for c in coeffs.iter().rev() {
        acc = acc * x.clone() + c.clone();
    }
    acc
}

/// A Dirichlet L-function L(s, χ)
///
/// # Convention: this is the L-function of χ AS A CHARACTER MOD N
///
/// This struct is the IMPRIMITIVE object
///
/// ```text
///     L(s, chi) = sum_{n >= 1} chi(n) n^{-s} = prod_{p not | N} (1 - chi(p) p^{-s})^{-1}
/// ```
///
/// where `chi(n) = 0` for every `n` with `gcd(n, N) > 1`, `N` the MODULUS.  If
/// `chi` is induced by the primitive `chi*` of conductor `f | N`, this is
/// `L(s, chi*) prod_{p | N, p not | f} (1 - chi*(p) p^{-s})` -- the Euler factors
/// at the primes dividing `N` are REMOVED, so it is not `L(s, chi*)` unless `chi`
/// is primitive.  Every method that sums "over `a` mod N" ([`Self::gauss_sum`],
/// [`Self::generalized_bernoulli_number`]) and every method that evaluates the
/// series or the Euler product follows this convention, and they agree with each
/// other: e.g. for the trivial character mod N,
/// `L(s, chi_0) = zeta(s) prod_{p | N} (1 - p^{-s})`, whose value at 0 is
/// `zeta(0) prod_{p|N} 0 = 0` for every `N > 1`, and `zeta(0) = -1/2` only at
/// `N = 1` -- which is what [`Self::value_at_zero`] returns (it used to return
/// -1/2 for EVERY modulus).
///
/// The functional equation, on the other hand, is a statement about the PRIMITIVE
/// character: `Lambda(s, chi*) = W(chi*) Lambda(1-s, conj chi*)` with the
/// conductor `f` in the gamma factor.  It is simply false for the imprimitive
/// `L(s, chi)` above.  So [`Self::root_number`], [`Self::completed_l_function`],
/// [`Self::approximate_functional_equation`] and [`Self::critical_line_value`]
/// REFUSE (panic) on an imprimitive character rather than return a number that
/// satisfies nothing; they are exactly the methods that use the conductor.
#[derive(Debug, Clone)]
pub struct DirichletLFunction {
    /// The associated Dirichlet character
    character: DirichletCharacter,
    /// Modulus N of the character: the range of every sum over a mod N
    modulus: u64,
    /// Conductor f of the character: the N of the functional equation
    conductor: u64,
    /// Whether the character is primitive
    is_primitive: bool,
}

impl DirichletLFunction {
    /// Create a new Dirichlet L-function for the given character
    ///
    /// # Arguments
    /// * `character` - The Dirichlet character χ
    ///
    /// PANICS for a non-trivial character: the conductor and the primitivity of
    /// one are not computable here (see [`DirichletCharacter::conductor`]), and
    /// neither is χ itself evaluable, so every method below would refuse anyway.
    /// Failing at construction is the honest place to fail.
    pub fn new(character: DirichletCharacter) -> Self {
        let modulus = character.modulus().to_u64().unwrap_or(1);
        let conductor = character.conductor().to_u64().unwrap_or(1);
        let is_primitive = character.is_primitive();

        DirichletLFunction {
            character,
            modulus,
            conductor,
            is_primitive,
        }
    }

    /// The modulus N of the character.
    pub fn modulus(&self) -> u64 {
        self.modulus
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
    pub fn evaluate_series(&self, s: Complex, max_terms: usize) -> Complex {
        let mut sum = Complex::zero();

        for n in 1..=max_terms {
            let chi_n = self.character.eval(&Integer::from(n as u64));
            if chi_n == 0 {
                continue;
            }

            // Compute n^(-s) = exp(-s * log(n))
            let n_f = n as f64;
            let log_n = n_f.ln();
            let n_to_minus_s = Complex::new(
                (-s.real() * log_n).exp() * (-s.imag() * log_n).cos(),
                (-s.real() * log_n).exp() * (-s.imag() * log_n).sin(),
            );

            // Convert chi_n (integer power of root of unity) to complex
            let chi_n_complex = self.character_to_complex(chi_n);

            sum = sum + chi_n_complex * n_to_minus_s;
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
    pub fn evaluate_euler_product(&self, s: Complex, max_prime: u64) -> Complex {
        let mut product = Complex::new(1.0, 0.0);

        for p in self.primes_up_to(max_prime) {
            let euler_factor = self.euler_factor(&Integer::from(p), s.clone());
            product = product * euler_factor;
        }

        product
    }

    /// Compute the Euler factor at a prime p
    ///
    /// L_p(s, χ) = (1 - χ(p)p^{-s})^{-1}
    pub fn euler_factor(&self, p: &Integer, s: Complex) -> Complex {
        let chi_p = self.character.eval(p);

        if chi_p == 0 {
            // Bad prime: chi(p) = 0 exactly when p divides the MODULUS of chi.
            return Complex::new(1.0, 0.0);
        }

        let p_f = p.to_f64().unwrap_or(2.0);

        // Compute p^(-s)
        let log_p = p_f.ln();
        let p_to_minus_s = Complex::new(
            (-s.real() * log_p).exp() * (-s.imag() * log_p).cos(),
            (-s.real() * log_p).exp() * (-s.imag() * log_p).sin(),
        );

        // Convert chi_p to complex
        let chi_p_complex = self.character_to_complex(chi_p);

        // Return (1 - χ(p)p^{-s})^{-1}
        Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) - chi_p_complex * p_to_minus_s)
    }

    /// Compute the functional equation
    ///
    /// Λ(s, χ) = N^{s/2} (π)^{-s/2} Γ((s+a)/2) L(s, χ)
    /// where a = 0 if χ(-1) = 1 (even), a = 1 if χ(-1) = -1 (odd)
    ///
    /// The functional equation is: Λ(s, χ) = W(χ) Λ(1-s, χ̄)
    ///
    /// PANICS on an imprimitive character; see [`Self::require_primitive`].
    pub fn completed_l_function(&self, s: Complex) -> Complex {
        self.require_primitive("completed_l_function");
        let N = self.conductor as f64;
        let a = if self.character.is_even() { 0.0 } else { 1.0 };

        // N^{s/2}
        let N_power = Complex::new(
            (s.real() / 2.0 * N.ln()).exp() * (s.imag() / 2.0 * N.ln()).cos(),
            (s.real() / 2.0 * N.ln()).exp() * (s.imag() / 2.0 * N.ln()).sin(),
        );

        // π^{-s/2}
        let pi_power = Complex::new(
            (-s.real() / 2.0 * PI.ln()).exp() * (-s.imag() / 2.0 * PI.ln()).cos(),
            (-s.real() / 2.0 * PI.ln()).exp() * (-s.imag() / 2.0 * PI.ln()).sin(),
        );

        // Γ((s+a)/2)
        let gamma_arg = (s.clone() + Complex::new(a, 0.0)) / Complex::new(2.0, 0.0);
        let gamma_value = self.complex_gamma(gamma_arg);

        // L(s, χ)
        let L_value = self.evaluate_series(s, 1000);

        N_power * pi_power * gamma_value * L_value
    }

    /// Compute special values of the L-function
    pub fn special_value(&self, s: f64) -> Complex {
        self.evaluate_series(Complex::new(s, 0.0), 2000)
    }

    /// Compute L(1, χ) - particularly important for class number formulas
    pub fn value_at_one(&self) -> Complex {
        if self.character.is_trivial() {
            // L(1, χ₀) diverges (pole)
            return Complex::new(f64::INFINITY, 0.0);
        }

        self.special_value(1.0)
    }

    /// L(0, χ) = -B_{1,χ}, with B_{1,χ} the generalized Bernoulli number of the
    /// character MOD N (see the struct-level note on the convention).
    ///
    /// `L(1-k, chi) = -B_{k, chi} / k` holds for EVERY character mod N, primitive
    /// or not, with `B_{k,chi} = N^{k-1} sum_{a=1}^{N} chi(a) B_k(a/N)`
    /// (Washington, *Cyclotomic Fields*, Thm 4.2); `k = 1` is the case here.
    ///
    /// In particular, for the trivial character mod N this returns
    /// `-B_{1,chi_0} = 0` for every `N > 1` and `-1/2` only at `N = 1` -- which is
    /// exactly `zeta(0) prod_{p | N} (1 - p^0)`, i.e. it agrees with
    /// [`Self::evaluate_series`]'s object.  It used to hard-return -1/2 for every
    /// modulus, contradicting the series on the same struct.
    pub fn value_at_zero(&self) -> Complex {
        -self.generalized_bernoulli_number(1)
    }

    /// Approximate functional equation for efficient computation
    ///
    /// L(s, χ) ≈ Σ_{n≤X} χ(n)/n^s + W(χ) N^{1/2-s} Σ_{n≤Y} χ̄(n)/n^{1-s}
    ///
    /// where X and Y are chosen optimally
    ///
    /// PANICS on an imprimitive character; see [`Self::require_primitive`].
    pub fn approximate_functional_equation(&self, s: Complex) -> Complex {
        self.require_primitive("approximate_functional_equation");
        // Optimal choice: X ≈ Y ≈ √N / (2π)
        let N = self.conductor as f64;
        let cutoff = (N.sqrt() / (2.0 * PI) * s.abs()).ceil() as usize;

        // First sum: direct series
        let mut sum1 = Complex::zero();
        for n in 1..=cutoff {
            let chi_n = self.character.eval(&Integer::from(n as u64));
            if chi_n == 0 {
                continue;
            }
            let n_f = n as f64;
            let n_to_minus_s = Complex::from_real(n_f).pow(&(-s.clone()));
            sum1 = sum1 + self.character_to_complex(chi_n) * n_to_minus_s;
        }

        // Second sum: from functional equation
        let W = self.root_number();
        let N_power = Complex::from_real(N).pow(&(Complex::from_real(0.5) - s.clone()));

        let mut sum2 = Complex::zero();
        for n in 1..=cutoff {
            let chi_n_bar = self.character.eval(&Integer::from(n as u64)); // Conjugate
            if chi_n_bar == 0 {
                continue;
            }
            let n_f = n as f64;
            let one_minus_s = Complex::new(1.0, 0.0) - s.clone();
            let n_to_minus_one_plus_s = Complex::from_real(n_f).pow(&(-one_minus_s));
            sum2 = sum2 + self.character_to_complex(-chi_n_bar) * n_to_minus_one_plus_s;
        }

        sum1 + W * N_power * sum2
    }

    /// Evaluate on the critical line Re(s) = 1/2
    ///
    /// # Arguments
    /// * `t` - The imaginary part (evaluates at s = 1/2 + it)
    ///
    /// PANICS on an imprimitive character (it goes through the approximate
    /// functional equation); see [`Self::require_primitive`].
    pub fn critical_line_value(&self, t: f64) -> Complex {
        let s = Complex::new(0.5, t);
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
            if prev_value.real() * value.real() < 0.0 || prev_value.imag() * value.imag() < 0.0 {
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
            let value = self.critical_line_value(mid).abs();

            if value < 1e-10 {
                return mid;
            }

            let value_a = self.critical_line_value(a).abs();
            let value_b = self.critical_line_value(b).abs();

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
    ///
    /// PANICS on an imprimitive character; see [`Self::require_primitive`].
    /// (|W| = 1 requires |τ(χ)| = √f, which holds only for primitive χ.)
    pub fn root_number(&self) -> Complex {
        self.require_primitive("root_number");
        let gauss_sum = self.gauss_sum();
        let N = self.conductor as f64;
        let a = if self.character.is_even() { 0.0 } else { 1.0 };

        let i_power = Complex::new(0.0, 1.0).pow(&Complex::new(a, 0.0));
        let sqrt_N = N.sqrt();

        gauss_sum / (i_power * Complex::new(sqrt_N, 0.0))
    }

    /// Compute the Gauss sum τ(χ)
    ///
    /// τ(χ) = Σ_{a mod N} χ(a) e^{2πia/N}, where N is the MODULUS of χ (not its
    /// conductor: the sum is over a full residue system mod N).
    pub fn gauss_sum(&self) -> Complex {
        let N = self.modulus;
        let mut sum = Complex::zero();

        for a in 0..N {
            let chi_a = self.character.eval(&Integer::from(a));
            if chi_a == 0 {
                continue;
            }

            // e^{2πia/N}
            let angle = 2.0 * PI * (a as f64) / (N as f64);
            let exponential = Complex::new(angle.cos(), angle.sin());

            sum = sum + self.character_to_complex(chi_a) * exponential;
        }

        sum
    }

    /// The generalized Bernoulli number B_{k,χ} of the character MOD N:
    ///
    /// ```text
    ///     B_{k,chi} = N^{k-1} sum_{a=1}^{N} chi(a) B_k(a/N)
    /// ```
    ///
    /// (the sum runs over a FULL residue system `a = 1, ..., N` -- `a = N` matters
    /// only at `N = 1`, where `chi(N) = chi(1) = 1` and `B_{1,chi} = B_1(1) = 1/2`;
    /// for `N > 1`, `chi(N) = 0`).  This is the normalization for which
    /// `L(1-k, chi) = -B_{k,chi}/k`.
    ///
    /// The values `B_k(a/N)` are computed EXACTLY as rationals
    /// ([`bernoulli_polynomial_at`]) and only then pushed into `f64`, so the only
    /// error here is the final rounding of each term.
    ///
    /// (The prefactor used to be `N^k / k`, i.e. off by a factor of `N^k / (k
    /// N^{k-1}) = N/k`, and the sum stopped at `N - 1`, dropping the only nonzero
    /// term at `N = 1`.  `B_k` itself was fabricated for `k >= 4`.)
    pub fn generalized_bernoulli_number(&self, k: u32) -> Complex {
        assert!(k >= 1, "B_{{k,chi}} is defined for k >= 1");
        let N = self.modulus;
        let mut sum = Complex::zero();

        for a in 1..=N {
            let chi_a = self.character.eval(&Integer::from(a));
            if chi_a == 0 {
                continue;
            }

            let x = Rational::new(Integer::from(a), Integer::from(N)).expect("N > 0");
            let b_k = bernoulli_polynomial_at(k, &x)
                .to_f64()
                .expect("B_k(a/N) is a finite rational");
            sum = sum + self.character_to_complex(chi_a) * Complex::new(b_k, 0.0);
        }

        // N^{k-1}
        let prefactor = (N as f64).powi(k as i32 - 1);
        sum * Complex::new(prefactor, 0.0)
    }

    /// Refuse the functional equation for an imprimitive character.
    ///
    /// `Lambda(s, chi) = W(chi) Lambda(1-s, conj chi)` is a statement about a
    /// PRIMITIVE `chi`.  This struct's L-function is the imprimitive series (see
    /// the struct-level note), so for `modulus != conductor` the completed
    /// L-function, the root number and the approximate functional equation are all
    /// meaningless -- there is no number to return, and returning one anyway (with
    /// the conductor in the gamma factor but the modulus in the Gauss sum, as this
    /// used to) satisfies nothing at all.
    fn require_primitive(&self, what: &str) {
        assert!(
            self.is_primitive,
            "DirichletLFunction::{what}: refusing. chi has modulus {} but conductor {}, so it is \
             IMPRIMITIVE, and this struct's L-function is the imprimitive series \
             sum_n chi(n) n^-s (Euler factors at p | {} removed). The functional equation -- and \
             hence the root number, the completed L-function and the approximate functional \
             equation -- holds only for the PRIMITIVE character chi* inducing chi; build the \
             L-function of chi* instead. (This used to return a value mixing the conductor {} \
             into the gamma factor with a Gauss sum taken mod {}, which satisfies no functional \
             equation.)",
            self.modulus, self.conductor, self.modulus, self.conductor, self.modulus
        );
    }

    /// Convert character value (integer power of root of unity) to complex number
    fn character_to_complex(&self, chi_n: i32) -> Complex {
        if chi_n == 0 {
            return Complex::zero();
        }
        if chi_n == 1 {
            return Complex::new(1.0, 0.0);
        }
        if chi_n == -1 {
            return Complex::new(-1.0, 0.0);
        }

        // For general case, we need the order of the character
        // chi_n represents the power k where χ(n) = e^{2πik/order}
        let order = self.character.order() as f64;
        let angle = 2.0 * PI * (chi_n as f64) / order;
        Complex::new(angle.cos(), angle.sin())
    }

    /// Complex gamma function (Stirling approximation)
    fn complex_gamma(&self, z: Complex) -> Complex {
        // Use Stirling's approximation for large |z|
        // Γ(z) ≈ √(2π/z) * (z/e)^z

        if z.real() <= 0.0 {
            // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
            let one_minus_z = Complex::new(1.0, 0.0) - z.clone();
            let pi_z = Complex::new(PI, 0.0) * z;
            let sin_pi_z = ((Complex::new(0.0, 1.0) * pi_z.clone()).exp()
                - (Complex::new(0.0, -1.0) * pi_z).exp())
                / Complex::new(0.0, 2.0);

            return Complex::new(PI, 0.0) / (sin_pi_z * self.complex_gamma(one_minus_z));
        }

        // Stirling approximation
        let two_pi = 2.0 * PI;
        let e = std::f64::consts::E;

        let sqrt_term = (Complex::new(two_pi, 0.0) / z.clone()).sqrt();
        let power_term = (z.clone() / Complex::new(e, 0.0)).pow(&z);

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

    #[test]
    fn test_trivial_character_l_function() {
        // L(s, χ₀) for the trivial character is related to the Riemann zeta function
        let chi = trivial_character(Integer::from(1));
        let L = DirichletLFunction::new(chi);

        // L(2, χ₀) should be close to ζ(2) = π²/6
        let value = L.special_value(2.0);
        let expected = PI * PI / 6.0;
        assert!((value.real() - expected).abs() < 0.01);
    }

    #[test]
    fn test_euler_product() {
        let chi = trivial_character(Integer::from(1));
        let L = DirichletLFunction::new(chi);

        let s = Complex::new(2.0, 0.0);
        let series_value = L.evaluate_series(s.clone(), 100);
        let product_value = L.evaluate_euler_product(s, 20);

        // Should be approximately equal
        assert!((series_value - product_value).abs() < 0.1);
    }

    #[test]
    fn test_critical_line_evaluation() {
        let chi = trivial_character(Integer::from(1));
        let L = DirichletLFunction::new(chi);

        // Evaluate at s = 1/2 + 14.134725i (known Riemann zeta zero)
        let value = L.critical_line_value(14.134725);

        // Should be close to zero for Riemann zeta
        assert!(value.abs() < 1.0); // Rough check
    }

    #[test]
    fn test_gauss_sum() {
        let chi = trivial_character(Integer::from(5));
        let L = DirichletLFunction::new(chi);

        let gauss = L.gauss_sum();
        // The Gauss sum of the principal (trivial) character mod N is the
        // Ramanujan sum c_N(1) = sum_{gcd(a,N)=1} e^{2 pi i a / N} = mu(N).
        // For N = 5 (prime), mu(5) = -1, so |tau| = 1 (NOT the modulus: the
        // |tau| = sqrt(N) identity holds only for PRIMITIVE characters, and
        // the trivial character mod 5 is imprimitive, of conductor 1).
        assert!((gauss.abs() - 1.0).abs() < 0.1);
    }

    #[test]
    #[ignore = "needs real algorithm: no valid ground truth for zeta(1/2+10i). \
                evaluate_series is a naive partial sum that DIVERGES for Re(s)<1, \
                and approximate_functional_equation is a crude ~2-term stub; a \
                correct test requires the Riemann-Siegel formula or a properly \
                balanced, smoothed approximate functional equation (Phase 4)."]
    fn test_approximate_functional_equation() {
        let chi = trivial_character(Integer::from(1));
        let L = DirichletLFunction::new(chi);

        let s = Complex::new(0.5, 10.0);
        let approx_value = L.approximate_functional_equation(s.clone());
        let series_value = L.evaluate_series(s, 500);

        // Should be reasonably close
        assert!((approx_value - series_value.clone()).abs() / series_value.abs() < 0.5);
    }

    /// GATE: the Bernoulli polynomials against PARI's `bernpol(k)`, k = 0..10.
    /// The coefficient lists below are literally PARI's `Vecrev(Vec(bernpol(k)))`
    /// (ascending order).  The old `bernoulli_polynomial` returned
    /// `x^k - k x^{k-1}/2` for k >= 4 -- e.g. `x^4 - 2x^3` instead of
    /// `x^4 - 2x^3 + x^2 - 1/30`.
    #[test]
    fn test_bernoulli_polynomial_against_pari() {
        fn r(n: i64, d: i64) -> Rational {
            Rational::new(Integer::from(n), Integer::from(d)).unwrap()
        }
        // PARI: for(k=0,10, print(Vecrev(Vec(bernpol(k)))))
        let expected: Vec<Vec<Rational>> = vec![
            vec![r(1, 1)],
            vec![r(-1, 2), r(1, 1)],
            vec![r(1, 6), r(-1, 1), r(1, 1)],
            vec![r(0, 1), r(1, 2), r(-3, 2), r(1, 1)],
            vec![r(-1, 30), r(0, 1), r(1, 1), r(-2, 1), r(1, 1)],
            vec![r(0, 1), r(-1, 6), r(0, 1), r(5, 3), r(-5, 2), r(1, 1)],
            vec![r(1, 42), r(0, 1), r(-1, 2), r(0, 1), r(5, 2), r(-3, 1), r(1, 1)],
            vec![
                r(0, 1),
                r(1, 6),
                r(0, 1),
                r(-7, 6),
                r(0, 1),
                r(7, 2),
                r(-7, 2),
                r(1, 1),
            ],
            vec![
                r(-1, 30),
                r(0, 1),
                r(2, 3),
                r(0, 1),
                r(-7, 3),
                r(0, 1),
                r(14, 3),
                r(-4, 1),
                r(1, 1),
            ],
            vec![
                r(0, 1),
                r(-3, 10),
                r(0, 1),
                r(2, 1),
                r(0, 1),
                r(-21, 5),
                r(0, 1),
                r(6, 1),
                r(-9, 2),
                r(1, 1),
            ],
            vec![
                r(5, 66),
                r(0, 1),
                r(-3, 2),
                r(0, 1),
                r(5, 1),
                r(0, 1),
                r(-7, 1),
                r(0, 1),
                r(15, 2),
                r(-5, 1),
                r(1, 1),
            ],
        ];

        for (k, want) in expected.iter().enumerate() {
            assert_eq!(
                &bernoulli_polynomial(k as u32),
                want,
                "bernpol({k}) coefficients (ascending)"
            );
        }

        // PARI: vector(11, k, bernfrac(k-1))
        let b = bernoulli_numbers(10);
        assert_eq!(b[0], r(1, 1));
        assert_eq!(b[1], r(-1, 2));
        assert_eq!(b[2], r(1, 6));
        assert_eq!(b[3], r(0, 1));
        assert_eq!(b[4], r(-1, 30));
        assert_eq!(b[6], r(1, 42));
        assert_eq!(b[8], r(-1, 30));
        assert_eq!(b[10], r(5, 66));
        // odd B_k vanish for k >= 3
        for k in [3usize, 5, 7, 9] {
            assert_eq!(b[k], Rational::zero(), "B_{k}");
        }

        // the defining functional equation B_k(x+1) - B_k(x) = k x^{k-1},
        // checked exactly at x = 3/7 -- this is a property the fabricated
        // version fails outright
        for k in 1..=10u32 {
            let x = r(3, 7);
            let x1 = x.clone() + Rational::one();
            let lhs = bernoulli_polynomial_at(k, &x1) - bernoulli_polynomial_at(k, &x);
            let mut rhs = Rational::new(Integer::from(k as u64), Integer::one()).unwrap();
            for _ in 0..(k - 1) {
                rhs = rhs * x.clone();
            }
            assert_eq!(lhs, rhs, "B_{k}(x+1) - B_{k}(x) = {k} x^{}", k - 1);
        }
    }

    /// GATE: L(0, chi_0 mod N) for the trivial character.
    ///
    /// L(s, chi_0 mod N) = zeta(s) prod_{p | N} (1 - p^{-s}) -- the imprimitive
    /// series this struct computes -- so at s = 0 it is
    /// zeta(0) * prod_{p|N} (1 - 1) = 0 for every N > 1, and -1/2 only at N = 1.
    /// PARI agrees: B_{1,chi_0 mod N} = 0 for N > 1, 1/2 for N = 1, and
    /// L(0,chi) = -B_{1,chi}.
    ///
    /// This used to hard-return -1/2 for EVERY modulus.
    #[test]
    fn test_value_at_zero() {
        // N = 1: zeta(0) = -1/2
        let L = DirichletLFunction::new(trivial_character(Integer::from(1)));
        assert!(
            (L.value_at_zero().real() - (-0.5)).abs() < 1e-12,
            "L(0, chi_0 mod 1) = zeta(0) = -1/2, got {}",
            L.value_at_zero().real()
        );
        assert!(L.value_at_zero().imag().abs() < 1e-12);

        // N > 1: the Euler factor at every p | N is (1 - p^0) = 0, so L(0) = 0
        for n in 2..=12u64 {
            let L = DirichletLFunction::new(trivial_character(Integer::from(n)));
            let v = L.value_at_zero();
            assert!(
                v.abs() < 1e-9,
                "L(0, chi_0 mod {n}) must be 0 (the old code returned -1/2), got {v:?}"
            );
        }
    }

    /// The generalized Bernoulli numbers of the trivial character mod N, against
    /// the CLOSED FORM
    ///
    /// ```text
    ///     B_{k, chi_0 mod N} = B_k(1) * prod_{p | N} (1 - p^{k-1}),
    /// ```
    ///
    /// which follows from L(1-k, chi_0) = zeta(1-k) prod_{p|N} (1 - p^{k-1})
    /// together with L(1-k, chi) = -B_{k,chi}/k.  Independently verified in PARI
    /// against the defining sum N^{k-1} sum_{a=1}^{N} chi_0(a) bernpol(k)(a/N) for
    /// all N = 1..10, k = 1..6 (samples: B_{1,chi_0 mod 1} = 1/2,
    /// B_{2,chi_0 mod 2} = -1/6, B_{4,chi_0 mod 5} = 62/15).
    ///
    /// Note B_k(1), not B_k: they differ at k = 1 (+1/2 vs -1/2).
    #[test]
    fn test_generalized_bernoulli_trivial_character() {
        for n in 1..=10u64 {
            for k in 1..=6u32 {
                let L = DirichletLFunction::new(trivial_character(Integer::from(n)));
                let got = L.generalized_bernoulli_number(k).real();

                let mut want = bernoulli_polynomial_at(k, &Rational::one()).to_f64().unwrap();
                let mut m = n;
                let mut p = 2u64;
                while p * p <= m {
                    if m % p == 0 {
                        want *= 1.0 - (p as f64).powi(k as i32 - 1);
                        while m % p == 0 {
                            m /= p;
                        }
                    }
                    p += 1;
                }
                if m > 1 {
                    want *= 1.0 - (m as f64).powi(k as i32 - 1);
                }

                assert!(
                    (got - want).abs() < 1e-6 * want.abs().max(1.0),
                    "B_{{{k}, chi_0 mod {n}}}: got {got}, want {want}"
                );
            }
        }
        // the PARI-computed samples, pinned exactly
        let b41 = DirichletLFunction::new(trivial_character(Integer::from(5)))
            .generalized_bernoulli_number(4)
            .real();
        assert!((b41 - 62.0 / 15.0).abs() < 1e-9, "B_{{4, chi_0 mod 5}} = 62/15");
        let b22 = DirichletLFunction::new(trivial_character(Integer::from(2)))
            .generalized_bernoulli_number(2)
            .real();
        assert!((b22 + 1.0 / 6.0).abs() < 1e-9, "B_{{2, chi_0 mod 2}} = -1/6");
    }

    /// The functional-equation methods refuse an imprimitive character rather than
    /// return a number that satisfies no functional equation.  The trivial
    /// character mod 5 has conductor 1, so it is imprimitive.
    #[test]
    #[should_panic(expected = "IMPRIMITIVE")]
    fn test_root_number_refuses_imprimitive() {
        let L = DirichletLFunction::new(trivial_character(Integer::from(5)));
        let _ = L.root_number();
    }

    #[test]
    #[should_panic(expected = "IMPRIMITIVE")]
    fn test_completed_l_function_refuses_imprimitive() {
        let L = DirichletLFunction::new(trivial_character(Integer::from(12)));
        let _ = L.completed_l_function(Complex::new(2.0, 0.0));
    }
}
