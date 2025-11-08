//! Probability distributions
//!
//! Common probability distributions including Normal, Binomial, Uniform, Poisson, and Exponential

use rand::Rng;
use std::f64::consts::{E, PI};

/// A probability distribution
pub trait Distribution {
    /// Compute the probability density function (PDF) at x
    fn pdf(&self, x: f64) -> f64;

    /// Compute the cumulative distribution function (CDF) at x
    fn cdf(&self, x: f64) -> f64;

    /// Compute the expected value (mean)
    fn mean(&self) -> f64;

    /// Compute the variance
    fn variance(&self) -> f64;

    /// Compute the standard deviation
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sample a random value from the distribution
    fn sample(&self) -> f64;
}

/// Normal (Gaussian) distribution N(μ, σ²)
#[derive(Clone, Debug)]
pub struct Normal {
    /// Mean μ
    pub mu: f64,
    /// Standard deviation σ
    pub sigma: f64,
}

impl Normal {
    /// Create a new normal distribution
    pub fn new(mu: f64, sigma: f64) -> Result<Self, String> {
        if sigma <= 0.0 {
            return Err("Standard deviation must be positive".to_string());
        }
        Ok(Normal { mu, sigma })
    }

    /// Create a standard normal distribution N(0, 1)
    pub fn standard() -> Self {
        Normal { mu: 0.0, sigma: 1.0 }
    }
}

impl Distribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let coefficient = 1.0 / (self.sigma * (2.0 * PI).sqrt());
        let exponent = -((x - self.mu).powi(2)) / (2.0 * self.sigma.powi(2));
        coefficient * E.powf(exponent)
    }

    fn cdf(&self, x: f64) -> f64 {
        // Using error function approximation
        let z = (x - self.mu) / (self.sigma * 2.0_f64.sqrt());
        0.5 * (1.0 + erf(z))
    }

    fn mean(&self) -> f64 {
        self.mu
    }

    fn variance(&self) -> f64 {
        self.sigma.powi(2)
    }

    fn sample(&self) -> f64 {
        // Box-Muller transform
        let mut rng = rand::thread_rng();
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        self.mu + self.sigma * z
    }
}

/// Binomial distribution B(n, p)
#[derive(Clone, Debug)]
pub struct Binomial {
    /// Number of trials n
    pub n: usize,
    /// Success probability p
    pub p: f64,
}

impl Binomial {
    /// Create a new binomial distribution
    pub fn new(n: usize, p: f64) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&p) {
            return Err("Probability must be in [0, 1]".to_string());
        }
        Ok(Binomial { n, p })
    }
}

impl Distribution for Binomial {
    fn pdf(&self, k: f64) -> f64 {
        let k_int = k as usize;
        if k < 0.0 || k_int > self.n || k != k.floor() {
            return 0.0;
        }

        let binom_coeff = binomial_coefficient(self.n, k_int);
        binom_coeff * self.p.powi(k_int as i32) * (1.0 - self.p).powi((self.n - k_int) as i32)
    }

    fn cdf(&self, k: f64) -> f64 {
        if k < 0.0 {
            return 0.0;
        }
        let k_int = k.floor() as usize;
        let upper = k_int.min(self.n);

        (0..=upper).map(|i| self.pdf(i as f64)).sum()
    }

    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }

    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }

    fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let mut successes = 0;
        for _ in 0..self.n {
            if rng.gen::<f64>() < self.p {
                successes += 1;
            }
        }
        successes as f64
    }
}

/// Uniform distribution U(a, b)
#[derive(Clone, Debug)]
pub struct Uniform {
    /// Lower bound a
    pub a: f64,
    /// Upper bound b
    pub b: f64,
}

impl Uniform {
    /// Create a new uniform distribution
    pub fn new(a: f64, b: f64) -> Result<Self, String> {
        if a >= b {
            return Err("Lower bound must be less than upper bound".to_string());
        }
        Ok(Uniform { a, b })
    }

    /// Create a standard uniform distribution U(0, 1)
    pub fn standard() -> Self {
        Uniform { a: 0.0, b: 1.0 }
    }
}

impl Distribution for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
            0.0
        } else {
            1.0 / (self.b - self.a)
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < self.a {
            0.0
        } else if x > self.b {
            1.0
        } else {
            (x - self.a) / (self.b - self.a)
        }
    }

    fn mean(&self) -> f64 {
        (self.a + self.b) / 2.0
    }

    fn variance(&self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }

    fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let u: f64 = rng.gen();
        self.a + u * (self.b - self.a)
    }
}

/// Poisson distribution Pois(λ)
#[derive(Clone, Debug)]
pub struct Poisson {
    /// Rate parameter λ
    pub lambda: f64,
}

impl Poisson {
    /// Create a new Poisson distribution
    pub fn new(lambda: f64) -> Result<Self, String> {
        if lambda <= 0.0 {
            return Err("Lambda must be positive".to_string());
        }
        Ok(Poisson { lambda })
    }
}

impl Distribution for Poisson {
    fn pdf(&self, k: f64) -> f64 {
        if k < 0.0 || k != k.floor() {
            return 0.0;
        }
        let k_int = k as usize;

        E.powf(-self.lambda) * self.lambda.powi(k_int as i32) / factorial(k_int)
    }

    fn cdf(&self, k: f64) -> f64 {
        if k < 0.0 {
            return 0.0;
        }
        let k_int = k.floor() as usize;

        (0..=k_int).map(|i| self.pdf(i as f64)).sum()
    }

    fn mean(&self) -> f64 {
        self.lambda
    }

    fn variance(&self) -> f64 {
        self.lambda
    }

    fn sample(&self) -> f64 {
        // Knuth's algorithm
        let mut rng = rand::thread_rng();
        let l = E.powf(-self.lambda);
        let mut k = 0;
        let mut p = 1.0;

        loop {
            k += 1;
            p *= rng.gen::<f64>();
            if p <= l {
                break;
            }
        }

        (k - 1) as f64
    }
}

/// Exponential distribution Exp(λ)
#[derive(Clone, Debug)]
pub struct Exponential {
    /// Rate parameter λ
    pub lambda: f64,
}

impl Exponential {
    /// Create a new exponential distribution
    pub fn new(lambda: f64) -> Result<Self, String> {
        if lambda <= 0.0 {
            return Err("Lambda must be positive".to_string());
        }
        Ok(Exponential { lambda })
    }
}

impl Distribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.lambda * E.powf(-self.lambda * x)
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - E.powf(-self.lambda * x)
        }
    }

    fn mean(&self) -> f64 {
        1.0 / self.lambda
    }

    fn variance(&self) -> f64 {
        1.0 / (self.lambda.powi(2))
    }

    fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        let u: f64 = rng.gen();
        -(u.ln()) / self.lambda
    }
}

// Helper functions

/// Error function (erf)
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * E.powf(-x * x);

    sign * y
}

/// Factorial
fn factorial(n: usize) -> f64 {
    if n == 0 || n == 1 {
        return 1.0;
    }
    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

/// Binomial coefficient C(n, k)
fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_creation() {
        let n = Normal::new(0.0, 1.0);
        assert!(n.is_ok());

        let n_invalid = Normal::new(0.0, -1.0);
        assert!(n_invalid.is_err());
    }

    #[test]
    fn test_standard_normal() {
        let n = Normal::standard();
        assert_eq!(n.mu, 0.0);
        assert_eq!(n.sigma, 1.0);
        assert_eq!(n.mean(), 0.0);
        assert_eq!(n.variance(), 1.0);
    }

    #[test]
    fn test_normal_pdf() {
        let n = Normal::standard();
        let pdf_0 = n.pdf(0.0);
        // At mean, PDF should be 1/sqrt(2π) ≈ 0.399
        assert!((pdf_0 - 0.3989).abs() < 0.001);
    }

    #[test]
    fn test_normal_cdf() {
        let n = Normal::standard();
        let cdf_0 = n.cdf(0.0);
        // At mean, CDF should be 0.5
        assert!((cdf_0 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_binomial_creation() {
        let b = Binomial::new(10, 0.5);
        assert!(b.is_ok());

        let b_invalid = Binomial::new(10, 1.5);
        assert!(b_invalid.is_err());
    }

    #[test]
    fn test_binomial_mean_variance() {
        let b = Binomial::new(10, 0.5).unwrap();
        assert_eq!(b.mean(), 5.0);
        assert_eq!(b.variance(), 2.5);
    }

    #[test]
    fn test_binomial_pdf() {
        let b = Binomial::new(3, 0.5).unwrap();
        // P(X = 1) = C(3,1) * 0.5^1 * 0.5^2 = 3 * 0.125 = 0.375
        let pdf_1 = b.pdf(1.0);
        assert!((pdf_1 - 0.375).abs() < 0.001);
    }

    #[test]
    fn test_uniform_creation() {
        let u = Uniform::new(0.0, 1.0);
        assert!(u.is_ok());

        let u_invalid = Uniform::new(1.0, 0.0);
        assert!(u_invalid.is_err());
    }

    #[test]
    fn test_uniform_pdf() {
        let u = Uniform::new(0.0, 2.0).unwrap();
        assert_eq!(u.pdf(1.0), 0.5);
        assert_eq!(u.pdf(-1.0), 0.0);
        assert_eq!(u.pdf(3.0), 0.0);
    }

    #[test]
    fn test_uniform_mean_variance() {
        let u = Uniform::new(0.0, 1.0).unwrap();
        assert_eq!(u.mean(), 0.5);
        assert!((u.variance() - 1.0 / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_creation() {
        let p = Poisson::new(3.0);
        assert!(p.is_ok());

        let p_invalid = Poisson::new(-1.0);
        assert!(p_invalid.is_err());
    }

    #[test]
    fn test_poisson_mean_variance() {
        let p = Poisson::new(5.0).unwrap();
        assert_eq!(p.mean(), 5.0);
        assert_eq!(p.variance(), 5.0);
    }

    #[test]
    fn test_exponential_creation() {
        let e = Exponential::new(2.0);
        assert!(e.is_ok());

        let e_invalid = Exponential::new(0.0);
        assert!(e_invalid.is_err());
    }

    #[test]
    fn test_exponential_mean_variance() {
        let e = Exponential::new(2.0).unwrap();
        assert_eq!(e.mean(), 0.5);
        assert_eq!(e.variance(), 0.25);
    }

    #[test]
    fn test_exponential_pdf() {
        let e = Exponential::new(1.0).unwrap();
        assert_eq!(e.pdf(0.0), 1.0);
        assert!(e.pdf(-1.0) == 0.0);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 2), 10.0);
        assert_eq!(binomial_coefficient(4, 0), 1.0);
        assert_eq!(binomial_coefficient(4, 4), 1.0);
    }

    #[test]
    fn test_normal_sample() {
        let n = Normal::new(10.0, 2.0).unwrap();
        // Just test that sampling doesn't panic
        let _sample = n.sample();
    }

    #[test]
    fn test_uniform_sample() {
        let u = Uniform::new(0.0, 10.0).unwrap();
        let sample = u.sample();
        assert!(sample >= 0.0 && sample <= 10.0);
    }
}
