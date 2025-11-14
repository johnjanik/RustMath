//! Special number sequences (Bernoulli numbers, Harmonic numbers, etc.)

use crate::rational::Rational;
use rustmath_core::{Result, Ring};
use rustmath_integers::Integer;

/// Compute the n-th Bernoulli number Bₙ
///
/// Bernoulli numbers are a sequence of rational numbers which occur frequently
/// in number theory and analysis.
///
/// Uses the recursive formula with binomial coefficients:
/// Bₙ = -1/(n+1) * Σ(k=0 to n-1) C(n+1, k) * Bₖ
///
/// First few Bernoulli numbers:
/// - B₀ = 1
/// - B₁ = -1/2 (or +1/2 depending on convention, we use -1/2)
/// - B₂ = 1/6
/// - B₃ = 0 (all odd Bₙ = 0 for n > 1)
/// - B₄ = -1/30
/// - B₆ = 1/42
///
/// # Examples
///
/// ```ignore
/// use rustmath_rationals::bernoulli;
/// use rustmath_rationals::Rational;
///
/// let b0 = bernoulli(0).unwrap();
/// assert_eq!(b0, Rational::from_integer(1));
///
/// let b2 = bernoulli(2).unwrap();
/// assert_eq!(b2, Rational::new(1, 6).unwrap());
/// ```
pub fn bernoulli(n: u32) -> Result<Rational> {
    if n == 0 {
        return Ok(Rational::from_integer(1));
    }
    if n == 1 {
        return Ok(Rational::new(-1, 2)?);
    }

    // Odd Bernoulli numbers (except B₁) are zero
    if n > 1 && n % 2 == 1 {
        return Ok(Rational::zero());
    }

    // Use dynamic programming to compute Bernoulli numbers
    let mut b_cache = vec![Rational::zero(); (n + 1) as usize];
    b_cache[0] = Rational::from_integer(1);

    for m in 1..=n {
        let mut sum = Rational::zero();

        // Compute Σ(k=0 to m-1) C(m+1, k) * Bₖ
        for k in 0..m {
            let binom = binomial(m + 1, k)?;
            sum = sum + binom * b_cache[k as usize].clone();
        }

        // Bₘ = -1/(m+1) * sum
        let denominator = Rational::from_integer(Integer::from(m + 1));
        b_cache[m as usize] = -sum / denominator;
    }

    Ok(b_cache[n as usize].clone())
}

/// Compute binomial coefficient C(n, k) as a Rational
fn binomial(n: u32, k: u32) -> Result<Rational> {
    if k > n {
        return Ok(Rational::zero());
    }
    if k == 0 || k == n {
        return Ok(Rational::from_integer(1));
    }

    // Use the formula C(n, k) = n! / (k! * (n-k)!)
    // But compute more efficiently: C(n, k) = n * (n-1) * ... * (n-k+1) / (k * (k-1) * ... * 1)
    let mut numerator = Integer::from(1);
    let mut denominator = Integer::from(1);

    for i in 0..k {
        numerator = numerator * Integer::from(n - i);
        denominator = denominator * Integer::from(i + 1);
    }

    Rational::new(numerator, denominator)
}

/// Compute the n-th harmonic number Hₙ = 1 + 1/2 + 1/3 + ... + 1/n
///
/// Harmonic numbers appear in many areas of mathematics and computer science,
/// particularly in the analysis of algorithms.
///
/// # Examples
///
/// ```ignore
/// use rustmath_rationals::harmonic;
/// use rustmath_rationals::Rational;
///
/// let h1 = harmonic(1).unwrap();
/// assert_eq!(h1, Rational::from(1));
///
/// let h2 = harmonic(2).unwrap();
/// assert_eq!(h2, Rational::new(3, 2).unwrap());
///
/// let h3 = harmonic(3).unwrap();
/// assert_eq!(h3, Rational::new(11, 6).unwrap());
/// ```
pub fn harmonic(n: u64) -> Result<Rational> {
    if n == 0 {
        return Ok(Rational::zero());
    }

    let mut sum = Rational::zero();
    for i in 1..=n {
        let term = Rational::new(Integer::from(1), Integer::from(i))?;
        sum = sum + term;
    }

    Ok(sum)
}

/// Compute the generalized harmonic number Hₙ,ₘ = Σ(k=1 to n) 1/k^m
///
/// The standard harmonic number is the special case m = 1.
///
/// # Examples
///
/// ```ignore
/// use rustmath_rationals::harmonic_generalized;
/// use rustmath_rationals::Rational;
///
/// // H(3, 1) = 1 + 1/2 + 1/3 = 11/6
/// let h = harmonic_generalized(3, 1).unwrap();
/// assert_eq!(h, Rational::new(11, 6).unwrap());
///
/// // H(3, 2) = 1 + 1/4 + 1/9 = 49/36
/// let h2 = harmonic_generalized(3, 2).unwrap();
/// assert_eq!(h2, Rational::new(49, 36).unwrap());
/// ```
pub fn harmonic_generalized(n: u64, m: u32) -> Result<Rational> {
    if n == 0 {
        return Ok(Rational::zero());
    }

    if m == 0 {
        // H(n, 0) = n (since 1/k^0 = 1)
        return Ok(Rational::from_integer(Integer::from(n)));
    }

    let mut sum = Rational::zero();
    for i in 1..=n {
        let k = Integer::from(i);
        let k_pow_m = k.pow(m);
        let term = Rational::new(Integer::from(1), k_pow_m)?;
        sum = sum + term;
    }

    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bernoulli_small() {
        // B₀ = 1
        assert_eq!(bernoulli(0).unwrap(), Rational::from_integer(1));

        // B₁ = -1/2
        assert_eq!(bernoulli(1).unwrap(), Rational::new(-1, 2).unwrap());

        // B₂ = 1/6
        assert_eq!(bernoulli(2).unwrap(), Rational::new(1, 6).unwrap());

        // B₃ = 0 (odd > 1)
        assert_eq!(bernoulli(3).unwrap(), Rational::zero());

        // B₄ = -1/30
        assert_eq!(bernoulli(4).unwrap(), Rational::new(-1, 30).unwrap());

        // B₅ = 0 (odd > 1)
        assert_eq!(bernoulli(5).unwrap(), Rational::zero());

        // B₆ = 1/42
        assert_eq!(bernoulli(6).unwrap(), Rational::new(1, 42).unwrap());
    }

    #[test]
    fn test_harmonic_small() {
        // H₀ = 0
        assert_eq!(harmonic(0).unwrap(), Rational::zero());

        // H₁ = 1
        assert_eq!(harmonic(1).unwrap(), Rational::from_integer(1));

        // H₂ = 1 + 1/2 = 3/2
        assert_eq!(harmonic(2).unwrap(), Rational::new(3, 2).unwrap());

        // H₃ = 1 + 1/2 + 1/3 = 11/6
        assert_eq!(harmonic(3).unwrap(), Rational::new(11, 6).unwrap());

        // H₄ = 1 + 1/2 + 1/3 + 1/4 = 25/12
        assert_eq!(harmonic(4).unwrap(), Rational::new(25, 12).unwrap());
    }

    #[test]
    fn test_harmonic_generalized() {
        // H(3, 1) = 1 + 1/2 + 1/3 = 11/6
        assert_eq!(harmonic_generalized(3, 1).unwrap(), Rational::new(11, 6).unwrap());

        // H(3, 2) = 1 + 1/4 + 1/9 = 49/36
        assert_eq!(harmonic_generalized(3, 2).unwrap(), Rational::new(49, 36).unwrap());

        // H(2, 3) = 1 + 1/8 = 9/8
        assert_eq!(harmonic_generalized(2, 3).unwrap(), Rational::new(9, 8).unwrap());

        // H(n, 0) = n
        assert_eq!(harmonic_generalized(5, 0).unwrap(), Rational::from_integer(5));
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 0).unwrap(), Rational::from_integer(1));
        assert_eq!(binomial(5, 5).unwrap(), Rational::from_integer(1));
        assert_eq!(binomial(5, 2).unwrap(), Rational::from_integer(10));
        assert_eq!(binomial(6, 3).unwrap(), Rational::from_integer(20));
        assert_eq!(binomial(10, 5).unwrap(), Rational::from_integer(252));
    }
}
