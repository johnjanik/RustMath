//! Elliptic Curve Method (ECM) for integer factorization
//!
//! The Elliptic Curve Method is a probabilistic factorization algorithm that uses
//! elliptic curves over Z/nZ to find non-trivial factors of composite numbers.
//!
//! # Algorithm Overview
//!
//! ECM works by:
//! 1. Selecting a random elliptic curve E and point P on that curve (mod n)
//! 2. Computing a large multiple kP for some integer k (stage 1)
//! 3. If during computation the modular inverse fails, gcd reveals a factor
//! 4. Optionally continuing with stage 2 for larger prime factors
//!
//! The algorithm is particularly effective when n has a factor p where p-1 or p+1
//! is smooth (has only small prime factors).
//!
//! # Examples
//!
//! ```
//! use rustmath_integers::Integer;
//! use rustmath_integers::ecm::ecm_factor;
//!
//! // Factor a composite number
//! let n = Integer::from(143); // 11 * 13
//! if let Some(factor) = ecm_factor(&n, 100, 1000, 10) {
//!     println!("Found factor: {}", factor);
//! }
//! ```

use crate::Integer;
use crate::prime::{is_prime, factor};
use rustmath_core::Ring;
use rand::Rng;

/// Point on an elliptic curve in projective coordinates (X:Y:Z)
///
/// Represents a point on an elliptic curve modulo n.
/// The affine coordinates are (X/Z, Y/Z) when Z ≠ 0.
/// If Z = 0, this represents the point at infinity.
#[derive(Debug, Clone)]
struct ECPoint {
    x: Integer,
    y: Integer,
    z: Integer,
}

impl ECPoint {
    /// Create the point at infinity
    fn infinity() -> Self {
        Self {
            x: Integer::zero(),
            y: Integer::one(),
            z: Integer::zero(),
        }
    }

    /// Check if this is the point at infinity
    fn is_infinity(&self) -> bool {
        self.z.is_zero()
    }

    /// Create a point from affine coordinates
    fn from_affine(x: Integer, y: Integer) -> Self {
        Self {
            x,
            y,
            z: Integer::one(),
        }
    }
}

/// Elliptic curve in short Weierstrass form: y² = x³ + ax + b (mod n)
#[derive(Debug, Clone)]
struct EllipticCurve {
    a: Integer,
    _b: Integer,
    n: Integer,
}

impl EllipticCurve {
    /// Create a new elliptic curve modulo n
    fn new(a: Integer, b: Integer, n: Integer) -> Self {
        Self { a, _b: b, n }
    }

    /// Compute modular inverse using extended GCD
    ///
    /// Returns None if gcd(a, n) > 1, which indicates we've found a factor
    fn mod_inverse(&self, a: &Integer) -> Option<Integer> {
        let (g, x, _) = a.extended_gcd(&self.n);

        if !g.is_one() {
            return None; // GCD > 1 means we found a factor!
        }

        let result = x % self.n.clone();
        Some(if result.signum() < 0 {
            result + self.n.clone()
        } else {
            result
        })
    }

    /// Add two points on the curve using the chord-and-tangent method
    ///
    /// Returns None if during computation we discover a factor of n
    fn add(&self, p: &ECPoint, q: &ECPoint) -> Option<ECPoint> {
        if p.is_infinity() {
            return Some(q.clone());
        }
        if q.is_infinity() {
            return Some(p.clone());
        }

        // For simplicity, convert to affine coordinates
        let p_x = &p.x % &self.n;
        let p_y = &p.y % &self.n;
        let q_x = &q.x % &self.n;
        let q_y = &q.y % &self.n;

        // Check if points are the same
        if p_x == q_x {
            if p_y == q_y {
                return self.double_point(p);
            } else {
                // Points are negatives of each other
                return Some(ECPoint::infinity());
            }
        }

        // Compute slope: λ = (q_y - p_y) / (q_x - p_x)
        let numerator = (q_y - p_y.clone()) % self.n.clone();
        let denominator = (q_x.clone() - p_x.clone()) % self.n.clone();

        let denominator = if denominator.signum() < 0 {
            denominator + self.n.clone()
        } else {
            denominator
        };

        let lambda = match self.mod_inverse(&denominator) {
            Some(inv) => (numerator * inv) % self.n.clone(),
            None => return None, // Found a factor!
        };

        // Compute result: x_r = λ² - x_p - x_q
        let x_r = (lambda.clone() * lambda.clone() - p_x.clone() - q_x.clone()) % self.n.clone();
        let x_r = if x_r.signum() < 0 { x_r + self.n.clone() } else { x_r };

        // Compute result: y_r = λ(x_p - x_r) - y_p
        let y_r = (lambda * (p_x - x_r.clone()) - p_y) % self.n.clone();
        let y_r = if y_r.signum() < 0 { y_r + self.n.clone() } else { y_r };

        Some(ECPoint::from_affine(x_r, y_r))
    }

    /// Double a point on the curve
    ///
    /// Returns None if during computation we discover a factor of n
    fn double_point(&self, p: &ECPoint) -> Option<ECPoint> {
        if p.is_infinity() {
            return Some(ECPoint::infinity());
        }

        let p_x = &p.x % &self.n;
        let p_y = &p.y % &self.n;

        if p_y.is_zero() {
            return Some(ECPoint::infinity());
        }

        // Compute slope: λ = (3x² + a) / (2y)
        let numerator = (Integer::from(3) * p_x.clone() * p_x.clone() + self.a.clone()) % self.n.clone();
        let numerator = if numerator.signum() < 0 {
            numerator + self.n.clone()
        } else {
            numerator
        };

        let denominator = (Integer::from(2) * p_y.clone()) % self.n.clone();
        let denominator = if denominator.signum() < 0 {
            denominator + self.n.clone()
        } else {
            denominator
        };

        let lambda = match self.mod_inverse(&denominator) {
            Some(inv) => (numerator * inv) % self.n.clone(),
            None => return None, // Found a factor!
        };

        // Compute result: x_r = λ² - 2x_p
        let x_r = (lambda.clone() * lambda.clone() - Integer::from(2) * p_x.clone()) % self.n.clone();
        let x_r = if x_r.signum() < 0 { x_r + self.n.clone() } else { x_r };

        // Compute result: y_r = λ(x_p - x_r) - y_p
        let y_r = (lambda * (p_x - x_r.clone()) - p_y) % self.n.clone();
        let y_r = if y_r.signum() < 0 { y_r + self.n.clone() } else { y_r };

        Some(ECPoint::from_affine(x_r, y_r))
    }

    /// Multiply a point by a scalar using binary method
    ///
    /// Returns (result_point, factor) where factor is Some if we found one
    fn scalar_multiply(&self, k: &Integer, p: &ECPoint) -> (Option<ECPoint>, Option<Integer>) {
        if k.is_zero() {
            return (Some(ECPoint::infinity()), None);
        }

        let mut result = ECPoint::infinity();
        let mut addend = p.clone();
        let mut scalar = k.clone();

        while !scalar.is_zero() {
            if scalar.is_odd() {
                match self.add(&result, &addend) {
                    Some(new_result) => result = new_result,
                    None => {
                        // Found a factor during addition
                        // Compute the actual factor
                        let factor = self.compute_factor_from_failed_inverse(&result, &addend);
                        return (None, Some(factor));
                    }
                }
            }

            match self.double_point(&addend) {
                Some(doubled) => addend = doubled,
                None => {
                    // Found a factor during doubling
                    let factor = self.compute_factor_from_failed_inverse(&addend, &addend);
                    return (None, Some(factor));
                }
            }

            scalar = scalar / Integer::from(2);
        }

        (Some(result), None)
    }

    /// Compute the factor when modular inverse fails
    fn compute_factor_from_failed_inverse(&self, p: &ECPoint, q: &ECPoint) -> Integer {
        let p_x = p.x.clone() % self.n.clone();
        let q_x = q.x.clone() % self.n.clone();

        let diff = if p_x == q_x {
            // Doubling case: denominator is 2*y
            (Integer::from(2) * p.y.clone()) % self.n.clone()
        } else {
            // Addition case: denominator is x_q - x_p
            (q_x - p_x) % self.n.clone()
        };

        let diff = if diff.signum() < 0 {
            diff + self.n.clone()
        } else {
            diff
        };

        diff.gcd(&self.n)
    }
}

/// Sieve of Eratosthenes to generate primes up to limit
fn sieve_of_eratosthenes(limit: u64) -> Vec<u64> {
    if limit < 2 {
        return vec![];
    }

    let mut is_prime = vec![true; (limit + 1) as usize];
    is_prime[0] = false;
    is_prime[1] = false;

    let sqrt_limit = (limit as f64).sqrt() as u64;
    for i in 2..=sqrt_limit {
        if is_prime[i as usize] {
            let mut j = i * i;
            while j <= limit {
                is_prime[j as usize] = false;
                j += i;
            }
        }
    }

    is_prime
        .iter()
        .enumerate()
        .filter(|(_, &prime)| prime)
        .map(|(i, _)| i as u64)
        .collect()
}

/// Compute k = product of small prime powers for stage 1
///
/// k = ∏ p^(floor(log_p(B1))) for all primes p ≤ B1
fn compute_stage1_multiplier(b1: u64) -> Integer {
    let primes = sieve_of_eratosthenes(b1);
    let mut k = Integer::one();

    for &p in &primes {
        // Find highest power of p that is ≤ b1
        let mut power = p;
        while power <= b1 / p {
            power *= p;
        }
        k = k * Integer::from(power as i64);
    }

    k
}

/// Attempt to factor n using a single elliptic curve
///
/// Returns Some(factor) if successful, None otherwise
fn try_one_curve(n: &Integer, b1: u64, b2: u64, seed: u64) -> Option<Integer> {
    let mut rng = rand::thread_rng();

    // Generate random curve parameters
    let sigma = Integer::from(rng.gen_range(6..1000000) as i64 + seed as i64);

    // Montgomery parameterization: use σ to generate curve
    // This is a standard technique to ensure we have a valid point
    let u = (sigma.clone() * sigma.clone() - Integer::from(5)) % n.clone();
    let v = (Integer::from(4) * sigma.clone()) % n.clone();

    let v_inv = {
        let (g, x, _) = v.extended_gcd(n);
        if !g.is_one() {
            return Some(g);
        }
        let result = x % n.clone();
        if result.signum() < 0 {
            result + n.clone()
        } else {
            result
        }
    };

    let _u3 = (u.clone() * u.clone() * u.clone()) % n.clone();
    let v3 = (v.clone() * v.clone() * v.clone()) % n.clone();

    let v3_inv = {
        let (g, x, _) = v3.extended_gcd(n);
        if !g.is_one() {
            return Some(g);
        }
        let result = x % n.clone();
        if result.signum() < 0 {
            result + n.clone()
        } else {
            result
        }
    };

    // Convert to Weierstrass form
    let x_p = (u.clone() * v_inv.clone()) % n.clone();
    let y_p = Integer::one();

    // Compute curve parameter a
    let a = ((v.clone() - u.clone()).pow(3) * (Integer::from(3) * u.clone() + v.clone())
             * v3_inv.clone() - Integer::from(2)) % n.clone();
    let a = if a.signum() < 0 { a + n.clone() } else { a };

    // Compute b from the point equation: y² = x³ + ax + b
    let b = (y_p.clone() * y_p.clone() - x_p.clone() * x_p.clone() * x_p.clone()
             - a.clone() * x_p.clone()) % n.clone();
    let b = if b.signum() < 0 { b + n.clone() } else { b };

    let curve = EllipticCurve::new(a, b, n.clone());
    let point = ECPoint::from_affine(x_p, y_p);

    // Stage 1: Compute k * P where k = product of small prime powers
    let k = compute_stage1_multiplier(b1);

    let (result, factor) = curve.scalar_multiply(&k, &point);
    if factor.is_some() {
        return factor;
    }

    let current_point = result?;

    // Stage 2: Check primes between b1 and b2
    // Simplified version - in production would use improved stage 2
    if b2 > b1 {
        let primes = sieve_of_eratosthenes(b2);
        let stage2_primes: Vec<u64> = primes.into_iter()
            .filter(|&p| p > b1 && p <= b2)
            .take(100) // Limit to avoid timeout
            .collect();

        for &p in &stage2_primes {
            let (new_point, factor) = curve.scalar_multiply(&Integer::from(p as i64), &current_point);
            if factor.is_some() {
                return factor;
            }
            if new_point.is_none() {
                break;
            }
        }
    }

    None
}

/// Factor n using the Elliptic Curve Method
///
/// # Arguments
///
/// * `n` - The number to factor
/// * `b1` - Stage 1 bound (typical: 2000-50000)
/// * `b2` - Stage 2 bound (typical: 10*b1 to 100*b1)
/// * `curves` - Number of curves to try
///
/// # Returns
///
/// A non-trivial factor of n, or None if factorization fails
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::ecm::ecm_factor;
///
/// let n = Integer::from(15770708441); // 135979 * 115979
/// if let Some(factor) = ecm_factor(&n, 10000, 100000, 20) {
///     println!("Found factor: {}", factor);
/// }
/// ```
pub fn ecm_factor(n: &Integer, b1: u64, b2: u64, curves: usize) -> Option<Integer> {
    // Quick checks
    if n.is_one() || n <= &Integer::one() {
        return None;
    }

    if n.is_even() {
        return Some(Integer::from(2));
    }

    // Check if n is prime
    if is_prime(n) {
        return None;
    }

    // Try trial division first for small factors
    let small_primes = [3i64, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    for &p in &small_primes {
        let p_int = Integer::from(p);
        if (n % &p_int).is_zero() {
            return Some(p_int);
        }
    }

    // Try multiple curves
    for i in 0..curves {
        if let Some(factor) = try_one_curve(n, b1, b2, i as u64) {
            // Make sure it's a non-trivial factor
            if !factor.is_one() && &factor != n {
                return Some(factor);
            }
        }
    }

    None
}

/// Complete factorization using ECM
///
/// Returns the complete prime factorization of n using ECM.
/// Falls back to trial division if ECM doesn't find factors.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::ecm::ecm_factor_complete;
///
/// let n = Integer::from(60); // 2² × 3 × 5
/// let factors = ecm_factor_complete(&n, 1000, 10000, 10);
/// // Returns [(2, 2), (3, 1), (5, 1)]
/// ```
pub fn ecm_factor_complete(n: &Integer, b1: u64, b2: u64, curves: usize) -> Vec<(Integer, u32)> {
    if n.is_zero() || n.is_one() {
        return vec![];
    }

    let mut factors = Vec::new();
    let mut remaining = n.abs();

    // Try ECM to find factors
    while remaining > Integer::one() && !is_prime(&remaining) {
        if let Some(factor) = ecm_factor(&remaining, b1, b2, curves) {
            // Count multiplicity
            let mut count = 0u32;
            while (&remaining % &factor).is_zero() {
                remaining = remaining / factor.clone();
                count += 1;
            }
            factors.push((factor, count));
        } else {
            // Fall back to standard factorization
            let mut standard_factors = factor(&remaining);
            factors.append(&mut standard_factors);
            break;
        }
    }

    // Add the remaining prime if any
    if remaining > Integer::one() {
        factors.push((remaining, 1));
    }

    factors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sieve_of_eratosthenes() {
        let primes = sieve_of_eratosthenes(20);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_ecm_small_composite() {
        // 143 = 11 × 13
        let n = Integer::from(143);
        let factor = ecm_factor(&n, 100, 1000, 10);
        assert!(factor.is_some());
        let f = factor.unwrap();
        assert!(f == Integer::from(11) || f == Integer::from(13));
    }

    #[test]
    fn test_ecm_larger_composite() {
        // 1037 = 17 × 61
        let n = Integer::from(1037);
        let factor = ecm_factor(&n, 500, 5000, 10);
        assert!(factor.is_some());
        let f = factor.unwrap();
        assert!((&n % &f).is_zero());
        assert!(f > Integer::one());
        assert!(&f < &n);
    }

    #[test]
    fn test_ecm_prime() {
        let p = Integer::from(97);
        let factor = ecm_factor(&p, 100, 1000, 10);
        assert!(factor.is_none());
    }

    #[test]
    fn test_ecm_even() {
        let n = Integer::from(144);
        let factor = ecm_factor(&n, 100, 1000, 5);
        assert_eq!(factor, Some(Integer::from(2)));
    }

    #[test]
    fn test_ecm_complete_factorization() {
        // 60 = 2² × 3 × 5
        let n = Integer::from(60);
        let factors = ecm_factor_complete(&n, 1000, 10000, 10);

        assert_eq!(factors.len(), 3);

        // Verify the factorization is correct
        let mut product = Integer::one();
        for (prime, exp) in &factors {
            for _ in 0..*exp {
                product = product * prime.clone();
            }
        }
        assert_eq!(product, n);
    }

    #[test]
    fn test_stage1_multiplier() {
        let k = compute_stage1_multiplier(10);
        // Should include 2³, 3², 5, 7
        // 2³ = 8, 3² = 9, so k should be divisible by 8, 9, 5, 7
        assert_eq!(k.clone() % Integer::from(8), Integer::zero());
        assert_eq!(k.clone() % Integer::from(9), Integer::zero());
        assert_eq!(k.clone() % Integer::from(5), Integer::zero());
        assert_eq!(k % Integer::from(7), Integer::zero());
    }
}
