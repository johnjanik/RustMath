//! # Eta Products
//!
//! This module provides eta products and eta quotients,
//! corresponding to SageMath's sage.modular.etaproducts module.
//!
//! The Dedekind eta function is η(τ) = q^(1/24) * ∏(1 - q^n) where q = e^(2πiτ).
//! An eta product is a product of powers of η(d*τ) for various divisors d of the level.

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// An element of the eta group
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EtaGroupElement {
    /// The level N
    level: Integer,
    /// Map from divisor d of N to the power r_d in η(d*τ)^{r_d}
    powers: HashMap<Integer, i64>,
}

impl EtaGroupElement {
    /// Create a new eta group element
    ///
    /// # Arguments
    /// * `level` - The level N
    /// * `powers` - Map from divisors to powers
    pub fn new(level: Integer, powers: HashMap<Integer, i64>) -> Self {
        EtaGroupElement { level, powers }
    }

    /// Get the level
    pub fn level(&self) -> &Integer {
        &self.level
    }

    /// Get the powers
    pub fn powers(&self) -> &HashMap<Integer, i64> {
        &self.powers
    }

    /// Get the power for a specific divisor
    pub fn get_power(&self, divisor: &Integer) -> i64 {
        *self.powers.get(divisor).unwrap_or(&0)
    }

    /// Set the power for a divisor
    pub fn set_power(&mut self, divisor: Integer, power: i64) {
        if power == 0 {
            self.powers.remove(&divisor);
        } else {
            self.powers.insert(divisor, power);
        }
    }

    /// Compute the order at infinity (q-expansion order)
    pub fn order_at_infinity(&self) -> Rational {
        let mut order = Rational::zero();

        for (d, &r) in &self.powers {
            // Each η(d*τ) contributes d/24 to the order
            let contribution = Rational::new(
                d.clone() * Integer::from(r),
                Integer::from(24),
            )
            .expect("denominator 24 is nonzero");
            order = order + contribution;
        }

        order
    }

    /// Compute the weight
    pub fn weight(&self) -> i64 {
        let mut w = 0i64;
        for &r in self.powers.values() {
            w += r;
        }
        w / 2
    }

    /// Check if this is a valid eta product (satisfies certain conditions)
    pub fn is_valid(&self) -> bool {
        // Check that all divisors actually divide the level
        for d in self.powers.keys() {
            if !(&self.level % d).is_zero() {
                return false;
            }
        }

        // Weight must be integral
        let total: i64 = self.powers.values().sum();
        if total % 2 != 0 {
            return false;
        }

        true
    }

    /// Multiply two eta products
    pub fn mul(&self, other: &EtaGroupElement) -> Option<EtaGroupElement> {
        if self.level != other.level {
            return None;
        }

        let mut new_powers = self.powers.clone();
        for (d, &r) in &other.powers {
            *new_powers.entry(d.clone()).or_insert(0) += r;
        }

        // Remove zero powers
        new_powers.retain(|_, &mut v| v != 0);

        Some(EtaGroupElement::new(self.level.clone(), new_powers))
    }

    /// Compute the inverse
    pub fn inverse(&self) -> EtaGroupElement {
        let mut new_powers = HashMap::new();
        for (d, &r) in &self.powers {
            new_powers.insert(d.clone(), -r);
        }
        EtaGroupElement::new(self.level.clone(), new_powers)
    }
}

/// The eta group for level N
#[derive(Debug, Clone)]
pub struct EtaGroup {
    /// The level
    level: Integer,
    /// Divisors of the level
    divisors: Vec<Integer>,
}

impl EtaGroup {
    /// Create a new eta group
    ///
    /// # Arguments
    /// * `level` - The level N
    pub fn new(level: Integer) -> Self {
        let divisors = compute_divisors(&level);
        EtaGroup { level, divisors }
    }

    /// Get the level
    pub fn level(&self) -> &Integer {
        &self.level
    }

    /// Get the divisors
    pub fn divisors(&self) -> &[Integer] {
        &self.divisors
    }

    /// Create the identity element
    pub fn identity(&self) -> EtaGroupElement {
        EtaGroupElement::new(self.level.clone(), HashMap::new())
    }

    /// Create an eta product from powers
    pub fn element(&self, powers: HashMap<Integer, i64>) -> EtaGroupElement {
        EtaGroupElement::new(self.level.clone(), powers)
    }
}

/// Create an eta group for level N
pub fn eta_group_class(N: Integer) -> EtaGroup {
    EtaGroup::new(N)
}

/// Create an eta product
///
/// # Arguments
/// * `N` - The level
/// * `powers` - Map from divisors to powers
pub fn eta_product(N: Integer, powers: HashMap<Integer, i64>) -> EtaGroupElement {
    EtaGroupElement::new(N, powers)
}

/// Compute divisors of n
fn compute_divisors(n: &Integer) -> Vec<Integer> {
    if n <= &Integer::zero() {
        return vec![];
    }

    let mut divisors = Vec::new();
    let mut i = Integer::one();
    let sqrt_n = n.sqrt().expect("n > 0 checked above");

    while &i <= &sqrt_n {
        if (n % &i).is_zero() {
            divisors.push(i.clone());
            let other = n / &i;
            if i != other {
                divisors.push(other);
            }
        }
        i = i + Integer::one();
    }

    divisors.sort();
    divisors
}

/// A family of cusps
#[derive(Debug, Clone)]
pub struct CuspFamily {
    /// The level
    level: Integer,
    /// Width of cusps in this family
    width: Integer,
    /// Cusps in this family
    cusps: Vec<(Integer, Integer)>, // (numerator, denominator) pairs
}

impl CuspFamily {
    /// Create a new cusp family
    pub fn new(level: Integer, width: Integer) -> Self {
        CuspFamily {
            level,
            width,
            cusps: Vec::new(),
        }
    }

    /// Get the level
    pub fn level(&self) -> &Integer {
        &self.level
    }

    /// Get the width
    pub fn width(&self) -> &Integer {
        &self.width
    }

    /// Get the cusps
    pub fn cusps(&self) -> &[(Integer, Integer)] {
        &self.cusps
    }

    /// Add a cusp to the family
    pub fn add_cusp(&mut self, numerator: Integer, denominator: Integer) {
        self.cusps.push((numerator, denominator));
    }

    /// Number of cusps in the family
    pub fn len(&self) -> usize {
        self.cusps.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.cusps.is_empty()
    }
}

/// Get all cusps for Gamma0(N)
pub fn all_cusps(N: Integer) -> Vec<(Integer, Integer)> {
    // Cusps of Gamma0(N) are represented as a/c where c | N and gcd(a, c) = 1
    let mut cusps = Vec::new();

    let divisors = compute_divisors(&N);
    for c in divisors {
        // For each divisor c, find all a with 0 <= a < c and gcd(a, c) = 1
        let mut a = Integer::zero();
        while &a < &c {
            if a.gcd(&c).is_one() {
                cusps.push((a.clone(), c.clone()));
            }
            a = a + Integer::one();
        }
    }

    // Add infinity (represented as 1/0)
    cusps.push((Integer::one(), Integer::zero()));

    cusps
}

/// Number of cusps of a given width for Gamma0(N)
pub fn num_cusps_of_width(N: &Integer, width: &Integer) -> usize {
    if !(N % width).is_zero() {
        return 0;
    }

    // Count cusps with the given width
    // This is related to the number of divisors
    let divisors = compute_divisors(N);
    divisors.iter().filter(|d| *d == width).count()
}

/// Compute the q-expansion of the eta function
///
/// # Arguments
/// * `prec` - Precision (number of terms)
///
/// # Returns
/// Coefficients of q^n for n = 1/24, 25/24, 49/24, ...
pub fn qexp_eta(prec: usize) -> Vec<i64> {
    // η(τ) = q^(1/24) * ∏(1 - q^n)
    // This computes coefficients for the product part
    let mut coeffs = vec![0i64; prec];

    if prec > 0 {
        coeffs[0] = 1;
    }

    // Compute product (1 - q)(1 - q^2)(1 - q^3)...
    for n in 1..prec {
        // Multiply by (1 - q^n)
        for k in (n..prec).rev() {
            coeffs[k] -= coeffs[k - n];
        }
    }

    coeffs
}

/// Find polynomial relations among eta products
pub fn eta_poly_relations(
    N: &Integer,
    degree: usize,
) -> Vec<Vec<i64>> {
    // This would find polynomial relations among eta products
    // For now, return empty (this is a complex computation)
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eta_group_element() {
        let mut powers = HashMap::new();
        powers.insert(Integer::one(), 24);

        let eta = EtaGroupElement::new(Integer::one(), powers);
        assert_eq!(eta.level(), &Integer::one());
        assert_eq!(eta.get_power(&Integer::one()), 24);
    }

    #[test]
    fn test_eta_group() {
        let G = EtaGroup::new(Integer::from(12));
        assert_eq!(G.level(), &Integer::from(12));
        assert!(!G.divisors().is_empty());
    }

    #[test]
    fn test_EtaGroup_class() {
        let G = eta_group_class(Integer::from(6));
        assert_eq!(G.level(), &Integer::from(6));
    }

    #[test]
    fn test_compute_divisors() {
        let divs = compute_divisors(&Integer::from(12));
        assert!(divs.contains(&Integer::one()));
        assert!(divs.contains(&Integer::from(12)));
        assert!(divs.contains(&Integer::from(2)));
        assert!(divs.contains(&Integer::from(3)));
        assert!(divs.contains(&Integer::from(4)));
        assert!(divs.contains(&Integer::from(6)));
    }

    #[test]
    fn test_eta_product() {
        // eta(tau)^24 = Delta(tau), the weight-12 modular discriminant: a
        // genuine integral-weight eta product (sum of exponents = 24, even).
        // The old test used eta^1, which has half-integral weight 1/2
        // (odd exponent sum) and is correctly rejected by `is_valid`.
        let mut powers = HashMap::new();
        powers.insert(Integer::one(), 24);

        let eta = eta_product(Integer::one(), powers);
        assert!(eta.is_valid());
    }

    #[test]
    fn test_all_cusps() {
        let cusps = all_cusps(Integer::from(2));
        assert!(!cusps.is_empty());
    }

    #[test]
    fn test_cusp_family() {
        let mut family = CuspFamily::new(Integer::from(12), Integer::from(4));
        family.add_cusp(Integer::one(), Integer::from(4));
        assert_eq!(family.len(), 1);
        assert!(!family.is_empty());
    }

    #[test]
    fn test_qexp_eta() {
        let coeffs = qexp_eta(10);
        assert_eq!(coeffs.len(), 10);
        assert_eq!(coeffs[0], 1);
    }

    #[test]
    fn test_order_at_infinity() {
        let mut powers = HashMap::new();
        powers.insert(Integer::one(), 24);

        let eta = EtaGroupElement::new(Integer::one(), powers);
        let order = eta.order_at_infinity();
        assert_eq!(order, Rational::one());
    }

    #[test]
    fn test_eta_multiply() {
        let mut powers1 = HashMap::new();
        powers1.insert(Integer::one(), 1);

        let mut powers2 = HashMap::new();
        powers2.insert(Integer::one(), 2);

        let eta1 = EtaGroupElement::new(Integer::one(), powers1);
        let eta2 = EtaGroupElement::new(Integer::one(), powers2);

        let product = eta1.mul(&eta2).unwrap();
        assert_eq!(product.get_power(&Integer::one()), 3);
    }
}
