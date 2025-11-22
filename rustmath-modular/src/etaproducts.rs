//! # Eta Products
//!
//! This module provides eta products and eta quotients,
//! corresponding to SageMath's sage.modular.etaproducts module.
//!
//! The Dedekind eta function is η(τ) = q^(1/24) * ∏(1 - q^n) where q = e^(2πiτ).
//! An eta product is a product of powers of η(d*τ) for various divisors d of the level.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Zero, One};
use num_integer::Integer;
use std::collections::HashMap;

/// An element of the eta group
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EtaGroupElement {
    /// The level N
    level: BigInt,
    /// Map from divisor d of N to the power r_d in η(d*τ)^{r_d}
    powers: HashMap<BigInt, i64>,
}

impl EtaGroupElement {
    /// Create a new eta group element
    ///
    /// # Arguments
    /// * `level` - The level N
    /// * `powers` - Map from divisors to powers
    pub fn new(level: BigInt, powers: HashMap<BigInt, i64>) -> Self {
        EtaGroupElement { level, powers }
    }

    /// Get the level
    pub fn level(&self) -> &BigInt {
        &self.level
    }

    /// Get the powers
    pub fn powers(&self) -> &HashMap<BigInt, i64> {
        &self.powers
    }

    /// Get the power for a specific divisor
    pub fn get_power(&self, divisor: &BigInt) -> i64 {
        *self.powers.get(divisor).unwrap_or(&0)
    }

    /// Set the power for a divisor
    pub fn set_power(&mut self, divisor: BigInt, power: i64) {
        if power == 0 {
            self.powers.remove(&divisor);
        } else {
            self.powers.insert(divisor, power);
        }
    }

    /// Compute the order at infinity (q-expansion order)
    pub fn order_at_infinity(&self) -> BigRational {
        let mut order = BigRational::zero();

        for (d, &r) in &self.powers {
            // Each η(d*τ) contributes d/24 to the order
            let contribution = BigRational::new(
                d.clone() * BigInt::from(r),
                BigInt::from(24),
            );
            order += contribution;
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
            if !self.level.is_multiple_of(d) {
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
    level: BigInt,
    /// Divisors of the level
    divisors: Vec<BigInt>,
}

impl EtaGroup {
    /// Create a new eta group
    ///
    /// # Arguments
    /// * `level` - The level N
    pub fn new(level: BigInt) -> Self {
        let divisors = compute_divisors(&level);
        EtaGroup { level, divisors }
    }

    /// Get the level
    pub fn level(&self) -> &BigInt {
        &self.level
    }

    /// Get the divisors
    pub fn divisors(&self) -> &[BigInt] {
        &self.divisors
    }

    /// Create the identity element
    pub fn identity(&self) -> EtaGroupElement {
        EtaGroupElement::new(self.level.clone(), HashMap::new())
    }

    /// Create an eta product from powers
    pub fn element(&self, powers: HashMap<BigInt, i64>) -> EtaGroupElement {
        EtaGroupElement::new(self.level.clone(), powers)
    }
}

/// Create an eta group for level N
pub fn eta_group_class(N: BigInt) -> EtaGroup {
    EtaGroup::new(N)
}

/// Create an eta product
///
/// # Arguments
/// * `N` - The level
/// * `powers` - Map from divisors to powers
pub fn eta_product(N: BigInt, powers: HashMap<BigInt, i64>) -> EtaGroupElement {
    EtaGroupElement::new(N, powers)
}

/// Compute divisors of n
fn compute_divisors(n: &BigInt) -> Vec<BigInt> {
    if n <= &BigInt::zero() {
        return vec![];
    }

    let mut divisors = Vec::new();
    let mut i = BigInt::one();
    let sqrt_n = n.sqrt();

    while &i <= &sqrt_n {
        if n.is_multiple_of(&i) {
            divisors.push(i.clone());
            let other = n / &i;
            if &i != &other {
                divisors.push(other);
            }
        }
        i += BigInt::one();
    }

    divisors.sort();
    divisors
}

/// A family of cusps
#[derive(Debug, Clone)]
pub struct CuspFamily {
    /// The level
    level: BigInt,
    /// Width of cusps in this family
    width: BigInt,
    /// Cusps in this family
    cusps: Vec<(BigInt, BigInt)>, // (numerator, denominator) pairs
}

impl CuspFamily {
    /// Create a new cusp family
    pub fn new(level: BigInt, width: BigInt) -> Self {
        CuspFamily {
            level,
            width,
            cusps: Vec::new(),
        }
    }

    /// Get the level
    pub fn level(&self) -> &BigInt {
        &self.level
    }

    /// Get the width
    pub fn width(&self) -> &BigInt {
        &self.width
    }

    /// Get the cusps
    pub fn cusps(&self) -> &[(BigInt, BigInt)] {
        &self.cusps
    }

    /// Add a cusp to the family
    pub fn add_cusp(&mut self, numerator: BigInt, denominator: BigInt) {
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
pub fn all_cusps(N: BigInt) -> Vec<(BigInt, BigInt)> {
    // Cusps of Gamma0(N) are represented as a/c where c | N and gcd(a, c) = 1
    let mut cusps = Vec::new();

    let divisors = compute_divisors(&N);
    for c in divisors {
        // For each divisor c, find all a with 0 <= a < c and gcd(a, c) = 1
        let mut a = BigInt::zero();
        while &a < &c {
            if a.gcd(&c).is_one() {
                cusps.push((a.clone(), c.clone()));
            }
            a += BigInt::one();
        }
    }

    // Add infinity (represented as 1/0)
    cusps.push((BigInt::one(), BigInt::zero()));

    cusps
}

/// Number of cusps of a given width for Gamma0(N)
pub fn num_cusps_of_width(N: &BigInt, width: &BigInt) -> usize {
    if !N.is_multiple_of(width) {
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
    N: &BigInt,
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
        powers.insert(BigInt::one(), 24);

        let eta = EtaGroupElement::new(BigInt::one(), powers);
        assert_eq!(eta.level(), &BigInt::one());
        assert_eq!(eta.get_power(&BigInt::one()), 24);
    }

    #[test]
    fn test_eta_group() {
        let G = EtaGroup::new(BigInt::from(12));
        assert_eq!(G.level(), &BigInt::from(12));
        assert!(!G.divisors().is_empty());
    }

    #[test]
    fn test_EtaGroup_class() {
        let G = EtaGroup_class(BigInt::from(6));
        assert_eq!(G.level(), &BigInt::from(6));
    }

    #[test]
    fn test_compute_divisors() {
        let divs = compute_divisors(&BigInt::from(12));
        assert!(divs.contains(&BigInt::one()));
        assert!(divs.contains(&BigInt::from(12)));
        assert!(divs.contains(&BigInt::from(2)));
        assert!(divs.contains(&BigInt::from(3)));
        assert!(divs.contains(&BigInt::from(4)));
        assert!(divs.contains(&BigInt::from(6)));
    }

    #[test]
    fn test_eta_product() {
        let mut powers = HashMap::new();
        powers.insert(BigInt::one(), 1);

        let eta = EtaProduct(BigInt::one(), powers);
        assert!(eta.is_valid());
    }

    #[test]
    fn test_all_cusps() {
        let cusps = AllCusps(BigInt::from(2));
        assert!(!cusps.is_empty());
    }

    #[test]
    fn test_cusp_family() {
        let mut family = CuspFamily::new(BigInt::from(12), BigInt::from(4));
        family.add_cusp(BigInt::one(), BigInt::from(4));
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
        powers.insert(BigInt::one(), 24);

        let eta = EtaGroupElement::new(BigInt::one(), powers);
        let order = eta.order_at_infinity();
        assert_eq!(order, BigRational::one());
    }

    #[test]
    fn test_eta_multiply() {
        let mut powers1 = HashMap::new();
        powers1.insert(BigInt::one(), 1);

        let mut powers2 = HashMap::new();
        powers2.insert(BigInt::one(), 2);

        let eta1 = EtaGroupElement::new(BigInt::one(), powers1);
        let eta2 = EtaGroupElement::new(BigInt::one(), powers2);

        let product = eta1.mul(&eta2).unwrap();
        assert_eq!(product.get_power(&BigInt::one()), 3);
    }
}
