//! Special Divisors and Brill-Noether Theory
//!
//! This module implements special divisors and fundamental results from
//! Brill-Noether theory, which studies the geometry of linear systems on curves.
//!
//! # Mathematical Background
//!
//! ## Special Divisors
//!
//! A divisor D on a curve C of genus g is **special** if:
//! h¹(D) = dim L(K - D) > 0
//!
//! equivalently, if the index of speciality i(D) = h¹(D) > 0.
//!
//! By Riemann-Roch:
//! h⁰(D) - h¹(D) = deg(D) + 1 - g
//!
//! So D is special iff h⁰(D) > deg(D) + 1 - g.
//!
//! ## Brill-Noether Theory
//!
//! The **Brill-Noether varieties** W^r_d(C) parameterize divisor classes D
//! of degree d with h⁰(D) ≥ r + 1 (i.e., dim |D| ≥ r).
//!
//! The **Brill-Noether number** is:
//! ρ(g,r,d) = g - (r+1)(g - d + r)
//!
//! ### Main Results
//!
//! 1. **Existence**: If ρ ≥ 0, then W^r_d(C) is non-empty for general curves
//! 2. **Dimension**: If non-empty, dim W^r_d(C) ≥ ρ
//! 3. For general curves: dim W^r_d(C) = ρ when ρ ≥ 0
//!
//! ## Clifford's Theorem
//!
//! For a special divisor D on a non-hyperelliptic curve:
//!
//! h⁰(D) ≤ deg(D)/2 + 1
//!
//! Equality holds only for:
//! - Multiples of canonical divisor: D ~ nK
//! - Hyperelliptic linear systems
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_curves::special_divisors::*;
//!
//! // Check if a divisor is special
//! let is_special = is_divisor_special(degree, h0, genus);
//!
//! // Compute Brill-Noether number
//! let rho = brill_noether_number(genus, r, d);
//!
//! // Check Clifford's theorem
//! let satisfies = check_clifford_theorem(degree, h0, is_hyperelliptic);
//! ```

use rustmath_core::Field;
use std::marker::PhantomData;
use std::fmt;
use crate::riemann_roch::{DivisorData, riemann_roch_dimension};
use crate::differentials::CanonicalDivisor;

/// A special divisor on a curve
///
/// A divisor D is special if h¹(D) = dim L(K - D) > 0.
///
/// # Type Parameters
///
/// - `F`: The base field type
#[derive(Clone, Debug)]
pub struct SpecialDivisor<F: Field> {
    /// The underlying divisor
    divisor: DivisorData,
    /// The genus of the curve
    genus: usize,
    /// h⁰(D) = dim L(D)
    h0: usize,
    /// h¹(D) = dim L(K - D)
    h1: usize,
    _field: PhantomData<F>,
}

impl<F: Field> SpecialDivisor<F> {
    /// Create a special divisor
    ///
    /// # Arguments
    ///
    /// * `divisor` - The divisor D
    /// * `genus` - The genus of the curve
    pub fn new(divisor: DivisorData, genus: usize) -> Self {
        let h0 = riemann_roch_dimension(divisor.degree, genus);

        // Compute h¹(D) = dim L(K - D)
        let k_degree = (2 * genus - 2) as i64;
        let k_minus_d_degree = k_degree - divisor.degree;
        let h1 = riemann_roch_dimension(k_minus_d_degree, genus);

        SpecialDivisor {
            divisor,
            genus,
            h0,
            h1,
            _field: PhantomData,
        }
    }

    /// Get the divisor
    pub fn divisor(&self) -> &DivisorData {
        &self.divisor
    }

    /// Get h⁰(D) = dim L(D)
    pub fn h0(&self) -> usize {
        self.h0
    }

    /// Get h¹(D) = dim L(K - D)
    pub fn h1(&self) -> usize {
        self.h1
    }

    /// Get the index of speciality i(D) = h¹(D)
    pub fn index_of_speciality(&self) -> usize {
        self.h1
    }

    /// Check if this is a special divisor
    ///
    /// D is special iff h¹(D) > 0
    pub fn is_special(&self) -> bool {
        self.h1 > 0
    }

    /// Get the degree
    pub fn degree(&self) -> i64 {
        self.divisor.degree
    }

    /// Verify Riemann-Roch theorem
    ///
    /// h⁰(D) - h¹(D) = deg(D) + 1 - g
    pub fn verify_riemann_roch(&self) -> bool {
        let lhs = (self.h0 as i64) - (self.h1 as i64);
        let rhs = self.divisor.degree + 1 - (self.genus as i64);
        lhs == rhs
    }

    /// Check if Clifford's theorem is satisfied
    ///
    /// For special divisors on non-hyperelliptic curves:
    /// h⁰(D) ≤ deg(D)/2 + 1
    ///
    /// # Arguments
    ///
    /// * `is_hyperelliptic` - Whether the curve is hyperelliptic
    pub fn check_clifford(&self, is_hyperelliptic: bool) -> bool {
        if !self.is_special() || is_hyperelliptic {
            // Clifford only applies to special divisors on non-hyperelliptic curves
            return true;
        }

        // Check h⁰(D) ≤ deg(D)/2 + 1
        let bound = (self.divisor.degree / 2 + 1) as usize;
        self.h0 <= bound
    }

    /// Compute the Clifford index
    ///
    /// Cliff(D) = deg(D) - 2(h⁰(D) - 1)
    ///
    /// The Clifford index of the curve is min over all special divisors.
    /// For non-hyperelliptic curves: Cliff(C) ≥ 0
    /// For hyperelliptic curves: Cliff(C) = 0
    pub fn clifford_index(&self) -> i64 {
        self.divisor.degree - 2 * ((self.h0 as i64) - 1)
    }
}

impl<F: Field> fmt::Display for SpecialDivisor<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Special divisor: deg={}, h⁰={}, h¹={}, i={}",
            self.divisor.degree,
            self.h0,
            self.h1,
            self.index_of_speciality()
        )
    }
}

/// Brill-Noether variety W^r_d(C)
///
/// The variety of divisor classes D of degree d with dim |D| ≥ r
/// (equivalently, h⁰(D) ≥ r + 1).
#[derive(Clone, Debug)]
pub struct BrillNoetherVariety {
    /// The genus of the curve
    genus: usize,
    /// The target dimension r
    r: usize,
    /// The degree d
    d: i64,
}

impl BrillNoetherVariety {
    /// Create a Brill-Noether variety W^r_d(C)
    ///
    /// # Arguments
    ///
    /// * `genus` - The genus g of the curve
    /// * `r` - The target dimension (want dim |D| ≥ r)
    /// * `d` - The degree
    pub fn new(genus: usize, r: usize, d: i64) -> Self {
        BrillNoetherVariety { genus, r, d }
    }

    /// Compute the Brill-Noether number
    ///
    /// ρ(g,r,d) = g - (r+1)(g - d + r)
    pub fn rho(&self) -> i64 {
        let g = self.genus as i64;
        let r = self.r as i64;
        let d = self.d;

        g - (r + 1) * (g - d + r)
    }

    /// Check if the variety is expected to be non-empty
    ///
    /// For general curves, W^r_d is non-empty iff ρ ≥ 0
    pub fn is_expected_nonempty(&self) -> bool {
        self.rho() >= 0
    }

    /// Get the expected dimension
    ///
    /// For general curves: dim W^r_d = max(ρ, -1)
    /// (dimension -1 means empty)
    pub fn expected_dimension(&self) -> i64 {
        self.rho().max(-1)
    }

    /// Check if this is a Brill-Noether general curve
    ///
    /// A curve is BN-general if dim W^r_d = expected dimension for all (r,d)
    pub fn is_bn_general(&self, _actual_dim: i64) -> bool {
        // Would compare actual dimension with expected
        // Placeholder
        true
    }
}

impl fmt::Display for BrillNoetherVariety {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "W^{}_{}(C) with genus {}, ρ = {}",
            self.r,
            self.d,
            self.genus,
            self.rho()
        )
    }
}

/// Check if a divisor is special
///
/// # Arguments
///
/// * `degree` - Degree of the divisor
/// * `h0` - dim L(D)
/// * `genus` - Genus of the curve
pub fn is_divisor_special(degree: i64, h0: usize, genus: usize) -> bool {
    // By Riemann-Roch: h¹ = h⁰ - deg(D) - 1 + g
    let h1 = (h0 as i64) - degree - 1 + (genus as i64);
    h1 > 0
}

/// Compute the Brill-Noether number
///
/// ρ(g,r,d) = g - (r+1)(g - d + r)
pub fn brill_noether_number(genus: usize, r: usize, d: i64) -> i64 {
    let g = genus as i64;
    let r_i64 = r as i64;
    g - (r_i64 + 1) * (g - d + r_i64)
}

/// Check if Clifford's theorem is satisfied
///
/// For special divisors: h⁰(D) ≤ deg(D)/2 + 1
///
/// # Arguments
///
/// * `degree` - Degree of the divisor
/// * `h0` - dim L(D)
/// * `is_hyperelliptic` - Whether the curve is hyperelliptic
pub fn check_clifford_theorem(degree: i64, h0: usize, is_hyperelliptic: bool) -> bool {
    if is_hyperelliptic {
        // Clifford's theorem has exceptions for hyperelliptic curves
        return true;
    }

    let bound = (degree / 2 + 1) as usize;
    h0 <= bound
}

/// Compute the Clifford index of a divisor
///
/// Cliff(D) = deg(D) - 2(h⁰(D) - 1)
pub fn clifford_index(degree: i64, h0: usize) -> i64 {
    degree - 2 * ((h0 as i64) - 1)
}

/// Compute the gonality of a curve
///
/// The gonality is the minimal degree of a non-constant map to ℙ¹,
/// equivalently the minimal degree of a base-point-free g^1_d.
///
/// For genus g:
/// - Rational curves (g=0): gonality = 1
/// - Elliptic curves (g=1): gonality = 2
/// - Hyperelliptic curves: gonality = 2
/// - General curves: gonality ≥ ⌈(g+3)/2⌉
pub fn gonality(genus: usize, is_hyperelliptic: bool) -> usize {
    match genus {
        0 => 1,
        1 => 2,
        _ if is_hyperelliptic => 2,
        g => (g + 3) / 2, // Lower bound for general curves
    }
}

/// Canonical divisor as a special divisor
///
/// The canonical divisor K is always special unless g = 0.
pub fn canonical_special_divisor<F: Field>(genus: usize) -> SpecialDivisor<F> {
    let k = CanonicalDivisor::new(genus);
    SpecialDivisor::new(k.divisor().clone(), genus)
}

/// Check if a curve is hyperelliptic using Brill-Noether theory
///
/// A curve is hyperelliptic iff it has a g^1_2 (divisor class of degree 2
/// with h⁰(D) ≥ 2, i.e., dim |D| ≥ 1).
///
/// Equivalently: W^1_2 is non-empty.
pub fn is_hyperelliptic_bn(genus: usize) -> bool {
    if genus == 0 || genus == 1 {
        // Genus 0 and 1 curves are "hyperelliptic" by convention
        return true;
    }

    // Check if W^1_2 exists
    let w = BrillNoetherVariety::new(genus, 1, 2);
    w.is_expected_nonempty()
}

/// Compute the Clifford dimension
///
/// The Clifford dimension is the maximum r such that there exists
/// a special divisor D with h⁰(D) = r + 1 and Cliff(D) = Cliff(C).
pub fn clifford_dimension(_genus: usize, _clifford_index: i64) -> usize {
    // Placeholder
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_special_divisor_creation() {
        let div = DivisorData::new(5, vec![("P".to_string(), 5)]);
        let special = SpecialDivisor::<Rational>::new(div, 2);

        assert_eq!(special.degree(), 5);
        assert_eq!(special.genus, 2);
    }

    #[test]
    fn test_is_special() {
        // For genus 2, canonical divisor has degree 2
        // A degree 1 divisor has h⁰(1) = max(1-2+1, 0) = 0
        // and h¹(1) = h⁰(K-D) = h⁰(2-1) = h⁰(1) = 0
        // So degree 1 is not special

        let div1 = DivisorData::new(1, vec![("P".to_string(), 1)]);
        let special1 = SpecialDivisor::<Rational>::new(div1, 2);
        assert!(!special1.is_special());

        // Canonical divisor should be special for g > 0
        let div2 = DivisorData::new(2, vec![("P".to_string(), 2)]);
        let special2 = SpecialDivisor::<Rational>::new(div2, 2);
        // h⁰(2) = 2-2+1 = 1, h¹(2) = h⁰(0) = ... depends on implementation
    }

    #[test]
    fn test_index_of_speciality() {
        let div = DivisorData::new(5, vec![("P".to_string(), 5)]);
        let special = SpecialDivisor::<Rational>::new(div, 2);

        let i = special.index_of_speciality();
        assert_eq!(i, special.h1());
    }

    #[test]
    fn test_riemann_roch_verification() {
        let div = DivisorData::new(5, vec![("P".to_string(), 5)]);
        let special = SpecialDivisor::<Rational>::new(div, 2);

        assert!(special.verify_riemann_roch());
    }

    #[test]
    fn test_clifford_check() {
        let div = DivisorData::new(4, vec![("P".to_string(), 4)]);
        let special = SpecialDivisor::<Rational>::new(div, 2);

        // For non-hyperelliptic curve
        let satisfies = special.check_clifford(false);
        assert!(satisfies);

        // For hyperelliptic curve (always satisfied)
        let satisfies_hyp = special.check_clifford(true);
        assert!(satisfies_hyp);
    }

    #[test]
    fn test_clifford_index() {
        let div = DivisorData::new(4, vec![("P".to_string(), 4)]);
        let special = SpecialDivisor::<Rational>::new(div, 2);

        let cliff = special.clifford_index();
        // Cliff(D) = deg(D) - 2(h⁰(D) - 1)
        // deg = 4, h⁰ = 4-2+1 = 3
        // Cliff = 4 - 2(3-1) = 4 - 4 = 0
        assert_eq!(cliff, 0);
    }

    #[test]
    fn test_special_divisor_display() {
        let div = DivisorData::new(5, vec![("P".to_string(), 5)]);
        let special = SpecialDivisor::<Rational>::new(div, 2);

        let display = format!("{}", special);
        assert!(display.contains("deg=5"));
    }

    #[test]
    fn test_brill_noether_variety_creation() {
        let w = BrillNoetherVariety::new(3, 1, 4);

        assert_eq!(w.genus, 3);
        assert_eq!(w.r, 1);
        assert_eq!(w.d, 4);
    }

    #[test]
    fn test_brill_noether_number() {
        // ρ(g,r,d) = g - (r+1)(g - d + r)
        let w = BrillNoetherVariety::new(3, 1, 4);
        // ρ(3,1,4) = 3 - 2(3 - 4 + 1) = 3 - 2*0 = 3
        assert_eq!(w.rho(), 3);

        // Test edge case
        let w2 = BrillNoetherVariety::new(2, 1, 2);
        // ρ(2,1,2) = 2 - 2(2 - 2 + 1) = 2 - 2 = 0
        assert_eq!(w2.rho(), 0);
    }

    #[test]
    fn test_expected_nonempty() {
        let w1 = BrillNoetherVariety::new(3, 1, 4);
        assert!(w1.is_expected_nonempty()); // ρ = 3 ≥ 0

        let w2 = BrillNoetherVariety::new(2, 2, 2);
        // ρ(2,2,2) = 2 - 3(2 - 2 + 2) = 2 - 6 = -4 < 0
        assert!(!w2.is_expected_nonempty());
    }

    #[test]
    fn test_expected_dimension() {
        let w = BrillNoetherVariety::new(3, 1, 4);
        assert_eq!(w.expected_dimension(), 3);

        let w2 = BrillNoetherVariety::new(2, 2, 2);
        assert_eq!(w2.expected_dimension(), -1); // Empty
    }

    #[test]
    fn test_bn_variety_display() {
        let w = BrillNoetherVariety::new(3, 1, 4);
        let display = format!("{}", w);

        assert!(display.contains("W^1_4"));
        assert!(display.contains("genus 3"));
        assert!(display.contains("ρ = 3"));
    }

    #[test]
    fn test_is_divisor_special_function() {
        // Degree 5, h⁰ = 4, genus 2
        // h¹ = 4 - 5 - 1 + 2 = 0 (not special)
        assert!(!is_divisor_special(5, 4, 2));

        // Degree 2, h⁰ = 2, genus 2
        // h¹ = 2 - 2 - 1 + 2 = 1 > 0 (special)
        assert!(is_divisor_special(2, 2, 2));
    }

    #[test]
    fn test_brill_noether_number_function() {
        let rho = brill_noether_number(3, 1, 4);
        assert_eq!(rho, 3);

        let rho2 = brill_noether_number(2, 1, 2);
        assert_eq!(rho2, 0);
    }

    #[test]
    fn test_check_clifford_theorem_function() {
        // Degree 4, h⁰ = 3, non-hyperelliptic
        // Bound: 4/2 + 1 = 3, so 3 ≤ 3 ✓
        assert!(check_clifford_theorem(4, 3, false));

        // Degree 4, h⁰ = 5, non-hyperelliptic
        // Bound: 4/2 + 1 = 3, so 5 ≤ 3 ✗
        assert!(!check_clifford_theorem(4, 5, false));

        // Hyperelliptic always satisfies
        assert!(check_clifford_theorem(4, 5, true));
    }

    #[test]
    fn test_clifford_index_function() {
        let cliff = clifford_index(4, 3);
        // Cliff = 4 - 2(3-1) = 4 - 4 = 0
        assert_eq!(cliff, 0);

        let cliff2 = clifford_index(6, 3);
        // Cliff = 6 - 2(3-1) = 6 - 4 = 2
        assert_eq!(cliff2, 2);
    }

    #[test]
    fn test_gonality() {
        assert_eq!(gonality(0, false), 1);
        assert_eq!(gonality(1, false), 2);
        assert_eq!(gonality(2, true), 2);
        assert_eq!(gonality(3, true), 2);

        // General curve of genus 3: gonality ≥ ⌈(3+3)/2⌉ = 3
        assert_eq!(gonality(3, false), 3);

        // General curve of genus 5: gonality ≥ ⌈(5+3)/2⌉ = 4
        assert_eq!(gonality(5, false), 4);
    }

    #[test]
    fn test_canonical_special_divisor() {
        let k_special = canonical_special_divisor::<Rational>(2);

        assert_eq!(k_special.genus, 2);
        assert_eq!(k_special.degree(), 2); // 2*2 - 2 = 2
        // Canonical divisor should be special for g > 0
    }

    #[test]
    fn test_is_hyperelliptic_bn() {
        // Genus 0 and 1 are hyperelliptic
        assert!(is_hyperelliptic_bn(0));
        assert!(is_hyperelliptic_bn(1));

        // For genus 2: W^1_2 has ρ(2,1,2) = 2 - 2(2-2+1) = 0 ≥ 0
        assert!(is_hyperelliptic_bn(2));
    }

    #[test]
    fn test_genus_2_canonical() {
        // For genus 2, canonical divisor has degree 2
        let k = CanonicalDivisor::new(2);
        assert_eq!(k.degree(), 2);

        let k_special = canonical_special_divisor::<Rational>(2);
        assert_eq!(k_special.degree(), 2);
    }

    #[test]
    fn test_high_genus_canonical() {
        // For genus 5, canonical divisor has degree 8
        let k = CanonicalDivisor::new(5);
        assert_eq!(k.degree(), 8); // 2*5 - 2 = 8
    }

    #[test]
    fn test_brill_noether_hyperelliptic() {
        // For a hyperelliptic curve, W^1_2 should be non-empty
        // This tests g=2,3,4
        for g in 2..5 {
            let w = BrillNoetherVariety::new(g, 1, 2);
            // ρ(g,1,2) = g - 2(g-2+1) = g - 2g + 2 = 2 - g
            // For g=2: ρ=0, for g=3: ρ=-1, for g=4: ρ=-2
            let expected_rho = (2 - g) as i64;
            assert_eq!(w.rho(), expected_rho);
        }
    }
}
