//! # Universal Cyclotomic Field Module
//!
//! Implementation of the Universal Cyclotomic Field (UCF), the smallest subfield
//! of the complex numbers containing all roots of unity.
//!
//! ## Overview
//!
//! The Universal Cyclotomic Field is the maximal abelian extension of ℚ within ℂ.
//! It consists of all complex numbers that can be expressed as finite ℚ-linear
//! combinations of roots of unity.
//!
//! **Mathematical Definition**:
//! UCF = ℚ(ζₙ : n ∈ ℕ) where ζₙ = e^(2πi/n)
//!
//! ## Key Properties
//!
//! - **Exact**: All computations are exact (no floating point)
//! - **Infinite Degree**: [UCF : ℚ] = ∞
//! - **Abelian**: All Galois conjugates are roots of unity
//! - **Contains**: All cyclotomic fields ℚ(ζₙ) for any n
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::universal_cyclotomic_field::{UniversalCyclotomicField, E};
//!
//! // Create the universal cyclotomic field
//! let UCF = UniversalCyclotomicField::new();
//!
//! // Create a primitive 5th root of unity: ζ₅
//! // let z5 = E(5, 1);
//! ```
//!
//! ## Applications
//!
//! - Character tables of finite groups
//! - Representation theory
//! - Number theory (cyclotomic extensions)
//! - Quantum computing (gates and circuits)

use rustmath_core::{Field, Ring};
use rustmath_rationals::Rational;
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

/// Errors for universal cyclotomic field operations
#[derive(Debug, Clone, Error, PartialEq)]
pub enum UCFError {
    #[error("Invalid root of unity order: {0}")]
    InvalidOrder(String),

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// The Universal Cyclotomic Field
///
/// This field contains all roots of unity and their ℚ-linear combinations.
/// It's the maximal abelian extension of ℚ.
#[derive(Debug, Clone, PartialEq)]
pub struct UniversalCyclotomicField;

impl UniversalCyclotomicField {
    /// Creates the universal cyclotomic field
    pub fn new() -> Self {
        UniversalCyclotomicField
    }

    /// Checks if this field is exact
    pub fn is_exact(&self) -> bool {
        true // UCF is exact, not numerical
    }

    /// Returns the characteristic (0 for characteristic zero fields)
    pub fn characteristic(&self) -> u64 {
        0
    }

    /// Checks if the field is infinite
    pub fn is_infinite(&self) -> bool {
        true // [UCF : ℚ] = ∞
    }
}

impl Default for UniversalCyclotomicField {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for UniversalCyclotomicField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Universal Cyclotomic Field")
    }
}

/// An element of the Universal Cyclotomic Field
///
/// Elements are represented as sparse ℚ-linear combinations of roots of unity:
/// Σ aᵢ · ζₙ^kᵢ where aᵢ ∈ ℚ, ζₙ = e^(2πi/n)
///
/// ## Representation
///
/// Uses a sparse map: (n, k) → coefficient, where:
/// - n is the order of the root of unity
/// - k is the power
/// - coefficient is the rational multiplier
#[derive(Debug, Clone, PartialEq)]
pub struct UniversalCyclotomicFieldElement {
    /// Sparse representation: (order, power) → coefficient
    /// ζₙ^k has coefficient stored at (n, k)
    terms: HashMap<(u64, u64), Rational>,
    /// The conductor (LCM of all orders)
    conductor: u64,
}

impl UniversalCyclotomicFieldElement {
    /// Creates a new UCF element from terms
    pub fn new(terms: HashMap<(u64, u64), Rational>) -> Self {
        let conductor = if terms.is_empty() {
            1
        } else {
            terms.keys().map(|(n, _)| *n).fold(1, num_integer::lcm)
        };

        UniversalCyclotomicFieldElement { terms, conductor }
    }

    /// Creates zero element
    pub fn zero() -> Self {
        UniversalCyclotomicFieldElement {
            terms: HashMap::new(),
            conductor: 1,
        }
    }

    /// Creates unit element (1)
    pub fn one() -> Self {
        let mut terms = HashMap::new();
        terms.insert((1, 0), Rational::from(1));
        UniversalCyclotomicFieldElement {
            terms,
            conductor: 1,
        }
    }

    /// Creates a primitive n-th root of unity: ζₙ
    pub fn root_of_unity(n: u64) -> Result<Self, UCFError> {
        if n == 0 {
            return Err(UCFError::InvalidOrder("Order must be positive".to_string()));
        }

        let mut terms = HashMap::new();
        terms.insert((n, 1), Rational::from(1));
        Ok(UniversalCyclotomicFieldElement {
            terms,
            conductor: n,
        })
    }

    /// Creates ζₙ^k (k-th power of primitive n-th root)
    pub fn root_of_unity_power(n: u64, k: u64) -> Result<Self, UCFError> {
        if n == 0 {
            return Err(UCFError::InvalidOrder("Order must be positive".to_string()));
        }

        let k_reduced = k % n; // ζₙ^n = 1
        if k_reduced == 0 {
            return Ok(Self::one());
        }

        let mut terms = HashMap::new();
        terms.insert((n, k_reduced), Rational::from(1));
        Ok(UniversalCyclotomicFieldElement {
            terms,
            conductor: n,
        })
    }

    /// Returns the conductor (smallest n such that element ∈ ℚ(ζₙ))
    pub fn conductor(&self) -> u64 {
        self.conductor
    }

    /// Checks if this element is zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty() || self.terms.values().all(|c| c.is_zero())
    }

    /// Checks if this element is rational
    pub fn is_rational(&self) -> bool {
        self.terms.keys().all(|(n, k)| *n == 1 || *k == 0)
    }

    /// Checks if this element is real
    pub fn is_real(&self) -> bool {
        // An element is real if it equals its complex conjugate
        // ζₙ^k is real only when k = 0 or k = n/2 (for even n)
        self.terms.keys().all(|(n, k)| {
            *k == 0 || (*n % 2 == 0 && *k == *n / 2)
        })
    }

    /// Complex conjugate
    ///
    /// For ζₙ^k, the conjugate is ζₙ^(n-k) = ζₙ^(-k)
    pub fn conjugate(&self) -> Self {
        let mut new_terms = HashMap::new();
        for ((n, k), coeff) in &self.terms {
            let new_k = if *k == 0 { 0 } else { *n - *k };
            new_terms.insert((*n, new_k), coeff.clone());
        }
        UniversalCyclotomicFieldElement::new(new_terms)
    }
}

impl fmt::Display for UniversalCyclotomicFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for ((n, k), coeff) in &self.terms {
            if coeff.is_zero() {
                continue;
            }

            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if *n == 1 || *k == 0 {
                // Rational term
                write!(f, "{}", coeff)?;
            } else {
                // Root of unity term
                if !coeff.is_one() {
                    write!(f, "{}·", coeff)?;
                }
                write!(f, "ζ_{}^{}", n, k)?;
            }
        }
        Ok(())
    }
}

/// Creates a primitive n-th root of unity
///
/// This is the main interface for constructing UCF elements.
///
/// # Arguments
/// * `n` - The order of the root of unity
/// * `k` - The power (default 1 if omitted)
///
/// # Returns
/// ζₙ^k where ζₙ = e^(2πi/n)
pub fn E(n: u64, k: Option<u64>) -> Result<UniversalCyclotomicFieldElement, UCFError> {
    let power = k.unwrap_or(1);
    UniversalCyclotomicFieldElement::root_of_unity_power(n, power)
}

/// Computes square root of an integer in UCF
///
/// Returns √n as a UCF element if possible.
pub fn ucf_sqrt_int(n: i64) -> Result<UniversalCyclotomicFieldElement, UCFError> {
    if n == 0 {
        return Ok(UniversalCyclotomicFieldElement::zero());
    }

    if n < 0 {
        // √(-n) = i·√n where i = ζ₄
        let sqrt_abs = ucf_sqrt_int(-n)?;
        // Multiply by i = ζ₄
        return Err(UCFError::NotImplemented(
            "Square roots of negative integers not yet implemented".to_string()
        ));
    }

    // For positive integers, we'd need prime factorization and Gauss sums
    // Simplified: just handle perfect squares
    let sqrt_floor = (n as f64).sqrt() as i64;
    if sqrt_floor * sqrt_floor == n {
        let mut terms = HashMap::new();
        terms.insert((1, 0), Rational::from(sqrt_floor));
        Ok(UniversalCyclotomicFieldElement::new(terms))
    } else {
        Err(UCFError::NotImplemented(
            "Non-perfect-square roots not yet implemented".to_string()
        ))
    }
}

/// Morphism from UCF to QQbar (algebraic numbers)
///
/// This would convert UCF elements to algebraic number representations.
#[derive(Debug, Clone)]
pub struct UCFtoQQbar;

impl UCFtoQQbar {
    /// Creates the morphism
    pub fn new() -> Self {
        UCFtoQQbar
    }
}

impl Default for UCFtoQQbar {
    fn default() -> Self {
        Self::new()
    }
}

/// Late import placeholder
///
/// In SageMath, this defers GAP initialization. In Rust, this is a no-op.
pub fn late_import() {
    // No-op in Rust implementation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_creation() {
        let ucf = UniversalCyclotomicField::new();
        assert!(ucf.is_exact());
        assert!(ucf.is_infinite());
        assert_eq!(ucf.characteristic(), 0);
    }

    #[test]
    fn test_field_display() {
        let ucf = UniversalCyclotomicField::default();
        assert_eq!(format!("{}", ucf), "Universal Cyclotomic Field");
    }

    #[test]
    fn test_element_zero() {
        let zero = UniversalCyclotomicFieldElement::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.conductor(), 1);
    }

    #[test]
    fn test_element_one() {
        let one = UniversalCyclotomicFieldElement::one();
        assert!(!one.is_zero());
        assert!(one.is_rational());
    }

    #[test]
    fn test_root_of_unity() {
        let z5 = UniversalCyclotomicFieldElement::root_of_unity(5).unwrap();
        assert_eq!(z5.conductor(), 5);
        assert!(!z5.is_rational());
    }

    #[test]
    fn test_root_of_unity_power() {
        let z8_2 = UniversalCyclotomicFieldElement::root_of_unity_power(8, 2).unwrap();
        assert_eq!(z8_2.conductor(), 8);
    }

    #[test]
    fn test_root_of_unity_power_reduces() {
        // ζ₅^5 = 1
        let z5_5 = UniversalCyclotomicFieldElement::root_of_unity_power(5, 5).unwrap();
        assert!(z5_5.is_rational());
    }

    #[test]
    fn test_e_function() {
        let z7 = E(7, None).unwrap();
        assert_eq!(z7.conductor(), 7);

        let z7_3 = E(7, Some(3)).unwrap();
        assert_eq!(z7_3.conductor(), 7);
    }

    #[test]
    fn test_conjugate() {
        let z4 = E(4, None).unwrap(); // i
        let conj = z4.conjugate();
        // ζ₄^1 conjugate is ζ₄^3 = -i
        assert_eq!(conj.conductor(), 4);
    }

    #[test]
    fn test_ucf_sqrt_int_zero() {
        let sqrt0 = ucf_sqrt_int(0).unwrap();
        assert!(sqrt0.is_zero());
    }

    #[test]
    fn test_ucf_sqrt_int_perfect_square() {
        let sqrt4 = ucf_sqrt_int(4).unwrap();
        assert!(sqrt4.is_rational());
    }

    #[test]
    fn test_ucf_sqrt_int_non_perfect() {
        assert!(ucf_sqrt_int(2).is_err());
    }

    #[test]
    fn test_ucftoqqbar_creation() {
        let _morphism = UCFtoQQbar::new();
    }

    #[test]
    fn test_late_import() {
        late_import(); // Should not panic
    }
}
