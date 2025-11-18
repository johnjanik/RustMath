//! # Tropical Semiring
//!
//! This module implements tropical semirings for tropical geometry.
//!
//! ## Overview
//!
//! A tropical semiring replaces the usual arithmetic operations with:
//! - Tropical addition: `a ⊕ b = min(a, b)` (or max for max-plus algebra)
//! - Tropical multiplication: `a ⊗ b = a + b`
//!
//! ## Theory
//!
//! Tropical geometry studies:
//! - Piecewise-linear analogues of algebraic varieties
//! - Degenerations of classical algebraic geometry
//! - Applications in optimization, phylogenetics, and economics
//!
//! The tropical semiring (ℝ ∪ {∞}, ⊕, ⊗) where:
//! - Additive identity: ∞
//! - Multiplicative identity: 0
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::semirings::tropical_semiring::{TropicalSemiring, TropicalSemiringElement};
//!
//! let semiring = TropicalSemiring::new_min(); // min-plus algebra
//! let a = TropicalSemiringElement::from_value(3.0);
//! let b = TropicalSemiringElement::from_value(5.0);
//!
//! // Tropical addition: min(3, 5) = 3
//! // Tropical multiplication: 3 + 5 = 8
//! ```

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul};

/// Type of tropical semiring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TropicalType {
    /// Min-plus algebra: ⊕ = min, ⊗ = +
    Min,
    /// Max-plus algebra: ⊕ = max, ⊗ = +
    Max,
}

/// Tropical semiring structure
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalSemiring {
    /// Type of tropical semiring (min or max)
    tropical_type: TropicalType,
}

impl TropicalSemiring {
    /// Create a min-plus tropical semiring
    ///
    /// # Examples
    /// ```
    /// use rustmath_rings::semirings::tropical_semiring::TropicalSemiring;
    ///
    /// let semiring = TropicalSemiring::new_min();
    /// ```
    pub fn new_min() -> Self {
        Self {
            tropical_type: TropicalType::Min,
        }
    }

    /// Create a max-plus tropical semiring
    pub fn new_max() -> Self {
        Self {
            tropical_type: TropicalType::Max,
        }
    }

    /// Get the tropical type
    pub fn tropical_type(&self) -> TropicalType {
        self.tropical_type
    }

    /// Get the additive identity (infinity)
    pub fn zero(&self) -> TropicalSemiringElement {
        TropicalSemiringElement::infinity(self.tropical_type)
    }

    /// Get the multiplicative identity (0)
    pub fn one(&self) -> TropicalSemiringElement {
        TropicalSemiringElement::from_value_with_type(0.0, self.tropical_type)
    }

    /// Create an element from a value
    pub fn from_value(&self, value: f64) -> TropicalSemiringElement {
        TropicalSemiringElement::from_value_with_type(value, self.tropical_type)
    }
}

impl fmt::Display for TropicalSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.tropical_type {
            TropicalType::Min => write!(f, "Tropical Semiring (min-plus)"),
            TropicalType::Max => write!(f, "Tropical Semiring (max-plus)"),
        }
    }
}

/// Element of a tropical semiring
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalSemiringElement {
    /// The value (or None for infinity)
    value: Option<f64>,
    /// Type of tropical semiring this element belongs to
    tropical_type: TropicalType,
}

impl TropicalSemiringElement {
    /// Create a finite tropical element
    ///
    /// # Examples
    /// ```
    /// use rustmath_rings::semirings::tropical_semiring::TropicalSemiringElement;
    ///
    /// let elem = TropicalSemiringElement::from_value(3.5);
    /// ```
    pub fn from_value(value: f64) -> Self {
        Self {
            value: Some(value),
            tropical_type: TropicalType::Min, // Default to min
        }
    }

    /// Create a tropical element with specified type
    pub fn from_value_with_type(value: f64, tropical_type: TropicalType) -> Self {
        Self {
            value: Some(value),
            tropical_type,
        }
    }

    /// Create the infinity element (additive identity)
    pub fn infinity(tropical_type: TropicalType) -> Self {
        Self {
            value: None,
            tropical_type,
        }
    }

    /// Check if this is infinity
    pub fn is_infinity(&self) -> bool {
        self.value.is_none()
    }

    /// Check if this is zero (multiplicative identity)
    pub fn is_zero(&self) -> bool {
        self.value == Some(0.0)
    }

    /// Get the underlying value (None if infinity)
    pub fn value(&self) -> Option<f64> {
        self.value
    }

    /// Get the tropical type
    pub fn tropical_type(&self) -> TropicalType {
        self.tropical_type
    }

    /// Tropical addition (min or max)
    pub fn tropical_add(&self, other: &Self) -> Self {
        if self.tropical_type != other.tropical_type {
            panic!("Cannot add elements from different tropical semirings");
        }

        let value = match (self.value, other.value) {
            (None, _) => other.value,
            (_, None) => self.value,
            (Some(a), Some(b)) => match self.tropical_type {
                TropicalType::Min => Some(a.min(b)),
                TropicalType::Max => Some(a.max(b)),
            },
        };

        Self {
            value,
            tropical_type: self.tropical_type,
        }
    }

    /// Tropical multiplication (addition)
    pub fn tropical_mul(&self, other: &Self) -> Self {
        if self.tropical_type != other.tropical_type {
            panic!("Cannot multiply elements from different tropical semirings");
        }

        let value = match (self.value, other.value) {
            (None, _) | (_, None) => None,
            (Some(a), Some(b)) => Some(a + b),
        };

        Self {
            value,
            tropical_type: self.tropical_type,
        }
    }

    /// Tropical power
    pub fn tropical_pow(&self, n: i32) -> Self {
        let value = self.value.map(|v| v * n as f64);
        Self {
            value,
            tropical_type: self.tropical_type,
        }
    }
}

impl Add for TropicalSemiringElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.tropical_add(&other)
    }
}

impl Mul for TropicalSemiringElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.tropical_mul(&other)
    }
}

impl PartialOrd for TropicalSemiringElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self.value, other.value) {
            (None, None) => Some(Ordering::Equal),
            (None, Some(_)) => Some(Ordering::Greater), // ∞ > any finite value
            (Some(_), None) => Some(Ordering::Less),
            (Some(a), Some(b)) => a.partial_cmp(&b),
        }
    }
}

impl fmt::Display for TropicalSemiringElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value {
            None => write!(f, "∞"),
            Some(v) => write!(f, "{}", v),
        }
    }
}

/// Morphism between tropical semirings
///
/// Maps elements from one tropical semiring to another.
#[derive(Debug, Clone)]
pub struct TropicalToTropical {
    /// Source semiring type
    source_type: TropicalType,
    /// Target semiring type
    target_type: TropicalType,
}

impl TropicalToTropical {
    /// Create a new morphism
    pub fn new(source_type: TropicalType, target_type: TropicalType) -> Self {
        Self {
            source_type,
            target_type,
        }
    }

    /// Apply the morphism
    pub fn apply(&self, element: &TropicalSemiringElement) -> TropicalSemiringElement {
        if element.tropical_type != self.source_type {
            panic!("Element type doesn't match source type");
        }

        // Convert between min and max by negating values
        if self.source_type == self.target_type {
            element.clone()
        } else {
            let value = element.value.map(|v| -v);
            TropicalSemiringElement {
                value,
                tropical_type: self.target_type,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_semiring_creation() {
        let min_sr = TropicalSemiring::new_min();
        assert_eq!(min_sr.tropical_type(), TropicalType::Min);

        let max_sr = TropicalSemiring::new_max();
        assert_eq!(max_sr.tropical_type(), TropicalType::Max);
    }

    #[test]
    fn test_zero_one() {
        let sr = TropicalSemiring::new_min();

        let zero = sr.zero();
        assert!(zero.is_infinity());

        let one = sr.one();
        assert!(one.is_zero());
    }

    #[test]
    fn test_tropical_element_creation() {
        let elem = TropicalSemiringElement::from_value(3.5);
        assert_eq!(elem.value(), Some(3.5));
        assert!(!elem.is_infinity());

        let inf = TropicalSemiringElement::infinity(TropicalType::Min);
        assert!(inf.is_infinity());
    }

    #[test]
    fn test_min_tropical_addition() {
        let a = TropicalSemiringElement::from_value_with_type(3.0, TropicalType::Min);
        let b = TropicalSemiringElement::from_value_with_type(5.0, TropicalType::Min);

        let sum = a.tropical_add(&b);
        assert_eq!(sum.value(), Some(3.0)); // min(3, 5) = 3
    }

    #[test]
    fn test_max_tropical_addition() {
        let a = TropicalSemiringElement::from_value_with_type(3.0, TropicalType::Max);
        let b = TropicalSemiringElement::from_value_with_type(5.0, TropicalType::Max);

        let sum = a.tropical_add(&b);
        assert_eq!(sum.value(), Some(5.0)); // max(3, 5) = 5
    }

    #[test]
    fn test_tropical_multiplication() {
        let a = TropicalSemiringElement::from_value(3.0);
        let b = TropicalSemiringElement::from_value(5.0);

        let product = a.tropical_mul(&b);
        assert_eq!(product.value(), Some(8.0)); // 3 + 5 = 8
    }

    #[test]
    fn test_tropical_power() {
        let a = TropicalSemiringElement::from_value(2.0);
        let pow = a.tropical_pow(3);
        assert_eq!(pow.value(), Some(6.0)); // 2 * 3 = 6
    }

    #[test]
    fn test_infinity_operations() {
        let inf = TropicalSemiringElement::infinity(TropicalType::Min);
        let a = TropicalSemiringElement::from_value_with_type(5.0, TropicalType::Min);

        // ∞ ⊕ a = a (min(∞, a) = a)
        let sum = inf.tropical_add(&a);
        assert_eq!(sum.value(), Some(5.0));

        // ∞ ⊗ a = ∞
        let product = inf.tropical_mul(&a);
        assert!(product.is_infinity());
    }

    #[test]
    fn test_operators() {
        let a = TropicalSemiringElement::from_value(3.0);
        let b = TropicalSemiringElement::from_value(5.0);

        let sum = a.clone() + b.clone();
        assert_eq!(sum.value(), Some(3.0));

        let product = a * b;
        assert_eq!(product.value(), Some(8.0));
    }

    #[test]
    fn test_ordering() {
        let a = TropicalSemiringElement::from_value(3.0);
        let b = TropicalSemiringElement::from_value(5.0);
        let inf = TropicalSemiringElement::infinity(TropicalType::Min);

        assert!(a < b);
        assert!(b < inf);
    }

    #[test]
    fn test_morphism() {
        let morphism = TropicalToTropical::new(TropicalType::Min, TropicalType::Max);
        let elem = TropicalSemiringElement::from_value_with_type(3.0, TropicalType::Min);

        let mapped = morphism.apply(&elem);
        assert_eq!(mapped.value(), Some(-3.0));
        assert_eq!(mapped.tropical_type(), TropicalType::Max);
    }

    #[test]
    #[should_panic(expected = "Cannot add elements from different tropical semirings")]
    fn test_mixed_type_addition() {
        let a = TropicalSemiringElement::from_value_with_type(3.0, TropicalType::Min);
        let b = TropicalSemiringElement::from_value_with_type(5.0, TropicalType::Max);

        let _ = a.tropical_add(&b);
    }

    #[test]
    fn test_display() {
        let elem = TropicalSemiringElement::from_value(3.5);
        assert_eq!(format!("{}", elem), "3.5");

        let inf = TropicalSemiringElement::infinity(TropicalType::Min);
        assert_eq!(format!("{}", inf), "∞");

        let sr = TropicalSemiring::new_min();
        assert!(format!("{}", sr).contains("min-plus"));
    }
}
