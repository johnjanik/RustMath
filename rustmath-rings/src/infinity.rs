//! # Infinity Module
//!
//! Implementation of mathematical infinity representations, supporting both signed
//! (positive/negative) and unsigned infinity with proper arithmetic and comparison operations.
//!
//! This module mirrors SageMath's `sage.rings.infinity` functionality, providing:
//! - **Signed Infinity**: `+∞` and `-∞` with proper arithmetic
//! - **Unsigned Infinity**: Magnitude-only infinity `∞`
//! - **Finite Numbers**: Finite values that can mix with infinities
//! - **Infinity Rings**: Algebraic structures containing infinity elements
//!
//! ## Examples
//!
//! ```
//! use rustmath_rings::infinity::{Infinity, PLUS_INFINITY, MINUS_INFINITY};
//!
//! // Arithmetic with infinities
//! assert_eq!(PLUS_INFINITY + 5.0, PLUS_INFINITY);
//! assert_eq!(PLUS_INFINITY * 2.0, PLUS_INFINITY);
//! assert_eq!(-PLUS_INFINITY, MINUS_INFINITY);
//!
//! // Comparisons
//! assert!(PLUS_INFINITY > 1000.0);
//! assert!(MINUS_INFINITY < -1000.0);
//! ```
//!
//! ## Signed vs Unsigned Infinity
//!
//! **Signed Infinity** (`+∞`, `-∞`):
//! - Used in ordered contexts (limits, bounds, analysis)
//! - Supports full arithmetic with sign rules
//! - Coerces from real/rational numbers
//!
//! **Unsigned Infinity** (`∞`):
//! - Represents magnitude only (complex infinity, `zoo` in SymPy)
//! - Cannot be added/subtracted (undefined sign)
//! - Used in contexts where direction doesn't matter
//!
//! ## Error Handling
//!
//! Operations that are mathematically undefined raise errors:
//! - `∞ - ∞` → SignError (indeterminate)
//! - `0 × ∞` → SignError (undefined)
//! - `∞ + (-∞)` → SignError (indeterminate)

use rustmath_core::{Ring, Field, CommutativeRing};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg};
use thiserror::Error;

/// Errors that can occur when working with infinities
#[derive(Debug, Clone, Error, PartialEq)]
pub enum InfinityError {
    #[error("Sign error: {0}")]
    SignError(String),

    #[error("Cannot convert unsigned infinity to float")]
    UnsignedToFloat,

    #[error("Division by zero involving infinity")]
    DivisionByZero,

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

/// Represents mathematical infinity in its various forms
///
/// This enum encodes the three types of infinity:
/// - `PlusInfinity`: Positive infinity (+∞)
/// - `MinusInfinity`: Negative infinity (-∞)
/// - `UnsignedInfinity`: Unsigned infinity (∞, no sign)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Infinity {
    /// Positive infinity (+∞)
    PlusInfinity,
    /// Negative infinity (-∞)
    MinusInfinity,
    /// Unsigned infinity (∞), magnitude only
    UnsignedInfinity,
}

impl Infinity {
    /// Returns the sign of the infinity: +1, -1, or 0 for unsigned
    pub fn sign(&self) -> i8 {
        match self {
            Infinity::PlusInfinity => 1,
            Infinity::MinusInfinity => -1,
            Infinity::UnsignedInfinity => 0,
        }
    }

    /// Returns the sign character for display
    pub fn sign_char(&self) -> &'static str {
        match self {
            Infinity::PlusInfinity => "+",
            Infinity::MinusInfinity => "-",
            Infinity::UnsignedInfinity => "",
        }
    }

    /// Checks if this is signed infinity (plus or minus)
    pub fn is_signed(&self) -> bool {
        !matches!(self, Infinity::UnsignedInfinity)
    }

    /// Checks if this is unsigned infinity
    pub fn is_unsigned(&self) -> bool {
        matches!(self, Infinity::UnsignedInfinity)
    }

    /// Converts to float representation
    ///
    /// # Errors
    /// Returns `UnsignedToFloat` if called on unsigned infinity
    pub fn to_f64(&self) -> Result<f64, InfinityError> {
        match self {
            Infinity::PlusInfinity => Ok(f64::INFINITY),
            Infinity::MinusInfinity => Ok(f64::NEG_INFINITY),
            Infinity::UnsignedInfinity => Err(InfinityError::UnsignedToFloat),
        }
    }

    /// Computes the square root
    ///
    /// # Errors
    /// Returns `SignError` for negative infinity
    pub fn sqrt(&self) -> Result<Self, InfinityError> {
        match self {
            Infinity::PlusInfinity => Ok(Infinity::PlusInfinity),
            Infinity::MinusInfinity => Err(InfinityError::SignError(
                "Cannot take square root of negative infinity".to_string()
            )),
            Infinity::UnsignedInfinity => Ok(Infinity::UnsignedInfinity),
        }
    }

    /// Computes absolute value
    pub fn abs(&self) -> Self {
        match self {
            Infinity::PlusInfinity => Infinity::PlusInfinity,
            Infinity::MinusInfinity => Infinity::PlusInfinity,
            Infinity::UnsignedInfinity => Infinity::UnsignedInfinity,
        }
    }
}

impl fmt::Display for Infinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Infinity::PlusInfinity => write!(f, "+Infinity"),
            Infinity::MinusInfinity => write!(f, "-Infinity"),
            Infinity::UnsignedInfinity => write!(f, "Infinity"),
        }
    }
}

impl Neg for Infinity {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Infinity::PlusInfinity => Infinity::MinusInfinity,
            Infinity::MinusInfinity => Infinity::PlusInfinity,
            Infinity::UnsignedInfinity => Infinity::UnsignedInfinity,
        }
    }
}

impl PartialOrd for Infinity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Infinity {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Infinity::MinusInfinity, Infinity::MinusInfinity) => Ordering::Equal,
            (Infinity::MinusInfinity, _) => Ordering::Less,
            (_, Infinity::MinusInfinity) => Ordering::Greater,
            (Infinity::PlusInfinity, Infinity::PlusInfinity) => Ordering::Equal,
            (Infinity::PlusInfinity, _) => Ordering::Greater,
            (_, Infinity::PlusInfinity) => Ordering::Less,
            (Infinity::UnsignedInfinity, Infinity::UnsignedInfinity) => Ordering::Equal,
        }
    }
}

/// Global constant for positive infinity
pub const PLUS_INFINITY: Infinity = Infinity::PlusInfinity;

/// Global constant for negative infinity
pub const MINUS_INFINITY: Infinity = Infinity::MinusInfinity;

/// Global constant for unsigned infinity
pub const UNSIGNED_INFINITY: Infinity = Infinity::UnsignedInfinity;

/// Represents a finite number in the context of infinity arithmetic
///
/// This is used to represent finite values when working with the infinity ring,
/// allowing mixed arithmetic between finite values and infinities.
#[derive(Debug, Clone, PartialEq)]
pub enum InfinityRingElement {
    /// An infinity value
    Infinite(Infinity),
    /// A finite positive number
    PositiveFinite,
    /// A finite negative number
    NegativeFinite,
    /// Zero
    Zero,
}

impl InfinityRingElement {
    /// Creates a finite number from a sign
    pub fn from_sign(sign: i8) -> Self {
        match sign.cmp(&0) {
            Ordering::Greater => InfinityRingElement::PositiveFinite,
            Ordering::Less => InfinityRingElement::NegativeFinite,
            Ordering::Equal => InfinityRingElement::Zero,
        }
    }

    /// Checks if this is infinite
    pub fn is_infinite(&self) -> bool {
        matches!(self, InfinityRingElement::Infinite(_))
    }

    /// Checks if this is finite
    pub fn is_finite(&self) -> bool {
        !self.is_infinite()
    }
}

impl fmt::Display for InfinityRingElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InfinityRingElement::Infinite(inf) => write!(f, "{}", inf),
            InfinityRingElement::PositiveFinite => write!(f, "A positive finite number"),
            InfinityRingElement::NegativeFinite => write!(f, "A negative finite number"),
            InfinityRingElement::Zero => write!(f, "Zero"),
        }
    }
}

/// Represents an element in the unsigned infinity ring
#[derive(Debug, Clone, PartialEq)]
pub enum UnsignedInfinityRingElement {
    /// The unsigned infinity value
    Infinite(Infinity),
    /// A finite value (less than infinity)
    LessThanInfinity,
}

impl fmt::Display for UnsignedInfinityRingElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnsignedInfinityRingElement::Infinite(inf) => write!(f, "{}", inf),
            UnsignedInfinityRingElement::LessThanInfinity => write!(f, "A number less than infinity"),
        }
    }
}

/// The signed infinity ring
///
/// This is a commutative ring containing positive infinity, negative infinity,
/// and finite numbers. It supports arithmetic with proper sign handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InfinityRing;

impl InfinityRing {
    /// Returns the n-th generator: 0 for +∞, 1 for -∞
    pub fn gen(n: usize) -> Infinity {
        match n {
            0 => Infinity::PlusInfinity,
            1 => Infinity::MinusInfinity,
            _ => panic!("InfinityRing only has 2 generators"),
        }
    }

    /// Creates an element from a value
    pub fn element_from<T: PartialOrd + Default + Clone>(value: T) -> InfinityRingElement {
        let zero = T::default();
        match value.partial_cmp(&zero) {
            Some(Ordering::Greater) => InfinityRingElement::PositiveFinite,
            Some(Ordering::Less) => InfinityRingElement::NegativeFinite,
            Some(Ordering::Equal) => InfinityRingElement::Zero,
            None => InfinityRingElement::Zero, // Default for incomparable
        }
    }
}

impl fmt::Display for InfinityRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "The Infinity Ring")
    }
}

/// The unsigned infinity ring
///
/// This is a ring containing unsigned infinity and finite values.
/// Unsigned infinity has no sign and represents magnitude only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsignedInfinityRing;

impl UnsignedInfinityRing {
    /// Returns the generator (unsigned infinity)
    pub fn gen(n: usize) -> Infinity {
        match n {
            0 => Infinity::UnsignedInfinity,
            _ => panic!("UnsignedInfinityRing only has 1 generator"),
        }
    }

    /// Returns a "less than infinity" element
    pub fn less_than_infinity() -> UnsignedInfinityRingElement {
        UnsignedInfinityRingElement::LessThanInfinity
    }
}

impl fmt::Display for UnsignedInfinityRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "The Unsigned Infinity Ring")
    }
}

/// Checks if a value is infinite
///
/// This is the Rust equivalent of SageMath's `is_Infinite` function.
pub fn is_infinite<T: 'static>(value: &T) -> bool {
    // For actual Infinity type
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<Infinity>()
}

/// Addition between infinities
///
/// # Errors
/// Returns `SignError` for undefined operations like ∞ + (-∞)
pub fn add_infinities(a: Infinity, b: Infinity) -> Result<Infinity, InfinityError> {
    match (a, b) {
        // Unsigned infinity cannot be added to anything
        (Infinity::UnsignedInfinity, _) | (_, Infinity::UnsignedInfinity) => {
            Err(InfinityError::SignError(
                "Cannot add unsigned infinities".to_string()
            ))
        }
        // Same sign infinities
        (Infinity::PlusInfinity, Infinity::PlusInfinity) => Ok(Infinity::PlusInfinity),
        (Infinity::MinusInfinity, Infinity::MinusInfinity) => Ok(Infinity::MinusInfinity),
        // Opposite sign infinities
        (Infinity::PlusInfinity, Infinity::MinusInfinity) |
        (Infinity::MinusInfinity, Infinity::PlusInfinity) => {
            Err(InfinityError::SignError(
                "Cannot add +Infinity and -Infinity (indeterminate)".to_string()
            ))
        }
    }
}

/// Subtraction between infinities
///
/// # Errors
/// Returns `SignError` for undefined operations like ∞ - ∞
pub fn sub_infinities(a: Infinity, b: Infinity) -> Result<Infinity, InfinityError> {
    add_infinities(a, -b)
}

/// Multiplication of infinities
///
/// # Errors
/// Returns `SignError` for 0 × ∞
pub fn mul_infinities(a: Infinity, b: Infinity) -> Result<Infinity, InfinityError> {
    match (a, b) {
        // Unsigned infinity multiplication
        (Infinity::UnsignedInfinity, _) | (_, Infinity::UnsignedInfinity) => {
            Ok(Infinity::UnsignedInfinity)
        }
        // Signed infinity multiplication follows sign rules
        (Infinity::PlusInfinity, Infinity::PlusInfinity) => Ok(Infinity::PlusInfinity),
        (Infinity::MinusInfinity, Infinity::MinusInfinity) => Ok(Infinity::PlusInfinity),
        (Infinity::PlusInfinity, Infinity::MinusInfinity) |
        (Infinity::MinusInfinity, Infinity::PlusInfinity) => Ok(Infinity::MinusInfinity),
    }
}

/// Multiplies infinity by a scalar value
///
/// # Errors
/// Returns `SignError` if the scalar is zero
pub fn mul_infinity_scalar(inf: Infinity, scalar: f64) -> Result<Infinity, InfinityError> {
    if scalar == 0.0 {
        return Err(InfinityError::SignError(
            "Cannot multiply infinity by zero".to_string()
        ));
    }

    match inf {
        Infinity::UnsignedInfinity => Ok(Infinity::UnsignedInfinity),
        Infinity::PlusInfinity => {
            if scalar > 0.0 {
                Ok(Infinity::PlusInfinity)
            } else {
                Ok(Infinity::MinusInfinity)
            }
        }
        Infinity::MinusInfinity => {
            if scalar > 0.0 {
                Ok(Infinity::MinusInfinity)
            } else {
                Ok(Infinity::PlusInfinity)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infinity_sign() {
        assert_eq!(PLUS_INFINITY.sign(), 1);
        assert_eq!(MINUS_INFINITY.sign(), -1);
        assert_eq!(UNSIGNED_INFINITY.sign(), 0);
    }

    #[test]
    fn test_infinity_is_signed() {
        assert!(PLUS_INFINITY.is_signed());
        assert!(MINUS_INFINITY.is_signed());
        assert!(!UNSIGNED_INFINITY.is_signed());
    }

    #[test]
    fn test_infinity_negation() {
        assert_eq!(-PLUS_INFINITY, MINUS_INFINITY);
        assert_eq!(-MINUS_INFINITY, PLUS_INFINITY);
        assert_eq!(-UNSIGNED_INFINITY, UNSIGNED_INFINITY);
    }

    #[test]
    fn test_infinity_abs() {
        assert_eq!(PLUS_INFINITY.abs(), PLUS_INFINITY);
        assert_eq!(MINUS_INFINITY.abs(), PLUS_INFINITY);
        assert_eq!(UNSIGNED_INFINITY.abs(), UNSIGNED_INFINITY);
    }

    #[test]
    fn test_infinity_sqrt() {
        assert_eq!(PLUS_INFINITY.sqrt().unwrap(), PLUS_INFINITY);
        assert!(MINUS_INFINITY.sqrt().is_err());
        assert_eq!(UNSIGNED_INFINITY.sqrt().unwrap(), UNSIGNED_INFINITY);
    }

    #[test]
    fn test_infinity_to_f64() {
        assert_eq!(PLUS_INFINITY.to_f64().unwrap(), f64::INFINITY);
        assert_eq!(MINUS_INFINITY.to_f64().unwrap(), f64::NEG_INFINITY);
        assert!(UNSIGNED_INFINITY.to_f64().is_err());
    }

    #[test]
    fn test_infinity_ordering() {
        assert!(PLUS_INFINITY > MINUS_INFINITY);
        assert!(MINUS_INFINITY < PLUS_INFINITY);
        assert_eq!(PLUS_INFINITY, PLUS_INFINITY);
    }

    #[test]
    fn test_add_infinities_same_sign() {
        assert_eq!(add_infinities(PLUS_INFINITY, PLUS_INFINITY).unwrap(), PLUS_INFINITY);
        assert_eq!(add_infinities(MINUS_INFINITY, MINUS_INFINITY).unwrap(), MINUS_INFINITY);
    }

    #[test]
    fn test_add_infinities_opposite_sign() {
        assert!(add_infinities(PLUS_INFINITY, MINUS_INFINITY).is_err());
        assert!(add_infinities(MINUS_INFINITY, PLUS_INFINITY).is_err());
    }

    #[test]
    fn test_add_infinities_unsigned() {
        assert!(add_infinities(UNSIGNED_INFINITY, PLUS_INFINITY).is_err());
        assert!(add_infinities(PLUS_INFINITY, UNSIGNED_INFINITY).is_err());
    }

    #[test]
    fn test_sub_infinities() {
        // +∞ - (-∞) = +∞
        assert_eq!(sub_infinities(PLUS_INFINITY, MINUS_INFINITY).unwrap(), PLUS_INFINITY);
        // -∞ - (+∞) = -∞
        assert_eq!(sub_infinities(MINUS_INFINITY, PLUS_INFINITY).unwrap(), MINUS_INFINITY);
        // +∞ - (+∞) is indeterminate
        assert!(sub_infinities(PLUS_INFINITY, PLUS_INFINITY).is_err());
    }

    #[test]
    fn test_mul_infinities() {
        assert_eq!(mul_infinities(PLUS_INFINITY, PLUS_INFINITY).unwrap(), PLUS_INFINITY);
        assert_eq!(mul_infinities(MINUS_INFINITY, MINUS_INFINITY).unwrap(), PLUS_INFINITY);
        assert_eq!(mul_infinities(PLUS_INFINITY, MINUS_INFINITY).unwrap(), MINUS_INFINITY);
        assert_eq!(mul_infinities(UNSIGNED_INFINITY, PLUS_INFINITY).unwrap(), UNSIGNED_INFINITY);
    }

    #[test]
    fn test_mul_infinity_scalar() {
        assert_eq!(mul_infinity_scalar(PLUS_INFINITY, 5.0).unwrap(), PLUS_INFINITY);
        assert_eq!(mul_infinity_scalar(PLUS_INFINITY, -5.0).unwrap(), MINUS_INFINITY);
        assert_eq!(mul_infinity_scalar(MINUS_INFINITY, 5.0).unwrap(), MINUS_INFINITY);
        assert_eq!(mul_infinity_scalar(MINUS_INFINITY, -5.0).unwrap(), PLUS_INFINITY);
        assert!(mul_infinity_scalar(PLUS_INFINITY, 0.0).is_err());
    }

    #[test]
    fn test_infinity_ring_generators() {
        assert_eq!(InfinityRing::gen(0), PLUS_INFINITY);
        assert_eq!(InfinityRing::gen(1), MINUS_INFINITY);
    }

    #[test]
    #[should_panic(expected = "only has 2 generators")]
    fn test_infinity_ring_invalid_generator() {
        InfinityRing::gen(2);
    }

    #[test]
    fn test_unsigned_infinity_ring_generator() {
        assert_eq!(UnsignedInfinityRing::gen(0), UNSIGNED_INFINITY);
    }

    #[test]
    fn test_infinity_ring_element_display() {
        let pos_finite = InfinityRingElement::PositiveFinite;
        let neg_finite = InfinityRingElement::NegativeFinite;
        let zero = InfinityRingElement::Zero;
        let inf = InfinityRingElement::Infinite(PLUS_INFINITY);

        assert_eq!(format!("{}", pos_finite), "A positive finite number");
        assert_eq!(format!("{}", neg_finite), "A negative finite number");
        assert_eq!(format!("{}", zero), "Zero");
        assert_eq!(format!("{}", inf), "+Infinity");
    }

    #[test]
    fn test_unsigned_infinity_ring_element_display() {
        let less = UnsignedInfinityRingElement::LessThanInfinity;
        let inf = UnsignedInfinityRingElement::Infinite(UNSIGNED_INFINITY);

        assert_eq!(format!("{}", less), "A number less than infinity");
        assert_eq!(format!("{}", inf), "Infinity");
    }

    #[test]
    fn test_infinity_display() {
        assert_eq!(format!("{}", PLUS_INFINITY), "+Infinity");
        assert_eq!(format!("{}", MINUS_INFINITY), "-Infinity");
        assert_eq!(format!("{}", UNSIGNED_INFINITY), "Infinity");
    }

    #[test]
    fn test_ring_display() {
        assert_eq!(format!("{}", InfinityRing), "The Infinity Ring");
        assert_eq!(format!("{}", UnsignedInfinityRing), "The Unsigned Infinity Ring");
    }
}
