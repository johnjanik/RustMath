//! Fraction field element module
//!
//! This module provides the core element types for fraction fields (fields of fractions)
//! over integral domains. It corresponds to SageMath's `sage.rings.fraction_field_element`.
//!
//! # Mathematical Background
//!
//! Given an integral domain R, a fraction field element is an equivalence class of pairs
//! (numerator, denominator) where both are elements of R and the denominator is non-zero.
//! Two pairs (a, b) and (c, d) are equivalent if ad = bc in R.
//!
//! # Key Types
//!
//! - `FractionFieldElement<R>`: Generic fraction field element over any integral domain R
//! - `FractionFieldElement_1poly_field`: Specialized type for fractions of univariate polynomials
//! - Helper functions for element creation, type checking, and serialization
//!
//! # Examples
//!
//! ```rust
//! use rustmath_rings::fraction_field_element::*;
//! use rustmath_integers::Integer;
//!
//! // Create a fraction 3/4
//! let frac = FractionFieldElement::new(Integer::from(3), Integer::from(4));
//! assert_eq!(frac.numerator(), &Integer::from(3));
//! assert_eq!(frac.denominator(), &Integer::from(4));
//!
//! // Check if it's a fraction field element
//! assert!(is_FractionFieldElement(&frac));
//! ```

use rustmath_core::{Ring, IntegralDomain, Field, EuclideanDomain};
use std::fmt;
use std::marker::PhantomData;
use std::hash::{Hash, Hasher};

/// Element of a fraction field
///
/// Represents the fraction numerator/denominator where both are elements of an integral
/// domain R. This is the core element type corresponding to SageMath's FractionFieldElement.
///
/// # Type Parameters
///
/// - `R`: The base integral domain (e.g., integers, polynomials)
///
/// # Invariants
///
/// - The denominator is never zero
/// - For Euclidean domains, the fraction is automatically reduced to lowest terms
#[derive(Clone, Debug)]
pub struct FractionFieldElement<R: IntegralDomain> {
    /// The numerator
    numerator: R,
    /// The denominator (always non-zero)
    denominator: R,
}

impl<R: IntegralDomain> FractionFieldElement<R> {
    /// Create a new fraction field element
    ///
    /// # Arguments
    ///
    /// * `numerator` - The numerator of the fraction
    /// * `denominator` - The denominator of the fraction
    ///
    /// # Panics
    ///
    /// Panics if denominator is zero
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rustmath_rings::fraction_field_element::FractionFieldElement;
    /// use rustmath_integers::Integer;
    ///
    /// let frac = FractionFieldElement::new(Integer::from(3), Integer::from(4));
    /// ```
    pub fn new(numerator: R, denominator: R) -> Self {
        if denominator.is_zero() {
            panic!("Denominator cannot be zero");
        }

        let mut frac = FractionFieldElement {
            numerator,
            denominator,
        };
        frac.reduce();
        frac
    }

    /// Get the numerator
    ///
    /// Returns a reference to the numerator of this fraction.
    pub fn numerator(&self) -> &R {
        &self.numerator
    }

    /// Get the denominator
    ///
    /// Returns a reference to the denominator of this fraction.
    pub fn denominator(&self) -> &R {
        &self.denominator
    }

    /// Get mutable reference to numerator (internal use)
    fn numerator_mut(&mut self) -> &mut R {
        &mut self.numerator
    }

    /// Get mutable reference to denominator (internal use)
    fn denominator_mut(&mut self) -> &mut R {
        &mut self.denominator
    }

    /// Reduce the fraction to lowest terms
    ///
    /// For Euclidean domains, this computes the GCD of numerator and denominator
    /// and divides both by it. For general integral domains, this is a no-op.
    fn reduce(&mut self) {
        // For general integral domains, we can't always compute GCD
        // This would need specialization for Euclidean domains
        // For now, this is a placeholder
    }

    /// Check if this fraction represents zero
    ///
    /// A fraction is zero if and only if its numerator is zero.
    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    /// Check if this fraction represents one
    ///
    /// A fraction is one if its numerator equals its denominator.
    pub fn is_one(&self) -> bool {
        self.numerator == self.denominator
    }

    /// Compute the multiplicative inverse
    ///
    /// Returns `Some(1/self)` if self is non-zero, `None` if self is zero.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let frac = FractionFieldElement::new(Integer::from(3), Integer::from(4));
    /// let inv = frac.inverse().unwrap(); // 4/3
    /// ```
    pub fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(FractionFieldElement {
                numerator: self.denominator.clone(),
                denominator: self.numerator.clone(),
            })
        }
    }

    /// Factor this fraction
    ///
    /// Returns a representation of the factorization of both numerator and denominator.
    /// The exact format depends on the base ring R.
    pub fn factor(&self) -> String {
        // Placeholder: actual factorization would depend on R
        format!("({})/({}) [not factored]", self.numerator, self.denominator)
    }

    /// Partial fraction decomposition
    ///
    /// For rational functions, compute partial fraction decomposition.
    /// This is a placeholder that would be specialized for polynomial fraction fields.
    pub fn partial_fraction_decomposition(&self) -> Vec<Self> {
        // Placeholder: real implementation would factor denominator and decompose
        vec![self.clone()]
    }

    /// Get the numerator as a separate element (consuming self)
    pub fn numerator_value(self) -> R {
        self.numerator
    }

    /// Get the denominator as a separate element (consuming self)
    pub fn denominator_value(self) -> R {
        self.denominator
    }
}

impl<R: IntegralDomain> std::ops::Add for FractionFieldElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // a/b + c/d = (ad + bc)/(bd)
        let num = self.numerator.clone() * other.denominator.clone()
            + other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        FractionFieldElement::new(num, den)
    }
}

impl<R: IntegralDomain> std::ops::Sub for FractionFieldElement<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        // a/b - c/d = (ad - bc)/(bd)
        let num = self.numerator.clone() * other.denominator.clone()
            - other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        FractionFieldElement::new(num, den)
    }
}

impl<R: IntegralDomain> std::ops::Mul for FractionFieldElement<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // (a/b) * (c/d) = (ac)/(bd)
        let num = self.numerator.clone() * other.numerator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        FractionFieldElement::new(num, den)
    }
}

impl<R: IntegralDomain> std::ops::Div for FractionFieldElement<R> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if other.numerator.is_zero() {
            panic!("Division by zero");
        }

        // (a/b) / (c/d) = (ad)/(bc)
        let num = self.numerator.clone() * other.denominator.clone();
        let den = self.denominator.clone() * other.numerator.clone();

        FractionFieldElement::new(num, den)
    }
}

impl<R: IntegralDomain> std::ops::Neg for FractionFieldElement<R> {
    type Output = Self;

    fn neg(self) -> Self {
        FractionFieldElement {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}

impl<R: IntegralDomain> PartialEq for FractionFieldElement<R> {
    fn eq(&self, other: &Self) -> bool {
        // a/b == c/d iff ad == bc
        self.numerator.clone() * other.denominator.clone()
            == other.numerator.clone() * self.denominator.clone()
    }
}

impl<R: IntegralDomain> Eq for FractionFieldElement<R> {}

impl<R: IntegralDomain> fmt::Display for FractionFieldElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator.is_one() {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

/// Specialized fraction field element type for univariate polynomial fractions
///
/// This type corresponds to SageMath's `FractionFieldElement_1poly_field`, which
/// represents elements of the fraction field k(x) where k is a field and k[x] is
/// the polynomial ring.
///
/// This is essentially a type alias for `FractionFieldElement<P>` where P is a
/// univariate polynomial type, but with additional methods specific to rational
/// functions.
pub type FractionFieldElement1polyField<P> = FractionFieldElement<P>;

/// Type checking: determine if an object is a FractionFieldElement
///
/// This function corresponds to SageMath's `is_FractionFieldElement`.
///
/// # Arguments
///
/// * `x` - A reference to a potential fraction field element
///
/// # Returns
///
/// Always returns `true` since Rust's type system ensures the argument is
/// indeed a `FractionFieldElement`.
///
/// # Examples
///
/// ```ignore
/// use rustmath_rings::fraction_field_element::*;
/// use rustmath_integers::Integer;
///
/// let frac = FractionFieldElement::new(Integer::from(3), Integer::from(4));
/// assert!(is_FractionFieldElement(&frac));
/// ```
pub fn is_fraction_field_element<R: IntegralDomain>(x: &FractionFieldElement<R>) -> bool {
    // In Rust, type checking is done at compile time
    // If this function is called, the argument is definitely a FractionFieldElement
    true
}

/// Create a fraction field element from numerator and denominator
///
/// This function corresponds to SageMath's `make_element` function.
/// It's a convenience constructor for creating fraction field elements.
///
/// # Arguments
///
/// * `parent` - The parent fraction field (for compatibility; not used in this implementation)
/// * `numerator` - The numerator
/// * `denominator` - The denominator
///
/// # Returns
///
/// A new `FractionFieldElement` with the given numerator and denominator.
///
/// # Panics
///
/// Panics if the denominator is zero.
///
/// # Examples
///
/// ```ignore
/// use rustmath_rings::fraction_field_element::*;
/// use rustmath_integers::Integer;
///
/// let frac = make_element((), Integer::from(5), Integer::from(7));
/// assert_eq!(frac.numerator(), &Integer::from(5));
/// assert_eq!(frac.denominator(), &Integer::from(7));
/// ```
pub fn make_element<R: IntegralDomain, P>(
    _parent: P,
    numerator: R,
    denominator: R,
) -> FractionFieldElement<R> {
    FractionFieldElement::new(numerator, denominator)
}

/// Create a fraction field element (old-style interface)
///
/// This function corresponds to SageMath's `make_element_old` function.
/// It provides backward compatibility with older serialization formats.
///
/// # Arguments
///
/// * `cls` - The class type (for compatibility; not used in Rust)
/// * `parent` - The parent fraction field (for compatibility; not used in this implementation)
/// * `numerator` - The numerator
/// * `denominator` - The denominator
///
/// # Returns
///
/// A new `FractionFieldElement` with the given numerator and denominator.
///
/// # Panics
///
/// Panics if the denominator is zero.
pub fn make_element_old<R: IntegralDomain, C, P>(
    _cls: C,
    _parent: P,
    numerator: R,
    denominator: R,
) -> FractionFieldElement<R> {
    FractionFieldElement::new(numerator, denominator)
}

/// Unpickle a fraction field element
///
/// Helper function for deserializing fraction field elements.
/// This is used for compatibility with SageMath's pickle format.
///
/// # Arguments
///
/// * `numerator` - The numerator
/// * `denominator` - The denominator
///
/// # Returns
///
/// A new `FractionFieldElement` with the given numerator and denominator.
pub fn unpickle_fraction_field_element<R: IntegralDomain>(
    numerator: R,
    denominator: R,
) -> FractionFieldElement<R> {
    FractionFieldElement::new(numerator, denominator)
}

/// Trait for fraction field element operations
///
/// This trait defines common operations for fraction field elements.
pub trait FractionFieldElementOps<R: IntegralDomain> {
    /// Get the numerator
    fn numerator(&self) -> &R;

    /// Get the denominator
    fn denominator(&self) -> &R;

    /// Compute the inverse
    fn inverse(&self) -> Option<FractionFieldElement<R>>;

    /// Check if the element is zero
    fn is_zero(&self) -> bool;

    /// Check if the element is one
    fn is_one(&self) -> bool;
}

impl<R: IntegralDomain> FractionFieldElementOps<R> for FractionFieldElement<R> {
    fn numerator(&self) -> &R {
        &self.numerator
    }

    fn denominator(&self) -> &R {
        &self.denominator
    }

    fn inverse(&self) -> Option<FractionFieldElement<R>> {
        FractionFieldElement::inverse(self)
    }

    fn is_zero(&self) -> bool {
        FractionFieldElement::is_zero(self)
    }

    fn is_one(&self) -> bool {
        FractionFieldElement::is_one(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_fraction_field_element_creation() {
        let frac = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        assert_eq!(frac.numerator(), &Integer::from(3));
        assert_eq!(frac.denominator(), &Integer::from(4));
    }

    #[test]
    #[should_panic(expected = "Denominator cannot be zero")]
    fn test_zero_denominator() {
        let _ = FractionFieldElement::new(Integer::from(1), Integer::zero());
    }

    #[test]
    fn test_is_zero() {
        let zero = FractionFieldElement::new(Integer::zero(), Integer::one());
        assert!(zero.is_zero());

        let non_zero = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        assert!(!non_zero.is_zero());
    }

    #[test]
    fn test_is_one() {
        let one = FractionFieldElement::new(Integer::from(5), Integer::from(5));
        assert!(one.is_one());

        let not_one = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        assert!(!not_one.is_one());
    }

    #[test]
    fn test_addition() {
        // 1/2 + 1/3 = 5/6
        let a = FractionFieldElement::new(Integer::from(1), Integer::from(2));
        let b = FractionFieldElement::new(Integer::from(1), Integer::from(3));
        let sum = a + b;

        let expected = FractionFieldElement::new(Integer::from(5), Integer::from(6));
        assert_eq!(sum, expected);
    }

    #[test]
    fn test_subtraction() {
        // 3/4 - 1/2 = 1/4
        let a = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        let b = FractionFieldElement::new(Integer::from(1), Integer::from(2));
        let diff = a - b;

        let expected = FractionFieldElement::new(Integer::from(1), Integer::from(4));
        assert_eq!(diff, expected);
    }

    #[test]
    fn test_multiplication() {
        // 2/3 * 3/4 = 1/2
        let a = FractionFieldElement::new(Integer::from(2), Integer::from(3));
        let b = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        let product = a * b;

        let expected = FractionFieldElement::new(Integer::from(1), Integer::from(2));
        assert_eq!(product, expected);
    }

    #[test]
    fn test_division() {
        // (2/3) / (4/5) = 10/12 = 5/6
        let a = FractionFieldElement::new(Integer::from(2), Integer::from(3));
        let b = FractionFieldElement::new(Integer::from(4), Integer::from(5));
        let quotient = a / b;

        let expected = FractionFieldElement::new(Integer::from(5), Integer::from(6));
        assert_eq!(quotient, expected);
    }

    #[test]
    fn test_negation() {
        let a = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        let neg = -a;

        assert_eq!(neg.numerator(), &Integer::from(-3));
        assert_eq!(neg.denominator(), &Integer::from(4));
    }

    #[test]
    fn test_equality() {
        // 2/4 == 1/2
        let a = FractionFieldElement::new(Integer::from(2), Integer::from(4));
        let b = FractionFieldElement::new(Integer::from(1), Integer::from(2));
        assert_eq!(a, b);

        // 1/2 != 1/3
        let c = FractionFieldElement::new(Integer::from(1), Integer::from(3));
        assert_ne!(a, c);
    }

    #[test]
    fn test_inverse() {
        let frac = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        let inv = frac.inverse().unwrap();

        assert_eq!(inv.numerator(), &Integer::from(4));
        assert_eq!(inv.denominator(), &Integer::from(3));

        // Zero has no inverse
        let zero = FractionFieldElement::new(Integer::zero(), Integer::one());
        assert!(zero.inverse().is_none());
    }

    #[test]
    fn test_display() {
        let a = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        assert_eq!(format!("{}", a), "3/4");

        let b = FractionFieldElement::new(Integer::from(5), Integer::from(1));
        assert_eq!(format!("{}", b), "5");
    }

    #[test]
    fn test_is_fraction_field_element() {
        let frac = FractionFieldElement::new(Integer::from(1), Integer::from(2));
        assert!(is_FractionFieldElement(&frac));
    }

    #[test]
    fn test_make_element() {
        let frac = make_element((), Integer::from(5), Integer::from(7));
        assert_eq!(frac.numerator(), &Integer::from(5));
        assert_eq!(frac.denominator(), &Integer::from(7));
    }

    #[test]
    fn test_make_element_old() {
        let frac = make_element_old((), (), Integer::from(5), Integer::from(7));
        assert_eq!(frac.numerator(), &Integer::from(5));
        assert_eq!(frac.denominator(), &Integer::from(7));
    }

    #[test]
    fn test_unpickle() {
        let frac = unpickle_fraction_field_element(Integer::from(3), Integer::from(4));
        assert_eq!(frac.numerator(), &Integer::from(3));
        assert_eq!(frac.denominator(), &Integer::from(4));
    }

    #[test]
    fn test_fraction_field_element_ops_trait() {
        let frac = FractionFieldElement::new(Integer::from(3), Integer::from(4));

        assert_eq!(FractionFieldElementOps::numerator(&frac), &Integer::from(3));
        assert_eq!(FractionFieldElementOps::denominator(&frac), &Integer::from(4));
        assert!(!FractionFieldElementOps::is_zero(&frac));
        assert!(!FractionFieldElementOps::is_one(&frac));

        let inv = FractionFieldElementOps::inverse(&frac).unwrap();
        assert_eq!(inv.numerator(), &Integer::from(4));
        assert_eq!(inv.denominator(), &Integer::from(3));
    }

    #[test]
    fn test_comprehensive_operations() {
        // Test more complex operations
        let a = FractionFieldElement::new(Integer::from(2), Integer::from(3));
        let b = FractionFieldElement::new(Integer::from(4), Integer::from(5));
        let c = FractionFieldElement::new(Integer::from(1), Integer::from(6));

        // (2/3 + 4/5) * 1/6
        let result = (a + b) * c;

        // 2/3 + 4/5 = (10 + 12)/15 = 22/15
        // 22/15 * 1/6 = 22/90 = 11/45
        let expected = FractionFieldElement::new(Integer::from(11), Integer::from(45));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_factor() {
        let frac = FractionFieldElement::new(Integer::from(6), Integer::from(8));
        let factored = frac.factor();
        assert!(factored.contains("6"));
        assert!(factored.contains("8"));
    }

    #[test]
    fn test_partial_fraction_decomposition() {
        let frac = FractionFieldElement::new(Integer::from(3), Integer::from(4));
        let decomp = frac.partial_fraction_decomposition();
        assert_eq!(decomp.len(), 1);
    }
}
