//! Localization of rings
//!
//! This module provides localization constructions for rings,
//! corresponding to SageMath's `sage.rings.localization`.
//!
//! # Mathematical Background
//!
//! Given a commutative ring R and a multiplicative subset S ⊆ R (not containing 0),
//! the localization S⁻¹R (also written R_S) is the ring of fractions:
//!
//! S⁻¹R = {r/s : r ∈ R, s ∈ S} / ~
//!
//! where r/s ~ r'/s' if there exists t ∈ S such that t(rs' - r's) = 0.
//!
//! ## Common Localizations
//!
//! - **At a prime ideal p**: S = R \ p, giving local ring R_p
//! - **At an element f**: S = {1, f, f², f³, ...}, giving R_f
//! - **Field of fractions**: S = R \ {0}, giving Frac(R)
//!
//! ## Properties
//!
//! - S⁻¹R is a ring with operations:
//!   - (r/s) + (r'/s') = (rs' + r's) / (ss')
//!   - (r/s) · (r'/s') = (rr') / (ss')
//! - Universal property: Any ring homomorphism φ: R → T inverting S
//!   factors uniquely through S⁻¹R
//! - Localization is exact (preserves exact sequences)
//!
//! ## Applications
//!
//! - Local rings in algebraic geometry
//! - Constructing fields of fractions
//! - Studying properties "locally" at primes
//! - Sheaf theory on schemes

use rustmath_core::Ring;
use std::marker::PhantomData;
use std::fmt;
use std::collections::HashSet;

/// Localization of a ring at a multiplicative set
///
/// This corresponds to SageMath's `Localization` class.
///
/// # Type Parameters
///
/// - `R`: The base ring type
///
/// # Mathematical Details
///
/// Elements are formal fractions r/s where s ∈ S.
/// Two fractions are equal if they satisfy the equivalence relation.
#[derive(Clone, Debug)]
pub struct Localization<R: Ring> {
    /// Name of the base ring
    base_ring: String,
    /// Description of the multiplicative set S
    multiplicative_set: String,
    /// Extra units (generators of S beyond 1)
    extra_units: Vec<String>,
    /// Whether this is a localization at a prime
    is_at_prime: bool,
    /// Ring marker
    ring_marker: PhantomData<R>,
}

impl<R: Ring> Localization<R> {
    /// Create a new localization
    ///
    /// # Arguments
    ///
    /// * `base_ring` - Name of the base ring
    /// * `multiplicative_set` - Description of S
    /// * `extra_units` - Generators of S
    ///
    /// # Returns
    ///
    /// A new Localization instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Localization Z[x]_f where f = x^2 + 1
    /// let loc = Localization::new(
    ///     "Z[x]".to_string(),
    ///     "{1, f, f^2, ...}".to_string(),
    ///     vec!["x^2 + 1".to_string()]
    /// );
    /// ```
    pub fn new(base_ring: String, multiplicative_set: String, extra_units: Vec<String>) -> Self {
        Localization {
            base_ring,
            multiplicative_set,
            extra_units,
            is_at_prime: false,
            ring_marker: PhantomData,
        }
    }

    /// Create localization at a prime ideal
    ///
    /// # Arguments
    ///
    /// * `base_ring` - Name of the base ring
    /// * `prime` - Prime ideal
    ///
    /// # Returns
    ///
    /// Localization at the prime
    pub fn at_prime(base_ring: String, prime: String) -> Self {
        let mult_set = format!("{} \\ ({})", base_ring, prime);
        Localization {
            base_ring: base_ring.clone(),
            multiplicative_set: mult_set,
            extra_units: vec![],
            is_at_prime: true,
            ring_marker: PhantomData,
        }
    }

    /// Create localization at an element
    ///
    /// # Arguments
    ///
    /// * `base_ring` - Name of the base ring
    /// * `element` - Element to localize at
    ///
    /// # Returns
    ///
    /// Localization at powers of the element
    pub fn at_element(base_ring: String, element: String) -> Self {
        let mult_set = format!("{{1, {}, {}^2, {}^3, ...}}", element, element, element);
        Localization {
            base_ring,
            multiplicative_set: mult_set,
            extra_units: vec![element],
            is_at_prime: false,
            ring_marker: PhantomData,
        }
    }

    /// Get the base ring
    ///
    /// # Returns
    ///
    /// Name of the base ring
    pub fn base_ring(&self) -> &str {
        &self.base_ring
    }

    /// Get the multiplicative set
    ///
    /// # Returns
    ///
    /// Description of S
    pub fn multiplicative_set(&self) -> &str {
        &self.multiplicative_set
    }

    /// Get extra units
    ///
    /// # Returns
    ///
    /// Reference to extra units
    pub fn extra_units(&self) -> &[String] {
        &self.extra_units
    }

    /// Check if localization is at a prime
    ///
    /// # Returns
    ///
    /// True if S = R \ p for some prime p
    pub fn is_at_prime(&self) -> bool {
        self.is_at_prime
    }

    /// Check if this gives a local ring
    ///
    /// # Returns
    ///
    /// True if the result is a local ring
    pub fn is_local(&self) -> bool {
        self.is_at_prime
    }

    /// Get notation for the localization
    ///
    /// # Returns
    ///
    /// Standard notation
    pub fn notation(&self) -> String {
        if self.is_at_prime {
            format!("{}_p", self.base_ring)
        } else if self.extra_units.len() == 1 {
            format!("{}_{}", self.base_ring, self.extra_units[0])
        } else {
            format!("S^{{-1}}{}", self.base_ring)
        }
    }
}

impl<R: Ring> fmt::Display for Localization<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_at_prime {
            write!(f, "Localization of {} at prime ideal", self.base_ring)
        } else if self.extra_units.len() == 1 {
            write!(f, "Localization of {} at {}", self.base_ring, self.extra_units[0])
        } else {
            write!(f, "Localization of {} at {}", self.base_ring, self.multiplicative_set)
        }
    }
}

/// Element of a localization
///
/// This corresponds to SageMath's `LocalizationElement`.
///
/// # Type Parameters
///
/// - `R`: The base ring type
#[derive(Clone, Debug)]
pub struct LocalizationElement<R: Ring> {
    /// Numerator
    numerator: String,
    /// Denominator (must be in S)
    denominator: String,
    /// Ring marker
    ring_marker: PhantomData<R>,
}

impl<R: Ring> LocalizationElement<R> {
    /// Create a new localization element
    ///
    /// # Arguments
    ///
    /// * `numerator` - Numerator r
    /// * `denominator` - Denominator s ∈ S
    ///
    /// # Returns
    ///
    /// Element r/s
    pub fn new(numerator: String, denominator: String) -> Self {
        LocalizationElement {
            numerator,
            denominator,
            ring_marker: PhantomData,
        }
    }

    /// Get numerator
    ///
    /// # Returns
    ///
    /// The numerator
    pub fn numerator(&self) -> &str {
        &self.numerator
    }

    /// Get denominator
    ///
    /// # Returns
    ///
    /// The denominator
    pub fn denominator(&self) -> &str {
        &self.denominator
    }
}

impl<R: Ring> fmt::Display for LocalizationElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == "1" {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "({}) / ({})", self.numerator, self.denominator)
        }
    }
}

/// Normalize extra units for localization
///
/// This corresponds to SageMath's `normalize_extra_units`.
///
/// # Arguments
///
/// * `units` - List of extra units
///
/// # Returns
///
/// Normalized list with duplicates removed
pub fn normalize_extra_units(units: Vec<String>) -> Vec<String> {
    let unique: HashSet<_> = units.into_iter().collect();
    let mut result: Vec<_> = unique.into_iter().collect();
    result.sort();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_localization_creation() {
        let loc: Localization<Rational> = Localization::new(
            "Z".to_string(),
            "Z\\{0}".to_string(),
            vec![],
        );

        assert_eq!(loc.base_ring(), "Z");
    }

    #[test]
    fn test_localization_at_prime() {
        let loc: Localization<Rational> = Localization::at_prime(
            "Z".to_string(),
            "(2)".to_string(),
        );

        assert!(loc.is_at_prime());
        assert!(loc.is_local());
    }

    #[test]
    fn test_localization_at_element() {
        let loc: Localization<Rational> = Localization::at_element(
            "Z[x]".to_string(),
            "x".to_string(),
        );

        assert!(!loc.is_at_prime());
        assert_eq!(loc.extra_units().len(), 1);
        assert_eq!(loc.extra_units()[0], "x");
    }

    #[test]
    fn test_localization_element() {
        let elem: LocalizationElement<Rational> =
            LocalizationElement::new("x+1".to_string(), "x-1".to_string());

        assert_eq!(elem.numerator(), "x+1");
        assert_eq!(elem.denominator(), "x-1");
    }

    #[test]
    fn test_localization_element_display() {
        let elem1: LocalizationElement<Rational> =
            LocalizationElement::new("x".to_string(), "1".to_string());
        let elem2: LocalizationElement<Rational> =
            LocalizationElement::new("x".to_string(), "y".to_string());

        assert_eq!(format!("{}", elem1), "x");
        assert!(format!("{}", elem2).contains("/"));
    }

    #[test]
    fn test_normalize_extra_units() {
        let units = vec![
            "x".to_string(),
            "y".to_string(),
            "x".to_string(),
            "z".to_string(),
        ];

        let normalized = normalize_extra_units(units);
        assert_eq!(normalized.len(), 3);
        assert!(!normalized.contains(&"duplicate".to_string()));
    }

    #[test]
    fn test_localization_notation() {
        let loc_prime: Localization<Rational> =
            Localization::at_prime("Z".to_string(), "(2)".to_string());
        let loc_elem: Localization<Rational> =
            Localization::at_element("R".to_string(), "f".to_string());

        let not_prime = loc_prime.notation();
        let not_elem = loc_elem.notation();

        assert!(not_prime.contains("Z"));
        assert!(not_elem.contains("f"));
    }

    #[test]
    fn test_localization_display() {
        let loc: Localization<Rational> = Localization::at_element(
            "Z[x]".to_string(),
            "x+1".to_string(),
        );

        let display = format!("{}", loc);
        assert!(display.contains("Localization"));
        assert!(display.contains("x+1"));
    }
}
