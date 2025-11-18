//! # Ring Extension Element Module
//!
//! Elements of ring extensions with specialized representations and operations.
//!
//! ## Overview
//!
//! This module provides element types for ring extensions, with specialized
//! implementations for basis-equipped and fraction field extensions.

use rustmath_core::Ring;
use std::fmt;
use std::marker::PhantomData;

/// Generic element of a ring extension
///
/// Represents an element in an extension L/K.
#[derive(Debug, Clone, PartialEq)]
pub struct RingExtensionElement<K, L>
where
    K: Ring,
    L: Ring,
{
    /// The underlying value in the backend ring
    value: PhantomData<L>,
    /// Reference to base ring
    _base: PhantomData<K>,
    description: String,
}

impl<K, L> RingExtensionElement<K, L>
where
    K: Ring,
    L: Ring,
{
    /// Creates a new extension element
    pub fn new(description: String) -> Self {
        RingExtensionElement {
            value: PhantomData,
            _base: PhantomData,
            description,
        }
    }
}

impl<K, L> fmt::Display for RingExtensionElement<K, L>
where
    K: Ring,
    L: Ring,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

/// Element of an extension with basis
///
/// Can be represented as coordinates relative to the basis.
#[derive(Debug, Clone, PartialEq)]
pub struct RingExtensionWithBasisElement<K, L>
where
    K: Ring,
    L: Ring,
{
    base: RingExtensionElement<K, L>,
    /// Coordinates in the basis
    coordinates: Vec<K>,
}

impl<K, L> RingExtensionWithBasisElement<K, L>
where
    K: Ring,
    L: Ring,
{
    /// Creates an element from coordinates
    pub fn from_coordinates(coordinates: Vec<K>) -> Self {
        RingExtensionWithBasisElement {
            base: RingExtensionElement::new("basis element".to_string()),
            coordinates,
        }
    }

    /// Returns the coordinates
    pub fn coordinates(&self) -> &[K] {
        &self.coordinates
    }
}

/// Element of a fraction field extension
#[derive(Debug, Clone, PartialEq)]
pub struct RingExtensionFractionFieldElement<K, L>
where
    K: Ring,
    L: Ring,
{
    base: RingExtensionElement<K, L>,
}

impl<K, L> RingExtensionFractionFieldElement<K, L>
where
    K: Ring,
    L: Ring,
{
    /// Creates a new fraction field element
    pub fn new(description: String) -> Self {
        RingExtensionFractionFieldElement {
            base: RingExtensionElement::new(description),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_element() {
        let elem: RingExtensionElement<i32, f64> =
            RingExtensionElement::new("test".to_string());
        assert_eq!(format!("{}", elem), "test");
    }

    #[test]
    fn test_basis_element_coordinates() {
        let coords = vec![1, 2, 3];
        let elem: RingExtensionWithBasisElement<i32, f64> =
            RingExtensionWithBasisElement::from_coordinates(coords);
        assert_eq!(elem.coordinates(), &[1, 2, 3]);
    }

    #[test]
    fn test_fraction_field_element() {
        let _elem: RingExtensionFractionFieldElement<i32, f64> =
            RingExtensionFractionFieldElement::new("frac".to_string());
    }
}
