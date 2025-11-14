//! # Monoid Base Classes
//!
//! This module provides the base trait and structures for monoids.

use std::fmt::Debug;
use std::hash::Hash;

/// A monoid is a set with an associative binary operation and an identity element
pub trait Monoid: Clone + Debug + Eq + Hash {
    /// The identity element of the monoid
    fn identity() -> Self;

    /// The monoid operation (multiplication)
    fn op(&self, other: &Self) -> Self;

    /// Check if this is the identity element
    fn is_identity(&self) -> bool {
        self == &Self::identity()
    }

    /// Compute the power of an element
    fn pow(&self, n: u64) -> Self {
        if n == 0 {
            return Self::identity();
        }

        let mut result = self.clone();
        let mut base = self.clone();
        let mut exp = n - 1;

        while exp > 0 {
            if exp % 2 == 1 {
                result = result.op(&base);
            }
            base = base.op(&base);
            exp /= 2;
        }

        result
    }
}

/// Base class for monoid implementations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Monoid_class<T: Monoid> {
    /// Identity element
    identity: T,
    /// Name of the monoid
    name: Option<String>,
}

impl<T: Monoid> Monoid_class<T> {
    /// Create a new monoid
    pub fn new() -> Self {
        Monoid_class {
            identity: T::identity(),
            name: None,
        }
    }

    /// Create a new monoid with a name
    pub fn with_name(name: String) -> Self {
        Monoid_class {
            identity: T::identity(),
            name: Some(name),
        }
    }

    /// Get the identity element
    pub fn identity(&self) -> &T {
        &self.identity
    }

    /// Get the name of the monoid
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Check if an element is in the monoid
    pub fn contains(&self, _element: &T) -> bool {
        // All elements of type T are in the monoid
        true
    }

    /// Get the order of the monoid (if finite)
    pub fn order(&self) -> Option<usize> {
        // Most monoids are infinite
        None
    }

    /// Check if the monoid is abelian (commutative)
    pub fn is_abelian(&self) -> bool {
        // Override in subclasses
        false
    }

    /// Check if the monoid is finite
    pub fn is_finite(&self) -> bool {
        self.order().is_some()
    }
}

impl<T: Monoid> Default for Monoid_class<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestMonoid {
        value: i32,
    }

    impl Monoid for TestMonoid {
        fn identity() -> Self {
            TestMonoid { value: 0 }
        }

        fn op(&self, other: &Self) -> Self {
            TestMonoid {
                value: self.value + other.value,
            }
        }
    }

    #[test]
    fn test_monoid_identity() {
        let id = TestMonoid::identity();
        assert_eq!(id.value, 0);
        assert!(id.is_identity());
    }

    #[test]
    fn test_monoid_op() {
        let a = TestMonoid { value: 3 };
        let b = TestMonoid { value: 5 };
        let c = a.op(&b);
        assert_eq!(c.value, 8);
    }

    #[test]
    fn test_monoid_pow() {
        let a = TestMonoid { value: 2 };
        let result = a.pow(5);
        assert_eq!(result.value, 10);
    }

    #[test]
    fn test_monoid_class() {
        let m = Monoid_class::<TestMonoid>::new();
        assert_eq!(m.identity().value, 0);
        assert!(!m.is_finite());
    }
}
