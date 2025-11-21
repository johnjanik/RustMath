//! # Core Valuation Module
//!
//! Provides the base traits and structures for discrete valuations.
//!
//! ## Theory
//!
//! A discrete valuation on a ring R is a function v: R → ℤ ∪ {∞} such that:
//! 1. v(xy) = v(x) + v(y)
//! 2. v(x+y) ≥ min(v(x), v(y))
//! 3. v(0) = ∞
//! 4. v is surjective onto ℤ ∪ {∞}

use rustmath_core::Ring;
use std::cmp::Ordering;
use std::fmt;

/// Value in the discrete valuation codomain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValuationValue {
    /// Finite valuation
    Finite(i64),
    /// Infinite valuation (for zero elements)
    Infinity,
}

impl ValuationValue {
    /// Check if this is infinity
    pub fn is_infinite(&self) -> bool {
        matches!(self, ValuationValue::Infinity)
    }

    /// Get the finite value, panic if infinite
    pub fn unwrap(&self) -> i64 {
        match self {
            ValuationValue::Finite(v) => *v,
            ValuationValue::Infinity => panic!("Cannot unwrap infinity"),
        }
    }

    /// Add two valuation values (for multiplication of elements)
    pub fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (ValuationValue::Infinity, _) | (_, ValuationValue::Infinity) => {
                ValuationValue::Infinity
            }
            (ValuationValue::Finite(a), ValuationValue::Finite(b)) => ValuationValue::Finite(a + b),
        }
    }

    /// Minimum of two valuation values (for addition of elements)
    pub fn min(&self, other: &Self) -> Self {
        match (self, other) {
            (ValuationValue::Infinity, x) | (x, ValuationValue::Infinity) => *x,
            (ValuationValue::Finite(a), ValuationValue::Finite(b)) => {
                ValuationValue::Finite((*a).min(*b))
            }
        }
    }
}

impl PartialOrd for ValuationValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ValuationValue {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (ValuationValue::Infinity, ValuationValue::Infinity) => Ordering::Equal,
            (ValuationValue::Infinity, ValuationValue::Finite(_)) => Ordering::Greater,
            (ValuationValue::Finite(_), ValuationValue::Infinity) => Ordering::Less,
            (ValuationValue::Finite(a), ValuationValue::Finite(b)) => a.cmp(b),
        }
    }
}

impl fmt::Display for ValuationValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValuationValue::Finite(v) => write!(f, "{}", v),
            ValuationValue::Infinity => write!(f, "∞"),
        }
    }
}

/// Discrete pseudo-valuation trait
///
/// A pseudo-valuation may have infinite value on non-zero elements.
pub trait DiscretePseudoValuation<R: Ring>: std::fmt::Debug + Clone {
    /// Evaluate the valuation on an element
    fn value(&self, element: &R) -> ValuationValue;

    /// Check if element is in the valuation ring
    fn is_in_valuation_ring(&self, element: &R) -> bool {
        match self.value(element) {
            ValuationValue::Finite(v) => v >= 0,
            ValuationValue::Infinity => true,
        }
    }

    /// Get the residue field characteristic (if applicable)
    fn residue_field_characteristic(&self) -> Option<usize> {
        None
    }
}

/// Discrete valuation trait
///
/// A proper discrete valuation (not just a pseudo-valuation).
pub trait DiscreteValuation<R: Ring>: DiscretePseudoValuation<R> {
    /// Get a uniformizer (element with valuation 1)
    fn uniformizer(&self) -> Option<R>;

    /// Get the value group
    fn value_group(&self) -> Vec<i64> {
        // Return ℤ by default
        vec![1]
    }
}

/// Infinite discrete pseudo-valuation
///
/// Takes infinite value on all non-zero elements (degenerate case).
#[derive(Debug, Clone, PartialEq)]
pub struct InfiniteDiscretePseudoValuation<R: Ring> {
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> InfiniteDiscretePseudoValuation<R> {
    /// Create a new infinite pseudo-valuation
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<R: Ring> Default for InfiniteDiscretePseudoValuation<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring> DiscretePseudoValuation<R> for InfiniteDiscretePseudoValuation<R> {
    fn value(&self, element: &R) -> ValuationValue {
        if element.is_zero() {
            ValuationValue::Infinity
        } else {
            ValuationValue::Infinity
        }
    }
}

/// Node in a MacLane approximation tree
///
/// Used for computing extensions of valuations to polynomial rings.
#[derive(Debug, Clone)]
pub struct MacLaneApproximantNode<R: Ring> {
    /// The valuation at this node
    pub valuation: Box<dyn DiscretePseudoValuation<R>>,
    /// Parent node
    pub parent: Option<Box<MacLaneApproximantNode<R>>>,
}

impl<R: Ring> MacLaneApproximantNode<R> {
    /// Create a new MacLane node
    pub fn new(valuation: Box<dyn DiscretePseudoValuation<R>>) -> Self {
        Self {
            valuation,
            parent: None,
        }
    }

    /// Create a node with a parent
    pub fn with_parent(
        valuation: Box<dyn DiscretePseudoValuation<R>>,
        parent: MacLaneApproximantNode<R>,
    ) -> Self {
        Self {
            valuation,
            parent: Some(Box::new(parent)),
        }
    }
}

/// Negative infinite discrete pseudo-valuation
///
/// A special degenerate case (rarely used).
#[derive(Debug, Clone, PartialEq)]
pub struct NegativeInfiniteDiscretePseudoValuation<R: Ring> {
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> NegativeInfiniteDiscretePseudoValuation<R> {
    /// Create a new negative infinite pseudo-valuation
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<R: Ring> Default for NegativeInfiniteDiscretePseudoValuation<R> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valuation_value() {
        let v1 = ValuationValue::Finite(5);
        let v2 = ValuationValue::Finite(3);
        let inf = ValuationValue::Infinity;

        assert!(!v1.is_infinite());
        assert!(inf.is_infinite());

        assert_eq!(v1.unwrap(), 5);

        // Test addition
        assert_eq!(v1.add(&v2), ValuationValue::Finite(8));
        assert_eq!(v1.add(&inf), ValuationValue::Infinity);

        // Test min
        assert_eq!(v1.min(&v2), ValuationValue::Finite(3));
        assert_eq!(v1.min(&inf), ValuationValue::Finite(5));
    }

    #[test]
    fn test_valuation_value_ordering() {
        let v1 = ValuationValue::Finite(3);
        let v2 = ValuationValue::Finite(5);
        let inf = ValuationValue::Infinity;

        assert!(v1 < v2);
        assert!(v2 < inf);
        assert!(v1 < inf);
    }

    #[test]
    fn test_infinite_pseudo_valuation() {
        use rustmath_integers::Integer;

        let val = InfiniteDiscretePseudoValuation::<Integer>::new();
        assert_eq!(val.value(&Integer::from(0)), ValuationValue::Infinity);
        assert_eq!(val.value(&Integer::from(5)), ValuationValue::Infinity);
    }

    #[test]
    #[should_panic(expected = "Cannot unwrap infinity")]
    fn test_unwrap_infinity() {
        let inf = ValuationValue::Infinity;
        let _ = inf.unwrap();
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ValuationValue::Finite(42)), "42");
        assert_eq!(format!("{}", ValuationValue::Infinity), "∞");
    }
}
