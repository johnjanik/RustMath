//! Fibers of vector bundles
//!
//! This module provides structures for working with individual fibers
//! of vector bundles. Each fiber is a vector space attached to a point
//! on the base manifold.

use crate::{ManifoldPoint, ManifoldError, Result};
use rustmath_core::Ring;
use std::marker::PhantomData;

/// A fiber of a vector bundle at a specific point
///
/// For a vector bundle E → M, the fiber over a point p ∈ M is
/// the vector space Ep = π^(-1)(p).
///
/// # Type Parameters
///
/// * `R` - The ring/field over which the vector space is defined
#[derive(Clone)]
pub struct VectorBundleFiber<R: Ring> {
    /// The base point over which this fiber sits
    base_point: String, // Simplified representation
    /// Dimension of this fiber (as a vector space)
    rank: usize,
    /// Name of the fiber
    name: String,
    _ring: PhantomData<R>,
}

impl<R: Ring> VectorBundleFiber<R> {
    /// Create a new vector bundle fiber
    ///
    /// # Arguments
    ///
    /// * `base_point` - The point on the base manifold
    /// * `rank` - The dimension of the fiber vector space
    /// * `name` - Name for this fiber
    pub fn new(base_point: String, rank: usize, name: String) -> Self {
        VectorBundleFiber {
            base_point,
            rank,
            name,
            _ring: PhantomData,
        }
    }

    /// Get the base point
    pub fn base_point(&self) -> &str {
        &self.base_point
    }

    /// Get the rank (dimension) of the fiber
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create a zero element in this fiber
    pub fn zero_element(&self) -> VectorBundleFiberElement<R> {
        VectorBundleFiberElement::new(
            vec![R::zero(); self.rank],
            self.clone(),
        )
    }

    /// Create a basis element (standard basis vector)
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the basis vector (0-based)
    pub fn basis_element(&self, index: usize) -> Result<VectorBundleFiberElement<R>> {
        if index >= self.rank {
            return Err(ManifoldError::InvalidDimension(
                format!("Basis index {} out of range for fiber of rank {}", index, self.rank)
            ));
        }

        let mut components = vec![R::zero(); self.rank];
        components[index] = R::one();

        Ok(VectorBundleFiberElement::new(components, self.clone()))
    }
}

/// An element (vector) in a vector bundle fiber
///
/// This represents a vector in the fiber Ep over a point p.
#[derive(Clone)]
pub struct VectorBundleFiberElement<R: Ring> {
    /// Components of the vector in some basis
    components: Vec<R>,
    /// The fiber this element belongs to
    fiber: VectorBundleFiber<R>,
}

impl<R: Ring> VectorBundleFiberElement<R> {
    /// Create a new fiber element
    ///
    /// # Arguments
    ///
    /// * `components` - The components of the vector
    /// * `fiber` - The fiber this element belongs to
    pub fn new(components: Vec<R>, fiber: VectorBundleFiber<R>) -> Self {
        VectorBundleFiberElement { components, fiber }
    }

    /// Get the components
    pub fn components(&self) -> &[R] {
        &self.components
    }

    /// Get the fiber this element belongs to
    pub fn fiber(&self) -> &VectorBundleFiber<R> {
        &self.fiber
    }

    /// Get a specific component
    pub fn component(&self, index: usize) -> Result<&R> {
        self.components.get(index).ok_or_else(|| {
            ManifoldError::InvalidDimension(
                format!("Component index {} out of range", index)
            )
        })
    }

    /// Add two fiber elements
    ///
    /// Both elements must belong to the same fiber
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.components.len() != other.components.len() {
            return Err(ManifoldError::InvalidOperation(
                "Cannot add elements from fibers of different dimensions".to_string()
            ));
        }

        let new_components: Vec<R> = self.components.iter()
            .zip(other.components.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Ok(VectorBundleFiberElement::new(new_components, self.fiber.clone()))
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        let new_components: Vec<R> = self.components.iter()
            .map(|c| c.clone() * scalar.clone())
            .collect();

        VectorBundleFiberElement::new(new_components, self.fiber.clone())
    }

    /// Negate the element
    pub fn negate(&self) -> Self {
        let new_components: Vec<R> = self.components.iter()
            .map(|c| R::zero() - c.clone())
            .collect();

        VectorBundleFiberElement::new(new_components, self.fiber.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_fiber_creation() {
        let fiber = VectorBundleFiber::<Rational>::new(
            "p0".to_string(),
            3,
            "E_p0".to_string(),
        );

        assert_eq!(fiber.rank(), 3);
        assert_eq!(fiber.base_point(), "p0");
        assert_eq!(fiber.name(), "E_p0");
    }

    #[test]
    fn test_zero_element() {
        let fiber = VectorBundleFiber::<Rational>::new(
            "p0".to_string(),
            3,
            "E_p0".to_string(),
        );

        let zero = fiber.zero_element();
        assert_eq!(zero.components().len(), 3);
        for comp in zero.components() {
            assert_eq!(*comp, Rational::from(0));
        }
    }

    #[test]
    fn test_basis_element() {
        let fiber = VectorBundleFiber::<Rational>::new(
            "p0".to_string(),
            3,
            "E_p0".to_string(),
        );

        let e1 = fiber.basis_element(1).unwrap();
        assert_eq!(e1.components()[0], Rational::from(0));
        assert_eq!(e1.components()[1], Rational::from(1));
        assert_eq!(e1.components()[2], Rational::from(0));
    }

    #[test]
    fn test_basis_element_out_of_range() {
        let fiber = VectorBundleFiber::<Rational>::new(
            "p0".to_string(),
            3,
            "E_p0".to_string(),
        );

        assert!(fiber.basis_element(3).is_err());
    }

    #[test]
    fn test_element_addition() {
        let fiber = VectorBundleFiber::<Rational>::new(
            "p0".to_string(),
            2,
            "E_p0".to_string(),
        );

        let v1 = VectorBundleFiberElement::new(
            vec![Rational::from(1), Rational::from(2)],
            fiber.clone(),
        );

        let v2 = VectorBundleFiberElement::new(
            vec![Rational::from(3), Rational::from(4)],
            fiber.clone(),
        );

        let sum = v1.add(&v2).unwrap();
        assert_eq!(sum.components()[0], Rational::from(4));
        assert_eq!(sum.components()[1], Rational::from(6));
    }

    #[test]
    fn test_scalar_multiplication() {
        let fiber = VectorBundleFiber::<Rational>::new(
            "p0".to_string(),
            2,
            "E_p0".to_string(),
        );

        let v = VectorBundleFiberElement::new(
            vec![Rational::from(2), Rational::from(3)],
            fiber,
        );

        let scaled = v.scalar_mul(&Rational::from(5));
        assert_eq!(scaled.components()[0], Rational::from(10));
        assert_eq!(scaled.components()[1], Rational::from(15));
    }

    #[test]
    fn test_negation() {
        let fiber = VectorBundleFiber::<Rational>::new(
            "p0".to_string(),
            2,
            "E_p0".to_string(),
        );

        let v = VectorBundleFiberElement::new(
            vec![Rational::from(2), Rational::from(-3)],
            fiber,
        );

        let neg = v.negate();
        assert_eq!(neg.components()[0], Rational::from(-2));
        assert_eq!(neg.components()[1], Rational::from(3));
    }
}
