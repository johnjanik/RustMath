//! Imaginary Groups
//!
//! This module implements groups of purely imaginary numbers. The imaginary group
//! consists of elements of the form k·i where k is an element of the base ring and
//! i is the imaginary unit.
//!
//! # Theory
//!
//! The imaginary group over a ring R is an additive group isomorphic to R, where
//! elements are represented as k·i. This is useful in asymptotic analysis where
//! complex exponents are decomposed into real and imaginary parts.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::imaginary_group::*;
//!
//! let group = ImaginaryGroup::<i32>::new();
//! let i = group.element(1);
//! let two_i = group.element(2);
//! let sum = i.add(&two_i);
//! assert_eq!(*sum.imaginary_part(), 3);
//! ```

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Neg, Sub};

/// A group of purely imaginary numbers over a base ring R
///
/// Elements are of the form k·i where k ∈ R and i is the imaginary unit.
#[derive(Clone, Debug)]
pub struct ImaginaryGroup<R> {
    _phantom: PhantomData<R>,
}

/// An element of the imaginary group
///
/// Represents k·i where k is from the base ring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImaginaryElement<R> {
    /// The coefficient of i
    imaginary_part: R,
}

impl<R> ImaginaryGroup<R> {
    /// Create a new imaginary group
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::imaginary_group::ImaginaryGroup;
    ///
    /// let group = ImaginaryGroup::<i32>::new();
    /// ```
    pub fn new() -> Self {
        ImaginaryGroup {
            _phantom: PhantomData,
        }
    }

    /// Create an element of the imaginary group
    ///
    /// # Arguments
    ///
    /// * `imaginary_part` - The coefficient k in k·i
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::imaginary_group::ImaginaryGroup;
    ///
    /// let group = ImaginaryGroup::<i32>::new();
    /// let element = group.element(5);  // Represents 5i
    /// ```
    pub fn element(&self, imaginary_part: R) -> ImaginaryElement<R> {
        ImaginaryElement { imaginary_part }
    }
}

impl<R> Default for ImaginaryGroup<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> ImaginaryElement<R> {
    /// Create a new imaginary element
    ///
    /// # Arguments
    ///
    /// * `imaginary_part` - The coefficient k in k·i
    pub fn new(imaginary_part: R) -> Self {
        ImaginaryElement { imaginary_part }
    }

    /// Get the imaginary part (the coefficient of i)
    pub fn imaginary_part(&self) -> &R {
        &self.imaginary_part
    }

    /// Get the imaginary part as a mutable reference
    pub fn imaginary_part_mut(&mut self) -> &mut R {
        &mut self.imaginary_part
    }

    /// Decompose into the imaginary part
    pub fn into_imaginary_part(self) -> R {
        self.imaginary_part
    }
}

impl<R> ImaginaryElement<R>
where
    R: Clone + Add<Output = R>,
{
    /// Add two imaginary elements
    ///
    /// (a·i) + (b·i) = (a + b)·i
    pub fn add(&self, other: &Self) -> Self {
        ImaginaryElement {
            imaginary_part: self.imaginary_part.clone() + other.imaginary_part.clone(),
        }
    }
}

impl<R> ImaginaryElement<R>
where
    R: Clone + Sub<Output = R>,
{
    /// Subtract two imaginary elements
    ///
    /// (a·i) - (b·i) = (a - b)·i
    pub fn sub(&self, other: &Self) -> Self {
        ImaginaryElement {
            imaginary_part: self.imaginary_part.clone() - other.imaginary_part.clone(),
        }
    }
}

impl<R> ImaginaryElement<R>
where
    R: Clone + Neg<Output = R>,
{
    /// Negate an imaginary element
    ///
    /// -(a·i) = (-a)·i
    pub fn neg(&self) -> Self {
        ImaginaryElement {
            imaginary_part: -self.imaginary_part.clone(),
        }
    }
}

impl<R> ImaginaryElement<R>
where
    R: Clone + Default + PartialEq,
{
    /// Check if this is the zero element
    ///
    /// The zero element is 0·i
    pub fn is_zero(&self) -> bool {
        self.imaginary_part == R::default()
    }
}

// Implement standard Add trait
impl<R> Add for ImaginaryElement<R>
where
    R: Clone + Add<Output = R>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        ImaginaryElement {
            imaginary_part: self.imaginary_part + other.imaginary_part,
        }
    }
}

// Implement standard Sub trait
impl<R> Sub for ImaginaryElement<R>
where
    R: Clone + Sub<Output = R>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        ImaginaryElement {
            imaginary_part: self.imaginary_part - other.imaginary_part,
        }
    }
}

// Implement standard Neg trait
impl<R> Neg for ImaginaryElement<R>
where
    R: Clone + Neg<Output = R>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        ImaginaryElement {
            imaginary_part: -self.imaginary_part,
        }
    }
}

impl<R> fmt::Display for ImaginaryElement<R>
where
    R: fmt::Display + PartialEq + Default + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let zero = R::default();

        if self.imaginary_part == zero {
            write!(f, "0")
        } else {
            write!(f, "{}*I", self.imaginary_part)
        }
    }
}

/// Create the additive identity element (0·i)
impl<R> ImaginaryElement<R>
where
    R: Default,
{
    /// Create the zero element
    pub fn zero() -> Self {
        ImaginaryElement {
            imaginary_part: R::default(),
        }
    }
}

/// Create the imaginary unit (1·i)
impl ImaginaryElement<i32> {
    /// Create the imaginary unit i
    pub fn i() -> Self {
        ImaginaryElement {
            imaginary_part: 1,
        }
    }
}

impl ImaginaryElement<i64> {
    /// Create the imaginary unit i
    pub fn i() -> Self {
        ImaginaryElement {
            imaginary_part: 1,
        }
    }
}

impl ImaginaryElement<f64> {
    /// Create the imaginary unit i
    pub fn i() -> Self {
        ImaginaryElement {
            imaginary_part: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let group = ImaginaryGroup::<i32>::new();
        let elem = group.element(5);

        assert_eq!(*elem.imaginary_part(), 5);
    }

    #[test]
    fn test_addition() {
        let a = ImaginaryElement::new(3);
        let b = ImaginaryElement::new(4);

        let sum = a.add(&b);
        assert_eq!(*sum.imaginary_part(), 7);

        // Test with trait
        let c = ImaginaryElement::new(5);
        let d = ImaginaryElement::new(2);
        let sum2 = c + d;
        assert_eq!(*sum2.imaginary_part(), 7);
    }

    #[test]
    fn test_subtraction() {
        let a = ImaginaryElement::new(10);
        let b = ImaginaryElement::new(3);

        let diff = a.sub(&b);
        assert_eq!(*diff.imaginary_part(), 7);

        // Test with trait
        let c = ImaginaryElement::new(15);
        let d = ImaginaryElement::new(5);
        let diff2 = c - d;
        assert_eq!(*diff2.imaginary_part(), 10);
    }

    #[test]
    fn test_negation() {
        let a = ImaginaryElement::new(5);
        let neg_a = a.neg();

        assert_eq!(*neg_a.imaginary_part(), -5);

        // Test with trait
        let b = ImaginaryElement::new(7);
        let neg_b = -b;
        assert_eq!(*neg_b.imaginary_part(), -7);
    }

    #[test]
    fn test_zero() {
        let zero = ImaginaryElement::<i32>::zero();
        assert!(zero.is_zero());
        assert_eq!(*zero.imaginary_part(), 0);
    }

    #[test]
    fn test_imaginary_unit() {
        let i = ImaginaryElement::<i32>::i();
        assert_eq!(*i.imaginary_part(), 1);

        let i64_i = ImaginaryElement::<i64>::i();
        assert_eq!(*i64_i.imaginary_part(), 1);

        let f64_i = ImaginaryElement::<f64>::i();
        assert_eq!(*f64_i.imaginary_part(), 1.0);
    }

    #[test]
    fn test_display() {
        let zero = ImaginaryElement::new(0);
        assert_eq!(format!("{}", zero), "0");

        let i = ImaginaryElement::new(1);
        assert_eq!(format!("{}", i), "1*I");

        let five_i = ImaginaryElement::new(5);
        assert_eq!(format!("{}", five_i), "5*I");

        let neg_i = ImaginaryElement::new(-1);
        assert_eq!(format!("{}", neg_i), "-1*I");
    }

    #[test]
    fn test_associativity() {
        let a = ImaginaryElement::new(2);
        let b = ImaginaryElement::new(3);
        let c = ImaginaryElement::new(4);

        // (a + b) + c
        let left_assoc = (a.clone() + b.clone()) + c.clone();

        // a + (b + c)
        let right_assoc = a + (b + c);

        assert_eq!(
            left_assoc.imaginary_part(),
            right_assoc.imaginary_part()
        );
    }

    #[test]
    fn test_commutativity() {
        let a = ImaginaryElement::new(5);
        let b = ImaginaryElement::new(7);

        let ab = a.clone() + b.clone();
        let ba = b + a;

        assert_eq!(ab.imaginary_part(), ba.imaginary_part());
    }

    #[test]
    fn test_identity() {
        let a = ImaginaryElement::new(42);
        let zero = ImaginaryElement::<i32>::zero();

        let sum = a.clone() + zero.clone();
        assert_eq!(sum.imaginary_part(), a.imaginary_part());

        let sum2 = zero + a.clone();
        assert_eq!(sum2.imaginary_part(), a.imaginary_part());
    }

    #[test]
    fn test_inverse() {
        let a = ImaginaryElement::new(13);
        let neg_a = -a.clone();

        let sum = a + neg_a;
        assert!(sum.is_zero());
    }

    #[test]
    fn test_mixed_operations() {
        let a = ImaginaryElement::new(10);
        let b = ImaginaryElement::new(5);
        let c = ImaginaryElement::new(3);

        // (10i - 5i) + 3i = 5i + 3i = 8i
        let result = (a - b) + c;
        assert_eq!(*result.imaginary_part(), 8);
    }

    #[test]
    fn test_into_imaginary_part() {
        let elem = ImaginaryElement::new(42);
        let value = elem.into_imaginary_part();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_different_types() {
        // Test with i64
        let a = ImaginaryElement::new(100i64);
        let b = ImaginaryElement::new(50i64);
        let sum = a + b;
        assert_eq!(*sum.imaginary_part(), 150i64);

        // Test with f64
        let c = ImaginaryElement::new(3.5f64);
        let d = ImaginaryElement::new(1.5f64);
        let sum2 = c + d;
        assert_eq!(*sum2.imaginary_part(), 5.0f64);
    }
}
