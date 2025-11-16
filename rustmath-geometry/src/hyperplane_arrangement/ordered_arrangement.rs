//! Ordered hyperplane arrangements
//!
//! This module provides hyperplane arrangements where the order of hyperplanes
//! is significant. This ordering is important for computing topological invariants
//! such as fundamental groups and braid monodromies.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::hyperplane_arrangement::ordered_arrangement::OrderedHyperplaneArrangementElement;
//! use rustmath_geometry::hyperplane_arrangement::hyperplane::Hyperplane;
//! use rustmath_integers::Integer;
//!
//! // Create hyperplanes in a specific order
//! let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
//! let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
//!
//! // The order is preserved
//! let arr = OrderedHyperplaneArrangementElement::new(vec![h1, h2]);
//! assert_eq!(arr.num_hyperplanes(), 2);
//! ```

use crate::hyperplane_arrangement::arrangement::HyperplaneArrangementElement;
use crate::hyperplane_arrangement::hyperplane::Hyperplane;
use rustmath_core::Ring;
use std::fmt;

/// An ordered hyperplane arrangement
///
/// This is similar to HyperplaneArrangementElement, but the order of the
/// hyperplanes is significant and preserved. This ordering is important for
/// topological computations like fundamental groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedHyperplaneArrangementElement<R: Ring> {
    /// The underlying unordered arrangement (ordering is implicit in the hyperplanes Vec)
    arrangement: HyperplaneArrangementElement<R>,
}

impl<R: Ring> OrderedHyperplaneArrangementElement<R> {
    /// Create a new ordered hyperplane arrangement
    ///
    /// The order of hyperplanes is preserved as given.
    ///
    /// # Arguments
    ///
    /// * `hyperplanes` - The ordered collection of hyperplanes
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::ordered_arrangement::OrderedHyperplaneArrangementElement;
    /// use rustmath_geometry::hyperplane_arrangement::hyperplane::Hyperplane;
    /// use rustmath_integers::Integer;
    ///
    /// let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
    /// let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
    /// let arr = OrderedHyperplaneArrangementElement::new(vec![h1, h2]);
    /// ```
    pub fn new(hyperplanes: Vec<Hyperplane<R>>) -> Self {
        let arrangement = HyperplaneArrangementElement::new(hyperplanes);
        Self { arrangement }
    }

    /// Create from an unordered arrangement
    ///
    /// Note: This uses the current order of hyperplanes in the arrangement.
    pub fn from_arrangement(arrangement: HyperplaneArrangementElement<R>) -> Self {
        Self { arrangement }
    }

    /// Get the hyperplanes in order
    pub fn hyperplanes(&self) -> &[Hyperplane<R>] {
        self.arrangement.hyperplanes()
    }

    /// Get the hyperplane at a specific position
    pub fn hyperplane(&self, index: usize) -> Option<&Hyperplane<R>> {
        self.hyperplanes().get(index)
    }

    /// Get the number of hyperplanes
    pub fn num_hyperplanes(&self) -> usize {
        self.arrangement.num_hyperplanes()
    }

    /// Get the ambient dimension
    pub fn ambient_dimension(&self) -> usize {
        self.arrangement.ambient_dimension()
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.arrangement.dimension()
    }

    /// Compute the rank
    pub fn rank(&self) -> usize {
        self.arrangement.rank()
    }

    /// Check if essential
    pub fn is_essential(&self) -> bool {
        self.arrangement.is_essential()
    }

    /// Check if central
    pub fn is_central(&self) -> bool {
        self.arrangement.is_central()
    }

    /// Check if linear
    pub fn is_linear(&self) -> bool {
        self.arrangement.is_linear()
    }

    /// Check if simplicial
    pub fn is_simplicial(&self) -> bool {
        self.arrangement.is_simplicial()
    }

    /// Union with another ordered arrangement
    ///
    /// Appends the hyperplanes from the other arrangement, preserving order.
    pub fn union(&self, other: &Self) -> Self {
        let mut hyperplanes = self.hyperplanes().to_vec();
        hyperplanes.extend(other.hyperplanes().iter().cloned());
        Self::new(hyperplanes)
    }

    /// Deletion of a hyperplane by index
    ///
    /// Returns the arrangement with the specified hyperplane removed,
    /// preserving the order of remaining hyperplanes.
    pub fn deletion(&self, index: usize) -> Self {
        let unordered_del = self.arrangement.deletion(index);
        Self::from_arrangement(unordered_del)
    }

    /// Restriction to a hyperplane
    ///
    /// Returns the ordered arrangement restricted to the specified hyperplane.
    pub fn restriction(&self, index: usize) -> Option<Self> {
        self.arrangement
            .restriction(index)
            .map(Self::from_arrangement)
    }

    /// Get the underlying unordered arrangement
    pub fn as_arrangement(&self) -> &HyperplaneArrangementElement<R> {
        &self.arrangement
    }

    /// Compute sign vector of a point with respect to the ordered hyperplanes
    pub fn sign_vector(&self, point: &[R]) -> Vec<i8> {
        self.arrangement.sign_vector(point)
    }

    /// Check if a point is contained in any hyperplane
    pub fn contains_point(&self, point: &[R]) -> bool {
        self.arrangement.contains_point(point)
    }

    /// Reorder the hyperplanes
    ///
    /// Creates a new ordered arrangement with hyperplanes in the specified order.
    ///
    /// # Arguments
    ///
    /// * `permutation` - The new order (indices into current hyperplane list)
    ///
    /// # Panics
    ///
    /// Panics if permutation is invalid (wrong length, out of bounds indices).
    pub fn reorder(&self, permutation: &[usize]) -> Self {
        if permutation.len() != self.num_hyperplanes() {
            panic!("Permutation length must equal number of hyperplanes");
        }

        let hyperplanes: Vec<_> = permutation
            .iter()
            .map(|&i| {
                self.hyperplanes()
                    .get(i)
                    .expect("Permutation index out of bounds")
                    .clone()
            })
            .collect();

        Self::new(hyperplanes)
    }

    /// Reverse the order of hyperplanes
    pub fn reverse(&self) -> Self {
        let mut hyperplanes = self.hyperplanes().to_vec();
        hyperplanes.reverse();
        Self::new(hyperplanes)
    }

    /// Insert a hyperplane at a specific position
    ///
    /// # Arguments
    ///
    /// * `index` - Position to insert at (0 = beginning)
    /// * `hyperplane` - The hyperplane to insert
    pub fn insert(&self, index: usize, hyperplane: Hyperplane<R>) -> Self {
        let mut hyperplanes = self.hyperplanes().to_vec();
        hyperplanes.insert(index, hyperplane);
        Self::new(hyperplanes)
    }

    /// Append a hyperplane at the end
    pub fn append(&self, hyperplane: Hyperplane<R>) -> Self {
        let mut hyperplanes = self.hyperplanes().to_vec();
        hyperplanes.push(hyperplane);
        Self::new(hyperplanes)
    }
}

impl<R: Ring + fmt::Display> fmt::Display for OrderedHyperplaneArrangementElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OrderedHyperplaneArrangement({} hyperplanes in {}D space)",
            self.num_hyperplanes(),
            self.ambient_dimension()
        )
    }
}

/// Parent class for ordered hyperplane arrangements
///
/// This manages the creation of ordered hyperplane arrangements.
#[derive(Clone, Debug)]
pub struct OrderedHyperplaneArrangements {
    /// Dimension of the ambient space
    dimension: usize,
}

impl OrderedHyperplaneArrangements {
    /// Create a new ordered hyperplane arrangements parent
    ///
    /// # Arguments
    ///
    /// * `dimension` - Dimension of the ambient vector space
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Create an ordered arrangement from hyperplanes
    pub fn arrangement<R: Ring>(
        &self,
        hyperplanes: Vec<Hyperplane<R>>,
    ) -> OrderedHyperplaneArrangementElement<R> {
        for h in &hyperplanes {
            if h.ambient_dimension() != self.dimension {
                panic!("Hyperplane dimension mismatch");
            }
        }
        OrderedHyperplaneArrangementElement::new(hyperplanes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;
    use rustmath_rationals::Rational;

    #[test]
    fn test_new_ordered_arrangement() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = OrderedHyperplaneArrangementElement::new(vec![h1, h2]);

        assert_eq!(arr.num_hyperplanes(), 2);
        assert_eq!(arr.ambient_dimension(), 2);
    }

    #[test]
    fn test_hyperplane_access() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = OrderedHyperplaneArrangementElement::new(vec![h1.clone(), h2.clone()]);

        assert_eq!(arr.hyperplane(0), Some(&h1));
        assert_eq!(arr.hyperplane(1), Some(&h2));
        assert_eq!(arr.hyperplane(2), None);
    }

    #[test]
    fn test_ordered_properties() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let arr = OrderedHyperplaneArrangementElement::new(vec![h1, h2]);

        assert!(arr.is_central());
        assert!(arr.is_essential());
        assert_eq!(arr.rank(), 2);
    }

    #[test]
    fn test_ordered_union() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let h3 = Hyperplane::new(vec![Integer::from(1), Integer::from(1)], Integer::from(0));

        let arr1 = OrderedHyperplaneArrangementElement::new(vec![h1.clone(), h2.clone()]);
        let arr2 = OrderedHyperplaneArrangementElement::new(vec![h3.clone()]);

        let union = arr1.union(&arr2);
        assert_eq!(union.num_hyperplanes(), 3);

        // Check order is preserved
        assert_eq!(union.hyperplane(0), Some(&h1));
        assert_eq!(union.hyperplane(1), Some(&h2));
        assert_eq!(union.hyperplane(2), Some(&h3));
    }

    #[test]
    fn test_reorder() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let h3 = Hyperplane::new(vec![Integer::from(1), Integer::from(1)], Integer::from(0));

        let arr = OrderedHyperplaneArrangementElement::new(vec![h1.clone(), h2.clone(), h3.clone()]);

        // Reverse order: [2, 1, 0]
        let reordered = arr.reorder(&[2, 1, 0]);
        assert_eq!(reordered.hyperplane(0), Some(&h3));
        assert_eq!(reordered.hyperplane(1), Some(&h2));
        assert_eq!(reordered.hyperplane(2), Some(&h1));
    }

    #[test]
    fn test_reverse() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));

        let arr = OrderedHyperplaneArrangementElement::new(vec![h1.clone(), h2.clone()]);
        let reversed = arr.reverse();

        assert_eq!(reversed.hyperplane(0), Some(&h2));
        assert_eq!(reversed.hyperplane(1), Some(&h1));
    }

    #[test]
    fn test_insert() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let h3 = Hyperplane::new(vec![Integer::from(1), Integer::from(1)], Integer::from(0));

        let arr = OrderedHyperplaneArrangementElement::new(vec![h1.clone(), h2.clone()]);
        let inserted = arr.insert(1, h3.clone());

        assert_eq!(inserted.num_hyperplanes(), 3);
        assert_eq!(inserted.hyperplane(0), Some(&h1));
        assert_eq!(inserted.hyperplane(1), Some(&h3));
        assert_eq!(inserted.hyperplane(2), Some(&h2));
    }

    #[test]
    fn test_append() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));

        let arr = OrderedHyperplaneArrangementElement::new(vec![h1.clone()]);
        let appended = arr.append(h2.clone());

        assert_eq!(appended.num_hyperplanes(), 2);
        assert_eq!(appended.hyperplane(0), Some(&h1));
        assert_eq!(appended.hyperplane(1), Some(&h2));
    }

    #[test]
    fn test_deletion() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let h2 = Hyperplane::new(vec![Integer::from(0), Integer::from(1)], Integer::from(0));
        let h3 = Hyperplane::new(vec![Integer::from(1), Integer::from(1)], Integer::from(0));

        let arr = OrderedHyperplaneArrangementElement::new(vec![h1.clone(), h2.clone(), h3.clone()]);
        let deleted = arr.deletion(1);

        assert_eq!(deleted.num_hyperplanes(), 2);
    }

    #[test]
    fn test_with_rationals() {
        let h1 = Hyperplane::new(
            vec![Rational::from(1), Rational::from(0)],
            Rational::from(0),
        );
        let h2 = Hyperplane::new(
            vec![Rational::from(0), Rational::from(1)],
            Rational::from(0),
        );
        let arr = OrderedHyperplaneArrangementElement::new(vec![h1, h2]);

        assert!(arr.is_central());
        assert_eq!(arr.rank(), 2);
    }

    #[test]
    fn test_parent() {
        let parent = OrderedHyperplaneArrangements::new(3);
        assert_eq!(parent.dimension(), 3);

        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0), Integer::from(0)], Integer::from(0));
        let arr = parent.arrangement(vec![h1]);
        assert_eq!(arr.ambient_dimension(), 3);
    }

    #[test]
    fn test_display() {
        let h1 = Hyperplane::new(vec![Integer::from(1), Integer::from(0)], Integer::from(0));
        let arr = OrderedHyperplaneArrangementElement::new(vec![h1]);

        let display = format!("{}", arr);
        assert!(display.contains("OrderedHyperplaneArrangement"));
    }
}
