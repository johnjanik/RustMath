//! Crystal operators and the Crystal trait
//!
//! This module defines the core Crystal trait that all crystal implementations must satisfy.
//! Crystal operators e_i and f_i are the raising and lowering operators.

use crate::weight::Weight;
use std::fmt::Debug;
use std::hash::Hash;

/// A crystal element
///
/// This is a marker trait for types that can be elements of a crystal.
pub trait CrystalElement: Clone + Debug + PartialEq + Eq + Hash {}

impl<T: Clone + Debug + PartialEq + Eq + Hash> CrystalElement for T {}

/// The Crystal trait
///
/// A crystal is a set B together with maps:
/// - wt: B → P (weight map to weight lattice)
/// - e_i, f_i: B → B ∪ {0} (raising and lowering operators for i = 1, ..., n)
/// - ε_i, φ_i: B → Z ∪ {-∞} (string functions)
///
/// These maps must satisfy the crystal axioms:
/// 1. φ_i(b) = ε_i(b) + ⟨wt(b), α_i^∨⟩
/// 2. wt(e_i b) = wt(b) + α_i if e_i b ≠ 0
/// 3. wt(f_i b) = wt(b) - α_i if f_i b ≠ 0
/// 4. ε_i(e_i b) = ε_i(b) - 1, φ_i(e_i b) = φ_i(b) + 1 if e_i b ≠ 0
/// 5. ε_i(f_i b) = ε_i(b) + 1, φ_i(f_i b) = φ_i(b) - 1 if f_i b ≠ 0
pub trait Crystal {
    /// The type of elements in this crystal
    type Element: CrystalElement;

    /// Weight of a crystal element
    fn weight(&self, b: &Self::Element) -> Weight;

    /// Raising operator e_i
    /// Returns None if e_i(b) = 0
    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element>;

    /// Lowering operator f_i
    /// Returns None if f_i(b) = 0
    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element>;

    /// String function ε_i(b) = max{k : e_i^k(b) ≠ 0}
    fn epsilon_i(&self, b: &Self::Element, i: usize) -> i64 {
        let mut count = 0;
        let mut current = b.clone();
        while let Some(next) = self.e_i(&current, i) {
            count += 1;
            current = next;
        }
        count
    }

    /// String function φ_i(b) = max{k : f_i^k(b) ≠ 0}
    fn phi_i(&self, b: &Self::Element, i: usize) -> i64 {
        let mut count = 0;
        let mut current = b.clone();
        while let Some(next) = self.f_i(&current, i) {
            count += 1;
            current = next;
        }
        count
    }

    /// Check if element is a highest weight element
    /// (i.e., e_i(b) = 0 for all i)
    fn is_highest_weight(&self, b: &Self::Element, rank: usize) -> bool {
        (0..rank).all(|i| self.e_i(b, i).is_none())
    }

    /// Check if element is a lowest weight element
    /// (i.e., f_i(b) = 0 for all i)
    fn is_lowest_weight(&self, b: &Self::Element, rank: usize) -> bool {
        (0..rank).all(|i| self.f_i(b, i).is_none())
    }

    /// Get all elements of the crystal (for finite crystals)
    fn elements(&self) -> Vec<Self::Element>;

    /// Get the highest weight elements
    fn highest_weight_elements(&self, rank: usize) -> Vec<Self::Element> {
        self.elements()
            .into_iter()
            .filter(|b| self.is_highest_weight(b, rank))
            .collect()
    }

    /// Apply e_i multiple times
    fn e_i_power(&self, b: &Self::Element, i: usize, k: usize) -> Option<Self::Element> {
        let mut current = b.clone();
        for _ in 0..k {
            current = self.e_i(&current, i)?;
        }
        Some(current)
    }

    /// Apply f_i multiple times
    fn f_i_power(&self, b: &Self::Element, i: usize, k: usize) -> Option<Self::Element> {
        let mut current = b.clone();
        for _ in 0..k {
            current = self.f_i(&current, i)?;
        }
        Some(current)
    }

    /// Check if two elements are in the same connected component
    fn same_component(&self, b1: &Self::Element, b2: &Self::Element, rank: usize) -> bool {
        // Two elements are in the same component if they can be reached from each other
        // by applying crystal operators
        // For now, we just check if they have the same highest weight element
        let hw1 = self.highest_weight_in_component(b1, rank);
        let hw2 = self.highest_weight_in_component(b2, rank);
        hw1 == hw2
    }

    /// Find the highest weight element in the same component as b
    fn highest_weight_in_component(&self, b: &Self::Element, rank: usize) -> Self::Element {
        let mut current = b.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..rank {
                if let Some(next) = self.e_i(&current, i) {
                    current = next;
                    changed = true;
                    break;
                }
            }
        }
        current
    }
}

/// A simple crystal for testing and examples
#[derive(Debug, Clone)]
pub struct SimpleCrystal {
    /// The rank of the root system
    pub rank: usize,
    /// Elements of the crystal (represented as weights)
    pub elements: Vec<Weight>,
}

impl Crystal for SimpleCrystal {
    type Element = Weight;

    fn weight(&self, b: &Self::Element) -> Weight {
        b.clone()
    }

    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        // Simple implementation: e_i adds the i-th fundamental weight
        let mut new_weight = b.clone();
        if i < new_weight.coords.len() {
            new_weight.coords[i] += 1;
            Some(new_weight)
        } else {
            None
        }
    }

    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        // Simple implementation: f_i subtracts the i-th fundamental weight
        let mut new_weight = b.clone();
        if i < new_weight.coords.len() && new_weight.coords[i] > 0 {
            new_weight.coords[i] -= 1;
            Some(new_weight)
        } else {
            None
        }
    }

    fn elements(&self) -> Vec<Self::Element> {
        self.elements.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_crystal() {
        let crystal = SimpleCrystal {
            rank: 2,
            elements: vec![
                Weight::new(vec![0, 0]),
                Weight::new(vec![1, 0]),
                Weight::new(vec![0, 1]),
            ],
        };

        let b = Weight::new(vec![0, 0]);

        // Apply f_1
        let b1 = crystal.f_i(&b, 0);
        assert!(b1.is_none()); // Can't go lower

        // Apply e_1
        let b2 = crystal.e_i(&b, 0);
        assert!(b2.is_some());
        assert_eq!(b2.unwrap().coords, vec![1, 0]);
    }

    #[test]
    fn test_highest_weight() {
        let crystal = SimpleCrystal {
            rank: 2,
            elements: vec![Weight::new(vec![0, 0]), Weight::new(vec![1, 0])],
        };

        let hw = Weight::new(vec![0, 0]);
        assert!(!crystal.is_highest_weight(&hw, 2)); // Can apply e_i

        let not_hw = Weight::new(vec![1, 0]);
        assert!(!crystal.is_highest_weight(&not_hw, 2));
    }
}
