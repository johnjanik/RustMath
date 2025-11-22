//! Affine crystals
//!
//! Affine crystals are crystals for affine Lie algebras. They have an additional
//! operator e_0 and f_0 corresponding to the affine simple root.
//!
//! For affine type A_n^(1), the affine crystal structure on tableaux involves
//! a "rotation" that treats 0 and n+1 as the same.

use crate::operators::Crystal;
use crate::weight::Weight;

/// An element in an affine crystal
///
/// For simplicity, we use a vector representation where the affine direction
/// is handled specially.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AffineElement {
    /// The underlying element (e.g., a tableau or weight)
    pub data: Vec<usize>,
    /// Level (affine weight component)
    pub level: i64,
}

impl AffineElement {
    /// Create a new affine element
    pub fn new(data: Vec<usize>, level: i64) -> Self {
        AffineElement { data, level }
    }
}

/// An affine crystal structure
///
/// This is a simplified implementation for type A_n^(1).
#[derive(Debug, Clone)]
pub struct AffineCrystal {
    /// Rank of the finite type (n for A_n^(1))
    pub rank: usize,
    /// Elements of the crystal
    pub elements: Vec<AffineElement>,
}

impl AffineCrystal {
    /// Create a new affine crystal
    pub fn new(rank: usize) -> Self {
        AffineCrystal {
            rank,
            elements: Vec::new(),
        }
    }

    /// Add an element to the crystal
    pub fn add_element(&mut self, element: AffineElement) {
        self.elements.push(element);
    }

    /// Generate the crystal from a highest weight element
    pub fn from_highest_weight(hw: AffineElement, rank: usize, max_depth: usize) -> Self {
        let mut crystal = AffineCrystal::new(rank);
        crystal.elements.push(hw.clone());

        let mut queue = vec![hw];
        let mut depth = 0;

        while !queue.is_empty() && depth < max_depth {
            let mut new_queue = Vec::new();
            for elem in queue {
                // Try all f_i operators (including f_0)
                for i in 0..=rank {
                    if let Some(new_elem) = crystal.apply_fi(&elem, i) {
                        if !crystal.elements.contains(&new_elem) {
                            crystal.elements.push(new_elem.clone());
                            new_queue.push(new_elem);
                        }
                    }
                }
            }
            queue = new_queue;
            depth += 1;
        }

        crystal
    }

    /// Apply f_i operator (helper for generation)
    fn apply_fi(&self, elem: &AffineElement, i: usize) -> Option<AffineElement> {
        if i > self.rank {
            return None;
        }

        if i == 0 {
            // Affine operator f_0
            self.apply_f0(elem)
        } else {
            // Finite operator f_i
            self.apply_fi_finite(elem, i)
        }
    }

    /// Apply affine operator f_0
    fn apply_f0(&self, elem: &AffineElement) -> Option<AffineElement> {
        // Simplified: decrease level and modify data
        let mut new_data = elem.data.clone();

        // Find the rightmost occurrence of 1 and change it to rank+1
        for i in (0..new_data.len()).rev() {
            if new_data[i] == 1 {
                new_data[i] = self.rank + 1;
                return Some(AffineElement::new(new_data, elem.level - 1));
            }
        }

        None
    }

    /// Apply finite operator f_i
    fn apply_fi_finite(&self, elem: &AffineElement, i: usize) -> Option<AffineElement> {
        if i == 0 || i > self.rank {
            return None;
        }

        let mut new_data = elem.data.clone();

        // Find the rightmost occurrence of i and change it to i+1
        for j in (0..new_data.len()).rev() {
            if new_data[j] == i {
                new_data[j] = i + 1;
                return Some(AffineElement::new(new_data, elem.level));
            }
        }

        None
    }

    /// Apply e_i operator
    fn apply_ei(&self, elem: &AffineElement, i: usize) -> Option<AffineElement> {
        if i > self.rank {
            return None;
        }

        if i == 0 {
            // Affine operator e_0
            self.apply_e0(elem)
        } else {
            // Finite operator e_i
            self.apply_ei_finite(elem, i)
        }
    }

    /// Apply affine operator e_0
    fn apply_e0(&self, elem: &AffineElement) -> Option<AffineElement> {
        let mut new_data = elem.data.clone();

        // Find the leftmost occurrence of rank+1 and change it to 1
        for i in 0..new_data.len() {
            if new_data[i] == self.rank + 1 {
                new_data[i] = 1;
                return Some(AffineElement::new(new_data, elem.level + 1));
            }
        }

        None
    }

    /// Apply finite operator e_i
    fn apply_ei_finite(&self, elem: &AffineElement, i: usize) -> Option<AffineElement> {
        if i == 0 || i > self.rank {
            return None;
        }

        let mut new_data = elem.data.clone();

        // Find the leftmost occurrence of i+1 and change it to i
        for j in 0..new_data.len() {
            if new_data[j] == i + 1 {
                new_data[j] = i;
                return Some(AffineElement::new(new_data, elem.level));
            }
        }

        None
    }

    /// Compute the energy function
    ///
    /// The energy function is a key invariant for affine crystals.
    pub fn energy(&self, elem: &AffineElement) -> i64 {
        // Simplified: just return the negative of the level
        -elem.level
    }

    /// Check if element is at level zero
    pub fn is_level_zero(&self, elem: &AffineElement) -> bool {
        elem.level == 0
    }

    /// Get classical component (forgetting affine structure)
    pub fn classical_component(&self, elem: &AffineElement) -> Vec<usize> {
        elem.data
            .iter()
            .map(|&x| if x > self.rank { x - self.rank } else { x })
            .collect()
    }
}

impl Crystal for AffineCrystal {
    type Element = AffineElement;

    fn weight(&self, b: &Self::Element) -> Weight {
        // Count occurrences of each letter
        let mut coords = vec![0i64; self.rank + 1];
        for &entry in &b.data {
            if entry > 0 && entry <= self.rank + 1 {
                coords[entry - 1] += 1;
            }
        }
        // Include level as the last coordinate
        coords[self.rank] = b.level;
        Weight::new(coords)
    }

    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        self.apply_ei(b, i)
    }

    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        self.apply_fi(b, i)
    }

    fn elements(&self) -> Vec<Self::Element> {
        self.elements.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_element() {
        let elem = AffineElement::new(vec![1, 2, 3], 0);
        assert_eq!(elem.level, 0);
        assert_eq!(elem.data, vec![1, 2, 3]);
    }

    #[test]
    fn test_affine_crystal() {
        let mut crystal = AffineCrystal::new(3);
        let elem = AffineElement::new(vec![1, 2, 2], 0);
        crystal.add_element(elem.clone());

        assert_eq!(crystal.elements.len(), 1);
        assert!(crystal.is_level_zero(&elem));
    }

    #[test]
    fn test_affine_operators() {
        let crystal = AffineCrystal::new(3);
        let elem = AffineElement::new(vec![1, 2, 3], 0);

        // Try f_1: should change first 1 to 2
        if let Some(new_elem) = crystal.f_i(&elem, 1) {
            assert_eq!(new_elem.data, vec![2, 2, 3]);
        }

        // Try f_0: should change 1 to 4 and decrease level
        if let Some(new_elem) = crystal.f_i(&elem, 0) {
            assert_eq!(new_elem.level, -1);
        }
    }

    #[test]
    fn test_energy_function() {
        let crystal = AffineCrystal::new(3);
        let elem1 = AffineElement::new(vec![1, 2], 0);
        let elem2 = AffineElement::new(vec![1, 2], -1);

        assert_eq!(crystal.energy(&elem1), 0);
        assert_eq!(crystal.energy(&elem2), 1);
    }
}
