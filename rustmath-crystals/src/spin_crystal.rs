//! Spin crystals for types B and D
//!
//! Spin crystals model the spin representations of the orthogonal and
//! special orthogonal Lie algebras (types B_n and D_n).
//!
//! Elements are represented as sequences of + and - signs.

use crate::operators::{Crystal, CrystalElement};
use crate::root_system::{RootSystem, RootSystemType};
use crate::weight::Weight;

/// A spin crystal element
///
/// Represented as a sequence of signs (+1 or -1).
/// For type B_n, we have n signs.
/// For type D_n, we have n signs with an even number of minuses.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpinElement {
    /// Signs: +1 or -1
    pub signs: Vec<i8>,
}

impl SpinElement {
    /// Create a new spin element
    pub fn new(signs: Vec<i8>) -> Self {
        // Validate signs are +1 or -1
        for &sign in &signs {
            assert!(sign == 1 || sign == -1, "Signs must be +1 or -1");
        }
        SpinElement { signs }
    }

    /// Create the highest weight element (all +1)
    pub fn highest_weight(n: usize) -> Self {
        SpinElement {
            signs: vec![1; n],
        }
    }

    /// Count the number of minus signs
    pub fn minus_count(&self) -> usize {
        self.signs.iter().filter(|&&s| s == -1).count()
    }

    /// Check if valid for type D (even number of minuses)
    pub fn is_valid_type_d(&self) -> bool {
        self.minus_count() % 2 == 0
    }

    /// Flip the sign at position i
    pub fn flip(&self, i: usize) -> SpinElement {
        let mut new_signs = self.signs.clone();
        new_signs[i] = -new_signs[i];
        SpinElement::new(new_signs)
    }

    /// Compute weight in the ε basis
    /// Weight = (ε_1, ε_2, ..., ε_n) where ε_i = sign[i]/2
    pub fn compute_weight(&self) -> Weight {
        let coords: Vec<i64> = self.signs.iter().map(|&s| s as i64).collect();
        Weight::new(coords)
    }
}

/// Spin crystal for type B_n
///
/// The spin representation of SO(2n+1).
#[derive(Debug, Clone)]
pub struct SpinCrystalB {
    /// Rank n
    pub rank: usize,
    /// Root system
    pub root_system: RootSystem,
}

impl SpinCrystalB {
    /// Create a new type B spin crystal
    pub fn new(n: usize) -> Self {
        SpinCrystalB {
            rank: n,
            root_system: RootSystem::new(RootSystemType::B(n)),
        }
    }

    /// Generate all spin elements
    pub fn all_elements(&self) -> Vec<SpinElement> {
        let mut elements = Vec::new();
        self.generate_recursive(&mut elements, Vec::new(), self.rank);
        elements
    }

    fn generate_recursive(&self, result: &mut Vec<SpinElement>, current: Vec<i8>, remaining: usize) {
        if remaining == 0 {
            result.push(SpinElement::new(current));
            return;
        }

        let mut with_plus = current.clone();
        with_plus.push(1);
        self.generate_recursive(result, with_plus, remaining - 1);

        let mut with_minus = current;
        with_minus.push(-1);
        self.generate_recursive(result, with_minus, remaining - 1);
    }
}

impl Crystal for SpinCrystalB {
    type Element = SpinElement;

    fn weight(&self, b: &Self::Element) -> Weight {
        b.compute_weight()
    }

    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i == 0 || i > self.rank {
            return None;
        }

        let idx = i - 1; // Convert to 0-indexed

        if idx < self.rank - 1 {
            // Standard case: swap if signs[i-1] = -, signs[i] = +
            if b.signs[idx] == -1 && b.signs[idx + 1] == 1 {
                let mut new_signs = b.signs.clone();
                new_signs[idx] = 1;
                new_signs[idx + 1] = -1;
                return Some(SpinElement::new(new_signs));
            }
        } else if idx == self.rank - 1 {
            // Last position for type B: flip if it's minus
            if b.signs[idx] == -1 {
                return Some(b.flip(idx));
            }
        }

        None
    }

    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i == 0 || i > self.rank {
            return None;
        }

        let idx = i - 1; // Convert to 0-indexed

        if idx < self.rank - 1 {
            // Standard case: swap if signs[i-1] = +, signs[i] = -
            if b.signs[idx] == 1 && b.signs[idx + 1] == -1 {
                let mut new_signs = b.signs.clone();
                new_signs[idx] = -1;
                new_signs[idx + 1] = 1;
                return Some(SpinElement::new(new_signs));
            }
        } else if idx == self.rank - 1 {
            // Last position for type B: flip if it's plus
            if b.signs[idx] == 1 {
                return Some(b.flip(idx));
            }
        }

        None
    }

    fn elements(&self) -> Vec<Self::Element> {
        self.all_elements()
    }
}

/// Spin crystal for type D_n
///
/// The spin representation of SO(2n). Elements must have an even number of minus signs.
#[derive(Debug, Clone)]
pub struct SpinCrystalD {
    /// Rank n
    pub rank: usize,
    /// Root system
    pub root_system: RootSystem,
    /// Which spin representation (+: even minuses, -: odd minuses would be for other component)
    pub parity: bool,
}

impl SpinCrystalD {
    /// Create a new type D spin crystal
    /// parity = true for even number of minuses (standard spin rep)
    pub fn new(n: usize, parity: bool) -> Self {
        SpinCrystalD {
            rank: n,
            root_system: RootSystem::new(RootSystemType::D(n)),
            parity,
        }
    }

    /// Generate all valid spin elements
    pub fn all_elements(&self) -> Vec<SpinElement> {
        let mut all = Vec::new();
        self.generate_recursive(&mut all, Vec::new(), self.rank);

        // Filter by parity
        all.into_iter()
            .filter(|elem| {
                let even_minuses = elem.minus_count() % 2 == 0;
                even_minuses == self.parity
            })
            .collect()
    }

    fn generate_recursive(&self, result: &mut Vec<SpinElement>, current: Vec<i8>, remaining: usize) {
        if remaining == 0 {
            let elem = SpinElement::new(current);
            let even_minuses = elem.minus_count() % 2 == 0;
            if even_minuses == self.parity {
                result.push(elem);
            }
            return;
        }

        let mut with_plus = current.clone();
        with_plus.push(1);
        self.generate_recursive(result, with_plus, remaining - 1);

        let mut with_minus = current;
        with_minus.push(-1);
        self.generate_recursive(result, with_minus, remaining - 1);
    }
}

impl Crystal for SpinCrystalD {
    type Element = SpinElement;

    fn weight(&self, b: &Self::Element) -> Weight {
        b.compute_weight()
    }

    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i == 0 || i > self.rank {
            return None;
        }

        let idx = i - 1; // Convert to 0-indexed

        if idx < self.rank - 2 {
            // Standard case: swap if signs[i-1] = -, signs[i] = +
            if b.signs[idx] == -1 && b.signs[idx + 1] == 1 {
                let mut new_signs = b.signs.clone();
                new_signs[idx] = 1;
                new_signs[idx + 1] = -1;
                let new_elem = SpinElement::new(new_signs);
                // Check parity is preserved
                if (new_elem.minus_count() % 2 == 0) == self.parity {
                    return Some(new_elem);
                }
            }
        } else if idx == self.rank - 2 {
            // Penultimate position: can swap with either last or second-to-last
            if b.signs[idx] == -1 && (b.signs[idx + 1] == 1) {
                let mut new_signs = b.signs.clone();
                new_signs[idx] = 1;
                new_signs[idx + 1] = -1;
                let new_elem = SpinElement::new(new_signs);
                if (new_elem.minus_count() % 2 == 0) == self.parity {
                    return Some(new_elem);
                }
            }
        }

        None
    }

    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i == 0 || i > self.rank {
            return None;
        }

        let idx = i - 1; // Convert to 0-indexed

        if idx < self.rank - 2 {
            // Standard case: swap if signs[i-1] = +, signs[i] = -
            if b.signs[idx] == 1 && b.signs[idx + 1] == -1 {
                let mut new_signs = b.signs.clone();
                new_signs[idx] = -1;
                new_signs[idx + 1] = 1;
                let new_elem = SpinElement::new(new_signs);
                // Check parity is preserved
                if (new_elem.minus_count() % 2 == 0) == self.parity {
                    return Some(new_elem);
                }
            }
        } else if idx == self.rank - 2 {
            // Penultimate position
            if b.signs[idx] == 1 && b.signs[idx + 1] == -1 {
                let mut new_signs = b.signs.clone();
                new_signs[idx] = -1;
                new_signs[idx + 1] = 1;
                let new_elem = SpinElement::new(new_signs);
                if (new_elem.minus_count() % 2 == 0) == self.parity {
                    return Some(new_elem);
                }
            }
        }

        None
    }

    fn elements(&self) -> Vec<Self::Element> {
        self.all_elements()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spin_element() {
        let elem = SpinElement::new(vec![1, -1, 1, -1]);
        assert_eq!(elem.signs, vec![1, -1, 1, -1]);
        assert_eq!(elem.minus_count(), 2);
        assert!(elem.is_valid_type_d());
    }

    #[test]
    fn test_spin_element_flip() {
        let elem = SpinElement::new(vec![1, 1, 1]);
        let flipped = elem.flip(1);
        assert_eq!(flipped.signs, vec![1, -1, 1]);
    }

    #[test]
    fn test_spin_crystal_b() {
        let crystal = SpinCrystalB::new(2);
        let elements = crystal.all_elements();

        // For B_2, we should have 2^2 = 4 elements
        assert_eq!(elements.len(), 4);

        // Test crystal operators
        let hw = SpinElement::highest_weight(2);
        assert!(crystal.is_highest_weight(&hw, 2));
    }

    #[test]
    fn test_spin_crystal_d() {
        let crystal = SpinCrystalD::new(3, true);
        let elements = crystal.all_elements();

        // For D_3 with even parity, we have elements with 0 or 2 minuses
        // 0 minuses: (+++): 1 element
        // 2 minuses: 3 choose 2 = 3 elements
        // Total: 4 elements
        assert_eq!(elements.len(), 4);

        // All elements should have even number of minuses
        for elem in &elements {
            assert!(elem.is_valid_type_d());
        }
    }

    #[test]
    fn test_spin_b_operators() {
        let crystal = SpinCrystalB::new(2);
        let elem = SpinElement::new(vec![1, 1]);

        // Apply f_2: should flip last sign
        let f2 = crystal.f_i(&elem, 2);
        assert!(f2.is_some());
        assert_eq!(f2.unwrap().signs, vec![1, -1]);

        // Apply f_1: should swap first two
        let f1 = crystal.f_i(&elem, 1);
        assert!(f1.is_none()); // Can't swap (1,1)
    }

    #[test]
    fn test_spin_weight() {
        let elem = SpinElement::new(vec![1, -1, 1]);
        let weight = elem.compute_weight();
        assert_eq!(weight.coords, vec![1, -1, 1]);
    }
}
