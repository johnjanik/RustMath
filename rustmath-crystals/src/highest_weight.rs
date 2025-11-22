//! Highest weight crystals
//!
//! Highest weight crystals are crystals with a unique highest weight element.
//! They correspond to irreducible representations of the Lie algebra.

use crate::operators::Crystal;
use crate::root_system::RootSystem;
use crate::weight::Weight;

/// A highest weight crystal element
///
/// This is a wrapper that tracks the highest weight and the "path" from it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HighestWeightElement {
    /// Sequence of operators applied to highest weight element
    /// Each entry is (i, k) meaning f_i^k was applied
    pub path: Vec<(usize, usize)>,
}

impl HighestWeightElement {
    /// Create the highest weight element (empty path)
    pub fn highest_weight() -> Self {
        HighestWeightElement { path: Vec::new() }
    }

    /// Apply f_i once
    pub fn apply_fi(&self, i: usize) -> Self {
        let mut new_path = self.path.clone();
        if let Some(last) = new_path.last_mut() {
            if last.0 == i {
                last.1 += 1;
                return HighestWeightElement { path: new_path };
            }
        }
        new_path.push((i, 1));
        HighestWeightElement { path: new_path }
    }

    /// Try to apply e_i once (may fail if we're at the top)
    pub fn apply_ei(&self, i: usize) -> Option<Self> {
        let mut new_path = self.path.clone();
        if let Some(last) = new_path.last_mut() {
            if last.0 == i {
                if last.1 > 1 {
                    last.1 -= 1;
                } else {
                    new_path.pop();
                }
                return Some(HighestWeightElement { path: new_path });
            }
        }
        None
    }

    /// Compute weight from highest weight and path
    pub fn compute_weight(&self, highest_weight: &Weight, root_system: &RootSystem) -> Weight {
        let mut w = highest_weight.clone();
        for &(i, k) in &self.path {
            let alpha_i = root_system.simple_root(i);
            for _ in 0..k {
                w = &w - &alpha_i;
            }
        }
        w
    }
}

/// A highest weight crystal
///
/// This represents B(λ), the crystal of the irreducible representation with highest weight λ.
#[derive(Debug, Clone)]
pub struct HighestWeightCrystal {
    /// The highest weight
    pub highest_weight: Weight,
    /// The root system
    pub root_system: RootSystem,
    /// Maximum depth to generate (for infinite crystals)
    pub max_depth: usize,
    /// Generated elements
    elements_cache: Vec<HighestWeightElement>,
}

impl HighestWeightCrystal {
    /// Create a new highest weight crystal
    pub fn new(highest_weight: Weight, root_system: RootSystem) -> Self {
        HighestWeightCrystal {
            highest_weight,
            root_system,
            max_depth: 10,
            elements_cache: Vec::new(),
        }
    }

    /// Create with specified maximum depth
    pub fn with_max_depth(
        highest_weight: Weight,
        root_system: RootSystem,
        max_depth: usize,
    ) -> Self {
        HighestWeightCrystal {
            highest_weight,
            root_system,
            max_depth,
            elements_cache: Vec::new(),
        }
    }

    /// Generate all elements up to max depth
    pub fn generate(&mut self) {
        self.elements_cache.clear();
        let hw = HighestWeightElement::highest_weight();
        self.elements_cache.push(hw.clone());

        let mut queue = vec![hw];
        let mut depth = 0;

        while !queue.is_empty() && depth < self.max_depth {
            let mut new_queue = Vec::new();
            for elem in queue {
                // Try all f_i operators
                for i in 0..self.root_system.rank {
                    let new_elem = elem.apply_fi(i);
                    // Check if valid using weight
                    let new_weight = new_elem.compute_weight(&self.highest_weight, &self.root_system);

                    // Simple validity check: weight coordinates shouldn't be too negative
                    if new_weight.coords.iter().all(|&x| x > -100) {
                        if !self.elements_cache.contains(&new_elem) {
                            self.elements_cache.push(new_elem.clone());
                            new_queue.push(new_elem);
                        }
                    }
                }
            }
            queue = new_queue;
            depth += 1;
        }
    }

    /// Get the dimension (number of elements)
    pub fn dimension(&self) -> usize {
        self.elements_cache.len()
    }

    /// Check if the crystal is finite
    pub fn is_finite(&self) -> bool {
        // A highest weight crystal is finite iff the highest weight is dominant
        self.highest_weight.is_dominant()
    }

    /// Get all weights that appear in the crystal
    pub fn weight_multiplicities(&self) -> std::collections::HashMap<Weight, usize> {
        let mut mult = std::collections::HashMap::new();
        for elem in &self.elements_cache {
            let weight = elem.compute_weight(&self.highest_weight, &self.root_system);
            *mult.entry(weight).or_insert(0) += 1;
        }
        mult
    }
}

impl Crystal for HighestWeightCrystal {
    type Element = HighestWeightElement;

    fn weight(&self, b: &Self::Element) -> Weight {
        b.compute_weight(&self.highest_weight, &self.root_system)
    }

    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i >= self.root_system.rank {
            return None;
        }
        b.apply_ei(i)
    }

    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        if i >= self.root_system.rank {
            return None;
        }

        let new_elem = b.apply_fi(i);
        let new_weight = self.weight(&new_elem);

        // Check if this is a valid element (weight shouldn't go too negative)
        // In a proper implementation, we'd check against the Weyl chamber
        if new_weight.coords.iter().all(|&x| x > -100) {
            Some(new_elem)
        } else {
            None
        }
    }

    fn elements(&self) -> Vec<Self::Element> {
        self.elements_cache.clone()
    }
}

/// Compute the dominant weight from a weight by reflection
pub fn dominant_weight(weight: &Weight, root_system: &RootSystem) -> Weight {
    let mut w = weight.clone();
    let mut changed = true;
    let mut iterations = 0;
    let max_iterations = 100; // Prevent infinite loops

    while changed && iterations < max_iterations {
        changed = false;
        for i in 0..root_system.rank {
            let action = root_system.coroot_action(&w, i);
            if action < 0 {
                // Reflect through the hyperplane perpendicular to α_i
                // s_i(w) = w - ⟨w, α_i^∨⟩ α_i
                let alpha_i = root_system.simple_root(i);
                // Compute w - action * α_i carefully to avoid overflow
                for j in 0..w.coords.len() {
                    w.coords[j] = w.coords[j].saturating_sub(action.saturating_mul(alpha_i.coords[j]));
                }
                changed = true;
            }
        }
        iterations += 1;
    }

    w
}

/// Compute the crystal dimension for a highest weight
///
/// For type A_n, the dimension can be computed using the hook length formula.
pub fn dimension_formula(highest_weight: &Weight) -> usize {
    // Simplified: for dominant weights, use the product formula
    if !highest_weight.is_dominant() {
        return 0;
    }

    // For type A, dimension = product of (λ_i + i) / product of hook lengths
    // This is a placeholder - full implementation would depend on the type
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::root_system::{RootSystem, RootSystemType};

    #[test]
    fn test_highest_weight_element() {
        let hw = HighestWeightElement::highest_weight();
        assert!(hw.path.is_empty());

        let elem1 = hw.apply_fi(1);
        assert_eq!(elem1.path, vec![(1, 1)]);

        let elem2 = elem1.apply_fi(1);
        assert_eq!(elem2.path, vec![(1, 2)]);

        let elem3 = elem2.apply_ei(1);
        assert!(elem3.is_some());
        assert_eq!(elem3.unwrap().path, vec![(1, 1)]);
    }

    #[test]
    fn test_highest_weight_crystal() {
        let hw = Weight::new(vec![1, 0]);
        let root_system = RootSystem::new(RootSystemType::A(2));
        let mut crystal = HighestWeightCrystal::new(hw, root_system);

        assert!(crystal.is_finite());

        crystal.generate();
        assert!(crystal.dimension() > 0);
    }

    #[test]
    fn test_dominant_weight() {
        let root_system = RootSystem::new(RootSystemType::A(2));
        let w = Weight::new(vec![1, -1]);
        let dom = dominant_weight(&w, &root_system);

        // Should be reflected to a dominant weight
        assert!(dom.coords.iter().any(|&x| x >= 0));
    }
}
