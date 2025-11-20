//! Elementary crystals
//!
//! This module provides fundamental crystal structures including:
//! - B(∞): The crystal of the negative part of the quantum group
//! - T_λ: One-dimensional crystals
//! - R_λ: Seminormal crystals
//! - Elementary crystals for each Dynkin node

use crate::operators::{Crystal, CrystalElement};
use crate::root_system::RootSystem;
use crate::weight::Weight;
use std::collections::BTreeMap;

/// Element of B(∞)
///
/// Represented as a monomial in Kashiwara operators
/// B(∞) = {f_{i_1}^{a_1} ⋯ f_{i_k}^{a_k} · 1}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InfinityCrystalElement {
    /// Sequence of (index i, power a_i) for f_i^{a_i}
    pub monomials: Vec<(usize, usize)>,
}

impl InfinityCrystalElement {
    /// Create the identity element (1)
    pub fn identity() -> Self {
        InfinityCrystalElement {
            monomials: Vec::new(),
        }
    }

    /// Create from a sequence of operators
    pub fn from_operators(ops: Vec<(usize, usize)>) -> Self {
        InfinityCrystalElement { monomials: ops }
    }

    /// Apply f_i to the element
    pub fn apply_fi(&self, i: usize) -> Self {
        let mut new_mons = self.monomials.clone();

        // Check if we can append f_i to the end
        if let Some(last) = new_mons.last_mut() {
            if last.0 == i {
                last.1 += 1;
                return InfinityCrystalElement { monomials: new_mons };
            }
        }

        new_mons.push((i, 1));
        InfinityCrystalElement { monomials: new_mons }
    }

    /// Try to apply e_i
    pub fn apply_ei(&self, i: usize) -> Option<Self> {
        if self.monomials.is_empty() {
            return None;
        }

        let mut new_mons = self.monomials.clone();

        // Check if the last monomial has index i
        if let Some(last) = new_mons.last_mut() {
            if last.0 == i {
                if last.1 > 1 {
                    last.1 -= 1;
                } else {
                    new_mons.pop();
                }
                return Some(InfinityCrystalElement { monomials: new_mons });
            }
        }

        None
    }

    /// Compute the weight
    pub fn compute_weight(&self, root_system: &RootSystem) -> Weight {
        let mut w = Weight::zero(root_system.rank);

        for &(i, power) in &self.monomials {
            let alpha_i = root_system.simple_root(i);
            for _ in 0..power {
                w = &w - &alpha_i;
            }
        }

        w
    }
}

/// The crystal B(∞)
///
/// This is the crystal of the negative part of the quantum group.
/// It has a unique highest weight element 1.
#[derive(Debug, Clone)]
pub struct InfinityCrystal {
    /// Root system
    pub root_system: RootSystem,
    /// Maximum depth for generation
    pub max_depth: usize,
    /// Generated elements
    elements_cache: Vec<InfinityCrystalElement>,
}

impl InfinityCrystal {
    /// Create a new B(∞) crystal
    pub fn new(root_system: RootSystem, max_depth: usize) -> Self {
        InfinityCrystal {
            root_system,
            max_depth,
            elements_cache: Vec::new(),
        }
    }

    /// Generate elements up to max_depth
    pub fn generate(&mut self) {
        self.elements_cache.clear();
        let identity = InfinityCrystalElement::identity();
        self.elements_cache.push(identity.clone());

        let mut queue = vec![identity];
        let mut depth = 0;

        while !queue.is_empty() && depth < self.max_depth {
            let mut new_queue = Vec::new();

            for elem in queue {
                for i in 0..self.root_system.rank {
                    let new_elem = elem.apply_fi(i);
                    if !self.elements_cache.contains(&new_elem) {
                        self.elements_cache.push(new_elem.clone());
                        new_queue.push(new_elem);
                    }
                }
            }

            queue = new_queue;
            depth += 1;
        }
    }
}

impl Crystal for InfinityCrystal {
    type Element = InfinityCrystalElement;

    fn weight(&self, b: &Self::Element) -> Weight {
        b.compute_weight(&self.root_system)
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
        Some(b.apply_fi(i))
    }

    fn elements(&self) -> Vec<Self::Element> {
        self.elements_cache.clone()
    }
}

/// One-dimensional crystal T_λ
///
/// This is a trivial crystal with a single element of weight λ.
/// All crystal operators act as zero: e_i = f_i = 0.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OneDimElement {
    /// The weight
    pub weight: Weight,
}

impl OneDimElement {
    /// Create a new one-dimensional element
    pub fn new(weight: Weight) -> Self {
        OneDimElement { weight }
    }
}

/// One-dimensional crystal
#[derive(Debug, Clone)]
pub struct OneDimCrystal {
    /// The weight λ
    pub weight: Weight,
}

impl OneDimCrystal {
    /// Create T_λ
    pub fn new(weight: Weight) -> Self {
        OneDimCrystal { weight }
    }
}

impl Crystal for OneDimCrystal {
    type Element = OneDimElement;

    fn weight(&self, _b: &Self::Element) -> Weight {
        self.weight.clone()
    }

    fn e_i(&self, _b: &Self::Element, _i: usize) -> Option<Self::Element> {
        None // All operators are zero
    }

    fn f_i(&self, _b: &Self::Element, _i: usize) -> Option<Self::Element> {
        None // All operators are zero
    }

    fn elements(&self) -> Vec<Self::Element> {
        vec![OneDimElement::new(self.weight.clone())]
    }
}

/// Elementary crystal B(i)
///
/// The elementary crystal B(i) is the crystal with elements {f_i^k · 1 | k >= 0}.
/// It models the sl_2 representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ElementaryElement {
    /// Index i
    pub i: usize,
    /// Power k (element is f_i^k · 1)
    pub k: usize,
}

impl ElementaryElement {
    /// Create a new elementary element
    pub fn new(i: usize, k: usize) -> Self {
        ElementaryElement { i, k }
    }

    /// Apply f_i
    pub fn apply_fi(&self, j: usize) -> Option<Self> {
        if j == self.i {
            Some(ElementaryElement::new(self.i, self.k + 1))
        } else {
            None
        }
    }

    /// Apply e_i
    pub fn apply_ei(&self, j: usize) -> Option<Self> {
        if j == self.i && self.k > 0 {
            Some(ElementaryElement::new(self.i, self.k - 1))
        } else {
            None
        }
    }

    /// Compute weight
    pub fn compute_weight(&self, root_system: &RootSystem) -> Weight {
        let alpha_i = root_system.simple_root(self.i);
        &Weight::zero(root_system.rank) - &(&alpha_i * self.k as i64)
    }
}

/// Elementary crystal B(i)
#[derive(Debug, Clone)]
pub struct ElementaryCrystal {
    /// The index i
    pub i: usize,
    /// Root system
    pub root_system: RootSystem,
    /// Maximum power
    pub max_power: usize,
}

impl ElementaryCrystal {
    /// Create B(i)
    pub fn new(i: usize, root_system: RootSystem, max_power: usize) -> Self {
        ElementaryCrystal {
            i,
            root_system,
            max_power,
        }
    }

    /// Get all elements
    pub fn all_elements(&self) -> Vec<ElementaryElement> {
        (0..=self.max_power)
            .map(|k| ElementaryElement::new(self.i, k))
            .collect()
    }
}

impl Crystal for ElementaryCrystal {
    type Element = ElementaryElement;

    fn weight(&self, b: &Self::Element) -> Weight {
        b.compute_weight(&self.root_system)
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
        if i == self.i && b.k < self.max_power {
            b.apply_fi(i)
        } else {
            None
        }
    }

    fn elements(&self) -> Vec<Self::Element> {
        self.all_elements()
    }
}

/// Seminormal crystal R_λ
///
/// This is a generalization of T_λ where some operators may act non-trivially.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SeminormalElement {
    /// Base weight
    pub base_weight: Weight,
    /// Additional data (for specific types)
    pub data: BTreeMap<usize, i64>,
}

impl SeminormalElement {
    /// Create a new seminormal element
    pub fn new(base_weight: Weight) -> Self {
        SeminormalElement {
            base_weight,
            data: BTreeMap::new(),
        }
    }

    /// Set data for index i
    pub fn set_data(&mut self, i: usize, value: i64) {
        self.data.insert(i, value);
    }
}

/// Seminormal crystal
#[derive(Debug, Clone)]
pub struct SeminormalCrystal {
    /// Base weight
    pub base_weight: Weight,
    /// Root system
    pub root_system: RootSystem,
}

impl SeminormalCrystal {
    /// Create R_λ
    pub fn new(base_weight: Weight, root_system: RootSystem) -> Self {
        SeminormalCrystal {
            base_weight,
            root_system,
        }
    }
}

impl Crystal for SeminormalCrystal {
    type Element = SeminormalElement;

    fn weight(&self, b: &Self::Element) -> Weight {
        b.base_weight.clone()
    }

    fn e_i(&self, _b: &Self::Element, _i: usize) -> Option<Self::Element> {
        // Implementation depends on the specific seminormal crystal
        None
    }

    fn f_i(&self, _b: &Self::Element, _i: usize) -> Option<Self::Element> {
        None
    }

    fn elements(&self) -> Vec<Self::Element> {
        vec![SeminormalElement::new(self.base_weight.clone())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::root_system::{RootSystem, RootSystemType};

    #[test]
    fn test_infinity_crystal_element() {
        let elem = InfinityCrystalElement::identity();
        assert!(elem.monomials.is_empty());

        let f1 = elem.apply_fi(1);
        assert_eq!(f1.monomials, vec![(1, 1)]);

        let f1_f1 = f1.apply_fi(1);
        assert_eq!(f1_f1.monomials, vec![(1, 2)]);

        let e1 = f1_f1.apply_ei(1);
        assert_eq!(e1, Some(f1));
    }

    #[test]
    fn test_infinity_crystal() {
        let root_system = RootSystem::new(RootSystemType::A(2));
        let mut crystal = InfinityCrystal::new(root_system, 3);

        crystal.generate();
        assert!(crystal.elements_cache.len() > 0);

        let id = InfinityCrystalElement::identity();
        assert!(crystal.is_highest_weight(&id, 2));
    }

    #[test]
    fn test_one_dim_crystal() {
        let weight = Weight::new(vec![1, 2]);
        let crystal = OneDimCrystal::new(weight.clone());

        let elem = OneDimElement::new(weight.clone());
        assert_eq!(crystal.weight(&elem), weight);

        // All operators should be zero
        assert!(crystal.e_i(&elem, 0).is_none());
        assert!(crystal.f_i(&elem, 0).is_none());
    }

    #[test]
    fn test_elementary_crystal() {
        let root_system = RootSystem::new(RootSystemType::A(2));
        let crystal = ElementaryCrystal::new(0, root_system, 5);

        let hw = ElementaryElement::new(0, 0);
        assert!(crystal.is_highest_weight(&hw, 2));

        let f0 = crystal.f_i(&hw, 0);
        assert!(f0.is_some());
        assert_eq!(f0.unwrap().k, 1);
    }

    #[test]
    fn test_elementary_element_weight() {
        let root_system = RootSystem::new(RootSystemType::A(2));
        let elem = ElementaryElement::new(0, 2);

        let weight = elem.compute_weight(&root_system);
        // Should be -2 * alpha_0
        assert_eq!(weight.coords[0], -4); // -2 * 2 (from Cartan matrix)
    }

    #[test]
    fn test_seminormal_crystal() {
        let weight = Weight::new(vec![1, 0]);
        let root_system = RootSystem::new(RootSystemType::A(2));
        let crystal = SeminormalCrystal::new(weight.clone(), root_system);

        let elem = SeminormalElement::new(weight.clone());
        assert_eq!(crystal.weight(&elem), weight);
    }
}
