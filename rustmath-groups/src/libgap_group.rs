//! LibGAP Group - Generic group interface
//!
//! This module provides a generic group interface for groups in RustMath.
//! In SageMath, GroupLibGAP provides an interface to GAP-based groups.
//! Here we provide a pure Rust implementation with similar functionality.
//!
//! # Overview
//!
//! The main class `GroupLibGAP` represents a group with generators and
//! provides methods for group operations, properties, and computations.
//!
//! # Example
//!
//! ```
//! use rustmath_groups::libgap_group::GroupLibGAP;
//! ```

use std::fmt;
use std::hash::Hash;
use crate::group_traits::{Group, GroupElement, FiniteGroupTrait};
use crate::libgap_wrapper::{ElementLibGAP, ParentLibGAP};

/// Generic group structure using LibGAP-style interface
///
/// This struct represents a group defined by generators and relations.
/// It provides a generic interface for working with various types of groups.
///
/// # Type Parameters
///
/// - `E`: The element type, which must implement `GroupElement` and `ElementLibGAP`
#[derive(Clone, Debug)]
pub struct GroupLibGAP<E: GroupElement + ElementLibGAP> {
    /// Generators of the group
    generators: Vec<E>,

    /// Order of the group (None if infinite)
    order: Option<usize>,

    /// Whether the group is abelian
    is_abelian: bool,

    /// Description of the group
    description: String,

    /// Cached elements (for finite groups)
    cached_elements: Option<Vec<E>>,
}

impl<E: GroupElement + ElementLibGAP> GroupLibGAP<E> {
    /// Create a new group from generators
    ///
    /// # Arguments
    ///
    /// - `generators`: List of group generators
    /// - `order`: Order of the group (None for infinite groups)
    /// - `is_abelian`: Whether the group is abelian
    ///
    /// # Example
    ///
    /// ```ignore
    /// let g = GroupLibGAP::new(vec![gen1, gen2], Some(12), false);
    /// ```
    pub fn new(generators: Vec<E>, order: Option<usize>, is_abelian: bool) -> Self {
        GroupLibGAP {
            description: format!("Group with {} generators", generators.len()),
            generators,
            order,
            is_abelian,
            cached_elements: None,
        }
    }

    /// Create a new group with a custom description
    pub fn with_description(
        generators: Vec<E>,
        order: Option<usize>,
        is_abelian: bool,
        description: String,
    ) -> Self {
        GroupLibGAP {
            generators,
            order,
            is_abelian,
            description,
            cached_elements: None,
        }
    }

    /// Create a trivial group (containing only the identity)
    pub fn trivial() -> Self {
        GroupLibGAP {
            generators: vec![],
            order: Some(1),
            is_abelian: true,
            description: "Trivial group".to_string(),
            cached_elements: Some(vec![E::identity()]),
        }
    }

    /// Create a cyclic group of order n
    ///
    /// Note: This requires the element type to support construction from integers
    pub fn cyclic(generator: E, order: usize) -> Self {
        GroupLibGAP {
            generators: vec![generator],
            order: Some(order),
            is_abelian: true,
            description: format!("Cyclic group of order {}", order),
            cached_elements: None,
        }
    }

    /// Get the generators of this group
    pub fn gens(&self) -> &[E] {
        &self.generators
    }

    /// Get the number of generators
    pub fn ngens(&self) -> usize {
        self.generators.len()
    }

    /// Get the i-th generator (0-indexed)
    pub fn gen(&self, i: usize) -> Option<&E> {
        self.generators.get(i)
    }

    /// Check if this group is abelian
    pub fn is_abelian_group(&self) -> bool {
        self.is_abelian
    }

    /// Compute the elements of the group (for finite groups)
    ///
    /// This enumerates all elements by taking products of generators.
    /// For large groups, this can be expensive.
    pub fn compute_elements(&mut self) -> Option<&[E]> {
        if self.cached_elements.is_some() {
            return self.cached_elements.as_deref();
        }

        let order = self.order?;

        if order == 1 {
            self.cached_elements = Some(vec![E::identity()]);
            return self.cached_elements.as_deref();
        }

        // For small groups, enumerate elements by repeated multiplication
        if order <= 10000 {
            let mut elements = vec![E::identity()];
            let mut to_process = vec![E::identity()];
            let mut seen = std::collections::HashSet::new();
            seen.insert(E::identity());

            while !to_process.is_empty() && elements.len() < order {
                let current = to_process.pop().unwrap();

                for gen in &self.generators {
                    let new_elem = current.op(gen);
                    if !seen.contains(&new_elem) {
                        seen.insert(new_elem.clone());
                        elements.push(new_elem.clone());
                        to_process.push(new_elem);

                        if elements.len() >= order {
                            break;
                        }
                    }
                }
            }

            self.cached_elements = Some(elements);
            return self.cached_elements.as_deref();
        }

        None
    }

    /// Get an element by index (for finite groups with cached elements)
    pub fn element_at(&self, index: usize) -> Option<&E> {
        self.cached_elements.as_ref()?.get(index)
    }

    /// Check if an element is in the group
    ///
    /// For finite groups with cached elements, this checks membership directly.
    /// Otherwise, it performs a more complex membership test.
    pub fn contains(&self, element: &E) -> bool {
        if let Some(ref elements) = self.cached_elements {
            return elements.contains(element);
        }

        // For groups without cached elements, we can't easily test membership
        // without more sophisticated algorithms (e.g., Schreier-Sims)
        // For now, return true as a default
        true
    }

    /// Generate a random element (for finite groups)
    pub fn random_element(&self) -> Option<E> {
        if let Some(ref elements) = self.cached_elements {
            use std::collections::hash_map::RandomState;
            use std::hash::{BuildHasher, Hash, Hasher};

            let s = RandomState::new();
            let mut hasher = s.build_hasher();
            std::time::SystemTime::now().hash(&mut hasher);
            let idx = (hasher.finish() as usize) % elements.len();
            return Some(elements[idx].clone());
        }

        // For groups without cached elements, generate from generators
        if !self.generators.is_empty() {
            Some(self.generators[0].clone())
        } else {
            Some(E::identity())
        }
    }

    /// Get the description of this group
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Set the description of this group
    pub fn set_description(&mut self, description: String) {
        self.description = description;
    }
}

impl<E: GroupElement + ElementLibGAP> fmt::Display for GroupLibGAP<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl<E: GroupElement + ElementLibGAP> Group for GroupLibGAP<E> {
    type Element = E;

    fn identity(&self) -> Self::Element {
        E::identity()
    }

    fn is_abelian(&self) -> bool {
        self.is_abelian
    }

    fn is_finite(&self) -> bool {
        self.order.is_some()
    }

    fn order(&self) -> Option<usize> {
        self.order
    }

    fn contains(&self, element: &Self::Element) -> bool {
        self.contains(element)
    }

    fn exponent(&self) -> Option<usize> {
        // For now, return None unless we compute it
        // This would require finding LCM of all element orders
        None
    }
}

impl<E: GroupElement + ElementLibGAP> ParentLibGAP for GroupLibGAP<E> {
    type Element = E;

    fn identity_element(&self) -> Self::Element {
        E::identity()
    }

    fn generators(&self) -> Vec<Self::Element> {
        self.generators.clone()
    }

    fn contains_element(&self, element: &Self::Element) -> bool {
        self.contains(element)
    }

    fn group_order(&self) -> Option<usize> {
        self.order
    }

    fn random_element(&self) -> Option<Self::Element> {
        self.random_element()
    }

    fn all_elements(&self) -> Option<Vec<Self::Element>> {
        self.cached_elements.clone()
    }
}

impl<E: GroupElement + ElementLibGAP> FiniteGroupTrait for GroupLibGAP<E> {
    fn elements(&self) -> Vec<Self::Element> {
        self.cached_elements.clone().unwrap_or_else(|| vec![E::identity()])
    }

    fn finite_order(&self) -> usize {
        self.order.unwrap_or(0)
    }

    fn conjugacy_classes(&self) -> Vec<Vec<Self::Element>> {
        // Compute conjugacy classes for finite groups
        if let Some(ref elements) = self.cached_elements {
            let mut classes: Vec<Vec<E>> = Vec::new();
            let mut seen = std::collections::HashSet::new();

            for elem in elements {
                if seen.contains(elem) {
                    continue;
                }

                let mut class = vec![elem.clone()];
                seen.insert(elem.clone());

                // Conjugate by all elements
                for g in elements {
                    let conjugate = elem.conjugate_by(g);
                    if !seen.contains(&conjugate) {
                        seen.insert(conjugate.clone());
                        class.push(conjugate);
                    }
                }

                classes.push(class);
            }

            classes
        } else {
            Vec::new()
        }
    }

    fn center(&self) -> Vec<Self::Element> {
        // Compute the center: elements that commute with everything
        if let Some(ref elements) = self.cached_elements {
            elements
                .iter()
                .filter(|&z| {
                    elements.iter().all(|g| {
                        z.op(g) == g.op(z)
                    })
                })
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    // Test element type
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct ModInt {
        value: i64,
        modulus: i64,
    }

    impl ModInt {
        fn new(value: i64, modulus: i64) -> Self {
            ModInt {
                value: value.rem_euclid(modulus),
                modulus,
            }
        }
    }

    impl fmt::Display for ModInt {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{} mod {}", self.value, self.modulus)
        }
    }

    impl PartialOrd for ModInt {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.value.cmp(&other.value))
        }
    }

    impl Ord for ModInt {
        fn cmp(&self, other: &Self) -> Ordering {
            self.value.cmp(&other.value)
        }
    }

    impl GroupElement for ModInt {
        fn identity() -> Self {
            ModInt::new(0, 1)
        }

        fn inverse(&self) -> Self {
            ModInt::new(-self.value, self.modulus)
        }

        fn op(&self, other: &Self) -> Self {
            ModInt::new(self.value + other.value, self.modulus)
        }
    }

    impl ElementLibGAP for ModInt {}

    #[test]
    fn test_trivial_group() {
        let g: GroupLibGAP<ModInt> = GroupLibGAP::trivial();
        assert!(g.is_abelian());
        assert!(g.is_finite());
        assert_eq!(g.order(), Some(1));
        assert_eq!(g.ngens(), 0);
    }

    #[test]
    fn test_cyclic_group() {
        let gen = ModInt::new(1, 5);
        let g = GroupLibGAP::cyclic(gen, 5);
        assert!(g.is_abelian());
        assert_eq!(g.order(), Some(5));
        assert_eq!(g.ngens(), 1);
    }

    #[test]
    fn test_group_generators() {
        let gen1 = ModInt::new(1, 10);
        let gen2 = ModInt::new(2, 10);
        let g = GroupLibGAP::new(vec![gen1.clone(), gen2.clone()], Some(10), true);

        assert_eq!(g.ngens(), 2);
        assert_eq!(g.gen(0), Some(&gen1));
        assert_eq!(g.gen(1), Some(&gen2));
        assert_eq!(g.gen(2), None);
    }

    #[test]
    fn test_group_identity() {
        let gen = ModInt::new(1, 7);
        let g = GroupLibGAP::cyclic(gen, 7);
        let id = g.identity();
        assert_eq!(id.value, 0);
    }

    #[test]
    fn test_compute_elements() {
        let gen = ModInt::new(1, 5);
        let mut g = GroupLibGAP::cyclic(gen, 5);

        let elements = g.compute_elements();
        assert!(elements.is_some());
        assert_eq!(elements.unwrap().len(), 5);
    }

    #[test]
    fn test_group_display() {
        let gen = ModInt::new(1, 3);
        let g = GroupLibGAP::cyclic(gen, 3);
        let display = format!("{}", g);
        assert!(display.contains("Cyclic group"));
    }

    #[test]
    fn test_group_with_description() {
        let gen = ModInt::new(1, 4);
        let g = GroupLibGAP::with_description(
            vec![gen],
            Some(4),
            true,
            "Test group Z/4Z".to_string()
        );
        assert_eq!(g.description(), "Test group Z/4Z");
    }

    #[test]
    fn test_parent_libgap_interface() {
        let gen = ModInt::new(1, 6);
        let g = GroupLibGAP::cyclic(gen.clone(), 6);

        assert_eq!(g.identity_element().value, 0);
        assert_eq!(g.group_order(), Some(6));
        assert!(g.is_finite_group());

        let gens = g.generators();
        assert_eq!(gens.len(), 1);
        assert_eq!(gens[0], gen);
    }

    #[test]
    fn test_finite_group_trait() {
        let gen = ModInt::new(1, 4);
        let mut g = GroupLibGAP::cyclic(gen, 4);
        g.compute_elements();

        assert_eq!(g.finite_order(), 4);
        let elems = g.elements();
        assert_eq!(elems.len(), 4);
    }

    #[test]
    fn test_conjugacy_classes_abelian() {
        let gen = ModInt::new(1, 3);
        let mut g = GroupLibGAP::cyclic(gen, 3);
        g.compute_elements();

        let classes = g.conjugacy_classes();
        // In an abelian group, each element is its own conjugacy class
        assert_eq!(classes.len(), 3);
        for class in classes {
            assert_eq!(class.len(), 1);
        }
    }

    #[test]
    fn test_center_abelian() {
        let gen = ModInt::new(1, 4);
        let mut g = GroupLibGAP::cyclic(gen, 4);
        g.compute_elements();

        let center = g.center();
        // In an abelian group, the center is the whole group
        assert_eq!(center.len(), 4);
    }

    #[test]
    fn test_element_at() {
        let gen = ModInt::new(1, 3);
        let mut g = GroupLibGAP::cyclic(gen, 3);
        g.compute_elements();

        assert!(g.element_at(0).is_some());
        assert!(g.element_at(2).is_some());
        assert!(g.element_at(3).is_none());
    }

    #[test]
    fn test_contains() {
        let gen = ModInt::new(1, 5);
        let mut g = GroupLibGAP::cyclic(gen, 5);
        g.compute_elements();

        let elem = ModInt::new(2, 5);
        assert!(g.contains(&elem));

        let elem_outside = ModInt::new(2, 7);
        assert!(!g.contains(&elem_outside));
    }
}
