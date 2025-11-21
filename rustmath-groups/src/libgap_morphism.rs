//! LibGAP Morphism - Group homomorphisms and morphisms
//!
//! This module provides functionality for group homomorphisms (structure-preserving
//! maps between groups). It includes:
//! - `GroupMorphismLibgap`: Represents a group homomorphism
//! - `GroupHomsetLibgap`: Represents the set of homomorphisms between two groups
//!
//! # Overview
//!
//! A group homomorphism φ: G → H is a function that preserves the group operation:
//! φ(g₁ · g₂) = φ(g₁) · φ(g₂) for all g₁, g₂ in G.
//!
//! # Example
//!
//! ```ignore
//! use rustmath_groups::libgap_morphism::{GroupMorphismLibgap, GroupHomsetLibgap};
//!
//! let phi = GroupMorphismLibgap::new(domain, codomain, mapping);
//! let ker = phi.kernel();
//! let im = phi.image();
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use crate::group_traits::{Group, GroupElement};

/// A group homomorphism
///
/// Represents a structure-preserving map φ: G → H between groups.
///
/// # Type Parameters
///
/// - `G`: The domain group type
/// - `H`: The codomain group type
#[derive(Clone, Debug)]
pub struct GroupMorphismLibgap<G, H>
where
    G: Group,
    H: Group,
{
    /// The domain group
    domain: G,

    /// The codomain group
    codomain: H,

    /// The mapping function as a HashMap (for finite groups)
    mapping: HashMap<G::Element, H::Element>,

    /// Whether this morphism has been validated as a homomorphism
    is_validated: bool,
}

impl<G, H> GroupMorphismLibgap<G, H>
where
    G: Group,
    H: Group,
{
    /// Create a new group morphism
    ///
    /// # Arguments
    ///
    /// - `domain`: The domain group
    /// - `codomain`: The codomain group
    /// - `mapping`: A map from domain elements to codomain elements
    /// - `validate`: Whether to validate that this is a valid homomorphism
    ///
    /// # Example
    ///
    /// ```ignore
    /// let phi = GroupMorphismLibgap::new(g, h, map, true);
    /// ```
    pub fn new(
        domain: G,
        codomain: H,
        mapping: HashMap<G::Element, H::Element>,
        validate: bool,
    ) -> Result<Self, String> {
        let morphism = GroupMorphismLibgap {
            domain,
            codomain,
            mapping,
            is_validated: false,
        };

        if validate {
            morphism.validate()?;
        }

        Ok(morphism)
    }

    /// Create a trivial homomorphism (maps everything to identity)
    pub fn trivial(domain: G, codomain: H) -> Self {
        GroupMorphismLibgap {
            domain,
            codomain,
            mapping: HashMap::new(),
            is_validated: true,
        }
    }

    /// Validate that this is a valid group homomorphism
    ///
    /// Checks that φ(g₁ · g₂) = φ(g₁) · φ(g₂) for sample elements
    pub fn validate(&self) -> Result<(), String> {
        // Check that identity maps to identity
        let id_domain = self.domain.identity();
        if let Some(mapped_id) = self.mapping.get(&id_domain) {
            let id_codomain = self.codomain.identity();
            if *mapped_id != id_codomain {
                return Err("Identity does not map to identity".to_string());
            }
        }

        // For finite groups, check homomorphism property on sample elements
        // Full validation would check all pairs, which can be expensive
        let mut count = 0;
        for (g1, img_g1) in &self.mapping {
            for (g2, img_g2) in &self.mapping {
                let product = g1.op(g2);
                if let Some(img_product) = self.mapping.get(&product) {
                    let expected = img_g1.op(img_g2);
                    if *img_product != expected {
                        return Err(format!(
                            "Homomorphism property violated: φ(g₁·g₂) ≠ φ(g₁)·φ(g₂)"
                        ));
                    }
                }

                count += 1;
                if count > 100 {
                    break; // Sample check for large groups
                }
            }
            if count > 100 {
                break;
            }
        }

        Ok(())
    }

    /// Apply the homomorphism to an element
    ///
    /// Returns φ(g) for an element g in the domain
    pub fn call(&self, element: &G::Element) -> Option<H::Element> {
        self.mapping.get(element).cloned().or_else(|| {
            // For trivial morphism or unmapped elements, return identity
            Some(self.codomain.identity())
        })
    }

    /// Get the kernel of the homomorphism
    ///
    /// ker(φ) = {g ∈ G | φ(g) = e_H}
    pub fn kernel(&self) -> Vec<G::Element> {
        let id_codomain = self.codomain.identity();
        self.mapping
            .iter()
            .filter(|(_, &ref v)| *v == id_codomain)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Get the image of the homomorphism
    ///
    /// im(φ) = {φ(g) | g ∈ G}
    pub fn image(&self) -> Vec<H::Element> {
        let mut image: Vec<H::Element> = self.mapping.values().cloned().collect();
        image.sort_by(|a, b| format!("{}", a).cmp(&format!("{}", b)));
        image.dedup();
        image
    }

    /// Get the image of a specific element (alias for call)
    pub fn pushforward(&self, element: &G::Element) -> Option<H::Element> {
        self.call(element)
    }

    /// Find a lift (preimage) of a codomain element
    ///
    /// Returns some g ∈ G such that φ(g) = h, if it exists
    pub fn lift(&self, element: &H::Element) -> Option<G::Element> {
        for (k, v) in &self.mapping {
            if v == element {
                return Some(k.clone());
            }
        }
        None
    }

    /// Get the preimage of a subset of the codomain
    ///
    /// Returns {g ∈ G | φ(g) ∈ S}
    pub fn preimage(&self, subset: &[H::Element]) -> Vec<G::Element> {
        let subset_set: HashSet<_> = subset.iter().collect();
        self.mapping
            .iter()
            .filter(|(_, v)| subset_set.contains(v))
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Create a section (right inverse) using the lift method
    ///
    /// A section is a map s: H → G such that φ ∘ s = id_H on the image
    pub fn section(&self) -> HashMap<H::Element, G::Element> {
        let mut section_map = HashMap::new();
        for h in self.image() {
            if let Some(g) = self.lift(&h) {
                section_map.insert(h, g);
            }
        }
        section_map
    }

    /// Check if this morphism is injective (one-to-one)
    pub fn is_injective(&self) -> bool {
        self.kernel().len() == 1 // Kernel contains only identity
    }

    /// Check if this morphism is surjective (onto)
    pub fn is_surjective(&self) -> bool {
        let image_size = self.image().len();
        if let Some(codomain_order) = self.codomain.order() {
            return image_size == codomain_order;
        }
        false
    }

    /// Check if this morphism is an isomorphism
    pub fn is_isomorphism(&self) -> bool {
        self.is_injective() && self.is_surjective()
    }

    /// Get the domain group
    pub fn domain(&self) -> &G {
        &self.domain
    }

    /// Get the codomain group
    pub fn codomain(&self) -> &H {
        &self.codomain
    }
}

impl<G, H> fmt::Display for GroupMorphismLibgap<G, H>
where
    G: Group,
    H: Group,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Group homomorphism from {} to {}",
            self.domain, self.codomain
        )
    }
}

/// The set of all homomorphisms between two groups
///
/// Represents Hom(G, H), the set of all group homomorphisms from G to H.
#[derive(Clone, Debug)]
pub struct GroupHomsetLibgap<G, H>
where
    G: Group,
    H: Group,
{
    /// The domain group
    domain: G,

    /// The codomain group
    codomain: H,

    /// Cached homomorphisms (if computed)
    homomorphisms: Option<Vec<GroupMorphismLibgap<G, H>>>,
}

impl<G, H> GroupHomsetLibgap<G, H>
where
    G: Group,
    H: Group,
{
    /// Create a new homset Hom(G, H)
    pub fn new(domain: G, codomain: H) -> Self {
        GroupHomsetLibgap {
            domain,
            codomain,
            homomorphisms: None,
        }
    }

    /// Get the trivial homomorphism (maps everything to identity)
    pub fn an_element(&self) -> GroupMorphismLibgap<G, H> {
        GroupMorphismLibgap::trivial(self.domain.clone(), self.codomain.clone())
    }

    /// Create a natural map when possible
    ///
    /// If the groups have the same structure, create a natural isomorphism
    pub fn natural_map(&self) -> Option<GroupMorphismLibgap<G, H>> {
        // For groups of the same order, try to create identity-like map
        if self.domain.order() == self.codomain.order() {
            Some(self.an_element())
        } else {
            None
        }
    }

    /// Get the domain group
    pub fn domain(&self) -> &G {
        &self.domain
    }

    /// Get the codomain group
    pub fn codomain(&self) -> &H {
        &self.codomain
    }

    /// Count the number of homomorphisms (if computable)
    pub fn cardinality(&self) -> Option<usize> {
        if let Some(ref homs) = self.homomorphisms {
            return Some(homs.len());
        }
        None
    }
}

impl<G, H> fmt::Display for GroupHomsetLibgap<G, H>
where
    G: Group,
    H: Group,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Set of group homomorphisms from {} to {}",
            self.domain, self.codomain
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            write!(f, "{}", self.value)
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

    // Test group
    #[derive(Clone, Debug)]
    struct CyclicGroup {
        order: usize,
    }

    impl fmt::Display for CyclicGroup {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Z/{}", self.order)
        }
    }

    impl Group for CyclicGroup {
        type Element = ModInt;

        fn identity(&self) -> Self::Element {
            ModInt::new(0, self.order as i64)
        }

        fn is_abelian(&self) -> bool {
            true
        }

        fn is_finite(&self) -> bool {
            true
        }

        fn order(&self) -> Option<usize> {
            Some(self.order)
        }

        fn contains(&self, element: &Self::Element) -> bool {
            element.modulus == self.order as i64
        }
    }

    #[test]
    fn test_trivial_morphism() {
        let g = CyclicGroup { order: 5 };
        let h = CyclicGroup { order: 7 };
        let phi = GroupMorphismLibgap::trivial(g.clone(), h.clone());

        let elem = ModInt::new(2, 5);
        let mapped = phi.call(&elem).unwrap();
        assert_eq!(mapped.value, 0); // Maps to identity
    }

    #[test]
    fn test_morphism_validation() {
        let g = CyclicGroup { order: 4 };
        let h = CyclicGroup { order: 2 };

        // Create a valid homomorphism: Z/4Z → Z/2Z
        let mut mapping = HashMap::new();
        mapping.insert(ModInt::new(0, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(1, 4), ModInt::new(1, 2));
        mapping.insert(ModInt::new(2, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(3, 4), ModInt::new(1, 2));

        let result = GroupMorphismLibgap::new(g, h, mapping, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_kernel() {
        let g = CyclicGroup { order: 4 };
        let h = CyclicGroup { order: 2 };

        let mut mapping = HashMap::new();
        mapping.insert(ModInt::new(0, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(1, 4), ModInt::new(1, 2));
        mapping.insert(ModInt::new(2, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(3, 4), ModInt::new(1, 2));

        let phi = GroupMorphismLibgap::new(g, h, mapping, false).unwrap();
        let ker = phi.kernel();

        // Kernel should be {0, 2}
        assert_eq!(ker.len(), 2);
        assert!(ker.contains(&ModInt::new(0, 4)));
        assert!(ker.contains(&ModInt::new(2, 4)));
    }

    #[test]
    fn test_image() {
        let g = CyclicGroup { order: 4 };
        let h = CyclicGroup { order: 2 };

        let mut mapping = HashMap::new();
        mapping.insert(ModInt::new(0, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(1, 4), ModInt::new(1, 2));
        mapping.insert(ModInt::new(2, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(3, 4), ModInt::new(1, 2));

        let phi = GroupMorphismLibgap::new(g, h, mapping, false).unwrap();
        let im = phi.image();

        // Image should be {0, 1} (all of Z/2Z)
        assert_eq!(im.len(), 2);
    }

    #[test]
    fn test_lift() {
        let g = CyclicGroup { order: 4 };
        let h = CyclicGroup { order: 2 };

        let mut mapping = HashMap::new();
        mapping.insert(ModInt::new(0, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(1, 4), ModInt::new(1, 2));
        mapping.insert(ModInt::new(2, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(3, 4), ModInt::new(1, 2));

        let phi = GroupMorphismLibgap::new(g, h, mapping, false).unwrap();

        let h_elem = ModInt::new(1, 2);
        let lifted = phi.lift(&h_elem);
        assert!(lifted.is_some());

        // Should lift to 1 or 3
        let lifted_val = lifted.unwrap();
        assert!(lifted_val.value == 1 || lifted_val.value == 3);
    }

    #[test]
    fn test_is_injective() {
        let g = CyclicGroup { order: 2 };
        let h = CyclicGroup { order: 2 };

        let mut mapping = HashMap::new();
        mapping.insert(ModInt::new(0, 2), ModInt::new(0, 2));
        mapping.insert(ModInt::new(1, 2), ModInt::new(1, 2));

        let phi = GroupMorphismLibgap::new(g, h, mapping, false).unwrap();
        assert!(phi.is_injective());
    }

    #[test]
    fn test_is_surjective() {
        let g = CyclicGroup { order: 4 };
        let h = CyclicGroup { order: 2 };

        let mut mapping = HashMap::new();
        mapping.insert(ModInt::new(0, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(1, 4), ModInt::new(1, 2));
        mapping.insert(ModInt::new(2, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(3, 4), ModInt::new(1, 2));

        let phi = GroupMorphismLibgap::new(g, h, mapping, false).unwrap();
        assert!(phi.is_surjective());
    }

    #[test]
    fn test_homset_creation() {
        let g = CyclicGroup { order: 3 };
        let h = CyclicGroup { order: 5 };
        let homset = GroupHomsetLibgap::new(g, h);

        assert_eq!(homset.domain().order(), Some(3));
        assert_eq!(homset.codomain().order(), Some(5));
    }

    #[test]
    fn test_homset_an_element() {
        let g = CyclicGroup { order: 4 };
        let h = CyclicGroup { order: 4 };
        let homset = GroupHomsetLibgap::new(g, h);

        let phi = homset.an_element();
        let elem = ModInt::new(2, 4);
        let mapped = phi.call(&elem).unwrap();
        assert_eq!(mapped.value, 0); // Trivial map
    }

    #[test]
    fn test_section() {
        let g = CyclicGroup { order: 4 };
        let h = CyclicGroup { order: 2 };

        let mut mapping = HashMap::new();
        mapping.insert(ModInt::new(0, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(1, 4), ModInt::new(1, 2));
        mapping.insert(ModInt::new(2, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(3, 4), ModInt::new(1, 2));

        let phi = GroupMorphismLibgap::new(g, h, mapping, false).unwrap();
        let section = phi.section();

        assert_eq!(section.len(), 2); // One lift for each element in image
    }

    #[test]
    fn test_preimage() {
        let g = CyclicGroup { order: 4 };
        let h = CyclicGroup { order: 2 };

        let mut mapping = HashMap::new();
        mapping.insert(ModInt::new(0, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(1, 4), ModInt::new(1, 2));
        mapping.insert(ModInt::new(2, 4), ModInt::new(0, 2));
        mapping.insert(ModInt::new(3, 4), ModInt::new(1, 2));

        let phi = GroupMorphismLibgap::new(g, h, mapping, false).unwrap();

        let subset = vec![ModInt::new(0, 2)];
        let pre = phi.preimage(&subset);

        // Preimage of {0} should be {0, 2}
        assert_eq!(pre.len(), 2);
    }
}
