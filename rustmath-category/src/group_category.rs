//! Group category and subcategories
//!
//! This module implements the category of Groups and its subcategories:
//! - Groups: Base category of all groups
//! - Groups.Commutative: Abelian groups
//! - Groups.Topological: Groups with topological structure
//! - Groups.CartesianProducts: Product groups
//! - ElementMethods: Methods for group elements
//! - ParentMethods: Methods for group parents

use crate::category::{
    CartesianProductsCategory, Category, CommutativeCategory, TopologicalCategory,
};
use rustmath_groups::{Group, GroupElement};
use std::fmt;
use std::hash::Hash;

/// The category of Groups
///
/// This category contains all groups and provides the base structure
/// for group-theoretic operations.
#[derive(Clone, Debug)]
pub struct GroupCategory;

impl GroupCategory {
    /// Create a new GroupCategory
    pub fn new() -> Self {
        GroupCategory
    }

    /// Get the Commutative subcategory
    pub fn commutative() -> GroupCategoryCommutative {
        GroupCategoryCommutative::new()
    }

    /// Get the Topological subcategory
    pub fn topological() -> GroupCategoryTopological {
        GroupCategoryTopological::new()
    }

    /// Get the CartesianProducts subcategory
    pub fn cartesian_products() -> GroupCategoryCartesianProducts {
        GroupCategoryCartesianProducts::new()
    }
}

impl Default for GroupCategory {
    fn default() -> Self {
        Self::new()
    }
}

impl Category for GroupCategory {
    fn name(&self) -> &str {
        "Groups"
    }

    fn axioms(&self) -> Vec<&str> {
        vec![
            "closure",
            "associativity",
            "identity",
            "inverse",
        ]
    }

    fn description(&self) -> String {
        "Category of groups with group homomorphisms".to_string()
    }
}

/// Parent methods for groups
///
/// This trait provides additional methods that can be implemented by group parents.
/// These methods extend the basic Group trait with category-aware functionality.
pub trait GroupParentMethods: Group {
    /// Get the category of this group
    fn category(&self) -> GroupCategory {
        GroupCategory::new()
    }

    /// Check if this group is in a specific category
    fn is_in_category<C: Category>(&self, _category: &C) -> bool {
        // Default implementation - can be overridden
        true
    }

    /// Get all generators of the group (if finitely generated)
    fn generators(&self) -> Option<Vec<Self::Element>> {
        None
    }

    /// Get the derived series of the group
    ///
    /// The derived series is: G ⊇ [G,G] ⊇ [[G,G],[G,G]] ⊇ ...
    fn derived_series(&self) -> Vec<Vec<Self::Element>> {
        Vec::new()
    }

    /// Get the lower central series
    ///
    /// The lower central series is: G ⊇ [G,G] ⊇ [G,[G,G]] ⊇ ...
    fn lower_central_series(&self) -> Vec<Vec<Self::Element>> {
        Vec::new()
    }

    /// Check if the group is cyclic
    fn is_cyclic(&self) -> bool {
        self.generator().is_some()
    }

    /// Check if the group is simple
    ///
    /// A simple group has no non-trivial normal subgroups
    fn is_simple(&self) -> bool {
        false
    }

    /// Check if the group is perfect
    ///
    /// A perfect group satisfies G = [G,G]
    fn is_perfect(&self) -> bool {
        false
    }

    /// Check if the group is solvable
    fn is_solvable(&self) -> bool {
        false
    }

    /// Check if the group is nilpotent
    fn is_nilpotent(&self) -> bool {
        false
    }

    /// Get the center of the group
    ///
    /// Z(G) = {z ∈ G | zg = gz for all g ∈ G}
    fn center(&self) -> Option<Vec<Self::Element>> {
        None
    }

    /// Get the commutator subgroup [G,G]
    fn commutator_subgroup(&self) -> Option<Vec<Self::Element>> {
        None
    }

    /// Get the Frattini subgroup
    ///
    /// The intersection of all maximal subgroups
    fn frattini_subgroup(&self) -> Option<Vec<Self::Element>> {
        None
    }
}

/// Element methods for group elements
///
/// This trait provides additional methods for group elements beyond the basic
/// GroupElement trait. These methods are category-aware and provide extended
/// functionality.
pub trait GroupElementMethods: GroupElement {
    /// Get the category of this element
    fn category(&self) -> GroupCategory {
        GroupCategory::new()
    }

    /// Compute multiple powers at once
    ///
    /// Returns a vector [self, self^2, self^3, ..., self^n]
    fn powers(&self, n: usize) -> Vec<Self> {
        let mut result = Vec::with_capacity(n);
        let mut current = self.clone();

        for _ in 0..n {
            result.push(current.clone());
            current = current.op(self);
        }

        result
    }

    /// Check if this element commutes with another
    fn commutes_with(&self, other: &Self) -> bool {
        let ab = self.op(other);
        let ba = other.op(self);
        ab == ba
    }

    /// Get the centralizer of this element in a set
    ///
    /// C_S(g) = {s ∈ S | sg = gs}
    fn centralizer_in(&self, elements: &[Self]) -> Vec<Self> {
        elements
            .iter()
            .filter(|e| self.commutes_with(e))
            .cloned()
            .collect()
    }

    /// Check if this element is central (commutes with all elements)
    fn is_central_in(&self, elements: &[Self]) -> bool {
        elements.iter().all(|e| self.commutes_with(e))
    }

    /// Compute the word length of this element with respect to generators
    ///
    /// Returns the minimal number of generators needed to express this element
    fn word_length(&self, _generators: &[Self]) -> Option<usize> {
        None // Default implementation
    }

    /// Express this element as a word in generators
    fn as_word(&self, _generators: &[Self]) -> Option<Vec<(usize, i64)>> {
        None // Default implementation: vector of (generator_index, exponent)
    }
}

/// The category of Commutative (Abelian) Groups
///
/// This is a subcategory of Groups where all elements commute.
#[derive(Clone, Debug)]
pub struct GroupCategoryCommutative;

impl GroupCategoryCommutative {
    pub fn new() -> Self {
        GroupCategoryCommutative
    }
}

impl Default for GroupCategoryCommutative {
    fn default() -> Self {
        Self::new()
    }
}

impl Category for GroupCategoryCommutative {
    fn name(&self) -> &str {
        "Groups.Commutative"
    }

    fn super_categories(&self) -> Vec<Box<dyn Category>> {
        vec![Box::new(GroupCategory::new())]
    }

    fn axioms(&self) -> Vec<&str> {
        vec![
            "closure",
            "associativity",
            "identity",
            "inverse",
            "commutativity",
        ]
    }

    fn description(&self) -> String {
        "Category of abelian (commutative) groups".to_string()
    }
}

impl CommutativeCategory for GroupCategoryCommutative {}

/// The category of Topological Groups
///
/// Groups with a compatible topological structure.
#[derive(Clone, Debug)]
pub struct GroupCategoryTopological;

impl GroupCategoryTopological {
    pub fn new() -> Self {
        GroupCategoryTopological
    }
}

impl Default for GroupCategoryTopological {
    fn default() -> Self {
        Self::new()
    }
}

impl Category for GroupCategoryTopological {
    fn name(&self) -> &str {
        "Groups.Topological"
    }

    fn super_categories(&self) -> Vec<Box<dyn Category>> {
        vec![Box::new(GroupCategory::new())]
    }

    fn axioms(&self) -> Vec<&str> {
        vec![
            "closure",
            "associativity",
            "identity",
            "inverse",
            "continuous_multiplication",
            "continuous_inversion",
        ]
    }

    fn description(&self) -> String {
        "Category of topological groups".to_string()
    }
}

impl TopologicalCategory for GroupCategoryTopological {}

/// The category of Cartesian Products of Groups
///
/// This category contains groups formed as Cartesian products.
#[derive(Clone, Debug)]
pub struct GroupCategoryCartesianProducts;

impl GroupCategoryCartesianProducts {
    pub fn new() -> Self {
        GroupCategoryCartesianProducts
    }
}

impl Default for GroupCategoryCartesianProducts {
    fn default() -> Self {
        Self::new()
    }
}

impl Category for GroupCategoryCartesianProducts {
    fn name(&self) -> &str {
        "Groups.CartesianProducts"
    }

    fn super_categories(&self) -> Vec<Box<dyn Category>> {
        vec![Box::new(GroupCategory::new())]
    }

    fn axioms(&self) -> Vec<&str> {
        vec![
            "closure",
            "associativity",
            "identity",
            "inverse",
            "componentwise_operation",
        ]
    }

    fn description(&self) -> String {
        "Category of Cartesian products of groups".to_string()
    }
}

impl CartesianProductsCategory for GroupCategoryCartesianProducts {}

/// A Cartesian product of two groups
///
/// For groups G and H, G × H is the group with:
/// - Elements: (g, h) where g ∈ G, h ∈ H
/// - Operation: (g₁, h₁) · (g₂, h₂) = (g₁·g₂, h₁·h₂)
#[derive(Clone, Debug)]
pub struct CartesianProductGroup<G1, G2>
where
    G1: Group,
    G2: Group,
{
    group1: G1,
    group2: G2,
}

impl<G1, G2> CartesianProductGroup<G1, G2>
where
    G1: Group,
    G2: Group,
{
    /// Create a new Cartesian product of two groups
    pub fn new(group1: G1, group2: G2) -> Self {
        CartesianProductGroup { group1, group2 }
    }

    /// Get the first component group
    pub fn first(&self) -> &G1 {
        &self.group1
    }

    /// Get the second component group
    pub fn second(&self) -> &G2 {
        &self.group2
    }
}

impl<G1, G2> fmt::Display for CartesianProductGroup<G1, G2>
where
    G1: Group,
    G2: Group,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} × {}", self.group1, self.group2)
    }
}

/// Element of a Cartesian product group
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CartesianProductElement<E1, E2>
where
    E1: GroupElement,
    E2: GroupElement,
{
    elem1: E1,
    elem2: E2,
}

impl<E1, E2> CartesianProductElement<E1, E2>
where
    E1: GroupElement,
    E2: GroupElement,
{
    /// Create a new element of the Cartesian product
    pub fn new(elem1: E1, elem2: E2) -> Self {
        CartesianProductElement { elem1, elem2 }
    }

    /// Get the first component
    pub fn first(&self) -> &E1 {
        &self.elem1
    }

    /// Get the second component
    pub fn second(&self) -> &E2 {
        &self.elem2
    }
}

impl<E1, E2> fmt::Display for CartesianProductElement<E1, E2>
where
    E1: GroupElement,
    E2: GroupElement,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.elem1, self.elem2)
    }
}

impl<E1, E2> std::ops::Mul for CartesianProductElement<E1, E2>
where
    E1: GroupElement,
    E2: GroupElement,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.op(&other)
    }
}

impl<E1, E2> GroupElement for CartesianProductElement<E1, E2>
where
    E1: GroupElement,
    E2: GroupElement,
{
    fn identity() -> Self {
        CartesianProductElement {
            elem1: E1::identity(),
            elem2: E2::identity(),
        }
    }

    fn inverse(&self) -> Self {
        CartesianProductElement {
            elem1: self.elem1.inverse(),
            elem2: self.elem2.inverse(),
        }
    }

    fn op(&self, other: &Self) -> Self {
        CartesianProductElement {
            elem1: self.elem1.op(&other.elem1),
            elem2: self.elem2.op(&other.elem2),
        }
    }

    fn pow(&self, n: i64) -> Self {
        CartesianProductElement {
            elem1: self.elem1.pow(n),
            elem2: self.elem2.pow(n),
        }
    }

    fn order(&self) -> Option<usize> {
        match (self.elem1.order(), self.elem2.order()) {
            (Some(o1), Some(o2)) => {
                // Order is LCM of component orders
                let gcd = gcd_usize(o1, o2);
                Some((o1 * o2) / gcd)
            }
            _ => None,
        }
    }
}

impl<G1, G2> Group for CartesianProductGroup<G1, G2>
where
    G1: Group,
    G2: Group,
{
    type Element = CartesianProductElement<G1::Element, G2::Element>;

    fn identity(&self) -> Self::Element {
        CartesianProductElement {
            elem1: self.group1.identity(),
            elem2: self.group2.identity(),
        }
    }

    fn is_abelian(&self) -> bool {
        self.group1.is_abelian() && self.group2.is_abelian()
    }

    fn is_finite(&self) -> bool {
        self.group1.is_finite() && self.group2.is_finite()
    }

    fn order(&self) -> Option<usize> {
        match (self.group1.order(), self.group2.order()) {
            (Some(o1), Some(o2)) => Some(o1 * o2),
            _ => None,
        }
    }

    fn contains(&self, element: &Self::Element) -> bool {
        self.group1.contains(&element.elem1) && self.group2.contains(&element.elem2)
    }

    fn is_multiplicative(&self) -> bool {
        self.group1.is_multiplicative() && self.group2.is_multiplicative()
    }

    fn exponent(&self) -> Option<usize> {
        match (self.group1.exponent(), self.group2.exponent()) {
            (Some(e1), Some(e2)) => {
                let gcd = gcd_usize(e1, e2);
                Some((e1 * e2) / gcd)
            }
            _ => None,
        }
    }
}

// Helper function for GCD
fn gcd_usize(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd_usize(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_category_name() {
        let cat = GroupCategory::new();
        assert_eq!(cat.name(), "Groups");
    }

    #[test]
    fn test_group_category_axioms() {
        let cat = GroupCategory::new();
        let axioms = cat.axioms();
        assert_eq!(axioms.len(), 4);
        assert!(axioms.contains(&"closure"));
        assert!(axioms.contains(&"associativity"));
        assert!(axioms.contains(&"identity"));
        assert!(axioms.contains(&"inverse"));
    }

    #[test]
    fn test_commutative_category() {
        let cat = GroupCategoryCommutative::new();
        assert_eq!(cat.name(), "Groups.Commutative");
        assert_eq!(cat.super_categories().len(), 1);

        let base_cat = GroupCategory::new();
        assert!(cat.is_subcategory_of(&base_cat));
    }

    #[test]
    fn test_topological_category() {
        let cat = GroupCategoryTopological::new();
        assert_eq!(cat.name(), "Groups.Topological");
        assert!(cat.has_topology());

        let base_cat = GroupCategory::new();
        assert!(cat.is_subcategory_of(&base_cat));
    }

    #[test]
    fn test_cartesian_products_category() {
        let cat = GroupCategoryCartesianProducts::new();
        assert_eq!(cat.name(), "Groups.CartesianProducts");
        assert!(cat.supports_cartesian_products());

        let base_cat = GroupCategory::new();
        assert!(cat.is_subcategory_of(&base_cat));
    }

    #[test]
    fn test_category_subcategories() {
        let groups = GroupCategory::new();
        let commutative = GroupCategory::commutative();
        let topological = GroupCategory::topological();
        let cartesian = GroupCategory::cartesian_products();

        assert!(commutative.is_subcategory_of(&groups));
        assert!(topological.is_subcategory_of(&groups));
        assert!(cartesian.is_subcategory_of(&groups));
    }

    // Test with a simple group implementation
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct SimpleElement(i32);

    impl fmt::Display for SimpleElement {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl GroupElement for SimpleElement {
        fn identity() -> Self {
            SimpleElement(0)
        }

        fn inverse(&self) -> Self {
            SimpleElement(-self.0)
        }

        fn op(&self, other: &Self) -> Self {
            SimpleElement(self.0 + other.0)
        }
    }

    #[derive(Clone, Debug)]
    struct SimpleGroup;

    impl fmt::Display for SimpleGroup {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Z")
        }
    }

    impl Group for SimpleGroup {
        type Element = SimpleElement;

        fn identity(&self) -> Self::Element {
            SimpleElement(0)
        }

        fn is_abelian(&self) -> bool {
            true
        }

        fn is_finite(&self) -> bool {
            false
        }

        fn order(&self) -> Option<usize> {
            None
        }

        fn contains(&self, _element: &Self::Element) -> bool {
            true
        }

        fn is_multiplicative(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_cartesian_product_element() {
        let e1 = SimpleElement(2);
        let e2 = SimpleElement(3);
        let prod = CartesianProductElement::new(e1.clone(), e2.clone());

        assert_eq!(prod.first(), &e1);
        assert_eq!(prod.second(), &e2);

        let identity = CartesianProductElement::<SimpleElement, SimpleElement>::identity();
        assert_eq!(identity.first().0, 0);
        assert_eq!(identity.second().0, 0);
    }

    #[test]
    fn test_cartesian_product_operation() {
        let e1 = CartesianProductElement::new(SimpleElement(2), SimpleElement(3));
        let e2 = CartesianProductElement::new(SimpleElement(4), SimpleElement(5));

        let result = e1.op(&e2);
        assert_eq!(result.first().0, 6);
        assert_eq!(result.second().0, 8);
    }

    #[test]
    fn test_cartesian_product_inverse() {
        let e = CartesianProductElement::new(SimpleElement(2), SimpleElement(3));
        let inv = e.inverse();

        assert_eq!(inv.first().0, -2);
        assert_eq!(inv.second().0, -3);
    }

    #[test]
    fn test_cartesian_product_group() {
        let g1 = SimpleGroup;
        let g2 = SimpleGroup;
        let prod = CartesianProductGroup::new(g1, g2);

        assert_eq!(prod.first().is_abelian(), true);
        assert_eq!(prod.second().is_abelian(), true);
        assert_eq!(prod.is_abelian(), true);
        assert_eq!(prod.is_finite(), false);
        assert_eq!(prod.order(), None);
    }

    #[test]
    fn test_element_methods_powers() {
        let e = SimpleElement(2);
        let powers = e.powers(5);

        assert_eq!(powers.len(), 5);
        assert_eq!(powers[0].0, 2);  // 2
        assert_eq!(powers[1].0, 4);  // 2+2
        assert_eq!(powers[2].0, 6);  // 2+2+2
        assert_eq!(powers[3].0, 8);  // 2+2+2+2
        assert_eq!(powers[4].0, 10); // 2+2+2+2+2
    }

    #[test]
    fn test_element_methods_commutes_with() {
        let e1 = SimpleElement(2);
        let e2 = SimpleElement(3);

        // In an abelian group, all elements commute
        assert!(e1.commutes_with(&e2));
    }
}
