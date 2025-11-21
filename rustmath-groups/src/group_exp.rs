//! Group Exponential Functor
//!
//! This module implements a functor that converts commutative additive groups
//! into isomorphic multiplicative groups. For a group G, the exponential
//! constructs elements denoted e^g where the multiplicative operation follows:
//! e^g · e^h = e^(g+h).
//!
//! This is analogous to the exponential map from additive to multiplicative
//! notation in mathematics, where exp(a) * exp(b) = exp(a + b).
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::group_exp::{GroupExp, GroupExpElement};
//! use rustmath_groups::additive_abelian_group::AdditiveAbelianGroup;
//!
//! // Create an additive group (e.g., Z_5)
//! let additive = AdditiveAbelianGroup::new(vec![5]);
//!
//! // Create the exponential group
//! let exp_group = GroupExp::new(additive.clone());
//!
//! // Elements multiply via addition in the base group
//! let e_a = GroupExpElement::new(&exp_group, additive.element(vec![2]));
//! let e_b = GroupExpElement::new(&exp_group, additive.element(vec![3]));
//! // e^2 * e^3 = e^5 = e^0 (mod 5)
//! ```

use std::fmt;
use std::hash::{Hash, Hasher};

use crate::additive_abelian_group::{AdditiveAbelianGroup, AdditiveAbelianGroupElement};
use crate::group_traits::Group;

/// The exponential of a commutative additive group
///
/// This structure wraps an additive group and provides a multiplicative
/// interface where multiplication corresponds to addition in the base group.
#[derive(Debug, Clone)]
pub struct GroupExp {
    /// The underlying additive group
    base_group: AdditiveAbelianGroup,
}

impl GroupExp {
    /// Create a new exponential group from an additive group
    ///
    /// # Arguments
    ///
    /// * `base_group` - The additive group to exponentiate
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::group_exp::GroupExp;
    /// use rustmath_groups::additive_abelian_group::AdditiveAbelianGroup;
    ///
    /// let additive = AdditiveAbelianGroup::new(vec![3]);
    /// let exp = GroupExp::new(additive);
    /// ```
    pub fn new(base_group: AdditiveAbelianGroup) -> Self {
        Self { base_group }
    }

    /// Returns a reference to the base additive group
    pub fn base_group(&self) -> &AdditiveAbelianGroup {
        &self.base_group
    }

    /// Returns the identity element (corresponding to zero in the base group)
    pub fn one(&self) -> GroupExpElement {
        GroupExpElement {
            parent: self.clone(),
            value: self.base_group.zero(),
        }
    }

    /// Create an element from a value in the base group
    pub fn element(&self, value: AdditiveAbelianGroupElement) -> GroupExpElement {
        GroupExpElement {
            parent: self.clone(),
            value,
        }
    }

    /// Returns the generators of this group as exponential elements
    pub fn generators(&self) -> Vec<GroupExpElement> {
        self.base_group
            .generators()
            .into_iter()
            .map(|g| self.element(g))
            .collect()
    }

    /// Compute the product of a list of elements
    pub fn product(&self, elements: &[GroupExpElement]) -> GroupExpElement {
        let mut result = self.one();
        for elem in elements {
            result = result.multiply(&elem);
        }
        result
    }
}

impl fmt::Display for GroupExp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GroupExp({})", self.base_group)
    }
}

impl PartialEq for GroupExp {
    fn eq(&self, other: &Self) -> bool {
        self.base_group == other.base_group
    }
}

impl Eq for GroupExp {}

impl Group for GroupExp {
    type Element = GroupExpElement;

    fn identity(&self) -> Self::Element {
        self.one()
    }

    fn is_finite(&self) -> bool {
        // The exponential group is finite if the base additive group is finite
        self.base_group.is_finite()
    }

    fn order(&self) -> Option<usize> {
        // Order equals the order of the base additive group
        self.base_group.order()
    }

    fn contains(&self, element: &Self::Element) -> bool {
        // Check if the element's parent matches this group
        // In a full implementation, we'd check if the value is in the base group
        self.base_group.contains(&element.value)
    }
}

/// An element of an exponential group
///
/// This represents an element e^g where g is an element of the base additive group.
/// Multiplication is defined as: e^a · e^b = e^(a+b).
#[derive(Debug, Clone)]
pub struct GroupExpElement {
    parent: GroupExp,
    value: AdditiveAbelianGroupElement,
}

impl GroupExpElement {
    /// Create a new exponential element
    pub fn new(parent: &GroupExp, value: AdditiveAbelianGroupElement) -> Self {
        Self {
            parent: parent.clone(),
            value,
        }
    }

    /// Returns a reference to the parent group
    pub fn parent(&self) -> &GroupExp {
        &self.parent
    }

    /// Returns a reference to the wrapped value
    pub fn value(&self) -> &AdditiveAbelianGroupElement {
        &self.value
    }

    /// Multiply this element with another: e^a · e^b = e^(a+b)
    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(
            self.parent.base_group, other.parent.base_group,
            "Cannot multiply elements from different groups"
        );
        Self {
            parent: self.parent.clone(),
            value: self.value.add(&other.value),
        }
    }

    /// Compute the inverse: (e^a)^{-1} = e^{-a}
    pub fn inverse(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            value: self.value.negate(),
        }
    }

    /// Check if this is the identity element (e^0)
    pub fn is_identity(&self) -> bool {
        self.value.is_zero()
    }

    /// Raise this element to a power: (e^a)^n = e^{na}
    pub fn pow(&self, n: i32) -> Self {
        Self {
            parent: self.parent.clone(),
            value: self.value.scalar_multiply(n),
        }
    }

    /// Compute the commutator [a, b] = aba^{-1}b^{-1}
    ///
    /// For abelian groups, this is always the identity.
    pub fn commutator(&self, other: &Self) -> Self {
        let ab = self.multiply(other);
        let a_inv = self.inverse();
        let b_inv = other.inverse();
        ab.multiply(&a_inv).multiply(&b_inv)
    }
}

impl fmt::Display for GroupExpElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e^({})", self.value)
    }
}

impl PartialEq for GroupExpElement {
    fn eq(&self, other: &Self) -> bool {
        self.parent == other.parent && self.value == other.value
    }
}

impl Eq for GroupExpElement {}

impl Hash for GroupExpElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the value
        // Note: We can't easily hash the parent, so we assume elements
        // are only compared within the same group
        self.value.hash(state);
    }
}

use crate::group_traits::GroupElement;

impl std::ops::Mul for GroupExpElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.op(&other)
    }
}

impl GroupElement for GroupExpElement {
    fn identity() -> Self {
        // Create a minimal default group (additive group Z_1)
        // Note: Users should prefer calling group.identity() for the specific group
        let additive = AdditiveAbelianGroup::new(vec![1]);
        let exp_group = GroupExp::new(additive.clone());
        GroupExpElement {
            parent: exp_group,
            value: additive.zero(),
        }
    }

    fn inverse(&self) -> Self {
        Self::inverse(self)
    }

    fn op(&self, other: &Self) -> Self {
        self.multiply(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_exp_creation() {
        let additive = AdditiveAbelianGroup::new(vec![5]);
        let exp = GroupExp::new(additive.clone());
        assert_eq!(exp.base_group(), &additive);
    }

    #[test]
    fn test_identity_element() {
        let additive = AdditiveAbelianGroup::new(vec![5]);
        let exp = GroupExp::new(additive);
        let one = exp.one();
        assert!(one.is_identity());
    }

    #[test]
    fn test_element_creation() {
        let additive = AdditiveAbelianGroup::new(vec![5]);
        let exp = GroupExp::new(additive.clone());
        let elem = exp.element(additive.element(vec![2]));
        assert!(!elem.is_identity());
    }

    #[test]
    fn test_multiplication() {
        let additive = AdditiveAbelianGroup::new(vec![5]);
        let exp = GroupExp::new(additive.clone());

        // e^2 * e^3 = e^5 = e^0 (mod 5)
        let e2 = exp.element(additive.element(vec![2]));
        let e3 = exp.element(additive.element(vec![3]));
        let result = e2.multiply(&e3);

        // Should be identity since 2+3 = 5 ≡ 0 (mod 5)
        assert!(result.is_identity());
    }

    #[test]
    fn test_inverse() {
        let additive = AdditiveAbelianGroup::new(vec![5]);
        let exp = GroupExp::new(additive.clone());

        let e2 = exp.element(additive.element(vec![2]));
        let e_inv = e2.inverse();

        // e^2 * e^{-2} = e^0
        let result = e2.multiply(&e_inv);
        assert!(result.is_identity());
    }

    #[test]
    fn test_power() {
        let additive = AdditiveAbelianGroup::new(vec![7]);
        let exp = GroupExp::new(additive.clone());

        let e2 = exp.element(additive.element(vec![2]));
        let e6 = e2.pow(3); // (e^2)^3 = e^6

        let expected = exp.element(additive.element(vec![6]));
        assert_eq!(e6, expected);
    }

    #[test]
    fn test_generators() {
        let additive = AdditiveAbelianGroup::new(vec![3, 5]);
        let exp = GroupExp::new(additive);
        let gens = exp.generators();
        assert_eq!(gens.len(), 2);
    }

    #[test]
    fn test_product() {
        let additive = AdditiveAbelianGroup::new(vec![10]);
        let exp = GroupExp::new(additive.clone());

        let e2 = exp.element(additive.element(vec![2]));
        let e3 = exp.element(additive.element(vec![3]));
        let e5 = exp.element(additive.element(vec![5]));

        // Product should be e^(2+3+5) = e^10 = e^0
        let result = exp.product(&[e2, e3, e5]);
        assert!(result.is_identity());
    }

    #[test]
    fn test_commutator() {
        // For abelian groups, commutators are always identity
        let additive = AdditiveAbelianGroup::new(vec![5]);
        let exp = GroupExp::new(additive.clone());

        let e2 = exp.element(additive.element(vec![2]));
        let e3 = exp.element(additive.element(vec![3]));

        let comm = e2.commutator(&e3);
        assert!(comm.is_identity());
    }

    #[test]
    fn test_associativity() {
        let additive = AdditiveAbelianGroup::new(vec![11]);
        let exp = GroupExp::new(additive.clone());

        let e2 = exp.element(additive.element(vec![2]));
        let e3 = exp.element(additive.element(vec![3]));
        let e5 = exp.element(additive.element(vec![5]));

        // (e^2 * e^3) * e^5 = e^2 * (e^3 * e^5)
        let left = e2.multiply(&e3).multiply(&e5);
        let right = e2.multiply(&e3.multiply(&e5));
        assert_eq!(left, right);
    }

    #[test]
    fn test_commutativity() {
        // Elements should commute since base group is abelian
        let additive = AdditiveAbelianGroup::new(vec![7]);
        let exp = GroupExp::new(additive.clone());

        let e2 = exp.element(additive.element(vec![2]));
        let e3 = exp.element(additive.element(vec![3]));

        assert_eq!(e2.multiply(&e3), e3.multiply(&e2));
    }

    #[test]
    fn test_display() {
        let additive = AdditiveAbelianGroup::new(vec![5]);
        let exp = GroupExp::new(additive.clone());
        let e2 = exp.element(additive.element(vec![2]));

        let display = format!("{}", e2);
        assert!(display.contains("e^"));
    }

    #[test]
    fn test_multiple_generators() {
        let additive = AdditiveAbelianGroup::new(vec![2, 3]);
        let exp = GroupExp::new(additive.clone());

        // Create element e^(1, 1)
        let e11 = exp.element(additive.element(vec![1, 1]));
        // Create element e^(1, 2)
        let e12 = exp.element(additive.element(vec![1, 2]));

        // e^(1,1) * e^(1,2) = e^(2,3) = e^(0,0)
        let result = e11.multiply(&e12);
        assert!(result.is_identity());
    }
}
