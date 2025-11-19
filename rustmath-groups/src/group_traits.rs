//! Base Group Traits
//!
//! This module defines the fundamental traits for group theory in RustMath.
//! It provides the trait hierarchy: Group -> AbelianGroup, FiniteGroup, etc.

use std::fmt;
use std::hash::Hash;

/// Base trait for all groups
///
/// A group is a set G with a binary operation · that satisfies:
/// 1. Closure: For all a, b in G, a · b is also in G
/// 2. Associativity: (a · b) · c = a · (b · c)
/// 3. Identity: There exists e in G such that e · a = a · e = a
/// 4. Inverse: For each a in G, there exists a⁻¹ such that a · a⁻¹ = a⁻¹ · a = e
pub trait Group: Clone + fmt::Debug + fmt::Display {
    /// The element type of this group
    type Element: GroupElement;

    /// Get the identity element
    fn identity(&self) -> Self::Element;

    /// Check if the group is abelian (commutative)
    ///
    /// Returns true if for all a, b: a · b = b · a
    fn is_abelian(&self) -> bool {
        false // Default: assume non-abelian unless overridden
    }

    /// Check if the group is finite
    fn is_finite(&self) -> bool;

    /// Get the order (cardinality) of the group
    ///
    /// Returns None if the group is infinite
    fn order(&self) -> Option<usize>;

    /// Check if the group is trivial (contains only the identity)
    fn is_trivial(&self) -> bool {
        self.order() == Some(1)
    }

    /// Check if this is a multiplicative group
    ///
    /// Returns true if the group operation is multiplication,
    /// false if it's addition
    fn is_multiplicative(&self) -> bool {
        true // Default: most groups use multiplicative notation
    }

    /// Get a generator (if the group is cyclic)
    ///
    /// Returns None if the group is not cyclic
    fn generator(&self) -> Option<Self::Element> {
        None
    }

    /// Check if the group contains a given element
    fn contains(&self, element: &Self::Element) -> bool;

    /// Get the exponent of the group
    ///
    /// The exponent is the LCM of all element orders.
    /// Returns None if the group is infinite.
    fn exponent(&self) -> Option<usize> {
        None
    }
}

/// Trait for abelian (commutative) groups
///
/// An abelian group satisfies a · b = b · a for all elements
pub trait AbelianGroupTrait: Group {
    /// Abelian groups are always commutative
    fn is_abelian_always(&self) -> bool {
        true
    }
}

/// Trait for finite groups
///
/// A finite group has a finite number of elements
pub trait FiniteGroupTrait: Group {
    /// Get all elements of the group
    fn elements(&self) -> Vec<Self::Element>;

    /// Get the actual order (must be finite)
    fn finite_order(&self) -> usize;

    /// Compute the conjugacy classes
    fn conjugacy_classes(&self) -> Vec<Vec<Self::Element>> {
        Vec::new() // Default implementation
    }

    /// Get the center of the group
    ///
    /// The center Z(G) = {z ∈ G | zg = gz for all g ∈ G}
    fn center(&self) -> Vec<Self::Element> {
        Vec::new() // Default implementation
    }
}

/// Trait for group elements
pub trait GroupElement: Clone + Eq + Hash + fmt::Debug + fmt::Display {
    /// Get the identity element
    fn identity() -> Self;

    /// Compute the inverse of this element
    fn inverse(&self) -> Self;

    /// Multiply two elements (or add for additive groups)
    fn op(&self, other: &Self) -> Self;

    /// Compute the power of an element (or scalar multiple for additive groups)
    ///
    /// For multiplicative groups: g^n
    /// For additive groups: n·g
    fn pow(&self, n: i64) -> Self {
        if n == 0 {
            return Self::identity();
        }

        if n < 0 {
            return self.inverse().pow(-n);
        }

        // Binary exponentiation
        let mut result = Self::identity();
        let mut base = self.clone();
        let mut exp = n as u64;

        while exp > 0 {
            if exp % 2 == 1 {
                result = result.op(&base);
            }
            base = base.clone().op(&base);
            exp /= 2;
        }

        result
    }

    /// Compute the order of this element
    ///
    /// Returns None if the element has infinite order
    fn order(&self) -> Option<usize> {
        if *self == Self::identity() {
            return Some(1);
        }

        let mut current = self.clone();
        let mut ord = 1;

        // Check up to some reasonable limit
        while ord < 100000 {
            current = current.op(self);
            ord += 1;

            if current == Self::identity() {
                return Some(ord);
            }
        }

        None // Assume infinite order if not found
    }

    /// Check if this is the identity element
    fn is_identity(&self) -> bool {
        *self == Self::identity()
    }

    /// Conjugate this element by another: x⁻¹ · self · x
    fn conjugate_by(&self, x: &Self) -> Self {
        let x_inv = x.inverse();
        x_inv.op(self).op(x)
    }

    /// Compute the commutator [self, other] = self·other·self⁻¹·other⁻¹
    fn commutator(&self, other: &Self) -> Self {
        let self_inv = self.inverse();
        let other_inv = other.inverse();
        self.op(other).op(&self_inv).op(&other_inv)
    }
}

/// Check if a given type implements the Group trait
///
/// This is a helper function for type checking
pub fn is_group<G: Group>(_g: &G) -> bool {
    true
}

/// Algebraic group trait (placeholder for future algebraic variety support)
///
/// An algebraic group is a group that is also an algebraic variety
pub trait AlgebraicGroupTrait: Group {
    /// Check if this is an algebraic group
    fn is_algebraic(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test implementation of GroupElement for integers mod n
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct ModN {
        value: i64,
        modulus: i64,
    }

    impl ModN {
        fn new(value: i64, modulus: i64) -> Self {
            ModN {
                value: value.rem_euclid(modulus),
                modulus,
            }
        }
    }

    impl fmt::Display for ModN {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{} (mod {})", self.value, self.modulus)
        }
    }

    impl GroupElement for ModN {
        fn identity() -> Self {
            ModN {
                value: 0,
                modulus: 1,
            }
        }

        fn inverse(&self) -> Self {
            ModN::new(-self.value, self.modulus)
        }

        fn op(&self, other: &Self) -> Self {
            assert_eq!(self.modulus, other.modulus);
            ModN::new(self.value + other.value, self.modulus)
        }
    }

    #[test]
    fn test_group_element_identity() {
        let id = ModN::new(0, 5);
        assert!(id.is_identity());
    }

    #[test]
    fn test_group_element_inverse() {
        let a = ModN::new(3, 5);
        let a_inv = a.inverse();
        let product = a.op(&a_inv);
        assert_eq!(product.value, 0);
    }

    #[test]
    fn test_group_element_op() {
        let a = ModN::new(2, 5);
        let b = ModN::new(3, 5);
        let sum = a.op(&b);
        assert_eq!(sum.value, 0); // 2 + 3 = 5 ≡ 0 (mod 5)
    }

    #[test]
    fn test_group_element_pow() {
        let a = ModN::new(2, 5);
        let a_cubed = a.pow(3);
        assert_eq!(a_cubed.value, 1); // 2 + 2 + 2 = 6 ≡ 1 (mod 5)
    }

    #[test]
    fn test_group_element_order() {
        let a = ModN::new(1, 5);
        let order = a.order();
        assert_eq!(order, Some(5)); // Order of 1 in Z/5Z is 5
    }

    #[test]
    fn test_conjugate_by() {
        let a = ModN::new(2, 7);
        let x = ModN::new(3, 7);
        let conjugate = a.conjugate_by(&x);
        // In abelian groups, conjugation doesn't change the element
        assert_eq!(conjugate, a);
    }

    #[test]
    fn test_commutator() {
        let a = ModN::new(2, 5);
        let b = ModN::new(3, 5);
        let comm = a.commutator(&b);
        // In abelian groups, commutator is always identity
        assert!(comm.is_identity());
    }

    // Simple test group implementation
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
        type Element = ModN;

        fn identity(&self) -> Self::Element {
            ModN::new(0, self.order as i64)
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

        fn generator(&self) -> Option<Self::Element> {
            Some(ModN::new(1, self.order as i64))
        }

        fn is_multiplicative(&self) -> bool {
            false // Cyclic groups use additive notation
        }

        fn exponent(&self) -> Option<usize> {
            Some(self.order)
        }
    }

    impl AbelianGroupTrait for CyclicGroup {}

    impl FiniteGroupTrait for CyclicGroup {
        fn elements(&self) -> Vec<Self::Element> {
            (0..self.order)
                .map(|i| ModN::new(i as i64, self.order as i64))
                .collect()
        }

        fn finite_order(&self) -> usize {
            self.order
        }
    }

    #[test]
    fn test_cyclic_group_basic() {
        let g = CyclicGroup { order: 5 };
        assert!(g.is_abelian());
        assert!(g.is_finite());
        assert_eq!(g.order(), Some(5));
        assert!(!g.is_trivial());
        assert!(!g.is_multiplicative());
    }

    #[test]
    fn test_cyclic_group_identity() {
        let g = CyclicGroup { order: 5 };
        let id = g.identity();
        assert_eq!(id.value, 0);
        assert!(id.is_identity());
    }

    #[test]
    fn test_cyclic_group_generator() {
        let g = CyclicGroup { order: 5 };
        let gen = g.generator().unwrap();
        assert_eq!(gen.value, 1);

        // Check that generator has full order
        assert_eq!(gen.order(), Some(5));
    }

    #[test]
    fn test_cyclic_group_elements() {
        let g = CyclicGroup { order: 5 };
        let elems = g.elements();
        assert_eq!(elems.len(), 5);

        // Check all elements are distinct
        for i in 0..5 {
            assert_eq!(elems[i].value, i as i64);
        }
    }

    #[test]
    fn test_cyclic_group_contains() {
        let g = CyclicGroup { order: 5 };
        let elem = ModN::new(3, 5);
        assert!(g.contains(&elem));

        let elem_wrong = ModN::new(3, 7);
        assert!(!g.contains(&elem_wrong));
    }

    #[test]
    fn test_is_group() {
        let g = CyclicGroup { order: 5 };
        assert!(is_group(&g));
    }

    #[test]
    fn test_exponent() {
        let g = CyclicGroup { order: 6 };
        assert_eq!(g.exponent(), Some(6));
    }

    #[test]
    fn test_trivial_group() {
        let g = CyclicGroup { order: 1 };
        assert!(g.is_trivial());
    }
}
