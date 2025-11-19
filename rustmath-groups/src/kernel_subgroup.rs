//! Kernel Subgroups
//!
//! This module implements kernel subgroups - the subgroup of elements
//! that map to the identity under a group homomorphism.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::kernel_subgroup::{KernelSubgroup, KernelSubgroupElement};
//! use rustmath_groups::group_traits::{Group, GroupElement};
//! ```

use std::fmt;
use std::hash::{Hash, Hasher};

/// A kernel subgroup element wraps an element from the ambient group
/// that maps to the identity under the defining morphism.
#[derive(Clone, Debug)]
pub struct KernelSubgroupElement<G: Clone> {
    /// The underlying element from the ambient group
    element: G,
}

impl<G: Clone> KernelSubgroupElement<G> {
    /// Create a new kernel subgroup element
    ///
    /// # Arguments
    ///
    /// * `element` - The element from the ambient group
    ///
    /// # Panics
    ///
    /// This should only be called with elements that are actually in the kernel.
    /// The caller is responsible for ensuring this invariant.
    pub fn new(element: G) -> Self {
        KernelSubgroupElement { element }
    }

    /// Get the underlying element from the ambient group (lift)
    pub fn lift(&self) -> &G {
        &self.element
    }

    /// Convert into the underlying ambient group element
    pub fn into_inner(self) -> G {
        self.element
    }
}

impl<G: Clone + PartialEq> PartialEq for KernelSubgroupElement<G> {
    fn eq(&self, other: &Self) -> bool {
        self.element == other.element
    }
}

impl<G: Clone + Eq> Eq for KernelSubgroupElement<G> {}

impl<G: Clone + Hash> Hash for KernelSubgroupElement<G> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.element.hash(state);
    }
}

impl<G: Clone + fmt::Display> fmt::Display for KernelSubgroupElement<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.element)
    }
}

/// A kernel subgroup represents the kernel of a group homomorphism,
/// i.e., the set of elements in the domain that map to the identity in the codomain.
///
/// This is a normal subgroup of the domain group.
///
/// # Type Parameters
///
/// * `G` - The type of elements in the ambient (domain) group
/// * `H` - The type of elements in the codomain group
/// * `F` - The type of the morphism (function from G to H)
#[derive(Clone)]
pub struct KernelSubgroup<G, H, F>
where
    G: Clone,
    H: Clone + PartialEq,
    F: Fn(&G) -> H + Clone,
{
    /// The group homomorphism whose kernel we represent
    morphism: F,
    /// The identity element in the codomain
    codomain_identity: H,
    /// Optional cached generators of the kernel
    generators: Option<Vec<G>>,
    /// Name for display purposes
    name: Option<String>,
    /// Phantom data to tie G and H types
    _phantom: std::marker::PhantomData<G>,
}

impl<G, H, F> KernelSubgroup<G, H, F>
where
    G: Clone,
    H: Clone + PartialEq,
    F: Fn(&G) -> H + Clone,
{
    /// Create a new kernel subgroup from a group homomorphism
    ///
    /// # Arguments
    ///
    /// * `morphism` - The group homomorphism
    /// * `codomain_identity` - The identity element in the codomain group
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::kernel_subgroup::KernelSubgroup;
    ///
    /// // Create a kernel of the sign homomorphism from S_n to {±1}
    /// let sign_morphism = |perm: &Vec<usize>| {
    ///     // Compute sign of permutation (simplified)
    ///     1  // placeholder
    /// };
    /// let kernel = KernelSubgroup::new(sign_morphism, 1);
    /// ```
    pub fn new(morphism: F, codomain_identity: H) -> Self {
        KernelSubgroup {
            morphism,
            codomain_identity,
            generators: None,
            name: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a kernel subgroup with a name
    pub fn new_with_name(morphism: F, codomain_identity: H, name: String) -> Self {
        KernelSubgroup {
            morphism,
            codomain_identity,
            generators: None,
            name: Some(name),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the defining morphism
    ///
    /// This returns a reference to the morphism that defines this kernel.
    pub fn defining_morphism(&self) -> &F {
        &self.morphism
    }

    /// Check if an element is in the kernel
    ///
    /// An element g is in the kernel if morphism(g) equals the identity in the codomain.
    ///
    /// # Arguments
    ///
    /// * `element` - The element to test for membership
    pub fn contains(&self, element: &G) -> bool {
        (self.morphism)(element) == self.codomain_identity
    }

    /// Retract an ambient group element to a kernel element
    ///
    /// # Arguments
    ///
    /// * `element` - An element from the ambient group
    ///
    /// # Returns
    ///
    /// * `Some(KernelSubgroupElement)` if the element is in the kernel
    /// * `None` if the element is not in the kernel
    pub fn retract(&self, element: G) -> Option<KernelSubgroupElement<G>> {
        if self.contains(&element) {
            Some(KernelSubgroupElement::new(element))
        } else {
            None
        }
    }

    /// Lift a kernel element to the ambient group
    ///
    /// # Arguments
    ///
    /// * `element` - An element of the kernel subgroup
    pub fn lift(&self, element: &KernelSubgroupElement<G>) -> &G {
        element.lift()
    }

    /// Set the generators of the kernel
    ///
    /// # Arguments
    ///
    /// * `gens` - A vector of generators for the kernel
    pub fn set_generators(&mut self, gens: Vec<G>) {
        self.generators = Some(gens);
    }

    /// Get the generators of the kernel, if they have been computed
    pub fn generators(&self) -> Option<&[G]> {
        self.generators.as_deref()
    }

    /// Get the identity element of the kernel (as a kernel element)
    ///
    /// Note: This requires the ambient group identity to be provided
    pub fn identity_from(&self, ambient_identity: G) -> KernelSubgroupElement<G> {
        // The identity of the ambient group is always in the kernel
        KernelSubgroupElement::new(ambient_identity)
    }

    /// Get the name of the kernel subgroup
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the name of the kernel subgroup
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }
}

impl<G, H, F> fmt::Debug for KernelSubgroup<G, H, F>
where
    G: Clone + fmt::Debug,
    H: Clone + PartialEq + fmt::Debug,
    F: Fn(&G) -> H + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref name) = self.name {
            write!(f, "KernelSubgroup({})", name)
        } else {
            write!(f, "KernelSubgroup")
        }
    }
}

impl<G, H, F> fmt::Display for KernelSubgroup<G, H, F>
where
    G: Clone,
    H: Clone + PartialEq,
    F: Fn(&G) -> H + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref name) = self.name {
            write!(f, "Kernel of {}", name)
        } else {
            write!(f, "Kernel subgroup")
        }
    }
}

/// Operations on kernel subgroup elements
impl<G> KernelSubgroupElement<G>
where
    G: Clone,
{
    /// Multiply two kernel elements (requires a group operation)
    ///
    /// This is a generic multiplication that works with any group operation.
    pub fn multiply_with<Op>(&self, other: &Self, op: Op) -> Self
    where
        Op: Fn(&G, &G) -> G,
    {
        KernelSubgroupElement {
            element: op(&self.element, &other.element),
        }
    }

    /// Compute the inverse of a kernel element (requires an inverse operation)
    pub fn inverse_with<Op>(&self, inv_op: Op) -> Self
    where
        Op: Fn(&G) -> G,
    {
        KernelSubgroupElement {
            element: inv_op(&self.element),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_subgroup_basic() {
        // Test with a simple homomorphism: Z -> Z/2Z (even/odd)
        let mod2_morphism = |n: &i64| n % 2;
        let kernel = KernelSubgroup::new(mod2_morphism, 0);

        // Even numbers are in the kernel
        assert!(kernel.contains(&0));
        assert!(kernel.contains(&2));
        assert!(kernel.contains(&4));
        assert!(kernel.contains(&-2));

        // Odd numbers are not in the kernel
        assert!(!kernel.contains(&1));
        assert!(!kernel.contains(&3));
        assert!(!kernel.contains(&-1));
    }

    #[test]
    fn test_kernel_element_operations() {
        let elem1 = KernelSubgroupElement::new(2);
        let elem2 = KernelSubgroupElement::new(4);

        // Test lift
        assert_eq!(*elem1.lift(), 2);
        assert_eq!(*elem2.lift(), 4);

        // Test multiplication (with integer addition as the group operation)
        let product = elem1.multiply_with(&elem2, |a, b| a + b);
        assert_eq!(*product.lift(), 6);

        // Test inverse (with negation)
        let inv = elem1.inverse_with(|a| -a);
        assert_eq!(*inv.lift(), -2);
    }

    #[test]
    fn test_kernel_retract() {
        let mod3_morphism = |n: &i64| n % 3;
        let kernel = KernelSubgroup::new(mod3_morphism, 0);

        // Retract a kernel element
        let elem = kernel.retract(6);
        assert!(elem.is_some());
        assert_eq!(*elem.unwrap().lift(), 6);

        // Try to retract a non-kernel element
        let non_elem = kernel.retract(7);
        assert!(non_elem.is_none());
    }

    #[test]
    fn test_kernel_generators() {
        let mod2_morphism = |n: &i64| n % 2;
        let mut kernel = KernelSubgroup::new(mod2_morphism, 0);

        // Initially no generators
        assert!(kernel.generators().is_none());

        // Set generators (kernel is generated by 2)
        kernel.set_generators(vec![2]);
        assert_eq!(kernel.generators(), Some([2].as_slice()));
    }

    #[test]
    fn test_kernel_identity() {
        let mod2_morphism = |n: &i64| n % 2;
        let kernel = KernelSubgroup::new(mod2_morphism, 0);

        // Get the identity element
        let identity = kernel.identity_from(0);
        assert_eq!(*identity.lift(), 0);
    }

    #[test]
    fn test_kernel_element_equality() {
        let elem1 = KernelSubgroupElement::new(42);
        let elem2 = KernelSubgroupElement::new(42);
        let elem3 = KernelSubgroupElement::new(17);

        assert_eq!(elem1, elem2);
        assert_ne!(elem1, elem3);
    }

    #[test]
    fn test_kernel_with_name() {
        let morphism = |n: &i64| n % 5;
        let kernel = KernelSubgroup::new_with_name(
            morphism,
            0,
            "mod 5 kernel".to_string()
        );

        assert_eq!(kernel.name(), Some("mod 5 kernel"));
    }

    #[test]
    fn test_determinant_kernel() {
        // Test kernel of determinant map from 2x2 matrices to scalars
        // This gives us SL(2) - special linear group

        type Matrix2x2 = [[i64; 2]; 2];

        let det_morphism = |m: &Matrix2x2| {
            m[0][0] * m[1][1] - m[0][1] * m[1][0]
        };

        let sl2_kernel = KernelSubgroup::new(det_morphism, 1);

        // Identity matrix is in SL(2)
        let identity = [[1, 0], [0, 1]];
        assert!(sl2_kernel.contains(&identity));

        // This matrix has determinant 1
        let mat1 = [[2, 1], [1, 1]];  // det = 2*1 - 1*1 = 1
        assert!(sl2_kernel.contains(&mat1));

        // This matrix has determinant 2, not in kernel
        let mat2 = [[2, 0], [0, 1]];  // det = 2
        assert!(!sl2_kernel.contains(&mat2));

        // This matrix has determinant -1, not in kernel
        let mat3 = [[0, 1], [1, 0]];  // det = -1
        assert!(!sl2_kernel.contains(&mat3));
    }

    #[test]
    fn test_sign_homomorphism_kernel() {
        // Kernel of sign homomorphism gives alternating group
        // We'll use a simple representation: permutations as vectors

        // Simple sign function for permutations (counts inversions mod 2)
        let sign = |perm: &Vec<usize>| -> i8 {
            let mut inversions = 0;
            for i in 0..perm.len() {
                for j in i+1..perm.len() {
                    if perm[i] > perm[j] {
                        inversions += 1;
                    }
                }
            }
            if inversions % 2 == 0 { 1 } else { -1 }
        };

        let alternating = KernelSubgroup::new(sign, 1);

        // Identity permutation [0,1,2] is even
        assert!(alternating.contains(&vec![0, 1, 2]));

        // (01) transposition [1,0,2] is odd
        assert!(!alternating.contains(&vec![1, 0, 2]));

        // (012) cycle [1,2,0] is even
        assert!(alternating.contains(&vec![1, 2, 0]));

        // (12) transposition [0,2,1] is odd
        assert!(!alternating.contains(&vec![0, 2, 1]));
    }
}
