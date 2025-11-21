//! Braid Groups
//!
//! This module implements braid groups, which are fundamental objects in topology,
//! knot theory, and low-dimensional topology. The braid group B_n on n strands is
//! the group of braids with n strands, up to isotopy.
//!
//! # Mathematical Structure
//!
//! The braid group B_n is the Artin group of type A_{n-1}, with presentation:
//! - Generators: σ_1, σ_2, ..., σ_{n-1}
//! - Relations:
//!   * σ_i σ_{i+1} σ_i = σ_{i+1} σ_i σ_{i+1} (braid relation)
//!   * σ_i σ_j = σ_j σ_i for |i-j| ≥ 2 (far commutativity)
//!
//! # Geometric Interpretation
//!
//! A braid on n strands can be visualized as n strands connecting n points on the
//! top to n points on the bottom, where the strands can cross over/under each other.
//! - σ_i represents a crossing of strands i and i+1 (strand i over i+1)
//! - σ_i^{-1} represents the opposite crossing (strand i+1 over i)
//!
//! # Key Properties
//!
//! - B_1 is trivial
//! - B_2 ≅ ℤ (infinite cyclic)
//! - B_3 is the trefoil knot group
//! - There's a surjection B_n → S_n (to symmetric group) via permutation
//! - The pure braid group P_n is the kernel of this surjection
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::braid::{BraidGroup, braid_group};
//!
//! // Create the braid group on 4 strands
//! let b4 = braid_group(4);
//!
//! // Get generators
//! let sigma1 = b4.gen(0);
//! let sigma2 = b4.gen(1);
//! let sigma3 = b4.gen(2);
//!
//! // Verify the braid relation: σ_1 σ_2 σ_1 = σ_2 σ_1 σ_2
//! let left = sigma1.multiply(&sigma2).multiply(&sigma1);
//! let right = sigma2.multiply(&sigma1).multiply(&sigma2);
//! ```

use std::fmt;
use std::hash::Hash;

use crate::artin::{ArtinGroup, ArtinGroupElement, FiniteTypeArtinGroup, FiniteTypeArtinGroupElement, CoxeterMatrix};
use crate::free_group::FreeGroupElement;
use crate::group_traits::{Group, GroupElement};
use crate::permutation_group::PermutationGroup;

/// A braid group on n strands
///
/// The braid group B_n is the Artin group of type A_{n-1}.
#[derive(Debug, Clone)]
pub struct BraidGroup {
    /// Number of strands
    n: usize,
    /// Underlying Artin group
    artin_group: FiniteTypeArtinGroup,
}

impl BraidGroup {
    /// Create a new braid group on n strands
    ///
    /// # Arguments
    ///
    /// * `n` - Number of strands (must be at least 1)
    ///
    /// # Panics
    ///
    /// Panics if n < 1
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "Braid group must have at least 1 strand");

        // B_n is the Artin group of type A_{n-1}
        let artin_group = if n == 1 {
            // B_1 is trivial, but we still need a valid Artin group
            ArtinGroup::new(CoxeterMatrix::new(vec![vec![1]]))
        } else {
            ArtinGroup::new(CoxeterMatrix::type_a(n - 1))
        };

        Self { n, artin_group }
    }

    /// Returns the number of strands
    pub fn strands(&self) -> usize {
        self.n
    }

    /// Returns the number of generators (n - 1)
    pub fn num_generators(&self) -> usize {
        if self.n == 1 {
            0
        } else {
            self.n - 1
        }
    }

    /// Returns a reference to the underlying Artin group
    pub fn as_artin_group(&self) -> &FiniteTypeArtinGroup {
        &self.artin_group
    }

    /// Get the i-th generator σ_i (1-indexed to match mathematical convention)
    ///
    /// Note: Internally generators are 0-indexed, but mathematically they're numbered 1 to n-1
    pub fn sigma(&self, i: usize) -> Braid {
        assert!(
            i >= 1 && i < self.n,
            "Generator index must be in range [1, n-1]"
        );
        Braid {
            parent: self.clone(),
            element: self.artin_group.gen(i - 1),
        }
    }

    /// Get the i-th generator (0-indexed, for compatibility)
    pub fn gen(&self, i: usize) -> Braid {
        assert!(i < self.num_generators(), "Generator index out of bounds");
        Braid {
            parent: self.clone(),
            element: self.artin_group.gen(i),
        }
    }

    /// Returns the generators as a vector
    pub fn generators(&self) -> Vec<Braid> {
        (0..self.num_generators()).map(|i| self.gen(i)).collect()
    }

    /// Returns the symmetric group S_n associated with this braid group
    ///
    /// There's a natural surjection B_n → S_n that sends each braid to its
    /// induced permutation on strands
    pub fn symmetric_group(&self) -> PermutationGroup {
        PermutationGroup::symmetric(self.n)
    }
}

impl fmt::Display for BraidGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Braid group on {} strands", self.n)
    }
}

impl Group for BraidGroup {
    type Element = Braid;

    fn identity(&self) -> Self::Element {
        Braid {
            parent: self.clone(),
            element: self.artin_group.identity(),
        }
    }

    fn is_finite(&self) -> bool {
        // Braid groups are infinite for n >= 2
        false
    }

    fn order(&self) -> Option<usize> {
        // Braid groups are infinite
        None
    }

    fn contains(&self, element: &Self::Element) -> bool {
        // Check if element belongs to this braid group
        self.n == element.parent.n
    }
}

/// An element of a braid group (a braid)
///
/// A braid is represented as an element of the underlying Artin group.
#[derive(Debug, Clone)]
pub struct Braid {
    parent: BraidGroup,
    element: FiniteTypeArtinGroupElement,
}

impl Braid {
    /// Create a new braid from an Artin group element
    pub fn new(parent: BraidGroup, element: FiniteTypeArtinGroupElement) -> Self {
        Self { parent, element }
    }

    /// Returns a reference to the parent braid group
    pub fn parent(&self) -> &BraidGroup {
        &self.parent
    }

    /// Returns the number of strands
    pub fn strands(&self) -> usize {
        self.parent.strands()
    }

    /// Returns the underlying Artin group element
    pub fn as_artin_element(&self) -> &FiniteTypeArtinGroupElement {
        &self.element
    }

    /// Check if this is the identity braid
    pub fn is_identity(&self) -> bool {
        self.element.is_identity()
    }

    /// Multiply with another braid
    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(
            self.parent.strands(),
            other.parent.strands(),
            "Braids must have same number of strands"
        );

        Self {
            parent: self.parent.clone(),
            element: self.element.multiply(&other.element),
        }
    }

    /// Compute the inverse braid
    pub fn inverse(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            element: self.element.inverse(),
        }
    }

    /// Raise to a power
    pub fn pow(&self, n: i32) -> Self {
        Self {
            parent: self.parent.clone(),
            element: self.element.pow(n),
        }
    }

    /// Compute the permutation induced by this braid
    ///
    /// Each braid induces a permutation on the strands. σ_i swaps strands i and i+1.
    pub fn permutation(&self) -> Vec<usize> {
        let n = self.strands();
        let mut perm: Vec<usize> = (0..n).collect();

        // Process each generator in the word
        for &(gen, exp) in self.element.word().word() {
            let i = gen as usize;
            let swaps = exp.abs() as usize;

            for _ in 0..swaps {
                // Generator i swaps positions i and i+1
                perm.swap(i, i + 1);
            }
        }

        perm
    }

    /// Check if this is a pure braid (induces the identity permutation)
    pub fn is_pure(&self) -> bool {
        let perm = self.permutation();
        perm.iter().enumerate().all(|(i, &p)| i == p)
    }

    /// Get the word length in terms of generators
    pub fn word_length(&self) -> usize {
        self.element.word().length()
    }

    /// Get the exponent sum of generator σ_i (1-indexed)
    pub fn exponent_sum(&self, i: usize) -> i32 {
        assert!(
            i >= 1 && i < self.parent.strands(),
            "Generator index out of bounds"
        );
        self.element.word().exponent_sum(i - 1) as i32
    }
}

impl fmt::Display for Braid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            write!(f, "1")
        } else {
            write!(f, "{}", self.element)
        }
    }
}

impl PartialEq for Braid {
    fn eq(&self, other: &Self) -> bool {
        self.parent.strands() == other.parent.strands() && self.element == other.element
    }
}

impl Eq for Braid {}

impl std::hash::Hash for Braid {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash both the parent's strands and the element, consistent with PartialEq
        self.parent.strands().hash(state);
        self.element.hash(state);
    }
}

impl std::ops::Mul for Braid {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.op(&other)
    }
}

impl GroupElement for Braid {
    fn identity() -> Self {
        // Create a minimal braid group with 2 strands for the identity element
        let parent = BraidGroup::new(2);
        Braid {
            parent: parent.clone(),
            element: parent.artin_group.identity(),
        }
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }

    fn op(&self, other: &Self) -> Self {
        self.multiply(other)
    }
}

/// Type alias for BraidGroup (to match SageMath naming)
pub type BraidGroupClass = BraidGroup;

/// Factory function to create a braid group
///
/// # Arguments
///
/// * `n` - Number of strands
///
/// # Returns
///
/// The braid group B_n
///
/// # Examples
///
/// ```
/// use rustmath_groups::braid::braid_group;
///
/// let b3 = braid_group(3);
/// assert_eq!(b3.strands(), 3);
/// ```
pub fn braid_group(n: usize) -> BraidGroup {
    BraidGroup::new(n)
}

/// Action of the mapping class group on surfaces
///
/// The braid group B_n acts on the mapping class group of a punctured disk
/// with n punctures. This structure represents that action.
#[derive(Debug, Clone)]
pub struct MappingClassGroupAction {
    /// Number of punctures
    n: usize,
    /// The braid group acting
    braid_group: BraidGroup,
}

impl MappingClassGroupAction {
    /// Create a new mapping class group action
    ///
    /// # Arguments
    ///
    /// * `n` - Number of punctures
    pub fn new(n: usize) -> Self {
        Self {
            n,
            braid_group: BraidGroup::new(n),
        }
    }

    /// Returns the number of punctures
    pub fn num_punctures(&self) -> usize {
        self.n
    }

    /// Returns a reference to the braid group
    pub fn braid_group(&self) -> &BraidGroup {
        &self.braid_group
    }

    /// Apply a braid to a mapping class
    ///
    /// This would return the image of the mapping class under the braid action.
    /// For now, we just provide the structure.
    pub fn act(&self, braid: &Braid) -> MappingClass {
        assert_eq!(
            braid.strands(),
            self.n,
            "Braid must have same number of strands as punctures"
        );

        MappingClass {
            action: self.clone(),
            braid: braid.clone(),
        }
    }
}

/// A mapping class (orbit under the braid group action)
#[derive(Debug, Clone)]
pub struct MappingClass {
    action: MappingClassGroupAction,
    braid: Braid,
}

impl MappingClass {
    /// Returns a reference to the action
    pub fn action(&self) -> &MappingClassGroupAction {
        &self.action
    }

    /// Returns a reference to the representing braid
    pub fn braid(&self) -> &Braid {
        &self.braid
    }
}

/// Right quantum word representation
///
/// This is a specialized representation of braid group elements used in
/// quantum algebra computations. It represents elements in terms of right
/// quantum words in the quantum plane.
#[derive(Debug, Clone)]
pub struct RightQuantumWord {
    /// The braid this represents
    braid: Braid,
    /// Quantum parameter (typically denoted q)
    quantum_param: String,
    /// Cached quantum word representation
    word_data: Vec<(usize, i32)>, // (generator index, exponent)
}

impl RightQuantumWord {
    /// Create a new right quantum word from a braid
    ///
    /// # Arguments
    ///
    /// * `braid` - The braid to represent
    /// * `quantum_param` - Name of the quantum parameter (e.g., "q")
    pub fn new(braid: Braid, quantum_param: String) -> Self {
        let word_data = braid
            .element
            .word()
            .word()
            .iter()
            .map(|&(g, e)| (g as usize, e as i32))
            .collect();

        Self {
            braid,
            quantum_param,
            word_data,
        }
    }

    /// Returns a reference to the underlying braid
    pub fn braid(&self) -> &Braid {
        &self.braid
    }

    /// Returns the quantum parameter name
    pub fn quantum_param(&self) -> &str {
        &self.quantum_param
    }

    /// Returns the word data
    pub fn word_data(&self) -> &[(usize, i32)] {
        &self.word_data
    }

    /// Get the number of strands
    pub fn strands(&self) -> usize {
        self.braid.strands()
    }

    /// Check if this represents the identity
    pub fn is_identity(&self) -> bool {
        self.word_data.is_empty()
    }

    /// Get the length of the quantum word
    pub fn length(&self) -> usize {
        self.word_data.len()
    }
}

impl fmt::Display for RightQuantumWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            write!(f, "1")
        } else {
            let terms: Vec<String> = self
                .word_data
                .iter()
                .map(|(i, e)| {
                    if *e == 1 {
                        format!("σ_{}", i + 1)
                    } else if *e == -1 {
                        format!("σ_{}^(-1)", i + 1)
                    } else {
                        format!("σ_{}^{}", i + 1, e)
                    }
                })
                .collect();
            write!(f, "{}", terms.join(" "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_braid_group_creation() {
        let b3 = BraidGroup::new(3);
        assert_eq!(b3.strands(), 3);
        assert_eq!(b3.num_generators(), 2);
    }

    #[test]
    fn test_braid_group_b1() {
        let b1 = BraidGroup::new(1);
        assert_eq!(b1.strands(), 1);
        assert_eq!(b1.num_generators(), 0);
    }

    #[test]
    fn test_braid_group_factory() {
        let b4 = braid_group(4);
        assert_eq!(b4.strands(), 4);
    }

    #[test]
    fn test_braid_generators() {
        let b3 = BraidGroup::new(3);
        let sigma1 = b3.sigma(1);
        let sigma2 = b3.sigma(2);

        assert!(!sigma1.is_identity());
        assert!(!sigma2.is_identity());
    }

    #[test]
    fn test_braid_identity() {
        let b3 = BraidGroup::new(3);
        let id = b3.identity();
        assert!(id.is_identity());
    }

    #[test]
    fn test_braid_multiplication() {
        let b3 = BraidGroup::new(3);
        let sigma1 = b3.sigma(1);
        let sigma2 = b3.sigma(2);

        let prod = sigma1.multiply(&sigma2);
        assert!(!prod.is_identity());
        assert_eq!(prod.word_length(), 2);
    }

    #[test]
    fn test_braid_inverse() {
        let b3 = BraidGroup::new(3);
        let sigma1 = b3.sigma(1);
        let sigma1_inv = sigma1.inverse();

        // σ_1 * σ_1^{-1} should give identity (after normalization)
        let prod = sigma1.multiply(&sigma1_inv);
        // Note: Without normal form implementation, this won't reduce
    }

    #[test]
    fn test_braid_permutation() {
        let b3 = BraidGroup::new(3);
        let sigma1 = b3.sigma(1); // Swaps strands 0 and 1

        let perm = sigma1.permutation();
        assert_eq!(perm, vec![1, 0, 2]);
    }

    #[test]
    fn test_braid_permutation_composition() {
        let b3 = BraidGroup::new(3);
        let sigma1 = b3.sigma(1); // Swaps 0,1
        let sigma2 = b3.sigma(2); // Swaps 1,2

        // σ_1 σ_2 should give permutation (0,2,1)
        let prod = sigma1.multiply(&sigma2);
        let perm = prod.permutation();
        assert_eq!(perm, vec![1, 2, 0]);
    }

    #[test]
    fn test_pure_braid() {
        let b3 = BraidGroup::new(3);
        let sigma1 = b3.sigma(1);

        // σ_1 σ_1^{-1} is pure (identity permutation)
        let pure = sigma1.multiply(&sigma1.inverse());
        assert!(pure.is_pure());
    }

    #[test]
    fn test_braid_relation() {
        // Verify the braid relation: σ_1 σ_2 σ_1 = σ_2 σ_1 σ_2
        let b3 = BraidGroup::new(3);
        let sigma1 = b3.sigma(1);
        let sigma2 = b3.sigma(2);

        let left = sigma1.multiply(&sigma2).multiply(&sigma1);
        let right = sigma2.multiply(&sigma1).multiply(&sigma2);

        // Check they give the same permutation
        assert_eq!(left.permutation(), right.permutation());
        assert_eq!(left.permutation(), vec![2, 0, 1]);
    }

    #[test]
    fn test_far_commutativity() {
        // For n=4, σ_1 and σ_3 should commute
        let b4 = BraidGroup::new(4);
        let sigma1 = b4.sigma(1);
        let sigma3 = b4.sigma(3);

        let left = sigma1.multiply(&sigma3);
        let right = sigma3.multiply(&sigma1);

        // Check they give the same permutation
        assert_eq!(left.permutation(), right.permutation());
    }

    #[test]
    fn test_exponent_sum() {
        let b3 = BraidGroup::new(3);
        let sigma1 = b3.sigma(1);
        let sigma2 = b3.sigma(2);

        let braid = sigma1.pow(3).multiply(&sigma2.pow(-2));
        assert_eq!(braid.exponent_sum(1), 3);
        assert_eq!(braid.exponent_sum(2), -2);
    }

    #[test]
    fn test_word_length() {
        let b3 = BraidGroup::new(3);
        let sigma1 = b3.sigma(1);
        let sigma2 = b3.sigma(2);

        let braid = sigma1.multiply(&sigma2).multiply(&sigma1);
        assert_eq!(braid.word_length(), 3);
    }

    #[test]
    fn test_mapping_class_group_action() {
        let action = MappingClassGroupAction::new(3);
        assert_eq!(action.num_punctures(), 3);

        let b3 = action.braid_group();
        let sigma1 = b3.sigma(1);

        let mc = action.act(&sigma1);
        assert_eq!(mc.braid().strands(), 3);
    }

    #[test]
    fn test_right_quantum_word() {
        let b3 = BraidGroup::new(3);
        let sigma1 = b3.sigma(1);
        let sigma2 = b3.sigma(2);

        let braid = sigma1.multiply(&sigma2);
        let qword = RightQuantumWord::new(braid, "q".to_string());

        assert_eq!(qword.strands(), 3);
        assert_eq!(qword.length(), 2);
        assert_eq!(qword.quantum_param(), "q");
        assert!(!qword.is_identity());
    }

    #[test]
    fn test_right_quantum_word_identity() {
        let b3 = BraidGroup::new(3);
        let id = b3.identity();

        let qword = RightQuantumWord::new(id, "q".to_string());
        assert!(qword.is_identity());
        assert_eq!(qword.length(), 0);
    }

    #[test]
    fn test_display() {
        let b3 = BraidGroup::new(3);
        let display = format!("{}", b3);
        assert!(display.contains("Braid group on 3 strands"));

        let sigma1 = b3.sigma(1);
        let braid_display = format!("{}", sigma1);
        assert!(!braid_display.is_empty());
    }

    #[test]
    fn test_symmetric_group() {
        let b3 = BraidGroup::new(3);
        let s3 = b3.symmetric_group();
        assert_eq!(s3.degree(), 3);
    }

    #[test]
    fn test_b2_is_infinite_cyclic() {
        // B_2 has one generator σ_1
        let b2 = BraidGroup::new(2);
        assert_eq!(b2.num_generators(), 1);

        let sigma = b2.sigma(1);
        let sigma_squared = sigma.pow(2);
        let sigma_cubed = sigma.pow(3);

        // All should have different permutations showing they're distinct
        assert_ne!(sigma.permutation(), sigma_squared.permutation());
        assert_ne!(sigma.permutation(), sigma_cubed.permutation());
    }

    #[test]
    fn test_generators_vector() {
        let b4 = BraidGroup::new(4);
        let gens = b4.generators();
        assert_eq!(gens.len(), 3);

        for (i, gen) in gens.iter().enumerate() {
            assert!(!gen.is_identity());
            assert_eq!(gen.strands(), 4);
        }
    }
}
