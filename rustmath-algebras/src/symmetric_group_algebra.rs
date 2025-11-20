//! Symmetric Group Algebra Implementation
//!
//! The symmetric group algebra R[Sₙ] where Sₙ is the symmetric group on n elements.
//! This module provides both ZSₙ (integer coefficients) and QSₙ (rational coefficients).
//!
//! The symmetric group is presented using Coxeter generators (simple transpositions):
//! - s₁, s₂, ..., sₙ₋₁ where sᵢ is the transposition (i, i+1)
//! - Relations:
//!   - sᵢ² = 1 (involutions)
//!   - sᵢsⱼ = sⱼsᵢ when |i - j| > 1 (distant generators commute)
//!   - sᵢsᵢ₊₁sᵢ = sᵢ₊₁sᵢsᵢ₊₁ (braid relations)
//!
//! Corresponds to sage.algebras.symmetric_group_algebra

use crate::group_algebra::{GroupAlgebra, GroupAlgebraElement, GroupElement};
use rustmath_combinatorics::Permutation;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};

/// A symmetric group element, represented as a permutation
///
/// This wraps the Permutation struct from rustmath-combinatorics and implements
/// the GroupElement trait for use in group algebras.
#[derive(Clone, Debug, Eq)]
pub struct SymmetricGroupElement {
    /// The underlying permutation
    perm: Permutation,
    /// The size of the symmetric group (n in Sₙ)
    n: usize,
}

impl SymmetricGroupElement {
    /// Create a new symmetric group element from a permutation
    pub fn from_permutation(perm: Permutation) -> Self {
        let n = perm.size();
        Self { perm, n }
    }

    /// Create the identity element in Sₙ
    pub fn identity_n(n: usize) -> Self {
        Self {
            perm: Permutation::identity(n),
            n,
        }
    }

    /// Create a simple transposition sᵢ = (i, i+1) in Sₙ
    ///
    /// This is a Coxeter generator. Index is 0-based, so s₀ = (0, 1), s₁ = (1, 2), etc.
    pub fn simple_transposition(n: usize, i: usize) -> Option<Self> {
        if i >= n - 1 {
            return None;
        }

        let mut vec: Vec<usize> = (0..n).collect();
        vec.swap(i, i + 1);

        Permutation::from_vec(vec).map(|perm| Self { perm, n })
    }

    /// Create a transposition (i, j) in Sₙ
    pub fn transposition(n: usize, i: usize, j: usize) -> Option<Self> {
        if i >= n || j >= n || i == j {
            return None;
        }

        let mut vec: Vec<usize> = (0..n).collect();
        vec.swap(i, j);

        Permutation::from_vec(vec).map(|perm| Self { perm, n })
    }

    /// Create from a cycle notation
    ///
    /// For example, cycle(5, &[0, 2, 3]) creates the cycle (0 2 3) in S₅
    pub fn from_cycle(n: usize, cycle: &[usize]) -> Option<Self> {
        if cycle.is_empty() {
            return Some(Self::identity_n(n));
        }

        let mut vec: Vec<usize> = (0..n).collect();

        // Apply the cycle
        for i in 0..cycle.len() {
            let from = cycle[i];
            let to = cycle[(i + 1) % cycle.len()];
            if from >= n || to >= n {
                return None;
            }
            vec[from] = to;
        }

        Permutation::from_vec(vec).map(|perm| Self { perm, n })
    }

    /// Get the underlying permutation
    pub fn permutation(&self) -> &Permutation {
        &self.perm
    }

    /// Get the size of the symmetric group
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get the sign of the permutation (+1 for even, -1 for odd)
    pub fn sign(&self) -> i32 {
        self.perm.sign()
    }

    /// Convert to cycle notation
    pub fn cycles(&self) -> Vec<Vec<usize>> {
        self.perm.cycles()
    }
}

impl PartialEq for SymmetricGroupElement {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.perm == other.perm
    }
}

impl Hash for SymmetricGroupElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n.hash(state);
        // Hash the permutation by hashing its vector representation
        for i in 0..self.n {
            if let Some(val) = self.perm.apply(i) {
                val.hash(state);
            }
        }
    }
}

impl Display for SymmetricGroupElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "()");
        }

        let cycles = self.cycles();
        if cycles.is_empty() {
            return write!(f, "()");
        }

        for (i, cycle) in cycles.iter().enumerate() {
            if i > 0 {
                write!(f, "")?;
            }
            write!(f, "(")?;
            for (j, &elem) in cycle.iter().enumerate() {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", elem)?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

impl GroupElement for SymmetricGroupElement {
    fn identity() -> Self {
        // Default to S₁ for the trait, but users should use identity_n
        Self::identity_n(1)
    }

    fn is_identity(&self) -> bool {
        self.perm == Permutation::identity(self.n)
    }

    fn mult(&self, other: &Self) -> Self {
        assert_eq!(
            self.n, other.n,
            "Cannot multiply elements from different symmetric groups"
        );

        // Composition of permutations
        let perm = self
            .perm
            .compose(&other.perm)
            .expect("Permutations should have same size");

        Self { perm, n: self.n }
    }

    fn inverse(&self) -> Self {
        Self {
            perm: self.perm.inverse(),
            n: self.n,
        }
    }
}

/// The symmetric group algebra over the integers: ℤ[Sₙ]
///
/// Elements are formal ℤ-linear combinations of permutations in Sₙ.
pub type ZSymmetricGroupAlgebra = GroupAlgebra<Integer, SymmetricGroupElement>;

/// An element of ℤ[Sₙ]
pub type ZSymmetricGroupElement = GroupAlgebraElement<Integer, SymmetricGroupElement>;

/// The symmetric group algebra over the rationals: ℚ[Sₙ]
///
/// Elements are formal ℚ-linear combinations of permutations in Sₙ.
pub type QSymmetricGroupAlgebra = GroupAlgebra<Rational, SymmetricGroupElement>;

/// An element of ℚ[Sₙ]
pub type QSymmetricGroupElement = GroupAlgebraElement<Rational, SymmetricGroupElement>;

/// Builder for symmetric group algebras
pub struct SymmetricGroupAlgebraBuilder;

impl SymmetricGroupAlgebraBuilder {
    /// Create ℤ[Sₙ]
    pub fn z_algebra(n: usize) -> ZSymmetricGroupAlgebra {
        GroupAlgebra::new()
    }

    /// Create ℚ[Sₙ]
    pub fn q_algebra(n: usize) -> QSymmetricGroupAlgebra {
        GroupAlgebra::new()
    }

    /// Create a basis element (permutation) in ℤ[Sₙ]
    pub fn z_basis_element(perm: SymmetricGroupElement) -> ZSymmetricGroupElement {
        GroupAlgebraElement::from_group_element(perm)
    }

    /// Create a basis element (permutation) in ℚ[Sₙ]
    pub fn q_basis_element(perm: SymmetricGroupElement) -> QSymmetricGroupElement {
        GroupAlgebraElement::from_group_element(perm)
    }

    /// Create the identity element in ℤ[Sₙ]
    pub fn z_identity(n: usize) -> ZSymmetricGroupElement {
        GroupAlgebraElement::from_group_element(SymmetricGroupElement::identity_n(n))
    }

    /// Create the identity element in ℚ[Sₙ]
    pub fn q_identity(n: usize) -> QSymmetricGroupElement {
        GroupAlgebraElement::from_group_element(SymmetricGroupElement::identity_n(n))
    }

    /// Create a Coxeter generator sᵢ in ℤ[Sₙ]
    pub fn z_simple_transposition(n: usize, i: usize) -> Option<ZSymmetricGroupElement> {
        SymmetricGroupElement::simple_transposition(n, i)
            .map(GroupAlgebraElement::from_group_element)
    }

    /// Create a Coxeter generator sᵢ in ℚ[Sₙ]
    pub fn q_simple_transposition(n: usize, i: usize) -> Option<QSymmetricGroupElement> {
        SymmetricGroupElement::simple_transposition(n, i)
            .map(GroupAlgebraElement::from_group_element)
    }
}

/// Verify Coxeter relations for the symmetric group
///
/// This function checks that the given generators satisfy:
/// - sᵢ² = 1
/// - sᵢsⱼ = sⱼsᵢ when |i - j| > 1
/// - sᵢsᵢ₊₁sᵢ = sᵢ₊₁sᵢsᵢ₊₁ (braid relation)
pub fn verify_coxeter_relations<R: Ring>(
    generators: &[GroupAlgebraElement<R, SymmetricGroupElement>],
) -> bool {
    let n = generators.len();

    // Check sᵢ² = 1
    for (i, si) in generators.iter().enumerate() {
        let si_squared = si.clone() * si.clone();
        if !si_squared.is_one() {
            eprintln!("Relation sᵢ² = 1 failed for i = {}", i);
            return false;
        }
    }

    // Check sᵢsⱼ = sⱼsᵢ when |i - j| > 1
    for i in 0..n {
        for j in 0..n {
            if (i as i32 - j as i32).abs() > 1 {
                let lhs = generators[i].clone() * generators[j].clone();
                let rhs = generators[j].clone() * generators[i].clone();
                if lhs != rhs {
                    eprintln!("Relation sᵢsⱼ = sⱼsᵢ failed for i = {}, j = {}", i, j);
                    return false;
                }
            }
        }
    }

    // Check braid relation: sᵢsᵢ₊₁sᵢ = sᵢ₊₁sᵢsᵢ₊₁
    for i in 0..n.saturating_sub(1) {
        let si = generators[i].clone();
        let si_plus_1 = generators[i + 1].clone();

        let lhs = si.clone() * si_plus_1.clone() * si.clone();
        let rhs = si_plus_1.clone() * si.clone() * si_plus_1.clone();

        if lhs != rhs {
            eprintln!("Braid relation failed for i = {}", i);
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_group_element_identity() {
        let e = SymmetricGroupElement::identity_n(5);
        assert!(e.is_identity());
        assert_eq!(e.size(), 5);
    }

    #[test]
    fn test_simple_transposition() {
        let n = 5;
        let s0 = SymmetricGroupElement::simple_transposition(n, 0).unwrap();
        let s1 = SymmetricGroupElement::simple_transposition(n, 1).unwrap();

        // Check that s0 = (0 1)
        assert_eq!(s0.perm.apply(0), Some(1));
        assert_eq!(s0.perm.apply(1), Some(0));
        assert_eq!(s0.perm.apply(2), Some(2));

        // Check that s1 = (1 2)
        assert_eq!(s1.perm.apply(1), Some(2));
        assert_eq!(s1.perm.apply(2), Some(1));
    }

    #[test]
    fn test_symmetric_group_multiplication() {
        let n = 4;
        let s0 = SymmetricGroupElement::simple_transposition(n, 0).unwrap();
        let s1 = SymmetricGroupElement::simple_transposition(n, 1).unwrap();

        // s0 * s1 = (0 1)(1 2) = (0 1 2)
        let product = s0.mult(&s1);
        assert_eq!(product.perm.apply(0), Some(1));
        assert_eq!(product.perm.apply(1), Some(2));
        assert_eq!(product.perm.apply(2), Some(0));
        assert_eq!(product.perm.apply(3), Some(3));
    }

    #[test]
    fn test_symmetric_group_inverse() {
        let n = 4;
        let s0 = SymmetricGroupElement::simple_transposition(n, 0).unwrap();

        let inv = s0.inverse();
        let product = s0.mult(&inv);

        assert!(product.is_identity());
    }

    #[test]
    fn test_coxeter_relation_involution() {
        // Test sᵢ² = 1
        let n = 5;
        for i in 0..n - 1 {
            let si = SymmetricGroupElement::simple_transposition(n, i).unwrap();
            let si_squared = si.mult(&si);
            assert!(
                si_squared.is_identity(),
                "s{}^2 should be identity",
                i
            );
        }
    }

    #[test]
    fn test_coxeter_relation_commutation() {
        // Test sᵢsⱼ = sⱼsᵢ when |i - j| > 1
        let n = 5;
        let s0 = SymmetricGroupElement::simple_transposition(n, 0).unwrap();
        let s2 = SymmetricGroupElement::simple_transposition(n, 2).unwrap();

        let lhs = s0.mult(&s2);
        let rhs = s2.mult(&s0);

        assert_eq!(lhs, rhs, "s0 and s2 should commute");
    }

    #[test]
    fn test_coxeter_relation_braid() {
        // Test sᵢsᵢ₊₁sᵢ = sᵢ₊₁sᵢsᵢ₊₁
        let n = 4;
        let s0 = SymmetricGroupElement::simple_transposition(n, 0).unwrap();
        let s1 = SymmetricGroupElement::simple_transposition(n, 1).unwrap();

        let lhs = s0.mult(&s1).mult(&s0);
        let rhs = s1.mult(&s0).mult(&s1);

        assert_eq!(lhs, rhs, "Braid relation should hold");
    }

    #[test]
    fn test_z_symmetric_group_algebra() {
        let n = 3;
        let alg = SymmetricGroupAlgebraBuilder::z_algebra(n);

        let e = SymmetricGroupAlgebraBuilder::z_identity(n);
        assert!(e.is_one());

        let s0 = SymmetricGroupAlgebraBuilder::z_simple_transposition(n, 0).unwrap();
        let s1 = SymmetricGroupAlgebraBuilder::z_simple_transposition(n, 1).unwrap();

        // Test addition
        let sum = s0.clone() + s1.clone();
        assert_eq!(sum.len(), 2);

        // Test multiplication
        let product = s0.clone() * s1.clone();
        assert_eq!(product.len(), 1); // Should be a single permutation
    }

    #[test]
    fn test_q_symmetric_group_algebra() {
        let n = 3;
        let alg = SymmetricGroupAlgebraBuilder::q_algebra(n);

        let e = SymmetricGroupAlgebraBuilder::q_identity(n);
        assert!(e.is_one());

        let s0 = SymmetricGroupAlgebraBuilder::q_simple_transposition(n, 0).unwrap();
        let s1 = SymmetricGroupAlgebraBuilder::q_simple_transposition(n, 1).unwrap();

        // Test scalar multiplication with rationals
        let half = Rational::new(Integer::from(1), Integer::from(2));
        let scaled = s0.scalar_mul(&half);

        // The coefficient should be 1/2
        let perm = SymmetricGroupElement::simple_transposition(n, 0).unwrap();
        let coeff = scaled.coeff(&perm);
        assert_eq!(coeff, half);
    }

    #[test]
    fn test_algebra_coxeter_relations() {
        let n = 4;

        // Test in ℤ[Sₙ]
        let mut z_generators = Vec::new();
        for i in 0..n - 1 {
            z_generators.push(SymmetricGroupAlgebraBuilder::z_simple_transposition(n, i).unwrap());
        }
        assert!(
            verify_coxeter_relations(&z_generators),
            "Coxeter relations should hold in ℤ[S₄]"
        );

        // Test in ℚ[Sₙ]
        let mut q_generators = Vec::new();
        for i in 0..n - 1 {
            q_generators.push(SymmetricGroupAlgebraBuilder::q_simple_transposition(n, i).unwrap());
        }
        assert!(
            verify_coxeter_relations(&q_generators),
            "Coxeter relations should hold in ℚ[S₄]"
        );
    }

    #[test]
    fn test_algebra_distributivity() {
        let n = 3;

        let e = SymmetricGroupAlgebraBuilder::z_identity(n);
        let s0 = SymmetricGroupAlgebraBuilder::z_simple_transposition(n, 0).unwrap();
        let s1 = SymmetricGroupAlgebraBuilder::z_simple_transposition(n, 1).unwrap();

        // Test distributivity: e * (s0 + s1) = e*s0 + e*s1
        let lhs = e.clone() * (s0.clone() + s1.clone());
        let rhs = e.clone() * s0.clone() + e.clone() * s1.clone();

        assert_eq!(lhs, rhs, "Distributivity should hold");
    }

    #[test]
    fn test_permutation_from_cycle() {
        let n = 5;
        let perm = SymmetricGroupElement::from_cycle(n, &[0, 2, 3]).unwrap();

        // (0 2 3) means 0→2, 2→3, 3→0
        assert_eq!(perm.perm.apply(0), Some(2));
        assert_eq!(perm.perm.apply(2), Some(3));
        assert_eq!(perm.perm.apply(3), Some(0));
        assert_eq!(perm.perm.apply(1), Some(1));
        assert_eq!(perm.perm.apply(4), Some(4));
    }

    #[test]
    fn test_sign_of_permutation() {
        let n = 4;

        // Identity is even
        let e = SymmetricGroupElement::identity_n(n);
        assert_eq!(e.sign(), 1);

        // Single transposition is odd
        let s0 = SymmetricGroupElement::simple_transposition(n, 0).unwrap();
        assert_eq!(s0.sign(), -1);

        // Product of two transpositions is even
        let s1 = SymmetricGroupElement::simple_transposition(n, 1).unwrap();
        let product = s0.mult(&s1);
        assert_eq!(product.sign(), 1);
    }

    #[test]
    fn test_all_simple_transpositions_s4() {
        let n = 4;

        // S₄ has 3 simple transpositions: s0, s1, s2
        for i in 0..n - 1 {
            let si = SymmetricGroupElement::simple_transposition(n, i).unwrap();
            assert_eq!(si.size(), n);
            assert_eq!(si.sign(), -1); // All transpositions are odd
        }

        // Should not be able to create s3 in S₄
        assert!(SymmetricGroupElement::simple_transposition(n, n - 1).is_none());
    }

    #[test]
    fn test_product_in_algebra() {
        let n = 3;

        // Create (s0 + s1) * (s0 + s1) in ℤ[S₃]
        let s0 = SymmetricGroupAlgebraBuilder::z_simple_transposition(n, 0).unwrap();
        let s1 = SymmetricGroupAlgebraBuilder::z_simple_transposition(n, 1).unwrap();

        let sum = s0.clone() + s1.clone();
        let product = sum.clone() * sum.clone();

        // (s0 + s1)² = s0² + s0*s1 + s1*s0 + s1²
        //            = 1 + s0*s1 + s1*s0 + 1
        //            = 2*1 + s0*s1 + s1*s0

        // Should have 3 terms: identity with coefficient 2, and two 3-cycles
        assert!(product.len() >= 2);
    }
}
