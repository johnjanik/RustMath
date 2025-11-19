//! Semimonomial Transformation Groups
//!
//! This module implements the semimonomial transformation group, which is the
//! semidirect product of the monomial group and the automorphism group of a ring.
//!
//! # Mathematical Structure
//!
//! The semimonomial transformation group of degree n over a ring R is:
//! ```text
//! G = (R×)^n ⋊ (S_n × Aut(R))
//! ```
//!
//! where:
//! - (R×)^n is the group of n-tuples of units
//! - S_n is the symmetric group
//! - Aut(R) is the automorphism group of R
//!
//! # Group Operations
//!
//! Elements are triples (φ, π, α) where multiplication is:
//! ```text
//! (φ, π, α)(ψ, σ, β) = (φ · ψ^(π,α), πσ, α ∘ β)
//! ```
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::semimonomial_transformation_group::SemimonomialTransformationGroup;
//!
//! // Create the group of degree 3 over integers
//! let group = SemimonomialTransformationGroup::new(3);
//! ```

use rustmath_combinatorics::Permutation;
use crate::semimonomial_transformation::SemimonomialTransformation;
use crate::group_traits::{Group, FiniteGroupTrait};
use std::fmt;
use std::marker::PhantomData;

/// The semimonomial transformation group of degree n over a ring R
///
/// This group consists of all semimonomial transformations (φ, π, α)
/// where φ ∈ (R×)^n, π ∈ S_n, and α ∈ Aut(R).
#[derive(Clone)]
pub struct SemimonomialTransformationGroup<T: Clone> {
    /// The degree n
    degree: usize,
    /// Number of automorphisms available
    num_automorphisms: usize,
    /// Phantom data for the ring type
    _phantom: PhantomData<T>,
}

impl<T: Clone> SemimonomialTransformationGroup<T> {
    /// Create a new semimonomial transformation group
    ///
    /// # Arguments
    /// * `degree` - The degree n
    /// * `num_automorphisms` - Number of ring automorphisms
    pub fn new(degree: usize, num_automorphisms: usize) -> Self {
        Self {
            degree,
            num_automorphisms,
            _phantom: PhantomData,
        }
    }

    /// Get the degree of the group
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the number of automorphisms
    pub fn num_automorphisms(&self) -> usize {
        self.num_automorphisms
    }

    /// Create the identity element
    pub fn identity_element(&self, one: T) -> SemimonomialTransformation<T> {
        SemimonomialTransformation::identity(self.degree, one)
    }

    /// Generate all permutation generators (adjacent transpositions)
    pub fn permutation_generators(&self) -> Vec<Permutation> {
        let mut generators = Vec::new();
        let n = self.degree;

        for i in 0..n.saturating_sub(1) {
            let mut perm_vec = (0..n).collect::<Vec<_>>();
            perm_vec.swap(i, i + 1);
            if let Some(perm) = Permutation::from_vec(perm_vec) {
                generators.push(perm);
            }
        }

        generators
    }

    /// Check if two elements are equal
    pub fn equal(&self, a: &SemimonomialTransformation<T>, b: &SemimonomialTransformation<T>) -> bool
    where
        T: PartialEq,
    {
        a.units() == b.units()
            && a.permutation() == b.permutation()
            && a.automorphism_id() == b.automorphism_id()
    }
}

impl<T: Clone + fmt::Debug> fmt::Debug for SemimonomialTransformationGroup<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemimonomialTransformationGroup")
            .field("degree", &self.degree)
            .field("num_automorphisms", &self.num_automorphisms)
            .finish()
    }
}

impl<T: Clone> fmt::Display for SemimonomialTransformationGroup<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SemimonomialTransformationGroup(degree={}, automorphisms={})",
            self.degree, self.num_automorphisms
        )
    }
}

/// Action of the semimonomial group on vectors
///
/// This structure represents the action of the group on R^n.
#[derive(Clone)]
pub struct SemimonomialActionVec<T: Clone> {
    /// The acting group
    group: SemimonomialTransformationGroup<T>,
}

impl<T: Clone> SemimonomialActionVec<T> {
    /// Create a new vector action
    pub fn new(group: SemimonomialTransformationGroup<T>) -> Self {
        Self { group }
    }

    /// Get the group
    pub fn group(&self) -> &SemimonomialTransformationGroup<T> {
        &self.group
    }

    /// Apply the action to a vector
    ///
    /// For (φ, π, α) acting on v:
    /// result[i] = φ[i]^(-1) · α(v[π(i)])
    pub fn act(
        &self,
        transform: &SemimonomialTransformation<T>,
        vector: &[T],
        apply_automorphism: impl Fn(&T) -> T,
        multiply: impl Fn(&T, &T) -> T,
        invert: impl Fn(&T) -> T,
    ) -> Vec<T> {
        crate::semimonomial_transformation::action_on_vector(
            transform,
            vector,
            apply_automorphism,
            multiply,
            invert,
        )
    }

    /// Check if a vector is fixed by the transformation
    pub fn is_fixed(
        &self,
        transform: &SemimonomialTransformation<T>,
        vector: &[T],
        apply_automorphism: impl Fn(&T) -> T,
        multiply: impl Fn(&T, &T) -> T,
        invert: impl Fn(&T) -> T,
    ) -> bool
    where
        T: PartialEq,
    {
        let result = self.act(transform, vector, apply_automorphism, multiply, invert);
        result == vector
    }
}

impl<T: Clone + fmt::Debug> fmt::Debug for SemimonomialActionVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemimonomialActionVec")
            .field("group", &self.group)
            .finish()
    }
}

impl<T: Clone> fmt::Display for SemimonomialActionVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Action of {} on vectors", self.group)
    }
}

/// Action of the semimonomial group on matrices
///
/// This structure represents the action of the group on matrices,
/// where the transformation is applied to each row.
#[derive(Clone)]
pub struct SemimonomialActionMat<T: Clone> {
    /// The acting group
    group: SemimonomialTransformationGroup<T>,
}

impl<T: Clone> SemimonomialActionMat<T> {
    /// Create a new matrix action
    pub fn new(group: SemimonomialTransformationGroup<T>) -> Self {
        Self { group }
    }

    /// Get the group
    pub fn group(&self) -> &SemimonomialTransformationGroup<T> {
        &self.group
    }

    /// Apply the action to a matrix (acts on each row)
    pub fn act(
        &self,
        transform: &SemimonomialTransformation<T>,
        matrix: &[Vec<T>],
        apply_automorphism: impl Fn(&T) -> T + Clone,
        multiply: impl Fn(&T, &T) -> T,
        invert: impl Fn(&T) -> T,
    ) -> Vec<Vec<T>> {
        crate::semimonomial_transformation::action_on_matrix(
            transform,
            matrix,
            apply_automorphism,
            multiply,
            invert,
        )
    }

    /// Check if a matrix is fixed by the transformation
    pub fn is_fixed(
        &self,
        transform: &SemimonomialTransformation<T>,
        matrix: &[Vec<T>],
        apply_automorphism: impl Fn(&T) -> T + Clone,
        multiply: impl Fn(&T, &T) -> T,
        invert: impl Fn(&T) -> T,
    ) -> bool
    where
        T: PartialEq,
    {
        let result = self.act(transform, matrix, apply_automorphism, multiply, invert);
        result == matrix
    }
}

impl<T: Clone + fmt::Debug> fmt::Debug for SemimonomialActionMat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemimonomialActionMat")
            .field("group", &self.group)
            .finish()
    }
}

impl<T: Clone> fmt::Display for SemimonomialActionMat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Action of {} on matrices", self.group)
    }
}

/// Compute the order of the semimonomial transformation group
///
/// For a finite ring R with |R×| = u (number of units),
/// the order is: u^n · n! · |Aut(R)|
pub fn group_order(degree: usize, num_units: usize, num_automorphisms: usize) -> Option<usize> {
    // Compute u^n
    let units_part = num_units.checked_pow(degree as u32)?;

    // Compute n!
    let mut factorial = 1usize;
    for i in 1..=degree {
        factorial = factorial.checked_mul(i)?;
    }

    // Multiply all parts
    units_part
        .checked_mul(factorial)?
        .checked_mul(num_automorphisms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_creation() {
        let group: SemimonomialTransformationGroup<i32> =
            SemimonomialTransformationGroup::new(3, 2);

        assert_eq!(group.degree(), 3);
        assert_eq!(group.num_automorphisms(), 2);
    }

    #[test]
    fn test_identity_element() {
        let group: SemimonomialTransformationGroup<i32> =
            SemimonomialTransformationGroup::new(3, 1);
        let id = group.identity_element(1);

        assert_eq!(id.degree(), 3);
        assert_eq!(id.units(), &[1, 1, 1]);
        assert_eq!(id.automorphism_id(), 0);
    }

    #[test]
    fn test_permutation_generators() {
        let group: SemimonomialTransformationGroup<i32> =
            SemimonomialTransformationGroup::new(4, 1);
        let gens = group.permutation_generators();

        assert_eq!(gens.len(), 3); // 4-1 adjacent transpositions
    }

    #[test]
    fn test_vector_action() {
        let group: SemimonomialTransformationGroup<i32> =
            SemimonomialTransformationGroup::new(3, 1);
        let action = SemimonomialActionVec::new(group);

        let perm = Permutation::identity(3);
        let transform = SemimonomialTransformation::new(vec![1, 1, 1], perm, 0);
        let vector = vec![1, 2, 3];

        let result = action.act(
            &transform,
            &vector,
            |x| *x, // Identity automorphism
            |a, b| a * b,
            |x| 1 / x,
        );

        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_matrix_action() {
        let group: SemimonomialTransformationGroup<i32> =
            SemimonomialTransformationGroup::new(2, 1);
        let action = SemimonomialActionMat::new(group);

        let perm = Permutation::identity(2);
        let transform = SemimonomialTransformation::new(vec![1, 1], perm, 0);
        let matrix = vec![vec![1, 2], vec![3, 4]];

        let result = action.act(
            &transform,
            &matrix,
            |x| *x,
            |a, b| a * b,
            |x| 1 / x,
        );

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn test_group_order() {
        // For degree 2, 2 units, 1 automorphism:
        // Order = 2^2 · 2! · 1 = 4 · 2 · 1 = 8
        assert_eq!(group_order(2, 2, 1), Some(8));

        // For degree 3, 3 units, 2 automorphisms:
        // Order = 3^3 · 3! · 2 = 27 · 6 · 2 = 324
        assert_eq!(group_order(3, 3, 2), Some(324));
    }

    #[test]
    fn test_display() {
        let group: SemimonomialTransformationGroup<i32> =
            SemimonomialTransformationGroup::new(3, 2);
        let display = format!("{}", group);

        assert!(display.contains("degree=3"));
        assert!(display.contains("automorphisms=2"));
    }

    #[test]
    fn test_equal() {
        let group: SemimonomialTransformationGroup<i32> =
            SemimonomialTransformationGroup::new(2, 1);

        let perm = Permutation::identity(2);
        let t1 = SemimonomialTransformation::new(vec![1, 2], perm.clone(), 0);
        let t2 = SemimonomialTransformation::new(vec![1, 2], perm, 0);

        assert!(group.equal(&t1, &t2));
    }
}
