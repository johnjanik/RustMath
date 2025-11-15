//! Abelian Lie Algebras
//!
//! An abelian Lie algebra is a Lie algebra where the Lie bracket of any
//! two elements is always zero: [x, y] = 0 for all x, y.
//!
//! This makes abelian Lie algebras:
//! - Nilpotent (the lower central series terminates at the first step)
//! - Solvable (the derived series terminates at the first step)
//! - Trivial to work with computationally
//!
//! Corresponds to sage.algebras.lie_algebras.abelian
//!
//! References:
//! - Humphreys, J. "Introduction to Lie Algebras and Representation Theory" (1972)

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Trait for Lie algebra structures
pub trait LieAlgebra<R: Ring>: Clone {
    /// Lie bracket: [x, y]
    fn bracket(&self, other: &Self) -> Self;

    /// Check if this is an abelian Lie algebra
    fn is_abelian() -> bool {
        false
    }

    /// Check if this is nilpotent
    fn is_nilpotent() -> bool {
        false
    }

    /// Check if this is solvable
    fn is_solvable() -> bool {
        false
    }
}

/// Finite-dimensional Abelian Lie Algebra
///
/// An abelian Lie algebra where [x, y] = 0 for all elements.
/// This is the simplest type of Lie algebra.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::AbelianLieAlgebra;
/// # use rustmath_integers::Integer;
/// let algebra: AbelianLieAlgebra<Integer> = AbelianLieAlgebra::new(3);
/// assert_eq!(algebra.dimension(), 3);
/// assert!(algebra.is_abelian());
/// ```
pub struct AbelianLieAlgebra<R: Ring> {
    /// Dimension of the algebra (number of basis elements)
    dimension: usize,
    /// Base ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> AbelianLieAlgebra<R> {
    /// Create a new abelian Lie algebra
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension (number of generators)
    pub fn new(dimension: usize) -> Self {
        AbelianLieAlgebra {
            dimension,
            coefficient_ring: PhantomData,
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if this is abelian (always true)
    pub fn is_abelian(&self) -> bool {
        true
    }

    /// Check if this is nilpotent (always true for abelian)
    pub fn is_nilpotent(&self) -> bool {
        true
    }

    /// Check if this is solvable (always true for abelian)
    pub fn is_solvable(&self) -> bool {
        true
    }

    /// Get the Lie bracket of two basis elements
    ///
    /// For abelian Lie algebras, this is always zero
    pub fn bracket_on_basis(&self, _i: usize, _j: usize) -> AbelianLieAlgebraElement<R>
    where
        R: From<i64>,
    {
        AbelianLieAlgebraElement::zero(self.dimension)
    }

    /// Get basis elements
    pub fn basis(&self) -> Vec<AbelianLieAlgebraElement<R>>
    where
        R: From<i64>,
    {
        (0..self.dimension)
            .map(|i| AbelianLieAlgebraElement::basis_element(i, self.dimension))
            .collect()
    }

    /// Get the zero element
    pub fn zero(&self) -> AbelianLieAlgebraElement<R>
    where
        R: From<i64>,
    {
        AbelianLieAlgebraElement::zero(self.dimension)
    }

    /// Construct the universal enveloping algebra (UEA)
    ///
    /// For an abelian Lie algebra, the UEA is isomorphic to a polynomial ring
    pub fn universal_enveloping_algebra_description(&self) -> String {
        format!("Polynomial ring in {} variables over base ring", self.dimension)
    }
}

impl<R: Ring + Clone> Display for AbelianLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Abelian Lie algebra on {} generators", self.dimension)
    }
}

/// Element of an abelian Lie algebra
///
/// Represented as a linear combination of basis elements
#[derive(Clone, Debug)]
pub struct AbelianLieAlgebraElement<R: Ring> {
    /// Coefficients for each basis element
    coefficients: Vec<R>,
}

impl<R: Ring + Clone> AbelianLieAlgebraElement<R> {
    /// Create a new element
    pub fn new(coefficients: Vec<R>) -> Self {
        AbelianLieAlgebraElement { coefficients }
    }

    /// Create the zero element
    pub fn zero(dimension: usize) -> Self
    where
        R: From<i64>,
    {
        AbelianLieAlgebraElement {
            coefficients: vec![R::from(0); dimension],
        }
    }

    /// Create a basis element
    pub fn basis_element(index: usize, dimension: usize) -> Self
    where
        R: From<i64>,
    {
        let mut coefficients = vec![R::from(0); dimension];
        if index < dimension {
            coefficients[index] = R::from(1);
        }
        AbelianLieAlgebraElement { coefficients }
    }

    /// Get coefficients
    pub fn coefficients(&self) -> &[R] {
        &self.coefficients
    }

    /// Get coefficient at index
    pub fn coefficient(&self, index: usize) -> Option<&R> {
        self.coefficients.get(index)
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.coefficients.iter().all(|c| c.is_zero())
    }

    /// Dimension of the element
    pub fn dimension(&self) -> usize {
        self.coefficients.len()
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R>,
    {
        assert_eq!(self.dimension(), other.dimension());
        let coefficients = self
            .coefficients
            .iter()
            .zip(other.coefficients.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        AbelianLieAlgebraElement { coefficients }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: std::ops::Mul<Output = R>,
    {
        let coefficients = self
            .coefficients
            .iter()
            .map(|c| c.clone() * scalar.clone())
            .collect();
        AbelianLieAlgebraElement { coefficients }
    }

    /// Lie bracket (always zero for abelian Lie algebras)
    pub fn bracket(&self, _other: &Self) -> Self
    where
        R: From<i64>,
    {
        Self::zero(self.dimension())
    }
}

impl<R: Ring + Clone> LieAlgebra<R> for AbelianLieAlgebraElement<R>
where
    R: From<i64>,
{
    fn bracket(&self, _other: &Self) -> Self {
        Self::zero(self.dimension())
    }

    fn is_abelian() -> bool {
        true
    }

    fn is_nilpotent() -> bool {
        true
    }

    fn is_solvable() -> bool {
        true
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for AbelianLieAlgebraElement<R> {
    fn eq(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
    }
}

impl<R: Ring + Clone + PartialEq> Eq for AbelianLieAlgebraElement<R> {}

impl<R: Ring + Clone + Display> Display for AbelianLieAlgebraElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut terms = Vec::new();
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if !coeff.is_zero() {
                terms.push(format!("{}*e{}", coeff, i));
            }
        }
        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + "))
        }
    }
}

/// Infinite-dimensional Abelian Lie Algebra
///
/// An abelian Lie algebra with infinitely many generators.
/// The Lie bracket is still always zero.
pub struct InfiniteDimensionalAbelianLieAlgebra<R: Ring> {
    /// Generator index set description
    index_description: String,
    /// Base ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> InfiniteDimensionalAbelianLieAlgebra<R> {
    /// Create a new infinite-dimensional abelian Lie algebra
    ///
    /// # Arguments
    ///
    /// * `index_description` - Description of the index set (e.g., "Z", "N", "Z^2")
    pub fn new(index_description: String) -> Self {
        InfiniteDimensionalAbelianLieAlgebra {
            index_description,
            coefficient_ring: PhantomData,
        }
    }

    /// Check if this is abelian (always true)
    pub fn is_abelian(&self) -> bool {
        true
    }

    /// Check if this is nilpotent (always true)
    pub fn is_nilpotent(&self) -> bool {
        true
    }

    /// Check if this is solvable (always true)
    pub fn is_solvable(&self) -> bool {
        true
    }

    /// Check if this is finite dimensional (always false)
    pub fn is_finite_dimensional(&self) -> bool {
        false
    }

    /// Get the index set description
    pub fn index_set(&self) -> &str {
        &self.index_description
    }
}

impl<R: Ring + Clone> Display for InfiniteDimensionalAbelianLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Infinite-dimensional abelian Lie algebra indexed by {}",
            self.index_description
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abelian_algebra_creation() {
        let algebra: AbelianLieAlgebra<i64> = AbelianLieAlgebra::new(5);
        assert_eq!(algebra.dimension(), 5);
        assert!(algebra.is_abelian());
        assert!(algebra.is_nilpotent());
        assert!(algebra.is_solvable());
    }

    #[test]
    fn test_element_creation() {
        let zero: AbelianLieAlgebraElement<i64> = AbelianLieAlgebraElement::zero(3);
        assert!(zero.is_zero());
        assert_eq!(zero.dimension(), 3);

        let e1: AbelianLieAlgebraElement<i64> = AbelianLieAlgebraElement::basis_element(1, 5);
        assert!(!e1.is_zero());
        assert_eq!(e1.coefficient(1), Some(&1));
        assert_eq!(e1.coefficient(0), Some(&0));
    }

    #[test]
    fn test_element_addition() {
        let e0: AbelianLieAlgebraElement<i64> = AbelianLieAlgebraElement::basis_element(0, 3);
        let e1: AbelianLieAlgebraElement<i64> = AbelianLieAlgebraElement::basis_element(1, 3);
        let sum = e0.add(&e1);
        assert_eq!(sum.coefficient(0), Some(&1));
        assert_eq!(sum.coefficient(1), Some(&1));
        assert_eq!(sum.coefficient(2), Some(&0));
    }

    #[test]
    fn test_scalar_multiplication() {
        let e0: AbelianLieAlgebraElement<i64> = AbelianLieAlgebraElement::basis_element(0, 3);
        let scaled = e0.scalar_mul(&5);
        assert_eq!(scaled.coefficient(0), Some(&5));
        assert_eq!(scaled.coefficient(1), Some(&0));
    }

    #[test]
    fn test_bracket_is_zero() {
        let e0: AbelianLieAlgebraElement<i64> = AbelianLieAlgebraElement::basis_element(0, 3);
        let e1: AbelianLieAlgebraElement<i64> = AbelianLieAlgebraElement::basis_element(1, 3);
        let bracket = e0.bracket(&e1);
        assert!(bracket.is_zero());
    }

    #[test]
    fn test_basis_generation() {
        let algebra: AbelianLieAlgebra<i64> = AbelianLieAlgebra::new(4);
        let basis = algebra.basis();
        assert_eq!(basis.len(), 4);
        for (i, elem) in basis.iter().enumerate() {
            assert_eq!(elem.coefficient(i), Some(&1));
        }
    }

    #[test]
    fn test_bracket_on_basis() {
        let algebra: AbelianLieAlgebra<i64> = AbelianLieAlgebra::new(3);
        let bracket = algebra.bracket_on_basis(0, 1);
        assert!(bracket.is_zero());
    }

    #[test]
    fn test_infinite_dimensional() {
        let inf_algebra: InfiniteDimensionalAbelianLieAlgebra<i64> =
            InfiniteDimensionalAbelianLieAlgebra::new("Z".to_string());
        assert!(!inf_algebra.is_finite_dimensional());
        assert!(inf_algebra.is_abelian());
        assert_eq!(inf_algebra.index_set(), "Z");
    }
}
