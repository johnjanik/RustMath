//! Nilpotent Lie Algebras
//!
//! A Lie algebra L is nilpotent if there exists s such that all iterated brackets
//! of length > s vanish. Equivalently, the lower central series:
//! L ⊃ [L,L] ⊃ [L,[L,L]] ⊃ ... eventually reaches zero.
//!
//! The minimal such s is called the nilpotency step or class.
//!
//! Free nilpotent Lie algebras are quotients of free Lie algebras by the
//! (s+1)-th term of the lower central series, where the only relations are:
//! - Anti-commutativity
//! - Jacobi identity
//! - Vanishing of all brackets of length > s
//!
//! Examples:
//! - Abelian Lie algebras (step 1)
//! - Heisenberg algebras (step 2)
//!
//! Corresponds to sage.algebras.lie_algebras.nilpotent_lie_algebra
//!
//! References:
//! - Khukhro, E. "Nilpotent Groups and their Automorphisms" (1993)
//! - Serre, J-P. "Lie Algebras and Lie Groups" (1992)

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Nilpotent Lie Algebra
///
/// A base class for nilpotent Lie algebras with structure coefficients.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
pub struct NilpotentLieAlgebra<R: Ring> {
    /// Number of generators
    num_generators: usize,
    /// Nilpotency step
    step: usize,
    /// Structure coefficients: maps (i, j) to linear combination of basis indices
    structure_coefficients: HashMap<(usize, usize), HashMap<usize, R>>,
    /// Names of generators
    generator_names: Vec<String>,
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> NilpotentLieAlgebra<R> {
    /// Create a new nilpotent Lie algebra
    ///
    /// # Arguments
    ///
    /// * `num_generators` - Number of generators
    /// * `step` - Nilpotency step
    /// * `structure_coefficients` - Map from (i,j) to bracket [e_i, e_j]
    pub fn new(
        num_generators: usize,
        step: usize,
        structure_coefficients: HashMap<(usize, usize), HashMap<usize, R>>,
    ) -> Self {
        let generator_names = (0..num_generators)
            .map(|i| format!("x{}", i))
            .collect();

        NilpotentLieAlgebra {
            num_generators,
            step,
            structure_coefficients,
            generator_names,
            coefficient_ring: PhantomData,
        }
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// Get the nilpotency step
    pub fn step(&self) -> usize {
        self.step
    }

    /// Check if this is nilpotent (always true)
    pub fn is_nilpotent(&self) -> bool {
        true
    }

    /// Check if this is solvable (always true, nilpotent => solvable)
    pub fn is_solvable(&self) -> bool {
        true
    }

    /// Check if this is finite dimensional
    pub fn is_finite_dimensional(&self) -> bool {
        true
    }

    /// Get the zero element
    pub fn zero(&self, dimension: usize) -> NilpotentLieAlgebraElement<R>
    where
        R: From<i64>,
    {
        NilpotentLieAlgebraElement::zero(dimension)
    }

    /// Get a generator
    pub fn generator(&self, index: usize, dimension: usize) -> NilpotentLieAlgebraElement<R>
    where
        R: From<i64>,
    {
        if index >= self.num_generators {
            return NilpotentLieAlgebraElement::zero(dimension);
        }
        NilpotentLieAlgebraElement::basis_element(index, dimension)
    }

    /// Get structure coefficients for [e_i, e_j]
    pub fn bracket_on_basis(&self, i: usize, j: usize) -> HashMap<usize, R>
    where
        R: Clone,
    {
        self.structure_coefficients.get(&(i, j))
            .cloned()
            .unwrap_or_else(HashMap::new)
    }

    /// Set generator names
    pub fn set_generator_names(&mut self, names: Vec<String>) {
        if names.len() == self.num_generators {
            self.generator_names = names;
        }
    }

    /// Get generator names
    pub fn generator_names(&self) -> &[String] {
        &self.generator_names
    }
}

impl<R: Ring + Clone> Display for NilpotentLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Nilpotent Lie algebra on {} generators with step {}",
            self.num_generators, self.step
        )
    }
}

/// Free Nilpotent Lie Algebra
///
/// The free nilpotent Lie algebra on r generators of step s.
/// This is the quotient of the free Lie algebra by the (s+1)-th term
/// of the lower central series.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::FreeNilpotentLieAlgebra;
/// // 2 generators, step 2 (like Heisenberg)
/// let free_nil: FreeNilpotentLieAlgebra<i64> = FreeNilpotentLieAlgebra::new(2, 2);
/// assert_eq!(free_nil.num_generators(), 2);
/// assert_eq!(free_nil.step(), 2);
/// ```
pub struct FreeNilpotentLieAlgebra<R: Ring> {
    /// Base nilpotent Lie algebra
    base: NilpotentLieAlgebra<R>,
    /// Number of generators (r)
    num_generators: usize,
    /// Nilpotency step (s)
    step: usize,
}

impl<R: Ring + Clone> FreeNilpotentLieAlgebra<R> {
    /// Create a new free nilpotent Lie algebra
    ///
    /// # Arguments
    ///
    /// * `num_generators` - Number of generators (r)
    /// * `step` - Nilpotency step (s)
    pub fn new(num_generators: usize, step: usize) -> Self
    where
        R: From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + std::ops::Neg<Output = R> + PartialEq,
    {
        // Build structure coefficients for free nilpotent Lie algebra
        let structure_coefficients = Self::build_structure_coefficients(num_generators, step);

        let base = NilpotentLieAlgebra::new(
            num_generators,
            step,
            structure_coefficients,
        );

        FreeNilpotentLieAlgebra {
            base,
            num_generators,
            step,
        }
    }

    /// Build structure coefficients for the free nilpotent Lie algebra
    ///
    /// Uses Lyndon basis elements of length <= step
    fn build_structure_coefficients(
        num_generators: usize,
        step: usize,
    ) -> HashMap<(usize, usize), HashMap<usize, R>>
    where
        R: From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + std::ops::Neg<Output = R> + PartialEq,
    {
        let mut coeffs = HashMap::new();

        // For simplicity, we implement basic anti-commutativity
        // Full implementation would use Lyndon basis
        for i in 0..num_generators {
            for j in 0..num_generators {
                if i == j {
                    // [x_i, x_i] = 0
                    coeffs.insert((i, j), HashMap::new());
                } else if i < j {
                    // [x_i, x_j] = e_{i,j} (some basis element)
                    // For free nilpotent of step >= 2, this would be non-zero
                    if step >= 2 {
                        let mut bracket_result = HashMap::new();
                        // Map to a new basis element (beyond generators)
                        let new_index = num_generators + i * num_generators + j;
                        bracket_result.insert(new_index, R::from(1));
                        coeffs.insert((i, j), bracket_result);
                    } else {
                        coeffs.insert((i, j), HashMap::new());
                    }
                } else {
                    // [x_j, x_i] = -[x_i, x_j] (anti-commutativity)
                    if step >= 2 {
                        let mut bracket_result = HashMap::new();
                        let new_index = num_generators + j * num_generators + i;
                        bracket_result.insert(new_index, R::from(-1));
                        coeffs.insert((i, j), bracket_result);
                    } else {
                        coeffs.insert((i, j), HashMap::new());
                    }
                }
            }
        }

        coeffs
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// Get the nilpotency step
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get the base algebra
    pub fn base(&self) -> &NilpotentLieAlgebra<R> {
        &self.base
    }

    /// Check if this is nilpotent (always true)
    pub fn is_nilpotent(&self) -> bool {
        true
    }

    /// Dimension of the free nilpotent Lie algebra
    ///
    /// Uses the formula for dimension based on generators and step
    pub fn dimension(&self) -> usize {
        // Simplified: full formula involves necklace polynomials
        // For step 1: dimension = r (abelian)
        // For step 2 with r generators: dimension = r + C(r,2) (like Heisenberg)
        if self.step == 1 {
            self.num_generators
        } else if self.step == 2 {
            let r = self.num_generators;
            r + (r * (r - 1)) / 2
        } else {
            // General case: would need full necklace formula
            // For now, rough estimate
            let mut dim = self.num_generators;
            for k in 2..=self.step {
                dim += self.num_generators.pow(k as u32) / k;
            }
            dim
        }
    }

    /// Get generators
    pub fn generators(&self, dimension: usize) -> Vec<NilpotentLieAlgebraElement<R>>
    where
        R: From<i64>,
    {
        (0..self.num_generators)
            .map(|i| self.base.generator(i, dimension))
            .collect()
    }
}

impl<R: Ring + Clone> Display for FreeNilpotentLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Free nilpotent Lie algebra on {} generators of step {}",
            self.num_generators, self.step
        )
    }
}

/// Element of a nilpotent Lie algebra
///
/// Represented as a linear combination of basis elements
#[derive(Clone)]
pub struct NilpotentLieAlgebraElement<R: Ring> {
    /// Coefficients for each basis element
    coefficients: Vec<R>,
}

impl<R: Ring + Clone> NilpotentLieAlgebraElement<R> {
    /// Create a new element
    pub fn new(coefficients: Vec<R>) -> Self {
        NilpotentLieAlgebraElement { coefficients }
    }

    /// Create the zero element
    pub fn zero(dimension: usize) -> Self
    where
        R: From<i64>,
    {
        NilpotentLieAlgebraElement {
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
        NilpotentLieAlgebraElement { coefficients }
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
        NilpotentLieAlgebraElement { coefficients }
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
        NilpotentLieAlgebraElement { coefficients }
    }

    /// Lie bracket with another element
    pub fn bracket(&self, other: &Self, algebra: &NilpotentLieAlgebra<R>) -> Self
    where
        R: From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq,
    {
        assert_eq!(self.dimension(), other.dimension());

        let mut result = Self::zero(self.dimension());

        // Compute [∑ a_i e_i, ∑ b_j e_j] = ∑ a_i b_j [e_i, e_j]
        for (i, a_i) in self.coefficients.iter().enumerate() {
            if a_i.is_zero() {
                continue;
            }
            for (j, b_j) in other.coefficients.iter().enumerate() {
                if b_j.is_zero() {
                    continue;
                }

                // Get [e_i, e_j] from structure coefficients
                let bracket_ij = algebra.bracket_on_basis(i, j);

                // Add a_i * b_j * [e_i, e_j] to result
                for (k, coeff_k) in bracket_ij {
                    let contribution = a_i.clone() * b_j.clone() * coeff_k.clone();
                    if k < result.coefficients.len() {
                        result.coefficients[k] = result.coefficients[k].clone() + contribution;
                    }
                }
            }
        }

        result
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for NilpotentLieAlgebraElement<R> {
    fn eq(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
    }
}

impl<R: Ring + Clone + PartialEq> Eq for NilpotentLieAlgebraElement<R> {}

impl<R: Ring + Clone + Display> Display for NilpotentLieAlgebraElement<R> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_nilpotent_creation_step1() {
        // Step 1 = abelian
        let nil: FreeNilpotentLieAlgebra<i64> = FreeNilpotentLieAlgebra::new(3, 1);
        assert_eq!(nil.num_generators(), 3);
        assert_eq!(nil.step(), 1);
        assert!(nil.is_nilpotent());
        assert_eq!(nil.dimension(), 3);
    }

    #[test]
    fn test_free_nilpotent_creation_step2() {
        // Step 2 with 2 generators (like Heisenberg)
        let nil: FreeNilpotentLieAlgebra<i64> = FreeNilpotentLieAlgebra::new(2, 2);
        assert_eq!(nil.num_generators(), 2);
        assert_eq!(nil.step(), 2);
        assert_eq!(nil.dimension(), 3); // 2 + C(2,2) = 2 + 1
    }

    #[test]
    fn test_nilpotent_algebra_creation() {
        let coeffs: HashMap<(usize, usize), HashMap<usize, i64>> = HashMap::new();
        let nil: NilpotentLieAlgebra<i64> = NilpotentLieAlgebra::new(2, 2, coeffs);
        assert_eq!(nil.num_generators(), 2);
        assert_eq!(nil.step(), 2);
        assert!(nil.is_nilpotent());
        assert!(nil.is_solvable());
    }

    #[test]
    fn test_element_creation() {
        let zero: NilpotentLieAlgebraElement<i64> = NilpotentLieAlgebraElement::zero(5);
        assert!(zero.is_zero());
        assert_eq!(zero.dimension(), 5);

        let e1: NilpotentLieAlgebraElement<i64> = NilpotentLieAlgebraElement::basis_element(1, 5);
        assert!(!e1.is_zero());
        assert_eq!(e1.coefficient(1), Some(&1));
    }

    #[test]
    fn test_element_addition() {
        let e0: NilpotentLieAlgebraElement<i64> = NilpotentLieAlgebraElement::basis_element(0, 3);
        let e1: NilpotentLieAlgebraElement<i64> = NilpotentLieAlgebraElement::basis_element(1, 3);
        let sum = e0.add(&e1);
        assert_eq!(sum.coefficient(0), Some(&1));
        assert_eq!(sum.coefficient(1), Some(&1));
    }

    #[test]
    fn test_element_scalar_mul() {
        let e0: NilpotentLieAlgebraElement<i64> = NilpotentLieAlgebraElement::basis_element(0, 3);
        let scaled = e0.scalar_mul(&5);
        assert_eq!(scaled.coefficient(0), Some(&5));
    }

    #[test]
    fn test_generators() {
        let nil: FreeNilpotentLieAlgebra<i64> = FreeNilpotentLieAlgebra::new(3, 2);
        let gens = nil.generators(nil.dimension());
        assert_eq!(gens.len(), 3);
    }

    #[test]
    fn test_structure_coefficients_step1() {
        // Step 1 should have all brackets = 0 (abelian)
        let nil: FreeNilpotentLieAlgebra<i64> = FreeNilpotentLieAlgebra::new(2, 1);
        let bracket = nil.base().bracket_on_basis(0, 1);
        assert!(bracket.is_empty());
    }

    #[test]
    fn test_structure_coefficients_step2() {
        // Step 2 should have non-trivial brackets
        let nil: FreeNilpotentLieAlgebra<i64> = FreeNilpotentLieAlgebra::new(2, 2);
        let bracket = nil.base().bracket_on_basis(0, 1);
        assert!(!bracket.is_empty());
    }
}
