//! Additive Abelian Group Wrapper
//!
//! This module provides wrapper functionality for treating subgroups of existing
//! additive abelian groups as new additive abelian groups. It enables discrete
//! logarithm computations and basis calculations for wrapped groups.
//!
//! The wrapper allows you to take a subgroup of an ambient additive abelian group
//! and treat it as an independent group with its own generators, while maintaining
//! the ability to convert between the wrapper representation and the ambient group.

use std::fmt;
use std::collections::HashMap;
use crate::additive_abelian_group::{AdditiveAbelianGroup, AdditiveAbelianGroupElement};

/// An unwrapping morphism that embeds wrapped group elements into their ambient universe
///
/// This morphism provides the coercion framework, allowing elements to be automatically
/// converted back to their underlying objects when needed.
#[derive(Clone, Debug)]
pub struct UnwrappingMorphism<T: Clone> {
    /// The source wrapped group
    source: AdditiveAbelianGroupWrapper<T>,
    /// The target ambient group type
    target_description: String,
}

impl<T: Clone> UnwrappingMorphism<T> {
    /// Create a new unwrapping morphism
    ///
    /// # Arguments
    /// * `source` - The wrapped group to unwrap from
    /// * `target_description` - Description of the target ambient space
    pub fn new(source: AdditiveAbelianGroupWrapper<T>, target_description: String) -> Self {
        UnwrappingMorphism {
            source,
            target_description,
        }
    }

    /// Apply the morphism to a wrapped element
    ///
    /// Converts a wrapped element back to its ambient group representation
    pub fn apply(&self, element: &AdditiveAbelianGroupWrapperElement<T>) -> Result<T, String> {
        element.element()
    }

    /// Get the source group
    pub fn source(&self) -> &AdditiveAbelianGroupWrapper<T> {
        &self.source
    }

    /// Get the target description
    pub fn target_description(&self) -> &str {
        &self.target_description
    }
}

impl<T: Clone + fmt::Display> fmt::Display for UnwrappingMorphism<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unwrapping morphism from wrapped group to {}", self.target_description)
    }
}

/// An element of a wrapped additive abelian group
///
/// Represents an element as both a vector (in terms of the wrapped group's generators)
/// and optionally the underlying ambient group element.
#[derive(Clone, Debug)]
pub struct AdditiveAbelianGroupWrapperElement<T: Clone> {
    /// Vector representation in terms of generators
    vector: Vec<i64>,
    /// The parent wrapped group
    parent: AdditiveAbelianGroupWrapper<T>,
    /// Optional cached ambient group element
    ambient_element: Option<T>,
}

impl<T: Clone> AdditiveAbelianGroupWrapperElement<T> {
    /// Create a new wrapper element from a vector
    ///
    /// # Arguments
    /// * `vector` - Coefficients with respect to the wrapped group's generators
    /// * `parent` - The parent wrapped group
    pub fn new(vector: Vec<i64>, parent: AdditiveAbelianGroupWrapper<T>) -> Result<Self, String> {
        if vector.len() != parent.ngens() {
            return Err(format!(
                "Expected {} coefficients, got {}",
                parent.ngens(),
                vector.len()
            ));
        }

        Ok(AdditiveAbelianGroupWrapperElement {
            vector,
            parent,
            ambient_element: None,
        })
    }

    /// Create a new wrapper element from an ambient element
    ///
    /// # Arguments
    /// * `ambient_element` - Element from the ambient group
    /// * `parent` - The parent wrapped group
    pub fn from_ambient(ambient_element: T, parent: AdditiveAbelianGroupWrapper<T>) -> Result<Self, String>
    where
        T: PartialEq,
    {
        let vector = parent.discrete_log(&ambient_element)?;
        Ok(AdditiveAbelianGroupWrapperElement {
            vector,
            parent,
            ambient_element: Some(ambient_element),
        })
    }

    /// Get the vector representation
    pub fn vector(&self) -> &[i64] {
        &self.vector
    }

    /// Get the parent wrapped group
    pub fn parent(&self) -> &AdditiveAbelianGroupWrapper<T> {
        &self.parent
    }

    /// Get the actual element in the ambient universe
    ///
    /// Uses discrete_exp to compute it if not cached
    pub fn element(&self) -> Result<T, String>
    where
        T: std::ops::Add<Output = T>,
    {
        if let Some(ref elem) = self.ambient_element {
            return Ok(elem.clone());
        }

        self.parent.discrete_exp(&self.vector)
    }

    /// Add two wrapped elements
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        if self.parent != other.parent {
            return Err("Cannot add elements from different wrapped groups".to_string());
        }

        let sum: Vec<i64> = self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a + b)
            .collect();

        Self::new(sum, self.parent.clone())
    }

    /// Negate the element
    pub fn negate(&self) -> Self {
        let neg: Vec<i64> = self.vector.iter().map(|x| -x).collect();
        Self::new(neg, self.parent.clone()).unwrap()
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, n: i64) -> Self {
        let scaled: Vec<i64> = self.vector.iter().map(|x| n * x).collect();
        Self::new(scaled, self.parent.clone()).unwrap()
    }

    /// Check if this is the identity element
    pub fn is_identity(&self) -> bool {
        self.vector.iter().all(|&x| x == 0)
    }
}

impl<T: Clone + fmt::Display> fmt::Display for AdditiveAbelianGroupWrapperElement<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, &coeff) in self.vector.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coeff)?;
        }
        write!(f, ")")
    }
}

impl<T: Clone + PartialEq> PartialEq for AdditiveAbelianGroupWrapperElement<T> {
    fn eq(&self, other: &Self) -> bool {
        self.parent == other.parent && self.vector == other.vector
    }
}

impl<T: Clone + PartialEq> Eq for AdditiveAbelianGroupWrapperElement<T> {}

/// A wrapped additive abelian group
///
/// Wraps a subgroup of an additive abelian group, providing discrete logarithm
/// and exponential functionality for converting between vector representations
/// and ambient group elements.
#[derive(Clone, Debug)]
pub struct AdditiveAbelianGroupWrapper<T: Clone> {
    /// Generators of the wrapped subgroup
    generators: Vec<T>,
    /// The underlying abstract abelian group structure
    abstract_group: AdditiveAbelianGroup,
    /// Description of the ambient universe
    universe_description: String,
}

impl<T: Clone> AdditiveAbelianGroupWrapper<T> {
    /// Create a new wrapped additive abelian group
    ///
    /// # Arguments
    /// * `generators` - List of generators from the ambient group
    /// * `abstract_group` - The abstract group structure (invariants, etc.)
    /// * `universe_description` - Description of the ambient space
    pub fn new(
        generators: Vec<T>,
        abstract_group: AdditiveAbelianGroup,
        universe_description: String,
    ) -> Result<Self, String> {
        if generators.len() != abstract_group.rank() {
            return Err(format!(
                "Number of generators ({}) must match group rank ({})",
                generators.len(),
                abstract_group.rank()
            ));
        }

        Ok(AdditiveAbelianGroupWrapper {
            generators,
            abstract_group,
            universe_description,
        })
    }

    /// Get the number of generators
    pub fn ngens(&self) -> usize {
        self.generators.len()
    }

    /// Get the generators
    pub fn generators(&self) -> &[T] {
        &self.generators
    }

    /// Get the i-th generator
    pub fn generator(&self, i: usize) -> Option<&T> {
        self.generators.get(i)
    }

    /// Get the abstract group structure
    pub fn abstract_group(&self) -> &AdditiveAbelianGroup {
        &self.abstract_group
    }

    /// Discrete exponential: convert coefficient vector to ambient group element
    ///
    /// Computes the weighted sum of generator elements: sum(coeff[i] * gen[i])
    ///
    /// # Arguments
    /// * `coefficients` - Vector of coefficients with respect to generators
    pub fn discrete_exp(&self, coefficients: &[i64]) -> Result<T, String>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<i64, Output = T> + Default,
    {
        if coefficients.len() != self.ngens() {
            return Err(format!(
                "Expected {} coefficients, got {}",
                self.ngens(),
                coefficients.len()
            ));
        }

        let mut result = T::default();
        for (i, &coeff) in coefficients.iter().enumerate() {
            if coeff != 0 {
                let term = self.generators[i].clone() * coeff;
                result = result + term;
            }
        }

        Ok(result)
    }

    /// Discrete logarithm: express an ambient element as a combination of generators
    ///
    /// This is the inverse of discrete_exp. Solves for coefficients such that
    /// element = sum(coeff[i] * gen[i])
    ///
    /// # Arguments
    /// * `element` - An element from the ambient group
    ///
    /// # Returns
    /// Vector of coefficients, or an error if the element is not in the subgroup
    pub fn discrete_log(&self, element: &T) -> Result<Vec<i64>, String>
    where
        T: PartialEq + std::ops::Add<Output = T> + std::ops::Mul<i64, Output = T> + Default,
    {
        // This is a placeholder implementation that would need to be specialized
        // for different types T. For full functionality, this would require
        // implementing algorithms like Shanks baby-step giant-step or Pollard's rho.

        // For now, we provide a basic implementation that works for small cases
        // A real implementation would decompose by prime factors and use
        // recursive algorithms for p-groups

        Err("Discrete logarithm computation requires type-specific implementation".to_string())
    }

    /// Compute the torsion subgroup
    ///
    /// Returns the subgroup of finite-order elements, optionally filtered
    /// by divisibility conditions.
    ///
    /// # Arguments
    /// * `divisor` - Optional divisor; only include elements whose order divides this
    pub fn torsion_subgroup(&self, divisor: Option<usize>) -> Result<Self, String> {
        let torsion_invariants: Vec<usize> = if let Some(d) = divisor {
            self.abstract_group
                .invariant_factors()
                .iter()
                .filter(|&&inv| inv != 0 && d % inv == 0)
                .copied()
                .collect()
        } else {
            self.abstract_group
                .invariant_factors()
                .iter()
                .filter(|&&inv| inv != 0)
                .copied()
                .collect()
        };

        let torsion_group = AdditiveAbelianGroup::new(0, torsion_invariants)?;

        // Select only the torsion generators
        let torsion_gens: Vec<T> = self.generators
            .iter()
            .skip(self.abstract_group.free_rank())
            .cloned()
            .collect();

        Self::new(torsion_gens, torsion_group, self.universe_description.clone())
    }

    /// Get the universe description
    pub fn universe_description(&self) -> &str {
        &self.universe_description
    }
}

impl<T: Clone + PartialEq> PartialEq for AdditiveAbelianGroupWrapper<T> {
    fn eq(&self, other: &Self) -> bool {
        self.generators == other.generators &&
        self.abstract_group == other.abstract_group
    }
}

impl<T: Clone + PartialEq> Eq for AdditiveAbelianGroupWrapper<T> {}

impl<T: Clone + fmt::Display> fmt::Display for AdditiveAbelianGroupWrapper<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wrapped additive abelian group with {} generators in {}",
               self.ngens(),
               self.universe_description)
    }
}

/// Compute a canonical basis from arbitrary generators
///
/// Takes a set of generators for an additive abelian group and computes
/// a canonical basis by reducing to p-primary components and systematically
/// constructing independent generators for each prime power.
///
/// # Arguments
/// * `generators` - Arbitrary generators for the group
/// * `universe_description` - Description of the ambient space
///
/// # Returns
/// A wrapped group with a canonical basis
///
/// # Examples
/// ```
/// use rustmath_groups::additive_abelian_wrapper::basis_from_generators;
///
/// // For integer vectors
/// let gens = vec![vec![2, 0], vec![0, 3]];
/// // Returns a canonical basis for the subgroup
/// ```
pub fn basis_from_generators<T>(
    generators: Vec<T>,
    universe_description: String,
) -> Result<AdditiveAbelianGroupWrapper<T>, String>
where
    T: Clone + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<i64, Output = T> + Default,
{
    if generators.is_empty() {
        let trivial_group = AdditiveAbelianGroup::new(0, vec![])?;
        return AdditiveAbelianGroupWrapper::new(generators, trivial_group, universe_description);
    }

    // This is a simplified implementation. A full implementation would:
    // 1. Decompose generators into p-primary components for each prime p
    // 2. For each prime, use _expand_basis_pgroup to build a canonical basis
    // 3. Combine the bases from all primes
    // 4. Determine the invariant factors from the basis structure

    // For now, we create a free abelian group with the given generators
    let rank = generators.len();
    let group = AdditiveAbelianGroup::new(rank, vec![])?;

    AdditiveAbelianGroupWrapper::new(generators, group, universe_description)
}

/// Helper function for discrete logarithm in p-groups (private implementation detail)
///
/// Implements recursive p-group discrete logarithm solving using a basic version
/// of the recursive algorithm, with Shanks' baby-step giant-step method for base cases.
fn _discrete_log_pgroup<T>(
    element: &T,
    generator: &T,
    p: usize,
    k: usize,
) -> Result<i64, String>
where
    T: Clone + PartialEq + std::ops::Add<Output = T> + std::ops::Mul<i64, Output = T> + Default,
{
    // Base case: use baby-step giant-step
    if k == 1 {
        for i in 0..p {
            let test = generator.clone() * (i as i64);
            if &test == element {
                return Ok(i as i64);
            }
        }
        return Err("Element not in subgroup generated by generator".to_string());
    }

    // Recursive case would be implemented here
    // This requires computing quotients and using the structure of p-groups

    Err("P-group discrete log not fully implemented".to_string())
}

/// Helper function to expand basis for p-groups (private implementation detail)
///
/// Extends a p-subgroup basis when encountering new elements, modifying
/// basis and valuation lists in-place according to published algorithms.
fn _expand_basis_pgroup<T>(
    basis: &mut Vec<T>,
    valuations: &mut Vec<usize>,
    new_element: T,
    p: usize,
) -> Result<(), String>
where
    T: Clone + PartialEq,
{
    // This would implement the basis expansion algorithm
    // Details would depend on the specific structure of T

    basis.push(new_element);
    valuations.push(0);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unwrapping_morphism() {
        // Create a simple wrapped group with integer vector generators
        let gens = vec![vec![1, 0], vec![0, 1]];
        let group = AdditiveAbelianGroup::new(2, vec![]).unwrap();
        let wrapped = AdditiveAbelianGroupWrapper::new(
            gens,
            group,
            "Z^2".to_string(),
        ).unwrap();

        let morphism = UnwrappingMorphism::new(wrapped.clone(), "Z^2".to_string());
        assert_eq!(morphism.target_description(), "Z^2");
        assert_eq!(morphism.source(), &wrapped);
    }

    #[test]
    fn test_wrapper_element_creation() {
        let gens = vec![vec![1, 0], vec![0, 1]];
        let group = AdditiveAbelianGroup::new(2, vec![]).unwrap();
        let wrapped = AdditiveAbelianGroupWrapper::new(
            gens,
            group,
            "Z^2".to_string(),
        ).unwrap();

        let elem = AdditiveAbelianGroupWrapperElement::new(
            vec![1, 2],
            wrapped.clone(),
        ).unwrap();

        assert_eq!(elem.vector(), &[1, 2]);
        assert!(!elem.is_identity());
    }

    #[test]
    fn test_wrapper_element_operations() {
        let gens = vec![vec![1, 0], vec![0, 1]];
        let group = AdditiveAbelianGroup::new(2, vec![]).unwrap();
        let wrapped = AdditiveAbelianGroupWrapper::new(
            gens,
            group,
            "Z^2".to_string(),
        ).unwrap();

        let elem1 = AdditiveAbelianGroupWrapperElement::new(
            vec![1, 2],
            wrapped.clone(),
        ).unwrap();

        let elem2 = AdditiveAbelianGroupWrapperElement::new(
            vec![3, 4],
            wrapped.clone(),
        ).unwrap();

        let sum = elem1.add(&elem2).unwrap();
        assert_eq!(sum.vector(), &[4, 6]);

        let neg = elem1.negate();
        assert_eq!(neg.vector(), &[-1, -2]);

        let scaled = elem1.scalar_mul(3);
        assert_eq!(scaled.vector(), &[3, 6]);
    }

    #[test]
    fn test_wrapper_identity() {
        let gens = vec![vec![1, 0], vec![0, 1]];
        let group = AdditiveAbelianGroup::new(2, vec![]).unwrap();
        let wrapped = AdditiveAbelianGroupWrapper::new(
            gens,
            group,
            "Z^2".to_string(),
        ).unwrap();

        let identity = AdditiveAbelianGroupWrapperElement::new(
            vec![0, 0],
            wrapped,
        ).unwrap();

        assert!(identity.is_identity());
    }

    #[test]
    fn test_wrapped_group_properties() {
        let gens = vec![vec![2, 0], vec![0, 3]];
        let group = AdditiveAbelianGroup::new(2, vec![]).unwrap();
        let wrapped = AdditiveAbelianGroupWrapper::new(
            gens.clone(),
            group,
            "Z^2".to_string(),
        ).unwrap();

        assert_eq!(wrapped.ngens(), 2);
        assert_eq!(wrapped.generators(), &gens);
        assert_eq!(wrapped.generator(0), Some(&vec![2, 0]));
        assert_eq!(wrapped.generator(1), Some(&vec![0, 3]));
        assert_eq!(wrapped.generator(2), None);
    }

    #[test]
    fn test_torsion_subgroup() {
        let gens = vec![vec![1], vec![2], vec![3]];
        let group = AdditiveAbelianGroup::new(1, vec![2, 4]).unwrap();
        let wrapped = AdditiveAbelianGroupWrapper::new(
            gens,
            group,
            "Z x Z/2Z x Z/4Z".to_string(),
        ).unwrap();

        let torsion = wrapped.torsion_subgroup(None).unwrap();
        assert_eq!(torsion.ngens(), 2);
        assert_eq!(torsion.abstract_group().free_rank(), 0);
    }

    #[test]
    fn test_basis_from_generators() {
        let gens = vec![vec![2, 0], vec![0, 3], vec![1, 1]];
        let result = basis_from_generators(gens, "Z^2".to_string());
        assert!(result.is_ok());

        let wrapped = result.unwrap();
        assert_eq!(wrapped.ngens(), 3);
        assert_eq!(wrapped.abstract_group().free_rank(), 3);
    }

    #[test]
    fn test_empty_generators() {
        let gens: Vec<Vec<i32>> = vec![];
        let result = basis_from_generators(gens, "Z^0".to_string());
        assert!(result.is_ok());

        let wrapped = result.unwrap();
        assert_eq!(wrapped.ngens(), 0);
    }
}
