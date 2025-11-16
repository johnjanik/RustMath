//! Nil Coxeter Algebra
//!
//! The Nil Coxeter algebra (also called 0-Hecke algebra) is an algebraic structure
//! associated with a Coxeter group or Weyl group. It is obtained as a specialization
//! of the Iwahori-Hecke algebra.
//!
//! # Mathematical Background
//!
//! For a Weyl group W with simple reflections s_1, ..., s_n, the Nil Coxeter algebra
//! has generators u_1, ..., u_n satisfying:
//!
//! 1. **Quadratic relations**: u_i^2 = 0 for all i (nilpotent property)
//! 2. **Braid relations**: The same braid relations as the Weyl group
//!
//! These generators correspond to nodes of the Dynkin diagram.
//!
//! ## Braid Relations
//!
//! For generators u_i and u_j:
//! - If nodes i and j are not connected: u_i u_j = u_j u_i
//! - If nodes i and j are connected with order m_{ij}: (u_i u_j)^{m_{ij}} = (u_j u_i)^{m_{ij}}
//!
//! ## Connection to Iwahori-Hecke Algebra
//!
//! The Nil Coxeter algebra is the specialization of the Iwahori-Hecke algebra
//! H(W, q) at q = 0. This explains the "0-Hecke" alternative name.
//!
//! # Applications
//!
//! - Noncommutative symmetric functions
//! - K-theory of flag varieties
//! - Schubert calculus
//! - Homogeneous functions and k-Schur functions
//!
//! # Examples
//!
//! ```
//! use rustmath_liealgebras::{NilCoxeterAlgebra, CartanType, CartanLetter};
//! use rustmath_rationals::Rational;
//!
//! // Create Nil Coxeter algebra for type A_2
//! let ct = CartanType::new(CartanLetter::A, 2).unwrap();
//! let nca = NilCoxeterAlgebra::new(ct, Rational::one());
//!
//! assert_eq!(nca.rank(), 2); // Two generators u_1, u_2
//! ```
//!
//! # References
//!
//! - Norton, P. "0-Hecke algebras" (1979)
//! - Krob, D. & Thibon, J.-Y. "Noncommutative symmetric functions IV: Quantum linear groups
//!   and Hecke algebras at q=0" (1997)
//! - SageMath: sage.algebras.nil_coxeter_algebra
//!
//! Corresponds to sage.algebras.nil_coxeter_algebra

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use crate::cartan_type::CartanType;
use crate::weyl_group::WeylGroup;

/// Nil Coxeter Algebra (0-Hecke Algebra)
///
/// An algebra with generators u_i corresponding to simple reflections of a Weyl group,
/// satisfying u_i^2 = 0 (nilpotent) and the braid relations.
///
/// # Type Parameters
///
/// * `R` - The base ring (typically Q, the rational numbers)
///
/// # Examples
///
/// ```
/// use rustmath_liealgebras::{NilCoxeterAlgebra, CartanType, CartanLetter};
///
/// // Create algebra for type A_2
/// let ct = CartanType::new(CartanLetter::A, 2).unwrap();
/// let nca = NilCoxeterAlgebra::new(ct, 1i64);
///
/// assert_eq!(nca.rank(), 2);
/// ```
#[derive(Clone)]
pub struct NilCoxeterAlgebra<R: Ring> {
    /// The Cartan type defining the Dynkin diagram
    cartan_type: CartanType,
    /// The base ring
    base_ring: R,
    /// The associated Weyl group
    weyl_group: WeylGroup,
    /// Prefix for generator names
    prefix: String,
}

impl<R: Ring + Clone> NilCoxeterAlgebra<R> {
    /// Create a new Nil Coxeter algebra
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The Cartan type (defines the Dynkin diagram)
    /// * `base_ring` - The base ring (typically rational numbers)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_liealgebras::{NilCoxeterAlgebra, CartanType, CartanLetter};
    ///
    /// let ct = CartanType::new(CartanLetter::A, 3).unwrap();
    /// let nca = NilCoxeterAlgebra::new(ct, 1i64);
    /// ```
    pub fn new(cartan_type: CartanType, base_ring: R) -> Self {
        let weyl_group = WeylGroup::new(cartan_type.clone());

        NilCoxeterAlgebra {
            cartan_type,
            base_ring,
            weyl_group,
            prefix: "u".to_string(),
        }
    }

    /// Create with custom generator prefix
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The Cartan type
    /// * `base_ring` - The base ring
    /// * `prefix` - Prefix for generator names (e.g., "u" gives u_1, u_2, ...)
    pub fn with_prefix(cartan_type: CartanType, base_ring: R, prefix: String) -> Self {
        let weyl_group = WeylGroup::new(cartan_type.clone());

        NilCoxeterAlgebra {
            cartan_type,
            base_ring,
            weyl_group,
            prefix,
        }
    }

    /// Get the rank (number of generators)
    ///
    /// This equals the rank of the root system.
    pub fn rank(&self) -> usize {
        self.cartan_type.rank()
    }

    /// Get the base ring
    pub fn base_ring(&self) -> &R {
        &self.base_ring
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &CartanType {
        &self.cartan_type
    }

    /// Get the Weyl group
    pub fn weyl_group(&self) -> &WeylGroup {
        &self.weyl_group
    }

    /// Get the i-th generator name
    ///
    /// Returns a name like "u_1", "u_2", etc.
    pub fn generator_name(&self, i: usize) -> String {
        format!("{}_{}", self.prefix, i + 1)
    }

    /// Get all generator names
    pub fn generator_names(&self) -> Vec<String> {
        (0..self.rank())
            .map(|i| self.generator_name(i))
            .collect()
    }

    /// Create a generator element
    ///
    /// Returns the i-th generator u_i as an element of the algebra.
    ///
    /// # Arguments
    ///
    /// * `i` - The generator index (0-indexed)
    ///
    /// # Returns
    ///
    /// The generator u_i, or None if index is out of range
    pub fn generator(&self, i: usize) -> Option<NilCoxeterAlgebraElement<R>>
    where
        R: From<i64>,
    {
        if i < self.rank() {
            let mut terms = HashMap::new();
            terms.insert(vec![i], R::from(1));
            Some(NilCoxeterAlgebraElement::new(terms))
        } else {
            None
        }
    }

    /// Get the zero element
    pub fn zero(&self) -> NilCoxeterAlgebraElement<R> {
        NilCoxeterAlgebraElement::zero()
    }

    /// Get the identity element (1)
    pub fn one(&self) -> NilCoxeterAlgebraElement<R>
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(vec![], R::from(1)); // Empty word represents 1
        NilCoxeterAlgebraElement::new(terms)
    }

    /// Check if two generators commute
    ///
    /// Generators u_i and u_j commute if nodes i and j are not adjacent in the Dynkin diagram.
    /// For now, this is a placeholder - a full implementation would check the Cartan matrix.
    pub fn generators_commute(&self, i: usize, j: usize) -> bool {
        // Simple implementation: only equal indices commute
        // A more complete implementation would check the Cartan matrix
        // to see if A[i,j] = 0 (nodes not connected)
        i == j
    }

    /// Get the braid relation order between two generators
    ///
    /// Returns m_{ij} such that (u_i u_j)^{m_{ij}} = (u_j u_i)^{m_{ij}}
    /// For now, this is a placeholder returning default values.
    pub fn braid_order(&self, i: usize, j: usize) -> usize {
        if i == j {
            2 // u_i^2 = 0, but in braid sense it's order 2
        } else {
            // Default to 3 for now
            // A full implementation would compute this from the Cartan matrix
            3
        }
    }
}

/// Element of a Nil Coxeter algebra
///
/// Elements are represented as linear combinations of words in the generators u_i,
/// subject to the relations u_i^2 = 0 and the braid relations.
///
/// # Representation
///
/// An element is stored as a HashMap mapping words (sequences of generator indices)
/// to coefficients. For example:
/// - `{[]: 1}` represents the identity element 1
/// - `{[0]: 1}` represents u_1
/// - `{[0, 1]: 2}` represents 2*u_1*u_2
///
/// # Type Parameters
///
/// * `R` - The base ring
#[derive(Clone, Debug)]
pub struct NilCoxeterAlgebraElement<R: Ring> {
    /// Terms: map from word (sequence of indices) to coefficient
    terms: HashMap<Vec<usize>, R>,
}

impl<R: Ring + Clone> NilCoxeterAlgebraElement<R> {
    /// Create a new element
    pub fn new(terms: HashMap<Vec<usize>, R>) -> Self {
        NilCoxeterAlgebraElement { terms }
    }

    /// Create the zero element
    pub fn zero() -> Self {
        NilCoxeterAlgebraElement {
            terms: HashMap::new(),
        }
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<Vec<usize>, R> {
        &self.terms
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.terms.is_empty() || self.terms.values().all(|c| c.is_zero())
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + PartialEq,
    {
        let mut result = self.terms.clone();

        for (word, coeff) in &other.terms {
            let new_coeff = if let Some(existing) = result.get(word) {
                existing.clone() + coeff.clone()
            } else {
                coeff.clone()
            };

            if !new_coeff.is_zero() {
                result.insert(word.clone(), new_coeff);
            } else {
                result.remove(word);
            }
        }

        NilCoxeterAlgebraElement { terms: result }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: std::ops::Mul<Output = R> + PartialEq,
    {
        if scalar.is_zero() {
            return Self::zero();
        }

        let terms = self
            .terms
            .iter()
            .map(|(word, coeff)| (word.clone(), coeff.clone() * scalar.clone()))
            .collect();

        NilCoxeterAlgebraElement { terms }
    }

    /// Negate the element
    pub fn negate(&self) -> Self
    where
        R: std::ops::Neg<Output = R>,
    {
        let terms = self
            .terms
            .iter()
            .map(|(word, coeff)| (word.clone(), -coeff.clone()))
            .collect();

        NilCoxeterAlgebraElement { terms }
    }

    /// Multiply two elements
    ///
    /// This requires implementing the braid relations and u_i^2 = 0.
    /// For now, this is a placeholder that performs naive concatenation.
    pub fn multiply(&self, _other: &Self) -> Self
    where
        R: std::ops::Mul<Output = R> + std::ops::Add<Output = R> + PartialEq + From<i64>,
    {
        // Full multiplication would:
        // 1. Multiply each pair of words by concatenation
        // 2. Apply u_i^2 = 0 (if a word contains repeated indices, result is 0)
        // 3. Apply braid relations to put words in normal form
        //
        // For now, return zero as placeholder
        Self::zero()
    }
}

impl<R: Ring + Clone + Display> Display for NilCoxeterAlgebraElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let mut first = true;
        for (word, coeff) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if word.is_empty() {
                // Identity element
                write!(f, "{}", coeff)?;
            } else if coeff.is_one() {
                write!(f, "u_{}", word.iter().map(|i| (i + 1).to_string()).collect::<Vec<_>>().join("*u_"))?;
            } else {
                write!(f, "{}*u_{}", coeff, word.iter().map(|i| (i + 1).to_string()).collect::<Vec<_>>().join("*u_"))?;
            }
        }

        Ok(())
    }
}

impl<R: Ring + Clone + Display> Display for NilCoxeterAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Nil Coxeter algebra of type {} over {}",
            self.cartan_type, self.base_ring
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartan_type::CartanLetter;

    #[test]
    fn test_nil_coxeter_creation() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let nca: NilCoxeterAlgebra<i64> = NilCoxeterAlgebra::new(ct, 1);

        assert_eq!(nca.rank(), 2);
    }

    #[test]
    fn test_generator_names() {
        let ct = CartanType::new(CartanLetter::A, 3).unwrap();
        let nca: NilCoxeterAlgebra<i64> = NilCoxeterAlgebra::new(ct, 1);

        let names = nca.generator_names();
        assert_eq!(names, vec!["u_1", "u_2", "u_3"]);
    }

    #[test]
    fn test_generator_creation() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let nca: NilCoxeterAlgebra<i64> = NilCoxeterAlgebra::new(ct, 1);

        let u1 = nca.generator(0);
        assert!(u1.is_some());

        let u3 = nca.generator(2);
        assert!(u3.is_none());
    }

    #[test]
    fn test_zero_and_one() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let nca: NilCoxeterAlgebra<i64> = NilCoxeterAlgebra::new(ct, 1);

        let zero = nca.zero();
        assert!(zero.is_zero());

        let one = nca.one();
        assert!(!one.is_zero());
    }

    #[test]
    fn test_element_addition() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let nca: NilCoxeterAlgebra<i64> = NilCoxeterAlgebra::new(ct, 1);

        let u1 = nca.generator(0).unwrap();
        let u2 = nca.generator(1).unwrap();

        let sum = u1.add(&u2);
        assert_eq!(sum.terms().len(), 2);
    }

    #[test]
    fn test_scalar_multiplication() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let nca: NilCoxeterAlgebra<i64> = NilCoxeterAlgebra::new(ct, 1);

        let u1 = nca.generator(0).unwrap();
        let scaled = u1.scalar_mul(&3);

        assert_eq!(scaled.terms().len(), 1);
    }

    #[test]
    fn test_display() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let nca: NilCoxeterAlgebra<i64> = NilCoxeterAlgebra::new(ct, 1);

        let display = format!("{}", nca);
        assert!(display.contains("Nil Coxeter algebra"));
        assert!(display.contains("A2"));
    }

    #[test]
    fn test_custom_prefix() {
        let ct = CartanType::new(CartanLetter::B, 2).unwrap();
        let nca: NilCoxeterAlgebra<i64> = NilCoxeterAlgebra::with_prefix(ct, 1, "v".to_string());

        assert_eq!(nca.generator_name(0), "v_1");
        assert_eq!(nca.generator_name(1), "v_2");
    }

    #[test]
    fn test_element_display() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let nca: NilCoxeterAlgebra<i64> = NilCoxeterAlgebra::new(ct, 1);

        let u1 = nca.generator(0).unwrap();
        let display = format!("{}", u1);
        assert!(display.contains("u_1"));
    }
}
