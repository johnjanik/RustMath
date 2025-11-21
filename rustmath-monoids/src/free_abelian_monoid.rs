//! # Free Abelian Monoids
//!
//! This module provides free abelian monoids and their elements.
//!
//! A free abelian monoid is a commutative monoid where each element can be
//! represented as a formal product of generators with non-negative integer exponents.
//!
//! ## Implementation Notes
//!
//! This implementation uses `Vec<Integer>` for dense exponent representation, where
//! the i-th element of the vector stores the exponent of the i-th generator.
//! This is in contrast to a sparse representation (using HashMap) which would be
//! more memory-efficient for elements with few non-zero exponents.
//!
//! ## Relationship to rustmath-monoids Architecture
//!
//! - Follows the pattern established by `FreeMonoid` in the `free_monoid` module
//! - Unlike `FreeMonoid` which uses word concatenation, this uses exponent addition
//! - The `Monoid` trait from the `monoid` module provides the algebraic structure
//! - Elements are represented densely with `Vec<Integer>` for exact arithmetic
//!
//! ## Comparison with SageMath
//!
//! This corresponds to `sage.monoids.free_abelian_monoid.FreeAbelianMonoid_class`
//! and related functionality in SageMath.

use rustmath_integers::Integer;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};

/// An element of a free abelian monoid
///
/// Elements are represented as vectors of exponents, where the i-th entry
/// is the exponent of the i-th generator. This is a dense representation.
///
/// # Example
/// ```
/// use rustmath_monoids::FreeAbelianMonoidElement;
/// use rustmath_integers::Integer;
///
/// // Create an element representing x_0^2 * x_1^3
/// let exps = vec![Integer::from(2), Integer::from(3)];
/// let elem = FreeAbelianMonoidElement::new(exps);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeAbelianMonoidElement {
    /// Exponents for each generator (dense representation)
    /// The i-th element is the exponent of the i-th generator
    exponents: Vec<Integer>,
}

impl FreeAbelianMonoidElement {
    /// Create a new element from exponents
    ///
    /// # Arguments
    /// * `exponents` - Vector of exponents, one for each generator
    pub fn new(exponents: Vec<Integer>) -> Self {
        FreeAbelianMonoidElement { exponents }
    }

    /// Create the identity element (all exponents zero)
    ///
    /// # Arguments
    /// * `rank` - Number of generators
    pub fn identity(rank: usize) -> Self {
        FreeAbelianMonoidElement {
            exponents: vec![Integer::from(0); rank],
        }
    }

    /// Create an identity element with minimal representation
    pub fn identity_minimal() -> Self {
        FreeAbelianMonoidElement {
            exponents: Vec::new(),
        }
    }

    /// Get the exponents
    pub fn exponents(&self) -> &[Integer] {
        &self.exponents
    }

    /// Get the exponent for a specific generator
    ///
    /// Returns 0 if the index is out of bounds
    pub fn get_exponent(&self, gen: usize) -> Integer {
        self.exponents.get(gen).cloned().unwrap_or_else(|| Integer::from(0))
    }

    /// Check if this is the identity (all exponents are zero)
    pub fn is_identity(&self) -> bool {
        self.exponents.iter().all(|e| e.is_zero())
    }

    /// Get the rank (number of generators)
    pub fn rank(&self) -> usize {
        self.exponents.len()
    }

    /// Multiply two elements (add exponents componentwise)
    ///
    /// If elements have different ranks, the result will have the maximum rank
    /// with missing exponents treated as zero.
    pub fn mul(&self, other: &Self) -> Self {
        let max_len = self.exponents.len().max(other.exponents.len());
        let mut new_exponents = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let exp1 = self.get_exponent(i);
            let exp2 = other.get_exponent(i);
            new_exponents.push(exp1 + exp2);
        }

        FreeAbelianMonoidElement {
            exponents: new_exponents,
        }
    }

    /// Monoid composition (same as multiplication for abelian monoids)
    pub fn compose(&self, other: &Self) -> Self {
        self.mul(other)
    }

    /// Compute a power of this element (multiply exponents by n)
    ///
    /// # Arguments
    /// * `n` - The power to raise to (must be non-negative for monoids)
    pub fn pow(&self, n: &Integer) -> Self {
        if n.is_zero() {
            return Self::identity(self.exponents.len());
        }

        let new_exponents: Vec<Integer> = self
            .exponents
            .iter()
            .map(|exp| exp.clone() * n.clone())
            .collect();

        FreeAbelianMonoidElement {
            exponents: new_exponents,
        }
    }

    /// Total degree (sum of all exponents)
    pub fn degree(&self) -> Integer {
        self.exponents
            .iter()
            .fold(Integer::from(0), |acc, exp| acc + exp.clone())
    }

    /// Support (indices of generators with non-zero exponents)
    pub fn support(&self) -> Vec<usize> {
        self.exponents
            .iter()
            .enumerate()
            .filter(|(_, exp)| !exp.is_zero())
            .map(|(i, _)| i)
            .collect()
    }

    /// Normalize the representation by removing trailing zeros
    pub fn normalize(&mut self) {
        while let Some(last) = self.exponents.last() {
            if last.is_zero() {
                self.exponents.pop();
            } else {
                break;
            }
        }
    }

    /// Create a normalized copy
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }

    /// Resize the exponent vector to match a given rank
    pub fn resize(&mut self, rank: usize) {
        self.exponents.resize(rank, Integer::from(0));
    }

    /// Create a resized copy
    pub fn resized(&self, rank: usize) -> Self {
        let mut result = self.clone();
        result.resize(rank);
        result
    }
}

impl Hash for FreeAbelianMonoidElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the normalized form to ensure x^1*y^0 and x^1 hash the same
        self.normalized().exponents.hash(state);
    }
}

/// A free abelian monoid on a set of generators
///
/// This corresponds to SageMath's `FreeAbelianMonoid_class`.
///
/// # Example
/// ```
/// use rustmath_monoids::FreeAbelianMonoid;
///
/// let M = FreeAbelianMonoid::new(vec!["x".to_string(), "y".to_string()]);
/// let x = M.gen(0).unwrap();
/// let y = M.gen(1).unwrap();
/// let xy = x.mul(&y);
/// assert_eq!(M.element_to_string(&xy), "x*y");
/// ```
#[derive(Debug, Clone)]
pub struct FreeAbelianMonoid {
    /// Number of generators (rank of the monoid)
    rank: usize,
    /// Names of generators
    generator_names: Vec<String>,
}

impl FreeAbelianMonoid {
    /// Create a new free abelian monoid with named generators
    ///
    /// # Arguments
    /// * `generators` - Names for the generators
    ///
    /// # Example
    /// ```
    /// use rustmath_monoids::FreeAbelianMonoid;
    /// let M = FreeAbelianMonoid::new(vec!["a".to_string(), "b".to_string()]);
    /// ```
    pub fn new(generators: Vec<String>) -> Self {
        let rank = generators.len();
        FreeAbelianMonoid {
            rank,
            generator_names: generators,
        }
    }

    /// Create a new free abelian monoid with n generators (x_0, x_1, ..., x_{n-1})
    ///
    /// # Arguments
    /// * `n` - Number of generators (rank)
    pub fn with_rank(n: usize) -> Self {
        let generator_names = (0..n).map(|i| format!("x_{}", i)).collect();
        FreeAbelianMonoid {
            rank: n,
            generator_names,
        }
    }

    /// Get the rank (number of generators)
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the number of generators (alias for rank)
    pub fn ngens(&self) -> usize {
        self.rank
    }

    /// Get the generator names
    pub fn generators(&self) -> &[String] {
        &self.generator_names
    }

    /// Get a generator name
    pub fn generator_name(&self, index: usize) -> Option<&str> {
        self.generator_names.get(index).map(|s| s.as_str())
    }

    /// Create the identity element
    pub fn identity(&self) -> FreeAbelianMonoidElement {
        FreeAbelianMonoidElement::identity(self.rank)
    }

    /// Create the identity element (alternative method name for compatibility)
    pub fn one(&self) -> FreeAbelianMonoidElement {
        self.identity()
    }

    /// Create a generator element
    ///
    /// Returns None if the index is out of bounds
    ///
    /// # Arguments
    /// * `index` - Index of the generator (0-based)
    pub fn gen(&self, index: usize) -> Option<FreeAbelianMonoidElement> {
        if index < self.rank {
            let mut exponents = vec![Integer::from(0); self.rank];
            exponents[index] = Integer::from(1);
            Some(FreeAbelianMonoidElement::new(exponents))
        } else {
            None
        }
    }

    /// Get all generators as a vector
    pub fn gens(&self) -> Vec<FreeAbelianMonoidElement> {
        (0..self.rank).filter_map(|i| self.gen(i)).collect()
    }

    /// Create an element from exponents
    ///
    /// The exponents vector should have length equal to the rank of the monoid.
    /// If it's shorter, missing exponents are treated as zero.
    /// If it's longer, extra exponents are ignored.
    pub fn element(&self, exponents: Vec<Integer>) -> FreeAbelianMonoidElement {
        let mut elem = FreeAbelianMonoidElement::new(exponents);
        elem.resize(self.rank);
        elem
    }

    /// Create an element from integer exponents
    pub fn element_from_ints(&self, exponents: Vec<i64>) -> FreeAbelianMonoidElement {
        let int_exps: Vec<Integer> = exponents.into_iter().map(Integer::from).collect();
        self.element(int_exps)
    }

    /// Check if an element belongs to this monoid
    pub fn contains(&self, elem: &FreeAbelianMonoidElement) -> bool {
        // All elements with non-negative exponents belong to the monoid
        elem.rank() <= self.rank && elem.exponents().iter().all(|e| e.signum() >= 0)
    }

    /// Format an element as a string
    pub fn element_to_string(&self, elem: &FreeAbelianMonoidElement) -> String {
        if elem.is_identity() {
            return "1".to_string();
        }

        let support = elem.support();
        if support.is_empty() {
            return "1".to_string();
        }

        let terms: Vec<String> = support
            .iter()
            .map(|&i| {
                let exp = elem.get_exponent(i);
                let name = if i < self.generator_names.len() {
                    &self.generator_names[i]
                } else {
                    "x"
                };

                if exp == Integer::from(1) {
                    name.to_string()
                } else {
                    format!("{}^{}", name, exp)
                }
            })
            .collect();

        terms.join("*")
    }

    /// Check if this monoid is abelian (always true for free abelian monoids)
    pub fn is_abelian(&self) -> bool {
        true
    }

    /// Check if this monoid is commutative (always true for free abelian monoids)
    pub fn is_commutative(&self) -> bool {
        true
    }

    /// Check if this monoid is finite (always false for free abelian monoids with rank > 0)
    pub fn is_finite(&self) -> bool {
        self.rank == 0
    }

    /// Get the order of the monoid (None for infinite monoids)
    pub fn order(&self) -> Option<Integer> {
        if self.rank == 0 {
            Some(Integer::from(1))
        } else {
            None
        }
    }
}

/// Factory function for creating free abelian monoids
///
/// This corresponds to SageMath's `FreeAbelianMonoid` factory function.
pub struct FreeAbelianMonoidFactory;

impl FreeAbelianMonoidFactory {
    pub fn create(generators: Vec<String>) -> FreeAbelianMonoid {
        FreeAbelianMonoid::new(generators)
    }
}

/// Create a free abelian monoid (SageMath compatibility)
///
/// This function provides an interface compatible with SageMath's
/// `FreeAbelianMonoid_class` constructor.
pub fn free_abelian_monoid_class(generators: Vec<String>) -> FreeAbelianMonoid {
    FreeAbelianMonoid::new(generators)
}

/// Check if an object is a free abelian monoid
///
/// In Rust's type system, if we have a `FreeAbelianMonoid` reference,
/// it is always a valid free abelian monoid.
pub fn is_free_abelian_monoid(_obj: &FreeAbelianMonoid) -> bool {
    true
}

/// Check if an object is a free abelian monoid element
///
/// In Rust's type system, if we have a `FreeAbelianMonoidElement` reference,
/// it is always a valid free abelian monoid element.
pub fn is_free_abelian_monoid_element(_obj: &FreeAbelianMonoidElement) -> bool {
    true
}

impl Display for FreeAbelianMonoid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Free abelian monoid on {} generators [{}]",
            self.rank,
            self.generator_names.join(", ")
        )
    }
}

impl Display for FreeAbelianMonoidElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            write!(f, "1")
        } else {
            let terms: Vec<String> = self
                .support()
                .iter()
                .map(|&i| {
                    let exp = self.get_exponent(i);
                    if exp == Integer::from(1) {
                        format!("x_{}", i)
                    } else {
                        format!("x_{}^{}", i, exp)
                    }
                })
                .collect();
            write!(f, "{}", terms.join("*"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_abelian_monoid_creation() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(M.rank(), 2);
        assert_eq!(M.ngens(), 2);
    }

    #[test]
    fn test_with_rank() {
        let M = FreeAbelianMonoid::with_rank(3);
        assert_eq!(M.rank(), 3);
        assert_eq!(M.generator_name(0), Some("x_0"));
        assert_eq!(M.generator_name(1), Some("x_1"));
        assert_eq!(M.generator_name(2), Some("x_2"));
    }

    #[test]
    fn test_identity() {
        let M = FreeAbelianMonoid::new(vec!["a".to_string(), "b".to_string()]);
        let id = M.identity();
        assert!(id.is_identity());
        assert_eq!(id.rank(), 2);
        assert_eq!(id.degree(), Integer::from(0));
    }

    #[test]
    fn test_one() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string()]);
        let one = M.one();
        assert!(one.is_identity());
    }

    #[test]
    fn test_generator() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string(), "y".to_string()]);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        assert_eq!(x.get_exponent(0), Integer::from(1));
        assert_eq!(x.get_exponent(1), Integer::from(0));
        assert_eq!(y.get_exponent(0), Integer::from(0));
        assert_eq!(y.get_exponent(1), Integer::from(1));
    }

    #[test]
    fn test_gen_out_of_bounds() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string()]);
        assert!(M.gen(0).is_some());
        assert!(M.gen(1).is_none());
    }

    #[test]
    fn test_gens() {
        let M = FreeAbelianMonoid::with_rank(3);
        let generators = M.gens();
        assert_eq!(generators.len(), 3);
        assert_eq!(generators[0].get_exponent(0), Integer::from(1));
        assert_eq!(generators[1].get_exponent(1), Integer::from(1));
        assert_eq!(generators[2].get_exponent(2), Integer::from(1));
    }

    #[test]
    fn test_multiplication() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string(), "y".to_string()]);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        let xy = x.mul(&y);
        assert_eq!(xy.get_exponent(0), Integer::from(1));
        assert_eq!(xy.get_exponent(1), Integer::from(1));

        // Commutativity
        let yx = y.mul(&x);
        assert_eq!(xy, yx);
    }

    #[test]
    fn test_composition() {
        let M = FreeAbelianMonoid::new(vec!["a".to_string(), "b".to_string()]);
        let a = M.gen(0).unwrap();
        let b = M.gen(1).unwrap();

        let ab = a.compose(&b);
        assert_eq!(ab.get_exponent(0), Integer::from(1));
        assert_eq!(ab.get_exponent(1), Integer::from(1));
    }

    #[test]
    fn test_associativity() {
        let M = FreeAbelianMonoid::with_rank(3);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();
        let z = M.gen(2).unwrap();

        let xy_z = x.mul(&y).mul(&z);
        let x_yz = x.mul(&y.mul(&z));
        assert_eq!(xy_z, x_yz);
    }

    #[test]
    fn test_identity_element() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string()]);
        let x = M.gen(0).unwrap();
        let id = M.identity();

        let x_id = x.mul(&id);
        let id_x = id.mul(&x);

        assert_eq!(x, x_id);
        assert_eq!(x, id_x);
    }

    #[test]
    fn test_pow() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string()]);
        let x = M.gen(0).unwrap();

        let x3 = x.pow(&Integer::from(3));
        assert_eq!(x3.get_exponent(0), Integer::from(3));

        let x0 = x.pow(&Integer::from(0));
        assert!(x0.is_identity());
    }

    #[test]
    fn test_pow_multiple_generators() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string(), "y".to_string()]);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        let x2y3 = x.pow(&Integer::from(2)).mul(&y.pow(&Integer::from(3)));
        let result = x2y3.pow(&Integer::from(2));

        assert_eq!(result.get_exponent(0), Integer::from(4));
        assert_eq!(result.get_exponent(1), Integer::from(6));
    }

    #[test]
    fn test_degree() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string(), "y".to_string()]);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        let x2y3 = x.pow(&Integer::from(2)).mul(&y.pow(&Integer::from(3)));
        assert_eq!(x2y3.degree(), Integer::from(5));

        let id = M.identity();
        assert_eq!(id.degree(), Integer::from(0));
    }

    #[test]
    fn test_support() {
        let M = FreeAbelianMonoid::with_rank(5);
        let x0 = M.gen(0).unwrap();
        let x2 = M.gen(2).unwrap();
        let x4 = M.gen(4).unwrap();

        let elem = x0.mul(&x2).mul(&x4);
        let support = elem.support();

        assert_eq!(support, vec![0, 2, 4]);

        let id = M.identity();
        assert!(id.support().is_empty());
    }

    #[test]
    fn test_element_to_string() {
        let M = FreeAbelianMonoid::new(vec!["a".to_string(), "b".to_string()]);
        let a = M.gen(0).unwrap();
        let b = M.gen(1).unwrap();

        assert_eq!(M.element_to_string(&a), "a");
        assert_eq!(M.element_to_string(&M.identity()), "1");

        let a2b = a.pow(&Integer::from(2)).mul(&b);
        let s = M.element_to_string(&a2b);
        assert!(s == "a^2*b" || s == "b*a^2");
    }

    #[test]
    fn test_element_from_ints() {
        let M = FreeAbelianMonoid::with_rank(3);
        let elem = M.element_from_ints(vec![2, 0, 3]);

        assert_eq!(elem.get_exponent(0), Integer::from(2));
        assert_eq!(elem.get_exponent(1), Integer::from(0));
        assert_eq!(elem.get_exponent(2), Integer::from(3));
    }

    #[test]
    fn test_contains() {
        let M = FreeAbelianMonoid::with_rank(2);
        let x = M.gen(0).unwrap();

        assert!(M.contains(&x));
        assert!(M.contains(&M.identity()));
    }

    #[test]
    fn test_normalize() {
        let exps = vec![Integer::from(1), Integer::from(2), Integer::from(0), Integer::from(0)];
        let mut elem = FreeAbelianMonoidElement::new(exps);

        assert_eq!(elem.rank(), 4);
        elem.normalize();
        assert_eq!(elem.rank(), 2);
        assert_eq!(elem.get_exponent(0), Integer::from(1));
        assert_eq!(elem.get_exponent(1), Integer::from(2));
    }

    #[test]
    fn test_resize() {
        let exps = vec![Integer::from(1), Integer::from(2)];
        let mut elem = FreeAbelianMonoidElement::new(exps);

        elem.resize(4);
        assert_eq!(elem.rank(), 4);
        assert_eq!(elem.get_exponent(0), Integer::from(1));
        assert_eq!(elem.get_exponent(1), Integer::from(2));
        assert_eq!(elem.get_exponent(2), Integer::from(0));
        assert_eq!(elem.get_exponent(3), Integer::from(0));
    }

    #[test]
    fn test_is_FreeAbelianMonoid() {
        let M = FreeAbelianMonoid::with_rank(3);
        assert!(is_free_abelian_monoid(&M));
    }

    #[test]
    fn test_is_FreeAbelianMonoidElement() {
        let elem = FreeAbelianMonoidElement::identity(2);
        assert!(is_free_abelian_monoid_element(&elem));
    }

    #[test]
    fn test_FreeAbelianMonoid_class() {
        let M = free_abelian_monoid_class(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(M.rank(), 2);
    }

    #[test]
    fn test_display_monoid() {
        let M = FreeAbelianMonoid::new(vec!["a".to_string(), "b".to_string()]);
        let display = format!("{}", M);
        assert!(display.contains("Free abelian monoid"));
        assert!(display.contains("2 generators"));
    }

    #[test]
    fn test_display_element() {
        let M = FreeAbelianMonoid::with_rank(2);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        let elem = x.pow(&Integer::from(2)).mul(&y);
        let display = format!("{}", elem);
        assert!(display.contains("x_0") && display.contains("x_1"));
    }

    #[test]
    fn test_is_abelian() {
        let M = FreeAbelianMonoid::with_rank(2);
        assert!(M.is_abelian());
        assert!(M.is_commutative());
    }

    #[test]
    fn test_is_finite() {
        let M0 = FreeAbelianMonoid::with_rank(0);
        assert!(M0.is_finite());
        assert_eq!(M0.order(), Some(Integer::from(1)));

        let M2 = FreeAbelianMonoid::with_rank(2);
        assert!(!M2.is_finite());
        assert_eq!(M2.order(), None);
    }

    #[test]
    fn test_large_exponents() {
        let M = FreeAbelianMonoid::with_rank(2);
        let x = M.gen(0).unwrap();

        // Test with very large exponent
        let large_exp = Integer::from(1000000);
        let x_large = x.pow(&large_exp);
        assert_eq!(x_large.get_exponent(0), large_exp);
    }

    #[test]
    fn test_many_generators() {
        let M = FreeAbelianMonoid::with_rank(100);
        assert_eq!(M.rank(), 100);

        let x50 = M.gen(50).unwrap();
        assert_eq!(x50.get_exponent(50), Integer::from(1));
    }

    #[test]
    fn test_element_equality() {
        let M = FreeAbelianMonoid::with_rank(3);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        let xy = x.mul(&y);
        let yx = y.mul(&x);

        assert_eq!(xy, yx); // Commutativity check
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::HashSet;

        let M = FreeAbelianMonoid::with_rank(2);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        let xy = x.mul(&y);
        let yx = y.mul(&x);

        let mut set = HashSet::new();
        set.insert(xy.clone());

        assert!(set.contains(&yx));
    }
}
