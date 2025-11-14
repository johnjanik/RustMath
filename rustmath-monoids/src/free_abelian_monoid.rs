//! # Free Abelian Monoids
//!
//! This module provides free abelian monoids and their elements.
//!
//! A free abelian monoid is a commutative monoid where each element can be
//! represented as a formal product of generators with non-negative integer exponents.

use num_bigint::BigInt;
use num_traits::{Zero, One};
use std::collections::HashMap;
use std::fmt::{self, Display};

/// An element of a free abelian monoid
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeAbelianMonoidElement {
    /// Exponents for each generator (sparse representation)
    exponents: HashMap<usize, BigInt>,
}

impl FreeAbelianMonoidElement {
    /// Create a new element from exponents
    pub fn new(exponents: HashMap<usize, BigInt>) -> Self {
        // Remove zero exponents
        let mut clean_exponents = HashMap::new();
        for (gen, exp) in exponents {
            if !exp.is_zero() {
                clean_exponents.insert(gen, exp);
            }
        }
        FreeAbelianMonoidElement {
            exponents: clean_exponents,
        }
    }

    /// Create the identity element
    pub fn identity() -> Self {
        FreeAbelianMonoidElement {
            exponents: HashMap::new(),
        }
    }

    /// Get the exponents
    pub fn exponents(&self) -> &HashMap<usize, BigInt> {
        &self.exponents
    }

    /// Get the exponent for a specific generator
    pub fn get_exponent(&self, gen: usize) -> BigInt {
        self.exponents.get(&gen).cloned().unwrap_or_else(BigInt::zero)
    }

    /// Check if this is the identity
    pub fn is_identity(&self) -> bool {
        self.exponents.is_empty()
    }

    /// Multiply two elements
    pub fn mul(&self, other: &Self) -> Self {
        let mut new_exponents = self.exponents.clone();

        for (gen, exp) in &other.exponents {
            *new_exponents.entry(*gen).or_insert_with(BigInt::zero) += exp;
        }

        // Clean up zeros
        new_exponents.retain(|_, exp| !exp.is_zero());

        FreeAbelianMonoidElement {
            exponents: new_exponents,
        }
    }

    /// Compute a power of this element
    pub fn pow(&self, n: &BigInt) -> Self {
        if n.is_zero() {
            return Self::identity();
        }

        let mut new_exponents = HashMap::new();
        for (gen, exp) in &self.exponents {
            new_exponents.insert(*gen, exp * n);
        }

        FreeAbelianMonoidElement {
            exponents: new_exponents,
        }
    }

    /// Total degree (sum of all exponents)
    pub fn degree(&self) -> BigInt {
        self.exponents.values().fold(BigInt::zero(), |acc, exp| acc + exp)
    }

    /// Support (indices of generators with non-zero exponents)
    pub fn support(&self) -> Vec<usize> {
        let mut indices: Vec<_> = self.exponents.keys().copied().collect();
        indices.sort();
        indices
    }
}

/// A free abelian monoid on a set of generators
#[derive(Debug, Clone)]
pub struct FreeAbelianMonoid {
    /// Number of generators
    num_generators: usize,
    /// Names of generators
    generator_names: Vec<String>,
}

impl FreeAbelianMonoid {
    /// Create a new free abelian monoid with named generators
    pub fn new(generators: Vec<String>) -> Self {
        let num_generators = generators.len();
        FreeAbelianMonoid {
            num_generators,
            generator_names: generators,
        }
    }

    /// Create a new free abelian monoid with n generators (x_0, x_1, ..., x_{n-1})
    pub fn with_rank(n: usize) -> Self {
        let generator_names = (0..n).map(|i| format!("x_{}", i)).collect();
        FreeAbelianMonoid {
            num_generators: n,
            generator_names,
        }
    }

    /// Get the number of generators
    pub fn rank(&self) -> usize {
        self.num_generators
    }

    /// Get the generator names
    pub fn generators(&self) -> &[String] {
        &self.generator_names
    }

    /// Create the identity element
    pub fn identity(&self) -> FreeAbelianMonoidElement {
        FreeAbelianMonoidElement::identity()
    }

    /// Create a generator element
    pub fn gen(&self, index: usize) -> Option<FreeAbelianMonoidElement> {
        if index < self.num_generators {
            let mut exponents = HashMap::new();
            exponents.insert(index, BigInt::one());
            Some(FreeAbelianMonoidElement::new(exponents))
        } else {
            None
        }
    }

    /// Create an element from exponents
    pub fn element(&self, exponents: HashMap<usize, BigInt>) -> FreeAbelianMonoidElement {
        FreeAbelianMonoidElement::new(exponents)
    }

    /// Format an element as a string
    pub fn element_to_string(&self, elem: &FreeAbelianMonoidElement) -> String {
        if elem.is_identity() {
            return "1".to_string();
        }

        let support = elem.support();
        let terms: Vec<String> = support
            .iter()
            .map(|&i| {
                let exp = elem.get_exponent(i);
                let name = if i < self.generator_names.len() {
                    &self.generator_names[i]
                } else {
                    "x"
                };

                if exp == BigInt::one() {
                    name.to_string()
                } else {
                    format!("{}^{}", name, exp)
                }
            })
            .collect();

        terms.join("*")
    }
}

/// Factory for creating free abelian monoids
pub struct FreeAbelianMonoidFactory;

impl FreeAbelianMonoidFactory {
    pub fn create(generators: Vec<String>) -> FreeAbelianMonoid {
        FreeAbelianMonoid::new(generators)
    }
}

/// Create a free abelian monoid
pub fn FreeAbelianMonoid_class(generators: Vec<String>) -> FreeAbelianMonoid {
    FreeAbelianMonoid::new(generators)
}

/// Check if an object is a free abelian monoid
pub fn is_FreeAbelianMonoid(obj: &FreeAbelianMonoid) -> bool {
    true
}

/// Check if an object is a free abelian monoid element
pub fn is_FreeAbelianMonoidElement(obj: &FreeAbelianMonoidElement) -> bool {
    true
}

impl Display for FreeAbelianMonoid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Free abelian monoid on {} generators [{}]",
            self.num_generators,
            self.generator_names.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_abelian_monoid() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(M.rank(), 2);
    }

    #[test]
    fn test_identity() {
        let M = FreeAbelianMonoid::new(vec!["a".to_string(), "b".to_string()]);
        let id = M.identity();
        assert!(id.is_identity());
    }

    #[test]
    fn test_generator() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string(), "y".to_string()]);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        assert_eq!(x.get_exponent(0), BigInt::one());
        assert_eq!(y.get_exponent(1), BigInt::one());
    }

    #[test]
    fn test_multiplication() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string(), "y".to_string()]);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        let xy = x.mul(&y);
        assert_eq!(xy.get_exponent(0), BigInt::one());
        assert_eq!(xy.get_exponent(1), BigInt::one());

        // Commutativity
        let yx = y.mul(&x);
        assert_eq!(xy, yx);
    }

    #[test]
    fn test_pow() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string()]);
        let x = M.gen(0).unwrap();

        let x3 = x.pow(&BigInt::from(3));
        assert_eq!(x3.get_exponent(0), BigInt::from(3));
    }

    #[test]
    fn test_degree() {
        let M = FreeAbelianMonoid::new(vec!["x".to_string(), "y".to_string()]);
        let x = M.gen(0).unwrap();
        let y = M.gen(1).unwrap();

        let x2y3 = x.pow(&BigInt::from(2)).mul(&y.pow(&BigInt::from(3)));
        assert_eq!(x2y3.degree(), BigInt::from(5));
    }

    #[test]
    fn test_element_to_string() {
        let M = FreeAbelianMonoid::new(vec!["a".to_string(), "b".to_string()]);
        let a = M.gen(0).unwrap();
        let b = M.gen(1).unwrap();

        assert_eq!(M.element_to_string(&a), "a");
        assert_eq!(M.element_to_string(&M.identity()), "1");

        let a2b = a.pow(&BigInt::from(2)).mul(&b);
        let s = M.element_to_string(&a2b);
        assert!(s.contains("a^2") || s.contains("b"));
    }

    #[test]
    fn test_is_FreeAbelianMonoid() {
        let M = FreeAbelianMonoid::with_rank(3);
        assert!(is_FreeAbelianMonoid(&M));
    }

    #[test]
    fn test_is_FreeAbelianMonoidElement() {
        let elem = FreeAbelianMonoidElement::identity();
        assert!(is_FreeAbelianMonoidElement(&elem));
    }
}
