//! Group Algebra Implementation
//!
//! The group algebra R[G] of a group G over a ring R.
//! Elements are formal linear combinations of group elements with coefficients in R.
//!
//! Corresponds to sage.algebras.group_algebra

use rustmath_core::{Ring, MathError, Result};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::Hash;
use std::ops::{Add, Mul, Neg, Sub};
use crate::traits::Algebra;

/// A group element trait
///
/// This trait represents the minimal requirements for group elements
pub trait GroupElement: Clone + Eq + Hash + Display + std::fmt::Debug {
    /// The identity element
    fn identity() -> Self;

    /// Check if this is the identity
    fn is_identity(&self) -> bool;

    /// Multiply two group elements
    fn mult(&self, other: &Self) -> Self;

    /// Inverse of this element
    fn inverse(&self) -> Self;
}

/// An element of a group algebra R[G]
#[derive(Clone, Debug)]
pub struct GroupAlgebraElement<R: Ring, G: GroupElement> {
    /// Coefficients for each group element
    terms: HashMap<G, R>,
}

impl<R: Ring, G: GroupElement> GroupAlgebraElement<R, G> {
    /// Create a new group algebra element
    pub fn new() -> Self {
        Self {
            terms: HashMap::new(),
        }
    }

    /// Create from a single term
    pub fn from_term(coeff: R, group_elem: G) -> Self {
        let mut terms = HashMap::new();
        if !coeff.is_zero() {
            terms.insert(group_elem, coeff);
        }
        Self { terms }
    }

    /// Create a group algebra element from a group element (coefficient 1)
    pub fn from_group_element(group_elem: G) -> Self {
        Self::from_term(R::one(), group_elem)
    }

    /// Add a term
    pub fn add_term(&mut self, coeff: R, group_elem: G) {
        if coeff.is_zero() {
            return;
        }

        let entry = self.terms.entry(group_elem).or_insert_with(R::zero);
        *entry = entry.clone() + coeff;

        // Clean up zero coefficients
        self.terms.retain(|_, c| !c.is_zero());
    }

    /// Get the coefficient of a group element
    pub fn coeff(&self, group_elem: &G) -> R {
        self.terms.get(group_elem).cloned().unwrap_or_else(R::zero)
    }

    /// Get all terms
    pub fn terms(&self) -> &HashMap<G, R> {
        &self.terms
    }

    /// Support (set of group elements with nonzero coefficients)
    pub fn support(&self) -> Vec<G> {
        self.terms.keys().cloned().collect()
    }

    /// Number of terms
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

impl<R: Ring, G: GroupElement> Default for GroupAlgebraElement<R, G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring, G: GroupElement> PartialEq for GroupAlgebraElement<R, G> {
    fn eq(&self, other: &Self) -> bool {
        if self.terms.len() != other.terms.len() {
            return false;
        }
        for (g, c) in &self.terms {
            let other_c = other.coeff(g);
            if *c != other_c {
                return false;
            }
        }
        true
    }
}

impl<R: Ring, G: GroupElement> Display for GroupAlgebraElement<R, G> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let mut terms_vec: Vec<_> = self.terms.iter().collect();
        // Sort by group element for consistent display
        terms_vec.sort_by_key(|(g, _)| format!("{}", g));

        for (i, (g, c)) in terms_vec.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            if g.is_identity() {
                write!(f, "{}", c)?;
            } else if c.is_one() {
                write!(f, "{}", g)?;
            } else {
                write!(f, "{}*{}", c, g)?;
            }
        }
        Ok(())
    }
}

impl<R: Ring, G: GroupElement> Add for GroupAlgebraElement<R, G> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self.clone();
        for (g, c) in other.terms {
            result.add_term(c, g);
        }
        result
    }
}

impl<R: Ring, G: GroupElement> Sub for GroupAlgebraElement<R, G> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<R: Ring, G: GroupElement> Neg for GroupAlgebraElement<R, G> {
    type Output = Self;

    fn neg(self) -> Self {
        let mut result = Self::new();
        for (g, c) in self.terms {
            result.add_term(-c, g);
        }
        result
    }
}

impl<R: Ring, G: GroupElement> Mul for GroupAlgebraElement<R, G> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut result = Self::new();

        for (g1, c1) in &self.terms {
            for (g2, c2) in &other.terms {
                let new_g = g1.mult(g2);
                let new_c = c1.clone() * c2.clone();
                result.add_term(new_c, new_g);
            }
        }

        result
    }
}

impl<R: Ring, G: GroupElement> Ring for GroupAlgebraElement<R, G> {
    fn zero() -> Self {
        Self::new()
    }

    fn one() -> Self {
        Self::from_term(R::one(), G::identity())
    }

    fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    fn is_one(&self) -> bool {
        self.terms.len() == 1 && {
            let (g, c) = self.terms.iter().next().unwrap();
            g.is_identity() && c.is_one()
        }
    }
}

impl<R: Ring, G: GroupElement> Algebra<R> for GroupAlgebraElement<R, G> {
    fn base_ring() -> R {
        R::zero()
    }

    fn scalar_mul(&self, scalar: &R) -> Self {
        let mut result = Self::new();
        for (g, c) in &self.terms {
            result.add_term(c.clone() * scalar.clone(), g.clone());
        }
        result
    }
}

/// A group algebra R[G]
pub struct GroupAlgebra<R: Ring, G: GroupElement> {
    /// Phantom data for the ring and group
    _phantom: std::marker::PhantomData<(R, G)>,
}

impl<R: Ring, G: GroupElement> GroupAlgebra<R, G> {
    /// Create a new group algebra
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create the zero element
    pub fn zero(&self) -> GroupAlgebraElement<R, G> {
        GroupAlgebraElement::new()
    }

    /// Create the one element
    pub fn one(&self) -> GroupAlgebraElement<R, G> {
        GroupAlgebraElement::from_term(R::one(), G::identity())
    }

    /// Create an element from a group element
    pub fn from_group_element(&self, g: G) -> GroupAlgebraElement<R, G> {
        GroupAlgebraElement::from_group_element(g)
    }

    /// Create a scalar element
    pub fn scalar(&self, r: R) -> GroupAlgebraElement<R, G> {
        GroupAlgebraElement::from_term(r, G::identity())
    }
}

impl<R: Ring, G: GroupElement> Default for GroupAlgebra<R, G> {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple cyclic group Z/nZ for testing
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CyclicGroup {
    value: usize,
    modulus: usize,
}

impl CyclicGroup {
    pub fn new(value: usize, modulus: usize) -> Self {
        Self {
            value: value % modulus,
            modulus,
        }
    }
}

impl Display for CyclicGroup {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "g^{}", self.value)
    }
}

impl GroupElement for CyclicGroup {
    fn identity() -> Self {
        Self {
            value: 0,
            modulus: 1,
        }
    }

    fn is_identity(&self) -> bool {
        self.value == 0
    }

    fn mult(&self, other: &Self) -> Self {
        assert_eq!(self.modulus, other.modulus);
        Self {
            value: (self.value + other.value) % self.modulus,
            modulus: self.modulus,
        }
    }

    fn inverse(&self) -> Self {
        Self {
            value: (self.modulus - self.value) % self.modulus,
            modulus: self.modulus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_cyclic_group() {
        let g1 = CyclicGroup::new(2, 5);
        let g2 = CyclicGroup::new(3, 5);
        let prod = g1.mult(&g2);
        assert_eq!(prod.value, 0); // 2 + 3 = 5 ≡ 0 (mod 5)
    }

    #[test]
    fn test_group_algebra_creation() {
        let alg: GroupAlgebra<Integer, CyclicGroup> = GroupAlgebra::new();
        let zero = alg.zero();
        let one = alg.one();

        assert!(zero.is_zero());
        assert!(one.is_one());
    }

    #[test]
    fn test_group_algebra_arithmetic() {
        let alg: GroupAlgebra<Integer, CyclicGroup> = GroupAlgebra::new();

        let g1 = CyclicGroup::new(1, 3);
        let g2 = CyclicGroup::new(2, 3);

        let x = alg.from_group_element(g1.clone());
        let y = alg.from_group_element(g2.clone());

        // Test addition
        let sum = x.clone() + y.clone();
        assert_eq!(sum.len(), 2);

        // Test multiplication
        let prod = x.clone() * y.clone();
        // g1 * g2 = g0 (identity in Z/3Z when 1+2=3≡0)
        assert_eq!(prod.len(), 1);
    }

    #[test]
    fn test_group_algebra_distributivity() {
        let alg: GroupAlgebra<Integer, CyclicGroup> = GroupAlgebra::new();

        let g0 = CyclicGroup::new(0, 3);
        let g1 = CyclicGroup::new(1, 3);
        let g2 = CyclicGroup::new(2, 3);

        let x = alg.from_group_element(g0);
        let y = alg.from_group_element(g1);
        let z = alg.from_group_element(g2);

        // Test distributivity: x * (y + z) = x*y + x*z
        let lhs = x.clone() * (y.clone() + z.clone());
        let rhs = x.clone() * y.clone() + x.clone() * z.clone();
        assert_eq!(lhs, rhs);
    }
}
