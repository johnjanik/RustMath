//! Lie Conformal Algebra Elements
//!
//! Provides the element structure for Lie conformal algebras.
//!
//! Elements are represented as linear combinations of basis elements with
//! coefficients in R[∂] (polynomials in the derivation operator).
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.lie_conformal_algebra_element

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::Hash;
use crate::lie_conformal_algebra::GeneratorIndex;

/// Element of a Lie conformal algebra
///
/// Represented as a linear combination of basis elements, where each
/// coefficient is a polynomial in ∂.
///
/// # Type Parameters
///
/// * `R` - The base ring
/// * `B` - The basis index type
///
/// # Mathematical Representation
///
/// An element is written as:
/// Σᵢ pᵢ(∂) ⊗ bᵢ
///
/// where pᵢ(∂) ∈ R[∂] and bᵢ are basis elements.
#[derive(Clone, Debug)]
pub struct LieConformalAlgebraElement<R: Ring, B: Clone + Eq + Hash> {
    /// Terms: map from basis element to polynomial in ∂
    /// The polynomial is represented as a Vec<R> where index i holds the coefficient of ∂^i
    terms: HashMap<B, Vec<R>>,
}

impl<R: Ring + Clone, B: Clone + Eq + Hash> LieConformalAlgebraElement<R, B> {
    /// Create a new element
    pub fn new(terms: HashMap<B, Vec<R>>) -> Self {
        LieConformalAlgebraElement { terms }
    }

    /// Create the zero element
    pub fn zero() -> Self {
        LieConformalAlgebraElement {
            terms: HashMap::new(),
        }
    }

    /// Create an element from a single basis element
    pub fn from_basis(basis: B) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(basis, vec![R::from(1)]);
        LieConformalAlgebraElement { terms }
    }

    /// Create an element with a polynomial coefficient
    pub fn with_polynomial(basis: B, poly: Vec<R>) -> Self {
        let mut terms = HashMap::new();
        terms.insert(basis, poly);
        LieConformalAlgebraElement { terms }
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<B, Vec<R>> {
        &self.terms
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.terms.is_empty() || self.terms.values().all(|poly| poly.iter().all(|c| c.is_zero()))
    }

    /// Get the degree (maximum ∂ power)
    pub fn degree(&self) -> usize {
        self.terms
            .values()
            .map(|poly| poly.len().saturating_sub(1))
            .max()
            .unwrap_or(0)
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R> + PartialEq,
    {
        let mut result = self.terms.clone();

        for (basis, poly) in &other.terms {
            let new_poly = if let Some(existing) = result.get(basis) {
                add_polynomials(existing, poly)
            } else {
                poly.clone()
            };

            if !is_zero_polynomial(&new_poly) {
                result.insert(basis.clone(), new_poly);
            } else {
                result.remove(basis);
            }
        }

        LieConformalAlgebraElement { terms: result }
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
            .map(|(basis, poly)| {
                let new_poly: Vec<R> = poly.iter().map(|c| c.clone() * scalar.clone()).collect();
                (basis.clone(), new_poly)
            })
            .collect();

        LieConformalAlgebraElement { terms }
    }

    /// Apply the derivation operator ∂
    pub fn apply_derivation(&self) -> Self
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + From<i64>,
    {
        let terms = self
            .terms
            .iter()
            .map(|(basis, poly)| {
                // ∂(p(∂) ⊗ b) = (∂p)(∂) ⊗ b + p(∂) ⊗ ∂b
                // For now, just differentiate the polynomial
                let new_poly = differentiate_polynomial(poly);
                (basis.clone(), new_poly)
            })
            .filter(|(_, poly)| !is_zero_polynomial(poly))
            .collect();

        LieConformalAlgebraElement { terms }
    }

    /// Negate the element
    pub fn negate(&self) -> Self
    where
        R: std::ops::Neg<Output = R>,
    {
        let terms = self
            .terms
            .iter()
            .map(|(basis, poly)| {
                let new_poly: Vec<R> = poly.iter().map(|c| -c.clone()).collect();
                (basis.clone(), new_poly)
            })
            .collect();

        LieConformalAlgebraElement { terms }
    }
}

/// Element wrapper for simpler cases
///
/// Wraps an element from an underlying algebra (e.g., for quotients or subalgebras).
#[derive(Clone, Debug)]
pub struct LCAElementWrapper<E> {
    /// The underlying element
    element: E,
}

impl<E> LCAElementWrapper<E> {
    /// Create a new wrapper
    pub fn new(element: E) -> Self {
        LCAElementWrapper { element }
    }

    /// Get the underlying element
    pub fn element(&self) -> &E {
        &self.element
    }

    /// Unwrap to get the element
    pub fn into_element(self) -> E {
        self.element
    }
}

// Helper functions for polynomial operations

/// Add two polynomials (represented as coefficient vectors)
fn add_polynomials<R: Ring + std::ops::Add<Output = R>>(p: &[R], q: &[R]) -> Vec<R> {
    let max_len = p.len().max(q.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let p_coeff = if i < p.len() { p[i].clone() } else { R::zero() };
        let q_coeff = if i < q.len() { q[i].clone() } else { R::zero() };
        result.push(p_coeff + q_coeff);
    }

    // Remove trailing zeros
    while result.len() > 1 && result.last().map_or(false, |c| c.is_zero()) {
        result.pop();
    }
    if result.is_empty() {
        result.push(R::zero());
    }

    result
}

/// Check if a polynomial is zero
fn is_zero_polynomial<R: Ring + PartialEq>(poly: &[R]) -> bool {
    poly.is_empty() || poly.iter().all(|c| c.is_zero())
}

/// Differentiate a polynomial (formal derivative with respect to ∂)
fn differentiate_polynomial<R: Ring + std::ops::Mul<Output = R> + From<i64>>(poly: &[R]) -> Vec<R> {
    if poly.len() <= 1 {
        return vec![R::zero()];
    }

    let mut result = Vec::with_capacity(poly.len() - 1);
    for i in 1..poly.len() {
        result.push(poly[i].clone() * R::from(i as i64));
    }

    if result.is_empty() {
        result.push(R::zero());
    }

    result
}

impl<R: Ring + Clone + Display, B: Clone + Eq + Hash + Display> Display
    for LieConformalAlgebraElement<R, B>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let mut first = true;
        for (basis, poly) in &self.terms {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }

            // Format polynomial
            if poly.len() == 1 && poly[0].is_one() {
                write!(f, "{}", basis)?;
            } else {
                write!(f, "(")?;
                for (i, coeff) in poly.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    if i == 0 {
                        write!(f, "{}", coeff)?;
                    } else if i == 1 {
                        write!(f, "{}*∂", coeff)?;
                    } else {
                        write!(f, "{}*∂^{}", coeff, i)?;
                    }
                }
                write!(f, ") ⊗ {}", basis)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_creation() {
        let elem: LieConformalAlgebraElement<i64, usize> =
            LieConformalAlgebraElement::from_basis(0);
        assert!(!elem.is_zero());
        assert_eq!(elem.terms().len(), 1);
    }

    #[test]
    fn test_element_addition() {
        let elem1: LieConformalAlgebraElement<i64, usize> =
            LieConformalAlgebraElement::from_basis(0);
        let elem2: LieConformalAlgebraElement<i64, usize> =
            LieConformalAlgebraElement::from_basis(1);

        let sum = elem1.add(&elem2);
        assert_eq!(sum.terms().len(), 2);
    }

    #[test]
    fn test_polynomial_operations() {
        let p = vec![1i64, 2, 3]; // 1 + 2∂ + 3∂²
        let q = vec![4i64, 5];    // 4 + 5∂

        let sum = add_polynomials(&p, &q);
        assert_eq!(sum, vec![5, 7, 3]); // 5 + 7∂ + 3∂²

        let deriv = differentiate_polynomial(&p);
        assert_eq!(deriv, vec![2, 6]); // 2 + 6∂
    }

    #[test]
    fn test_zero_polynomial() {
        assert!(is_zero_polynomial(&[0i64]));
        assert!(is_zero_polynomial(&[0i64, 0, 0]));
        assert!(!is_zero_polynomial(&[1i64]));
        assert!(!is_zero_polynomial(&[0i64, 1]));
    }
}
