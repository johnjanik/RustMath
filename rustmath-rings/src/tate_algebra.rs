//! # Tate Algebras
//!
//! This module implements Tate algebras - rings of strictly convergent power series.
//!
//! ## Overview
//!
//! A Tate algebra over a field K with respect to variables x₁, ..., xₙ is the ring
//! of formal power series:
//! ```text
//! K⟨x₁, ..., xₙ⟩ = {Σ aₘ xᵐ : aₘ → 0 as |m| → ∞}
//! ```
//!
//! These are "strictly convergent" power series in non-archimedean analysis.
//!
//! ## Theory
//!
//! Tate algebras are fundamental in:
//! - Rigid analytic geometry (non-archimedean analogue of complex manifolds)
//! - p-adic analysis and geometry
//! - Study of affinoid algebras
//!
//! Properties:
//! - Noetherian
//! - Regular
//! - Jacobson
//! - Every maximal ideal corresponds to a point in the "unit polydisc"
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::tate_algebra::TateAlgebra;
//!
//! // Create Tate algebra Q_p⟨x, y⟩
//! let algebra = TateAlgebra::new(vec!["x".to_string(), "y".to_string()]);
//! ```

use rustmath_core::{Field, Ring};
use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;

/// Monoid of terms in a Tate algebra
///
/// Represents monomials x₁^{a₁} ... xₙ^{aₙ}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TateTermMonoid {
    /// Number of variables
    nvars: usize,
    /// Variable names
    variables: Vec<String>,
}

impl TateTermMonoid {
    /// Create a new term monoid
    pub fn new(variables: Vec<String>) -> Self {
        let nvars = variables.len();
        Self { nvars, variables }
    }

    /// Get the number of variables
    pub fn nvars(&self) -> usize {
        self.nvars
    }

    /// Get variable names
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Create a term from exponents
    pub fn term(&self, exponents: Vec<usize>) -> TateTerm {
        if exponents.len() != self.nvars {
            panic!("Exponent vector length must match number of variables");
        }
        TateTerm { exponents }
    }

    /// Get the identity element (all exponents 0)
    pub fn one(&self) -> TateTerm {
        TateTerm {
            exponents: vec![0; self.nvars],
        }
    }
}

/// A term (monomial) in the Tate algebra
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TateTerm {
    /// Exponents for each variable
    exponents: Vec<usize>,
}

impl TateTerm {
    /// Get the exponents
    pub fn exponents(&self) -> &[usize] {
        &self.exponents
    }

    /// Get the total degree
    pub fn degree(&self) -> usize {
        self.exponents.iter().sum()
    }

    /// Multiply two terms
    pub fn mul(&self, other: &Self) -> Self {
        if self.exponents.len() != other.exponents.len() {
            panic!("Terms must have same number of variables");
        }

        let exponents = self
            .exponents
            .iter()
            .zip(other.exponents.iter())
            .map(|(a, b)| a + b)
            .collect();

        Self { exponents }
    }
}

/// Generic Tate algebra over a field K
///
/// Represents K⟨x₁, ..., xₙ⟩
#[derive(Debug, Clone, PartialEq)]
pub struct TateAlgebra<K: Field> {
    /// Base field (type-level)
    _field: PhantomData<K>,
    /// Term monoid
    term_monoid: TateTermMonoid,
}

impl<K: Field> TateAlgebra<K> {
    /// Create a new Tate algebra
    ///
    /// # Arguments
    /// * `variables` - Names of the variables
    ///
    /// # Examples
    /// ```
    /// use rustmath_rings::tate_algebra::TateAlgebra;
    /// use rustmath_rationals::Rational;
    ///
    /// let algebra = TateAlgebra::<Rational>::new(vec!["x".to_string(), "y".to_string()]);
    /// ```
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            _field: PhantomData,
            term_monoid: TateTermMonoid::new(variables),
        }
    }

    /// Get the term monoid
    pub fn term_monoid(&self) -> &TateTermMonoid {
        &self.term_monoid
    }

    /// Get the number of variables
    pub fn nvars(&self) -> usize {
        self.term_monoid.nvars()
    }

    /// Get variable names
    pub fn variables(&self) -> &[String] {
        self.term_monoid.variables()
    }

    /// Create an element from a map of terms to coefficients
    pub fn from_terms(&self, terms: BTreeMap<TateTerm, K>) -> TateElement<K> {
        TateElement::new(terms)
    }

    /// Create a constant element
    pub fn constant(&self, value: K) -> TateElement<K> {
        let mut terms = BTreeMap::new();
        let zero_term = self.term_monoid.one();
        terms.insert(zero_term, value);
        TateElement::new(terms)
    }

    /// Create a variable (x_i)
    pub fn variable(&self, index: usize) -> TateElement<K> {
        if index >= self.nvars() {
            panic!("Variable index out of range");
        }

        let mut exponents = vec![0; self.nvars()];
        exponents[index] = 1;
        let term = TateTerm { exponents };

        let mut terms = BTreeMap::new();
        terms.insert(term, K::one());
        TateElement::new(terms)
    }
}

impl<K: Field + fmt::Display> fmt::Display for TateAlgebra<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tate Algebra over {} in variables {}",
            std::any::type_name::<K>(),
            self.variables().join(", ")
        )
    }
}

/// Element of a Tate algebra
///
/// Represents a strictly convergent power series
#[derive(Debug, Clone, PartialEq)]
pub struct TateElement<K: Field> {
    /// Terms: maps monomials to coefficients
    terms: BTreeMap<TateTerm, K>,
}

impl<K: Field> TateElement<K> {
    /// Create a new Tate algebra element
    pub fn new(terms: BTreeMap<TateTerm, K>) -> Self {
        // Filter out zero coefficients
        let terms: BTreeMap<_, _> = terms
            .into_iter()
            .filter(|(_, coeff)| !coeff.is_zero())
            .collect();

        Self { terms }
    }

    /// Get the terms
    pub fn terms(&self) -> &BTreeMap<TateTerm, K> {
        &self.terms
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.terms.clone();

        for (term, coeff) in &other.terms {
            result
                .entry(term.clone())
                .and_modify(|c| *c = c.clone() + coeff.clone())
                .or_insert_with(|| coeff.clone());
        }

        Self::new(result)
    }

    /// Multiply two elements
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = BTreeMap::new();

        for (term1, coeff1) in &self.terms {
            for (term2, coeff2) in &other.terms {
                let new_term = term1.mul(term2);
                let new_coeff = coeff1.clone() * coeff2.clone();

                result
                    .entry(new_term)
                    .and_modify(|c| *c = c.clone() + new_coeff.clone())
                    .or_insert(new_coeff);
            }
        }

        Self::new(result)
    }

    /// Negate
    pub fn neg(&self) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|(term, coeff)| (term.clone(), -coeff.clone()))
            .collect();
        Self::new(terms)
    }
}

impl<K: Field + fmt::Display> fmt::Display for TateElement<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (term, coeff) in &self.terms {
            if !first {
                write!(f, " + ")?;
            }

            write!(f, "{}", coeff)?;

            for (i, exp) in term.exponents().iter().enumerate() {
                if *exp > 0 {
                    write!(f, "*x{}^{}", i, exp)?;
                }
            }

            first = false;
        }

        Ok(())
    }
}

/// Factory for creating Tate algebras
///
/// Provides a convenient interface for Tate algebra construction
#[derive(Debug, Clone)]
pub struct TateAlgebraFactory;

impl TateAlgebraFactory {
    /// Create a new Tate algebra
    pub fn create<K: Field>(variables: Vec<String>) -> TateAlgebra<K> {
        TateAlgebra::new(variables)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_tate_term_monoid() {
        let monoid = TateTermMonoid::new(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(monoid.nvars(), 2);

        let term = monoid.term(vec![2, 3]);
        assert_eq!(term.degree(), 5);
    }

    #[test]
    fn test_tate_term_mul() {
        let t1 = TateTerm {
            exponents: vec![1, 2],
        };
        let t2 = TateTerm {
            exponents: vec![3, 1],
        };

        let product = t1.mul(&t2);
        assert_eq!(product.exponents, vec![4, 3]);
        assert_eq!(product.degree(), 7);
    }

    #[test]
    fn test_tate_algebra() {
        let algebra = TateAlgebra::<Rational>::new(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(algebra.nvars(), 2);
        assert_eq!(algebra.variables(), &["x", "y"]);
    }

    #[test]
    fn test_constant() {
        let algebra = TateAlgebra::<Rational>::new(vec!["x".to_string()]);
        let c = algebra.constant(Rational::new(5, 1));

        assert_eq!(c.terms().len(), 1);
        assert!(!c.is_zero());
    }

    #[test]
    fn test_variable() {
        let algebra = TateAlgebra::<Rational>::new(vec!["x".to_string(), "y".to_string()]);
        let x = algebra.variable(0);
        let y = algebra.variable(1);

        assert_eq!(x.terms().len(), 1);
        assert_eq!(y.terms().len(), 1);
    }

    #[test]
    fn test_element_addition() {
        let algebra = TateAlgebra::<Rational>::new(vec!["x".to_string()]);
        let c1 = algebra.constant(Rational::new(3, 1));
        let c2 = algebra.constant(Rational::new(2, 1));

        let sum = c1.add(&c2);
        // Sum should have constant term 5
        assert_eq!(sum.terms().len(), 1);
    }

    #[test]
    fn test_element_multiplication() {
        let algebra = TateAlgebra::<Rational>::new(vec!["x".to_string()]);
        let x = algebra.variable(0);
        let c = algebra.constant(Rational::new(2, 1));

        let product = x.mul(&c);
        assert_eq!(product.terms().len(), 1);
    }

    #[test]
    fn test_element_negation() {
        let algebra = TateAlgebra::<Rational>::new(vec!["x".to_string()]);
        let c = algebra.constant(Rational::new(5, 1));
        let neg = c.neg();

        assert_eq!(neg.terms().len(), 1);
    }

    #[test]
    fn test_factory() {
        let algebra = TateAlgebraFactory::create::<Rational>(vec!["x".to_string()]);
        assert_eq!(algebra.nvars(), 1);
    }

    #[test]
    fn test_is_zero() {
        let zero = TateElement::<Rational>::new(BTreeMap::new());
        assert!(zero.is_zero());

        let algebra = TateAlgebra::<Rational>::new(vec!["x".to_string()]);
        let c = algebra.constant(Rational::new(5, 1));
        assert!(!c.is_zero());
    }
}
