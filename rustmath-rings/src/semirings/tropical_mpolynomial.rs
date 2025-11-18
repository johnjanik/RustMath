//! # Tropical Multivariate Polynomials
//!
//! This module implements multivariate polynomials over tropical semirings.
//!
//! ## Overview
//!
//! A tropical multivariate polynomial is a piecewise-linear function in multiple variables.
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::semirings::tropical_mpolynomial::TropicalMPolynomialSemiring;
//! use rustmath_rings::semirings::tropical_semiring::TropicalType;
//!
//! let ring = TropicalMPolynomialSemiring::new(TropicalType::Min, vec!["x".to_string(), "y".to_string()]);
//! ```

use super::tropical_semiring::{TropicalSemiringElement, TropicalType};
use std::collections::BTreeMap;
use std::fmt;

/// Exponent vector for a monomial
pub type Exponent = Vec<usize>;

/// Semiring of tropical multivariate polynomials
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalMPolynomialSemiring {
    tropical_type: TropicalType,
    variables: Vec<String>,
}

impl TropicalMPolynomialSemiring {
    /// Create a new tropical multivariate polynomial semiring
    pub fn new(tropical_type: TropicalType, variables: Vec<String>) -> Self {
        Self {
            tropical_type,
            variables,
        }
    }

    /// Get the number of variables
    pub fn nvars(&self) -> usize {
        self.variables.len()
    }

    /// Get variable names
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Create a polynomial from a map of exponents to coefficients
    pub fn from_terms(&self, terms: BTreeMap<Exponent, f64>) -> TropicalMPolynomial {
        TropicalMPolynomial::new(terms, self.tropical_type, self.nvars())
    }
}

/// A tropical multivariate polynomial
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalMPolynomial {
    /// Maps exponent vectors to coefficients
    terms: BTreeMap<Exponent, f64>,
    tropical_type: TropicalType,
    nvars: usize,
}

impl TropicalMPolynomial {
    /// Create a new tropical multivariate polynomial
    pub fn new(terms: BTreeMap<Exponent, f64>, tropical_type: TropicalType, nvars: usize) -> Self {
        // Validate exponent vectors
        for exp in terms.keys() {
            if exp.len() != nvars {
                panic!("Exponent vector length must match number of variables");
            }
        }

        Self {
            terms,
            tropical_type,
            nvars,
        }
    }

    /// Get the number of variables
    pub fn nvars(&self) -> usize {
        self.nvars
    }

    /// Evaluate at a point
    pub fn eval(&self, point: &[f64]) -> TropicalSemiringElement {
        if point.len() != self.nvars {
            panic!("Point dimension must match number of variables");
        }

        let mut result = TropicalSemiringElement::infinity(self.tropical_type);

        for (exp, coeff) in &self.terms {
            // Compute coefficient + sum(exp[i] * point[i])
            let mut value = *coeff;
            for (i, e) in exp.iter().enumerate() {
                value += (*e as f64) * point[i];
            }

            let term = TropicalSemiringElement::from_value_with_type(value, self.tropical_type);
            result = result.tropical_add(&term);
        }

        result
    }

    /// Add two tropical multivariate polynomials
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.terms.clone();

        for (exp, coeff) in &other.terms {
            result
                .entry(exp.clone())
                .and_modify(|c| {
                    *c = match self.tropical_type {
                        TropicalType::Min => c.min(*coeff),
                        TropicalType::Max => c.max(*coeff),
                    }
                })
                .or_insert(*coeff);
        }

        Self::new(result, self.tropical_type, self.nvars)
    }

    /// Get all terms
    pub fn terms(&self) -> &BTreeMap<Exponent, f64> {
        &self.terms
    }
}

impl fmt::Display for TropicalMPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "∞");
        }

        let mut first = true;
        for (exp, coeff) in &self.terms {
            if !first {
                write!(f, " ⊕ ")?;
            }

            write!(f, "{}", coeff)?;

            for (i, e) in exp.iter().enumerate() {
                if *e > 0 {
                    write!(f, "⊗x{}^{}", i, e)?;
                }
            }

            first = false;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpolynomial_semiring() {
        let ring = TropicalMPolynomialSemiring::new(
            TropicalType::Min,
            vec!["x".to_string(), "y".to_string()],
        );
        assert_eq!(ring.nvars(), 2);
    }

    #[test]
    fn test_mpolynomial_eval() {
        let mut terms = BTreeMap::new();
        terms.insert(vec![0, 0], 0.0); // constant 0
        terms.insert(vec![1, 0], 1.0); // 1 ⊗ x
        terms.insert(vec![0, 1], 2.0); // 2 ⊗ y

        let poly = TropicalMPolynomial::new(terms, TropicalType::Min, 2);

        let val = poly.eval(&[0.0, 0.0]);
        assert_eq!(val.value(), Some(0.0)); // min(0, 1, 2) = 0

        let val = poly.eval(&[1.0, 1.0]);
        assert_eq!(val.value(), Some(0.0)); // min(0, 2, 3) = 0
    }

    #[test]
    fn test_mpolynomial_addition() {
        let mut terms1 = BTreeMap::new();
        terms1.insert(vec![1, 0], 1.0);

        let mut terms2 = BTreeMap::new();
        terms2.insert(vec![0, 1], 2.0);

        let p1 = TropicalMPolynomial::new(terms1, TropicalType::Min, 2);
        let p2 = TropicalMPolynomial::new(terms2, TropicalType::Min, 2);

        let sum = p1.add(&p2);
        assert_eq!(sum.terms().len(), 2);
    }

    #[test]
    #[should_panic(expected = "Exponent vector length must match number of variables")]
    fn test_invalid_exponent_length() {
        let mut terms = BTreeMap::new();
        terms.insert(vec![1, 0, 0], 1.0); // 3 variables

        let _ = TropicalMPolynomial::new(terms, TropicalType::Min, 2); // but nvars=2
    }
}
