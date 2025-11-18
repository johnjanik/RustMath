//! # Tropical Polynomials
//!
//! This module implements univariate polynomials over tropical semirings.
//!
//! ## Overview
//!
//! A tropical polynomial has the form:
//! ```text
//! p(x) = a₀ ⊕ (a₁ ⊗ x) ⊕ (a₂ ⊗ x²) ⊕ ...
//! ```
//!
//! In min-plus algebra, this becomes:
//! ```text
//! p(x) = min(a₀, a₁ + x, a₂ + 2x, ...)
//! ```
//!
//! ## Theory
//!
//! - Tropical polynomials are piecewise-linear convex (min) or concave (max) functions
//! - The "roots" are points where the minimum is attained by at least two terms
//! - Degree is the maximum index with a finite coefficient
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::semirings::tropical_polynomial::{TropicalPolynomial, TropicalPolynomialSemiring};
//! use rustmath_rings::semirings::tropical_semiring::TropicalType;
//!
//! let ring = TropicalPolynomialSemiring::new(TropicalType::Min);
//! ```

use super::tropical_semiring::{TropicalSemiringElement, TropicalType};
use std::fmt;

/// Semiring of tropical polynomials
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalPolynomialSemiring {
    /// Base tropical semiring type
    tropical_type: TropicalType,
    /// Variable name
    variable: String,
}

impl TropicalPolynomialSemiring {
    /// Create a new tropical polynomial semiring
    pub fn new(tropical_type: TropicalType) -> Self {
        Self {
            tropical_type,
            variable: String::from("x"),
        }
    }

    /// Create with custom variable name
    pub fn with_variable(tropical_type: TropicalType, variable: String) -> Self {
        Self {
            tropical_type,
            variable,
        }
    }

    /// Get the tropical type
    pub fn tropical_type(&self) -> TropicalType {
        self.tropical_type
    }

    /// Get the variable name
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Create a polynomial from coefficients
    pub fn from_coefficients(&self, coefficients: Vec<Option<f64>>) -> TropicalPolynomial {
        TropicalPolynomial::new(coefficients, self.tropical_type)
    }

    /// Create the zero polynomial (all coefficients infinity)
    pub fn zero(&self) -> TropicalPolynomial {
        TropicalPolynomial::new(vec![], self.tropical_type)
    }

    /// Create a monomial x^degree with coefficient
    pub fn monomial(&self, coefficient: f64, degree: usize) -> TropicalPolynomial {
        let mut coeffs = vec![None; degree + 1];
        coeffs[degree] = Some(coefficient);
        TropicalPolynomial::new(coeffs, self.tropical_type)
    }
}

impl fmt::Display for TropicalPolynomialSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tropical Polynomial Semiring in {} ({:?})",
            self.variable, self.tropical_type
        )
    }
}

/// A tropical polynomial
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalPolynomial {
    /// Coefficients [a₀, a₁, a₂, ...] (None represents infinity)
    coefficients: Vec<Option<f64>>,
    /// Tropical semiring type
    tropical_type: TropicalType,
}

impl TropicalPolynomial {
    /// Create a new tropical polynomial
    pub fn new(coefficients: Vec<Option<f64>>, tropical_type: TropicalType) -> Self {
        Self {
            coefficients,
            tropical_type,
        }
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> Option<usize> {
        self.coefficients
            .iter()
            .enumerate()
            .rev()
            .find(|(_, c)| c.is_some())
            .map(|(i, _)| i)
    }

    /// Get coefficient at a given degree
    pub fn coefficient(&self, degree: usize) -> TropicalSemiringElement {
        let value = if degree < self.coefficients.len() {
            self.coefficients[degree]
        } else {
            None
        };
        TropicalSemiringElement::from_value_with_type(
            value.unwrap_or(f64::INFINITY),
            self.tropical_type,
        )
    }

    /// Evaluate the polynomial at a point
    pub fn eval(&self, x: f64) -> TropicalSemiringElement {
        let mut result = TropicalSemiringElement::infinity(self.tropical_type);

        for (i, coeff) in self.coefficients.iter().enumerate() {
            if let Some(c) = coeff {
                let term = TropicalSemiringElement::from_value_with_type(*c, self.tropical_type)
                    .tropical_mul(&TropicalSemiringElement::from_value_with_type(
                        x,
                        self.tropical_type,
                    ).tropical_pow(i as i32));
                result = result.tropical_add(&term);
            }
        }

        result
    }

    /// Add two tropical polynomials
    pub fn add(&self, other: &Self) -> Self {
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = if i < self.coefficients.len() {
                self.coefficients[i]
            } else {
                None
            };
            let b = if i < other.coefficients.len() {
                other.coefficients[i]
            } else {
                None
            };

            let sum = match (a, b) {
                (None, None) => None,
                (Some(x), None) | (None, Some(x)) => Some(x),
                (Some(x), Some(y)) => match self.tropical_type {
                    TropicalType::Min => Some(x.min(y)),
                    TropicalType::Max => Some(x.max(y)),
                },
            };

            result.push(sum);
        }

        Self::new(result, self.tropical_type)
    }

    /// Multiply two tropical polynomials
    pub fn mul(&self, other: &Self) -> Self {
        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            return Self::new(vec![], self.tropical_type);
        }

        let result_len = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![None; result_len];

        for (i, a) in self.coefficients.iter().enumerate() {
            for (j, b) in other.coefficients.iter().enumerate() {
                if let (Some(av), Some(bv)) = (a, b) {
                    let product = av + bv; // Tropical multiplication
                    result[i + j] = match result[i + j] {
                        None => Some(product),
                        Some(current) => Some(match self.tropical_type {
                            TropicalType::Min => current.min(product),
                            TropicalType::Max => current.max(product),
                        }),
                    };
                }
            }
        }

        Self::new(result, self.tropical_type)
    }

    /// Find the roots (points where minimum is attained by multiple terms)
    pub fn roots(&self) -> Vec<f64> {
        let mut roots = Vec::new();

        // For each pair of terms, find where they intersect
        for i in 0..self.coefficients.len() {
            for j in (i + 1)..self.coefficients.len() {
                if let (Some(ai), Some(aj)) = (self.coefficients[i], self.coefficients[j]) {
                    // Solve ai + i*x = aj + j*x
                    if i != j {
                        let x = (aj - ai) / ((i as f64) - (j as f64));
                        roots.push(x);
                    }
                }
            }
        }

        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        roots
    }
}

impl fmt::Display for TropicalPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if let Some(c) = coeff {
                if !first {
                    write!(f, " ⊕ ")?;
                }

                if i == 0 {
                    write!(f, "{}", c)?;
                } else if i == 1 {
                    write!(f, "({} ⊗ x)", c)?;
                } else {
                    write!(f, "({} ⊗ x^{})", c, i)?;
                }

                first = false;
            }
        }

        if first {
            write!(f, "∞")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_polynomial_semiring() {
        let ring = TropicalPolynomialSemiring::new(TropicalType::Min);
        assert_eq!(ring.tropical_type(), TropicalType::Min);
        assert_eq!(ring.variable(), "x");
    }

    #[test]
    fn test_polynomial_degree() {
        let poly = TropicalPolynomial::new(vec![Some(1.0), Some(2.0), None, Some(3.0)], TropicalType::Min);
        assert_eq!(poly.degree(), Some(3));

        let zero = TropicalPolynomial::new(vec![None, None], TropicalType::Min);
        assert_eq!(zero.degree(), None);
    }

    #[test]
    fn test_polynomial_eval() {
        // p(x) = 0 ⊕ (1 ⊗ x) = min(0, 1 + x)
        let poly = TropicalPolynomial::new(vec![Some(0.0), Some(1.0)], TropicalType::Min);

        let val = poly.eval(0.0);
        assert_eq!(val.value(), Some(0.0)); // min(0, 1) = 0

        let val = poly.eval(1.0);
        assert_eq!(val.value(), Some(0.0)); // min(0, 2) = 0

        let val = poly.eval(-2.0);
        assert_eq!(val.value(), Some(-1.0)); // min(0, -1) = -1
    }

    #[test]
    fn test_polynomial_addition() {
        let p1 = TropicalPolynomial::new(vec![Some(1.0), Some(2.0)], TropicalType::Min);
        let p2 = TropicalPolynomial::new(vec![Some(0.0), Some(3.0)], TropicalType::Min);

        let sum = p1.add(&p2);
        assert_eq!(sum.coefficients, vec![Some(0.0), Some(2.0)]);
    }

    #[test]
    fn test_polynomial_multiplication() {
        let p1 = TropicalPolynomial::new(vec![Some(1.0), Some(2.0)], TropicalType::Min);
        let p2 = TropicalPolynomial::new(vec![Some(0.0), Some(1.0)], TropicalType::Min);

        let product = p1.mul(&p2);
        // (1 ⊕ 2⊗x) ⊗ (0 ⊕ 1⊗x) = 1 ⊕ (min(2,2)⊗x) ⊕ 3⊗x²
        assert_eq!(product.degree(), Some(2));
    }

    #[test]
    fn test_monomial() {
        let ring = TropicalPolynomialSemiring::new(TropicalType::Min);
        let mono = ring.monomial(5.0, 3);

        assert_eq!(mono.degree(), Some(3));
        assert_eq!(mono.coefficients[3], Some(5.0));
    }

    #[test]
    fn test_roots() {
        let poly = TropicalPolynomial::new(vec![Some(0.0), Some(1.0)], TropicalType::Min);
        let roots = poly.roots();

        // 0 = 1 + x => x = -1
        assert_eq!(roots.len(), 1);
        assert!((roots[0] - (-1.0)).abs() < 1e-10);
    }
}
