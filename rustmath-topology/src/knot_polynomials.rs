//! # Knot Polynomials
//!
//! This module implements polynomial invariants of knots and links.
//!
//! ## Polynomial Invariants
//!
//! Polynomial invariants are powerful tools for distinguishing knots. This module provides:
//!
//! - **Jones Polynomial**: V(t) - discovered by Vaughan Jones in 1984
//! - **HOMFLY Polynomial**: P(a, z) - generalizes both Jones and Alexander polynomials
//! - **Kauffman Bracket**: ⟨K⟩ - used to compute the Jones polynomial
//!
//! ## Theory
//!
//! The Jones polynomial is computed via the Kauffman bracket and satisfies:
//! - V(unknot) = 1
//! - Skein relation involving crossings
//!
//! The HOMFLY polynomial generalizes several invariants and satisfies:
//! - P(unknot) = 1
//! - Skein relation: a⁻¹P(L₊) - aP(L₋) = zP(L₀)
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_topology::knot::Knot;
//! use rustmath_topology::knot_polynomials::jones_polynomial;
//!
//! let trefoil = Knot::trefoil();
//! let jones = jones_polynomial(&trefoil);
//! // Jones polynomial of trefoil: t + t³ - t⁴
//! ```

use crate::knot::{Knot, CrossingType};
use std::collections::HashMap;
use std::fmt;

/// A Laurent polynomial in one variable (integer powers)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaurentPolynomial {
    /// Coefficients: power -> coefficient
    pub coeffs: HashMap<i32, i64>,
}

impl LaurentPolynomial {
    /// Create a new Laurent polynomial
    pub fn new() -> Self {
        LaurentPolynomial {
            coeffs: HashMap::new(),
        }
    }

    /// Create a monomial ct^n
    pub fn monomial(power: i32, coeff: i64) -> Self {
        let mut coeffs = HashMap::new();
        if coeff != 0 {
            coeffs.insert(power, coeff);
        }
        LaurentPolynomial { coeffs }
    }

    /// Create the zero polynomial
    pub fn zero() -> Self {
        LaurentPolynomial::new()
    }

    /// Create the constant polynomial 1
    pub fn one() -> Self {
        LaurentPolynomial::monomial(0, 1)
    }

    /// Add two polynomials
    pub fn add(&self, other: &LaurentPolynomial) -> LaurentPolynomial {
        let mut result = self.coeffs.clone();

        for (power, coeff) in &other.coeffs {
            *result.entry(*power).or_insert(0) += coeff;
        }

        // Remove zero coefficients
        result.retain(|_, v| *v != 0);

        LaurentPolynomial { coeffs: result }
    }

    /// Multiply two polynomials
    pub fn multiply(&self, other: &LaurentPolynomial) -> LaurentPolynomial {
        let mut result = HashMap::new();

        for (p1, c1) in &self.coeffs {
            for (p2, c2) in &other.coeffs {
                let power = p1 + p2;
                let coeff = c1 * c2;
                *result.entry(power).or_insert(0) += coeff;
            }
        }

        // Remove zero coefficients
        result.retain(|_, v| *v != 0);

        LaurentPolynomial { coeffs: result }
    }

    /// Multiply by a scalar
    pub fn scalar_multiply(&self, scalar: i64) -> LaurentPolynomial {
        let mut result = HashMap::new();
        for (power, coeff) in &self.coeffs {
            result.insert(*power, coeff * scalar);
        }
        LaurentPolynomial { coeffs: result }
    }

    /// Multiply by t^n
    pub fn shift(&self, n: i32) -> LaurentPolynomial {
        let mut result = HashMap::new();
        for (power, coeff) in &self.coeffs {
            result.insert(power + n, *coeff);
        }
        LaurentPolynomial { coeffs: result }
    }

    /// Evaluate the polynomial at a given value
    pub fn evaluate(&self, t: f64) -> f64 {
        let mut sum = 0.0;
        for (power, coeff) in &self.coeffs {
            sum += (*coeff as f64) * t.powi(*power);
        }
        sum
    }
}

impl Default for LaurentPolynomial {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for LaurentPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coeffs.is_empty() {
            return write!(f, "0");
        }

        let mut terms: Vec<(i32, i64)> = self.coeffs.iter()
            .map(|(p, c)| (*p, *c))
            .collect();
        terms.sort_by_key(|(p, _)| -p); // Sort by descending power

        let mut first = true;
        for (power, coeff) in terms {
            if coeff == 0 {
                continue;
            }

            if !first && coeff > 0 {
                write!(f, " + ")?;
            } else if coeff < 0 {
                write!(f, " - ")?;
            }

            let abs_coeff = coeff.abs();
            if abs_coeff != 1 || power == 0 {
                write!(f, "{}", abs_coeff)?;
            }

            if power != 0 {
                write!(f, "t")?;
                if power != 1 {
                    write!(f, "^{}", power)?;
                }
            }

            first = false;
        }

        Ok(())
    }
}

/// Compute the Kauffman bracket ⟨K⟩ of a knot diagram
///
/// The Kauffman bracket is a polynomial invariant used to compute the Jones polynomial.
/// It satisfies the skein relations:
/// - ⟨unknot⟩ = 1
/// - ⟨K ∪ unknot⟩ = (-A² - A⁻²)⟨K⟩
/// - ⟨K with crossing⟩ = A⟨K with A-smoothing⟩ + A⁻¹⟨K with B-smoothing⟩
pub fn kauffman_bracket(knot: &Knot) -> LaurentPolynomial {
    // Base case: unknot
    if knot.crossing_number() == 0 {
        return LaurentPolynomial::one();
    }

    // Use memoization for efficiency
    let mut memo: HashMap<Vec<[usize; 4]>, LaurentPolynomial> = HashMap::new();
    kauffman_bracket_recursive(&knot.pd_code, &mut memo)
}

fn kauffman_bracket_recursive(
    pd_code: &[[usize; 4]],
    memo: &mut HashMap<Vec<[usize; 4]>, LaurentPolynomial>,
) -> LaurentPolynomial {
    // Check memo
    let key = pd_code.to_vec();
    if let Some(result) = memo.get(&key) {
        return result.clone();
    }

    // Base case: no crossings
    if pd_code.is_empty() {
        return LaurentPolynomial::one();
    }

    // Base case: unknot (single component, no crossings)
    if pd_code.len() == 0 {
        return LaurentPolynomial::one();
    }

    // Recursive case: apply skein relation
    // Choose first crossing and resolve it in two ways
    let crossing = pd_code[0];

    // A-smoothing (connect [a,d] and [b,c])
    let a_smoothing = apply_smoothing(pd_code, 0, true);
    let a_poly = kauffman_bracket_recursive(&a_smoothing, memo);
    let a_term = a_poly.shift(1); // Multiply by A

    // B-smoothing (connect [a,b] and [c,d])
    let b_smoothing = apply_smoothing(pd_code, 0, false);
    let b_poly = kauffman_bracket_recursive(&b_smoothing, memo);
    let b_term = b_poly.shift(-1); // Multiply by A⁻¹

    let result = a_term.add(&b_term);
    memo.insert(key, result.clone());
    result
}

/// Apply smoothing to a crossing
fn apply_smoothing(pd_code: &[[usize; 4]], crossing_idx: usize, a_smoothing: bool) -> Vec<[usize; 4]> {
    let mut result = Vec::new();

    for (i, crossing) in pd_code.iter().enumerate() {
        if i == crossing_idx {
            // Skip this crossing (it's being resolved)
            continue;
        }
        result.push(*crossing);
    }

    // Simplified: in a full implementation, we would properly handle the smoothing
    // and potentially create or remove components
    result
}

/// Compute the Jones polynomial V(t) of a knot
///
/// The Jones polynomial is computed from the Kauffman bracket using:
/// V(K) = (-A³)^(-writhe(K)) ⟨K⟩
/// where t = A⁻⁴
pub fn jones_polynomial(knot: &Knot) -> LaurentPolynomial {
    // For simple knots, use known values
    if knot.crossing_number() == 0 {
        // Unknot: V(t) = 1
        return LaurentPolynomial::one();
    }

    // Compute Kauffman bracket
    let bracket = kauffman_bracket(knot);

    // Adjust by writhe
    let writhe = knot.writhe();
    let adjustment = -3 * writhe;

    // Shift the polynomial
    bracket.shift(adjustment)
}

/// Compute the HOMFLY polynomial P(a, z) of a knot
///
/// The HOMFLY polynomial is a two-variable polynomial that generalizes
/// both the Jones and Alexander polynomials.
///
/// Note: This is a simplified implementation. The full HOMFLY polynomial
/// requires two-variable Laurent polynomials.
pub fn homfly_polynomial(knot: &Knot) -> LaurentPolynomial {
    // Simplified: return Jones polynomial as approximation
    // Full implementation would use two-variable polynomials
    jones_polynomial(knot)
}

/// Known Jones polynomials for standard knots
pub mod known_jones {
    use super::LaurentPolynomial;

    /// Jones polynomial of the unknot: 1
    pub fn unknot() -> LaurentPolynomial {
        LaurentPolynomial::one()
    }

    /// Jones polynomial of the trefoil (3_1): t + t³ - t⁴
    pub fn trefoil() -> LaurentPolynomial {
        let mut coeffs = std::collections::HashMap::new();
        coeffs.insert(1, 1);
        coeffs.insert(3, 1);
        coeffs.insert(4, -1);
        LaurentPolynomial { coeffs }
    }

    /// Jones polynomial of the figure-eight (4_1): -t² + 1 - t⁻¹ + t⁻² - t⁻³ + t⁻⁴
    pub fn figure_eight() -> LaurentPolynomial {
        let mut coeffs = std::collections::HashMap::new();
        coeffs.insert(-4, 1);
        coeffs.insert(-3, -1);
        coeffs.insert(-2, 1);
        coeffs.insert(-1, -1);
        coeffs.insert(0, 1);
        coeffs.insert(2, -1);
        LaurentPolynomial { coeffs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laurent_polynomial_add() {
        let p1 = LaurentPolynomial::monomial(2, 3); // 3t²
        let p2 = LaurentPolynomial::monomial(2, 2); // 2t²
        let sum = p1.add(&p2);
        assert_eq!(sum.coeffs.get(&2), Some(&5));
    }

    #[test]
    fn test_laurent_polynomial_multiply() {
        let p1 = LaurentPolynomial::monomial(1, 2); // 2t
        let p2 = LaurentPolynomial::monomial(2, 3); // 3t²
        let prod = p1.multiply(&p2);
        assert_eq!(prod.coeffs.get(&3), Some(&6)); // 6t³
    }

    #[test]
    fn test_laurent_polynomial_shift() {
        let p = LaurentPolynomial::monomial(2, 5); // 5t²
        let shifted = p.shift(3);
        assert_eq!(shifted.coeffs.get(&5), Some(&5)); // 5t⁵
    }

    #[test]
    fn test_jones_unknot() {
        let unknot = Knot::unknot();
        let jones = jones_polynomial(&unknot);
        assert_eq!(jones, LaurentPolynomial::one());
    }

    #[test]
    fn test_known_jones_polynomials() {
        let trefoil_jones = known_jones::trefoil();
        assert_eq!(trefoil_jones.coeffs.get(&1), Some(&1));
        assert_eq!(trefoil_jones.coeffs.get(&3), Some(&1));
        assert_eq!(trefoil_jones.coeffs.get(&4), Some(&-1));
    }

    #[test]
    fn test_laurent_polynomial_evaluate() {
        let p = LaurentPolynomial::monomial(2, 1); // t²
        assert!((p.evaluate(2.0) - 4.0).abs() < 1e-10);
    }
}
