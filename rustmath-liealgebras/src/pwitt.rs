//! Polynomial Witt Lie Algebra
//!
//! The polynomial Witt algebra is a subalgebra of the Witt algebra consisting
//! of polynomial derivations on C[t] (as opposed to Laurent polynomials).
//!
//! The polynomial Witt algebra has basis elements d_n for n ≥ 0,
//! representing the derivations d_n = -t^{n+1}d/dt on the polynomial ring.
//!
//! The Lie bracket satisfies:
//! - [d_m, d_n] = (m - n)d_{m+n}  for m, n ≥ 0
//!
//! This is a proper subalgebra of the full Witt algebra, containing only
//! the non-negative index generators.
//!
//! Corresponds to sage.algebras.lie_algebras.examples.pwitt
//!
//! References:
//! - Kac, V. "Infinite-dimensional Lie algebras" (1990)
//! - Fialowski, A. "Deformations of some infinite-dimensional Lie algebras" (1990)

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Polynomial Witt Lie Algebra
///
/// The infinite-dimensional polynomial Witt algebra with generators
/// d_n for n ≥ 0, representing polynomial derivations.
///
/// This is the positive (or polynomial) part of the Witt algebra.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::pwitt::PolynomialWittAlgebra;
/// let pwitt: PolynomialWittAlgebra<i64> = PolynomialWittAlgebra::new();
/// assert!(!pwitt.is_finite_dimensional());
/// ```
pub struct PolynomialWittAlgebra<R: Ring> {
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> PolynomialWittAlgebra<R> {
    /// Create a new polynomial Witt algebra
    pub fn new() -> Self {
        PolynomialWittAlgebra {
            coefficient_ring: PhantomData,
        }
    }

    /// Check if this is finite dimensional (always false)
    pub fn is_finite_dimensional(&self) -> bool {
        false
    }

    /// Check if this is nilpotent (always false)
    pub fn is_nilpotent(&self) -> bool {
        false
    }

    /// Check if this is solvable (always false)
    pub fn is_solvable(&self) -> bool {
        false
    }

    /// Get the zero element
    pub fn zero(&self) -> PolynomialWittElement<R>
    where
        R: From<i64>,
    {
        PolynomialWittElement::zero()
    }

    /// Get the generator d_n (for n ≥ 0)
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the generator (must be non-negative)
    ///
    /// # Returns
    ///
    /// The generator d_n with coefficient 1, or None if n < 0
    pub fn generator(&self, n: u64) -> PolynomialWittElement<R>
    where
        R: From<i64>,
    {
        PolynomialWittElement::generator(n)
    }

    /// Compute the Lie bracket [d_m, d_n]
    ///
    /// Returns (m - n)d_{m+n}
    ///
    /// # Arguments
    ///
    /// * `m` - Index of first generator (non-negative)
    /// * `n` - Index of second generator (non-negative)
    ///
    /// # Returns
    ///
    /// The bracket [d_m, d_n] = (m - n)d_{m+n}
    pub fn bracket_generators(&self, m: u64, n: u64) -> PolynomialWittElement<R>
    where
        R: From<i64>,
    {
        let m_i64 = m as i64;
        let n_i64 = n as i64;
        let coeff = m_i64 - n_i64;

        if coeff == 0 {
            PolynomialWittElement::zero()
        } else {
            let mut terms = HashMap::new();
            terms.insert(m + n, R::from(coeff));
            PolynomialWittElement { terms }
        }
    }

    /// Compute the bracket of two polynomial Witt elements
    pub fn bracket(
        &self,
        x: &PolynomialWittElement<R>,
        y: &PolynomialWittElement<R>,
    ) -> PolynomialWittElement<R>
    where
        R: From<i64>
            + std::ops::Add<Output = R>
            + std::ops::Sub<Output = R>
            + std::ops::Mul<Output = R>
            + PartialEq,
    {
        x.bracket(y)
    }
}

impl<R: Ring + Clone> Default for PolynomialWittAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring + Clone> Display for PolynomialWittAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Polynomial Witt algebra")
    }
}

/// Element of the Polynomial Witt Algebra
///
/// Represents a finite linear combination of generators d_n for n ≥ 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialWittElement<R: Ring> {
    /// Map from generator index (≥ 0) to coefficient
    /// Represents ∑ terms[n] * d_n
    terms: HashMap<u64, R>,
}

impl<R: Ring + Clone> PolynomialWittElement<R> {
    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        PolynomialWittElement {
            terms: HashMap::new(),
        }
    }

    /// Create a generator d_n with coefficient 1 (for n ≥ 0)
    pub fn generator(n: u64) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(n, R::from(1));
        PolynomialWittElement { terms }
    }

    /// Create a custom element from a map
    pub fn from_terms(terms: HashMap<u64, R>) -> Self {
        PolynomialWittElement { terms }
    }

    /// Get the coefficient of d_n
    pub fn coefficient(&self, n: u64) -> Option<&R> {
        self.terms.get(&n)
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i64>,
    {
        self.terms.is_empty() || self.terms.values().all(|c| c == &R::from(0))
    }

    /// Add two elements
    pub fn add(&self, other: &PolynomialWittElement<R>) -> PolynomialWittElement<R>
    where
        R: std::ops::Add<Output = R> + PartialEq + From<i64>,
    {
        let mut result = self.terms.clone();
        for (idx, coeff) in &other.terms {
            result
                .entry(*idx)
                .and_modify(|e| *e = e.clone() + coeff.clone())
                .or_insert_with(|| coeff.clone());
        }
        // Remove zero coefficients
        result.retain(|_, v| v != &R::from(0));
        PolynomialWittElement { terms: result }
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: &R) -> PolynomialWittElement<R>
    where
        R: std::ops::Mul<Output = R> + PartialEq + From<i64>,
    {
        if scalar == &R::from(0) {
            return PolynomialWittElement::zero();
        }
        let mut result = HashMap::new();
        for (idx, coeff) in &self.terms {
            result.insert(*idx, scalar.clone() * coeff.clone());
        }
        PolynomialWittElement { terms: result }
    }

    /// Compute the Lie bracket [self, other]
    ///
    /// Uses bilinearity: [∑ aᵢdᵢ, ∑ bⱼdⱼ] = ∑ aᵢbⱼ[dᵢ, dⱼ]
    pub fn bracket(&self, other: &PolynomialWittElement<R>) -> PolynomialWittElement<R>
    where
        R: From<i64>
            + std::ops::Add<Output = R>
            + std::ops::Sub<Output = R>
            + std::ops::Mul<Output = R>
            + PartialEq,
    {
        let mut result_terms = HashMap::new();

        for (m, coeff_m) in &self.terms {
            for (n, coeff_n) in &other.terms {
                // [d_m, d_n] = (m - n)d_{m+n}
                let m_i64 = *m as i64;
                let n_i64 = *n as i64;
                let bracket_coeff = m_i64 - n_i64;

                if bracket_coeff != 0 {
                    let result_idx = m + n;
                    let contribution = R::from(bracket_coeff)
                        * coeff_m.clone()
                        * coeff_n.clone();

                    result_terms
                        .entry(result_idx)
                        .and_modify(|e: &mut R| *e = e.clone() + contribution.clone())
                        .or_insert(contribution);
                }
            }
        }

        // Remove zero coefficients
        result_terms.retain(|_, v| v != &R::from(0));

        PolynomialWittElement {
            terms: result_terms,
        }
    }
}

impl<R: Ring + Clone> Display for PolynomialWittElement<R>
where
    R: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let mut indices: Vec<_> = self.terms.keys().collect();
        indices.sort();

        let mut first = true;
        for idx in indices {
            if let Some(coeff) = self.terms.get(idx) {
                if !first {
                    write!(f, " + ")?;
                }
                write!(f, "{}*d_{}", coeff, idx)?;
                first = false;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_polynomial_witt_algebra_creation() {
        let pwitt: PolynomialWittAlgebra<Integer> = PolynomialWittAlgebra::new();
        assert!(!pwitt.is_finite_dimensional());
        assert!(!pwitt.is_nilpotent());
        assert!(!pwitt.is_solvable());
    }

    #[test]
    fn test_polynomial_witt_generators() {
        let pwitt: PolynomialWittAlgebra<Integer> = PolynomialWittAlgebra::new();
        let d0 = pwitt.generator(0);
        let d1 = pwitt.generator(1);
        let d2 = pwitt.generator(2);

        assert!(!d0.is_zero());
        assert_eq!(d0.coefficient(0), Some(&Integer::from(1)));
        assert_eq!(d1.coefficient(1), Some(&Integer::from(1)));
        assert_eq!(d2.coefficient(2), Some(&Integer::from(1)));
    }

    #[test]
    fn test_polynomial_witt_bracket() {
        let pwitt: PolynomialWittAlgebra<Integer> = PolynomialWittAlgebra::new();

        // [d_1, d_2] = (1-2)d_3 = -d_3
        let bracket = pwitt.bracket_generators(1, 2);
        assert_eq!(bracket.coefficient(3), Some(&Integer::from(-1)));

        // [d_2, d_1] = (2-1)d_3 = d_3
        let bracket2 = pwitt.bracket_generators(2, 1);
        assert_eq!(bracket2.coefficient(3), Some(&Integer::from(1)));

        // [d_0, d_1] = (0-1)d_1 = -d_1
        let bracket3 = pwitt.bracket_generators(0, 1);
        assert_eq!(bracket3.coefficient(1), Some(&Integer::from(-1)));

        // [d_0, d_0] = 0
        let bracket4 = pwitt.bracket_generators(0, 0);
        assert!(bracket4.is_zero());
    }

    #[test]
    fn test_polynomial_witt_element_operations() {
        let d1 = PolynomialWittElement::<Integer>::generator(1);
        let d2 = PolynomialWittElement::<Integer>::generator(2);

        // Addition
        let sum = d1.add(&d2);
        assert_eq!(sum.coefficient(1), Some(&Integer::from(1)));
        assert_eq!(sum.coefficient(2), Some(&Integer::from(1)));

        // Scalar multiplication
        let scaled = d1.scale(&Integer::from(3));
        assert_eq!(scaled.coefficient(1), Some(&Integer::from(3)));
    }

    #[test]
    fn test_polynomial_witt_element_bracket() {
        let d1 = PolynomialWittElement::<Integer>::generator(1);
        let d2 = PolynomialWittElement::<Integer>::generator(2);

        // [d_1, d_2] = -d_3
        let bracket = d1.bracket(&d2);
        assert_eq!(bracket.coefficient(3), Some(&Integer::from(-1)));

        // Anti-symmetry: [d_2, d_1] = -[d_1, d_2] = d_3
        let bracket_rev = d2.bracket(&d1);
        assert_eq!(bracket_rev.coefficient(3), Some(&Integer::from(1)));
    }

    #[test]
    fn test_polynomial_witt_jacobi_identity() {
        // Test Jacobi identity: [x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0
        let d0 = PolynomialWittElement::<Integer>::generator(0);
        let d1 = PolynomialWittElement::<Integer>::generator(1);
        let d2 = PolynomialWittElement::<Integer>::generator(2);

        let yz = d1.bracket(&d2);
        let x_yz = d0.bracket(&yz);

        let zx = d2.bracket(&d0);
        let y_zx = d1.bracket(&zx);

        let xy = d0.bracket(&d1);
        let z_xy = d2.bracket(&xy);

        let total = x_yz.add(&y_zx).add(&z_xy);
        assert!(total.is_zero());
    }

    #[test]
    fn test_polynomial_witt_display() {
        let d1 = PolynomialWittElement::<Integer>::generator(1);
        let display = format!("{}", d1);
        assert!(display.contains("d_1"));
    }

    #[test]
    fn test_polynomial_witt_bracket_closure() {
        // Verify that polynomial Witt is closed under bracket
        // [d_m, d_n] = (m-n)d_{m+n} should always have index m+n ≥ 0
        let d0 = PolynomialWittElement::<Integer>::generator(0);
        let d1 = PolynomialWittElement::<Integer>::generator(1);
        let d2 = PolynomialWittElement::<Integer>::generator(2);

        let b01 = d0.bracket(&d1);
        assert!(!b01.is_zero()); // [d_0, d_1] = -d_1

        let b12 = d1.bracket(&d2);
        assert!(!b12.is_zero()); // [d_1, d_2] = -d_3

        // All indices should be non-negative
        for key in b01.terms.keys() {
            assert!(*key >= 0);
        }
        for key in b12.terms.keys() {
            assert!(*key >= 0);
        }
    }
}
