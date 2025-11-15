//! Witt Lie Algebra
//!
//! The Witt algebra is an infinite-dimensional Lie algebra consisting of
//! derivations on the Laurent polynomial ring C[t, t⁻¹].
//!
//! The Witt algebra has basis elements d_n for n ∈ ℤ (or L_n in physics notation),
//! representing the derivations d_n = -t^{n+1}d/dt.
//!
//! The Lie bracket satisfies:
//! - [d_m, d_n] = (m - n)d_{m+n}
//!
//! The Witt algebra is also known as the centerless Virasoro algebra.
//! It appears in the study of conformal field theory and modular forms.
//!
//! Corresponds to sage.algebras.lie_algebras.examples.witt
//!
//! References:
//! - Kac, V. "Infinite-dimensional Lie algebras" (1990)
//! - Iohara, Koga "Representation Theory of the Virasoro Algebra" (2011)

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Witt Lie Algebra
///
/// The infinite-dimensional Witt algebra with generators d_n (n ∈ ℤ),
/// representing derivations on the Laurent polynomial ring.
///
/// This is the centerless Virasoro algebra.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::witt::WittAlgebra;
/// let witt: WittAlgebra<i64> = WittAlgebra::new();
/// assert!(!witt.is_finite_dimensional());
/// ```
pub struct WittAlgebra<R: Ring> {
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> WittAlgebra<R> {
    /// Create a new Witt algebra
    pub fn new() -> Self {
        WittAlgebra {
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

    /// Check if this is simple (always true over characteristic 0)
    pub fn is_simple(&self) -> bool {
        true
    }

    /// Get the zero element
    pub fn zero(&self) -> WittElement<R>
    where
        R: From<i64>,
    {
        WittElement::zero()
    }

    /// Get the generator d_n
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the generator
    ///
    /// # Returns
    ///
    /// The generator d_n with coefficient 1
    pub fn generator(&self, n: i64) -> WittElement<R>
    where
        R: From<i64>,
    {
        WittElement::generator(n)
    }

    /// Compute the Lie bracket [d_m, d_n]
    ///
    /// Returns (m - n)d_{m+n}
    ///
    /// # Arguments
    ///
    /// * `m` - Index of first generator
    /// * `n` - Index of second generator
    ///
    /// # Returns
    ///
    /// The bracket [d_m, d_n] = (m - n)d_{m+n}
    pub fn bracket_generators(&self, m: i64, n: i64) -> WittElement<R>
    where
        R: From<i64> + std::ops::Sub<Output = R>,
    {
        let coeff = m - n;
        if coeff == 0 {
            WittElement::zero()
        } else {
            let mut terms = HashMap::new();
            terms.insert(m + n, R::from(coeff));
            WittElement { terms }
        }
    }

    /// Compute the bracket of two Witt elements
    pub fn bracket(&self, x: &WittElement<R>, y: &WittElement<R>) -> WittElement<R>
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

impl<R: Ring + Clone> Default for WittAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring + Clone> Display for WittAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Witt algebra")
    }
}

/// Element of the Witt Algebra
///
/// Represents a finite linear combination of generators d_n.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WittElement<R: Ring> {
    /// Map from generator index to coefficient
    /// Represents ∑ terms[n] * d_n
    terms: HashMap<i64, R>,
}

impl<R: Ring + Clone> WittElement<R> {
    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        WittElement {
            terms: HashMap::new(),
        }
    }

    /// Create a generator d_n with coefficient 1
    pub fn generator(n: i64) -> Self
    where
        R: From<i64>,
    {
        let mut terms = HashMap::new();
        terms.insert(n, R::from(1));
        WittElement { terms }
    }

    /// Create a custom element from a map
    pub fn from_terms(terms: HashMap<i64, R>) -> Self {
        WittElement { terms }
    }

    /// Get the coefficient of d_n
    pub fn coefficient(&self, n: i64) -> Option<&R> {
        self.terms.get(&n)
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i64>,
    {
        self.terms.is_empty()
            || self.terms.values().all(|c| c == &R::from(0))
    }

    /// Add two elements
    pub fn add(&self, other: &WittElement<R>) -> WittElement<R>
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
        WittElement { terms: result }
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: &R) -> WittElement<R>
    where
        R: std::ops::Mul<Output = R> + PartialEq + From<i64>,
    {
        if scalar == &R::from(0) {
            return WittElement::zero();
        }
        let mut result = HashMap::new();
        for (idx, coeff) in &self.terms {
            result.insert(*idx, scalar.clone() * coeff.clone());
        }
        WittElement { terms: result }
    }

    /// Compute the Lie bracket [self, other]
    ///
    /// Uses bilinearity: [∑ aᵢdᵢ, ∑ bⱼdⱼ] = ∑ aᵢbⱼ[dᵢ, dⱼ]
    pub fn bracket(&self, other: &WittElement<R>) -> WittElement<R>
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
                let bracket_coeff = *m - *n;
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

        WittElement {
            terms: result_terms,
        }
    }
}

impl<R: Ring + Clone> Display for WittElement<R>
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
    fn test_witt_algebra_creation() {
        let witt: WittAlgebra<Integer> = WittAlgebra::new();
        assert!(!witt.is_finite_dimensional());
        assert!(!witt.is_nilpotent());
        assert!(!witt.is_solvable());
        assert!(witt.is_simple());
    }

    #[test]
    fn test_witt_generators() {
        let witt: WittAlgebra<Integer> = WittAlgebra::new();
        let d0 = witt.generator(0);
        let d1 = witt.generator(1);
        let d_minus1 = witt.generator(-1);

        assert!(!d0.is_zero());
        assert_eq!(d0.coefficient(0), Some(&Integer::from(1)));
        assert_eq!(d1.coefficient(1), Some(&Integer::from(1)));
        assert_eq!(d_minus1.coefficient(-1), Some(&Integer::from(1)));
    }

    #[test]
    fn test_witt_bracket() {
        let witt: WittAlgebra<Integer> = WittAlgebra::new();

        // [d_1, d_2] = (1-2)d_3 = -d_3
        let bracket = witt.bracket_generators(1, 2);
        assert_eq!(bracket.coefficient(3), Some(&Integer::from(-1)));

        // [d_2, d_1] = (2-1)d_3 = d_3
        let bracket2 = witt.bracket_generators(2, 1);
        assert_eq!(bracket2.coefficient(3), Some(&Integer::from(1)));

        // [d_1, d_{-1}] = (1-(-1))d_0 = 2d_0
        let bracket3 = witt.bracket_generators(1, -1);
        assert_eq!(bracket3.coefficient(0), Some(&Integer::from(2)));

        // [d_0, d_0] = 0
        let bracket4 = witt.bracket_generators(0, 0);
        assert!(bracket4.is_zero());
    }

    #[test]
    fn test_witt_element_operations() {
        let d1 = WittElement::<Integer>::generator(1);
        let d2 = WittElement::<Integer>::generator(2);

        // Addition
        let sum = d1.add(&d2);
        assert_eq!(sum.coefficient(1), Some(&Integer::from(1)));
        assert_eq!(sum.coefficient(2), Some(&Integer::from(1)));

        // Scalar multiplication
        let scaled = d1.scale(&Integer::from(3));
        assert_eq!(scaled.coefficient(1), Some(&Integer::from(3)));
    }

    #[test]
    fn test_witt_element_bracket() {
        let d1 = WittElement::<Integer>::generator(1);
        let d2 = WittElement::<Integer>::generator(2);

        // [d_1, d_2] = -d_3
        let bracket = d1.bracket(&d2);
        assert_eq!(bracket.coefficient(3), Some(&Integer::from(-1)));

        // Anti-symmetry: [d_2, d_1] = -[d_1, d_2] = d_3
        let bracket_rev = d2.bracket(&d1);
        assert_eq!(bracket_rev.coefficient(3), Some(&Integer::from(1)));
    }

    #[test]
    fn test_witt_jacobi_identity() {
        // Test Jacobi identity: [x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0
        let d0 = WittElement::<Integer>::generator(0);
        let d1 = WittElement::<Integer>::generator(1);
        let d2 = WittElement::<Integer>::generator(2);

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
    fn test_witt_display() {
        let d1 = WittElement::<Integer>::generator(1);
        let display = format!("{}", d1);
        assert!(display.contains("d_1"));
    }
}
