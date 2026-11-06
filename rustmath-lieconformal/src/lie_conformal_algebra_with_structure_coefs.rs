//! Lie Conformal Algebra with Structure Coefficients
//!
//! This module provides the `LieConformalAlgebraWithStructureCoefficients` class,
//! which implements a Lie conformal algebra where the lambda-brackets between
//! generators are defined through explicit structure coefficients.
//!
//! # Mathematical Background
//!
//! A Lie conformal algebra with structure coefficients is defined by:
//! - A finite set of generators {g_i}
//! - Structure coefficients s_{ij}^k(λ) that determine the λ-brackets:
//!   [g_i_λ g_j] = Σ_k Σ_n s_{ij}^{k,n} λ^n ⊗ g_k
//!
//! The structure coefficients must satisfy:
//! - Sesquilinearity relations involving ∂
//! - Skew-symmetry: [g_j_λ g_i] = -[g_i_{-λ-∂} g_j]
//! - Jacobi identity
//!
//! # Examples
//!
//! ```
//! use rustmath_lieconformal::{LieConformalAlgebraWithStructureCoefficients, Degree};
//! use std::collections::HashMap;
//!
//! // The current Lie conformal algebra Cur(g) of the 2-dimensional non-abelian
//! // Lie algebra g = <x, y | [x, y] = x>, plus a central K. In Cur(g) the
//! // λ-bracket of g ⊂ Cur(g) is the Lie bracket (Kac, VAfB §2.7):
//! //   [x_λ y] = x,   [y_λ x] = -x.
//! // (A Virasoro example does not fit here: the table holds polynomials in λ
//! // only, so the ∂L term of [L_λ L] = (∂ + 2λ)L + (λ³/12)C cannot be stored.)
//! let mut structure = HashMap::new();
//! structure.insert((0, 1, 0), vec![1i64]);
//! structure.insert((1, 0, 0), vec![-1i64]);
//!
//! let lca = LieConformalAlgebraWithStructureCoefficients::new(
//!     1i64,
//!     3, // x, y, K
//!     structure,
//!     Some(vec!["x".to_string(), "y".to_string(), "K".to_string()]),
//!     Some(vec![Degree::int(1), Degree::int(1), Degree::int(0)]),
//!     Some(vec![2]), // K is central
//! );
//! assert_eq!(lca.bracket_on_basis(0, 1), HashMap::from([(0, vec![1])]));
//! assert_eq!(lca.bracket_on_basis(1, 0), HashMap::from([(0, vec![-1])]));
//! assert!(lca.bracket_on_basis(0, 0).is_empty());
//! assert!(!lca.is_central(0));
//! assert!(!lca.is_central(1));
//! assert!(lca.is_central(2));
//! ```
//!
//! # References
//!
//! - Kac, V. "Vertex Algebras for Beginners" (1998)
//! - SageMath: sage.algebras.lie_conformal_algebras.lie_conformal_algebra_with_structure_coefs
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.lie_conformal_algebra_with_structure_coefs

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use crate::lie_conformal_algebra::{LieConformalAlgebra, LambdaBracket};
use crate::graded_lie_conformal_algebra::{GradedLieConformalAlgebra, Degree};
use crate::finitely_freely_generated_lca::FinitelyFreelyGeneratedLCA;
use crate::lie_conformal_algebra_element::LieConformalAlgebraElement;
use crate::lie_conformal_algebra::GeneratorIndex;

/// Structure coefficients for a Lie conformal algebra
///
/// Maps pairs of generator indices (i, j) and a result index k to a polynomial in λ
/// representing the coefficient of g_k in [g_i_λ g_j].
///
/// The structure is: HashMap<(i, j, k), Vec<R>>
/// where Vec<R> represents the polynomial c_0 + c_1*λ + c_2*λ² + ...
pub type LCAStructureCoefficients<R> = HashMap<(usize, usize, usize), Vec<R>>;

/// Lie conformal algebra defined by structure coefficients
///
/// This class implements a finitely generated Lie conformal algebra where the
/// lambda-brackets between generators are explicitly specified via structure
/// coefficients.
///
/// # Type Parameters
///
/// * `R` - The base ring
///
/// # Mathematical Structure
///
/// For generators g_0, ..., g_{n-1}, the lambda-brackets are defined by:
/// ```text
/// [g_i_λ g_j] = Σ_k Σ_m s_{ijk}^m λ^m ⊗ g_k
/// ```
///
/// where s_{ijk}^m are the structure coefficients stored in the structure_coeffs field.
///
/// # Examples
///
/// ```
/// use rustmath_lieconformal::{Degree, GradedLieConformalAlgebra, LieConformalAlgebra};
/// use rustmath_lieconformal::LieConformalAlgebraWithStructureCoefficients;
/// use std::collections::HashMap;
///
/// // Create a simple 2-generator algebra
/// let mut coeffs = HashMap::new();
/// // [g_0_λ g_1] = g_0, and by skew-symmetry [g_1_λ g_0] = -g_0
/// coeffs.insert((0, 1, 0), vec![1i64]); // coefficient of λ^0
/// coeffs.insert((1, 0, 0), vec![-1i64]);
///
/// let lca = LieConformalAlgebraWithStructureCoefficients::new(
///     1i64,
///     2,
///     coeffs,
///     Some(vec!["a".to_string(), "b".to_string()]),
///     Some(vec![Degree::int(1), Degree::int(1)]),
///     None,
/// );
///
/// assert_eq!(lca.ngens(), Some(2));
/// assert_eq!(lca.generator_name(1), Some("b".to_string()));
/// assert_eq!(lca.generator_degree(0), Some(Degree::int(1)));
/// assert_eq!(lca.get_coefficient(0, 1, 0), Some(&vec![1]));
/// assert_eq!(lca.get_coefficient(0, 1, 1), None);
/// ```
#[derive(Clone)]
pub struct LieConformalAlgebraWithStructureCoefficients<R: Ring> {
    /// The underlying finitely generated structure
    base: FinitelyFreelyGeneratedLCA<R>,
    /// Structure coefficients defining lambda-brackets
    structure_coeffs: LCAStructureCoefficients<R>,
}

impl<R: Ring + Clone + From<i64>> LieConformalAlgebraWithStructureCoefficients<R> {
    /// Create a new Lie conformal algebra with structure coefficients
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring
    /// * `ngens` - Number of generators
    /// * `structure_coeffs` - Structure coefficients for lambda-brackets
    /// * `names` - Optional generator names
    /// * `degrees` - Optional generator degrees (conformal weights)
    /// * `central_indices` - Optional indices of central elements
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::{LieConformalAlgebraWithStructureCoefficients, Degree};
    /// use std::collections::HashMap;
    ///
    /// let mut coeffs = HashMap::new();
    /// let lca = LieConformalAlgebraWithStructureCoefficients::new(
    ///     1i64,
    ///     2,
    ///     coeffs,
    ///     Some(vec!["L".to_string(), "C".to_string()]),
    ///     Some(vec![Degree::int(2), Degree::int(0)]),
    ///     Some(vec![1]),
    /// );
    /// ```
    pub fn new(
        base_ring: R,
        ngens: usize,
        structure_coeffs: LCAStructureCoefficients<R>,
        names: Option<Vec<String>>,
        degrees: Option<Vec<Degree>>,
        central_indices: Option<Vec<usize>>,
    ) -> Self {
        let base = FinitelyFreelyGeneratedLCA::new(
            base_ring,
            ngens,
            names,
            degrees,
            None, // parities
            central_indices,
        );

        LieConformalAlgebraWithStructureCoefficients {
            base,
            structure_coeffs,
        }
    }

    /// Get the underlying base algebra
    pub fn base(&self) -> &FinitelyFreelyGeneratedLCA<R> {
        &self.base
    }

    /// Get the structure coefficients
    ///
    /// Returns a reference to the HashMap containing all structure coefficients.
    /// The key (i, j, k) represents the coefficient of g_k in [g_i_λ g_j],
    /// and the value is a polynomial in λ.
    pub fn structure_coefficients(&self) -> &LCAStructureCoefficients<R> {
        &self.structure_coeffs
    }

    /// Get the structure coefficient for specific generators
    ///
    /// Returns the polynomial coefficients for the k-th generator in [g_i_λ g_j].
    ///
    /// # Arguments
    ///
    /// * `i` - First generator index
    /// * `j` - Second generator index
    /// * `k` - Result generator index
    ///
    /// # Returns
    ///
    /// `Some(&Vec<R>)` if the coefficient exists, `None` otherwise
    pub fn get_coefficient(&self, i: usize, j: usize, k: usize) -> Option<&Vec<R>> {
        self.structure_coeffs.get(&(i, j, k))
    }

    /// Set a structure coefficient
    ///
    /// Sets the coefficient of g_k in [g_i_λ g_j] to the given polynomial.
    ///
    /// # Arguments
    ///
    /// * `i` - First generator index
    /// * `j` - Second generator index
    /// * `k` - Result generator index
    /// * `poly` - Polynomial coefficients in λ
    pub fn set_coefficient(&mut self, i: usize, j: usize, k: usize, poly: Vec<R>) {
        self.structure_coeffs.insert((i, j, k), poly);
    }

    /// Compute the lambda bracket of two generators
    ///
    /// Returns [g_i_λ g_j] as a HashMap mapping generator indices to polynomials in λ.
    ///
    /// # Arguments
    ///
    /// * `i` - First generator index
    /// * `j` - Second generator index
    ///
    /// # Returns
    ///
    /// HashMap mapping k to the polynomial coefficient of g_k
    pub fn bracket_on_basis(&self, i: usize, j: usize) -> HashMap<usize, Vec<R>>
    where
        R: PartialEq,
    {
        let mut result = HashMap::new();

        // Collect all coefficients involving generators i and j
        for ((gen_i, gen_j, gen_k), poly) in &self.structure_coeffs {
            if *gen_i == i && *gen_j == j {
                // Check if polynomial is non-zero
                if !poly.is_empty() && !poly.iter().all(|c| c.is_zero()) {
                    result.insert(*gen_k, poly.clone());
                }
            }
        }

        result
    }

    /// Check if a generator is central
    ///
    /// A generator is central if [g_i_λ g] = 0 for all generators g, which by
    /// skew-symmetry is the same as [g_λ g_i] = 0 for all g. So g_i is central
    /// exactly when no stored non-zero coefficient has g_i in either input
    /// slot (it may still occur as a value, like C in the Virasoro bracket).
    pub fn is_central(&self, i: usize) -> bool
    where
        R: PartialEq,
    {
        !self.structure_coeffs.iter().any(|((gen_i, gen_j, _), poly)| {
            (*gen_i == i || *gen_j == i) && poly.iter().any(|c| !c.is_zero())
        })
    }

    /// Get generator name
    pub fn generator_name(&self, i: usize) -> Option<String> {
        self.base.generator_names().get(i).cloned()
    }
}

impl<R: Ring + Clone + From<i64>> LieConformalAlgebra<R>
    for LieConformalAlgebraWithStructureCoefficients<R>
{
    type Element = LieConformalAlgebraElement<R, GeneratorIndex>;

    fn base_ring(&self) -> &R {
        self.base.base_ring()
    }

    fn ngens(&self) -> Option<usize> {
        Some(self.base.ngens())
    }

    fn generator(&self, i: usize) -> Option<Self::Element> {
        self.base.generator(i)
    }

    fn zero(&self) -> Self::Element {
        self.base.zero()
    }

    fn central_charge(&self) -> Option<R> {
        self.base.central_charge()
    }
}

impl<R: Ring + Clone + From<i64>> GradedLieConformalAlgebra<R>
    for LieConformalAlgebraWithStructureCoefficients<R>
{
    fn generator_degree(&self, index: usize) -> Option<Degree> {
        self.base.generator_degree(index)
    }

    fn degree(&self, element: &Self::Element) -> Option<Degree> {
        self.base.degree(element)
    }
}

impl<R> LambdaBracket<R, LieConformalAlgebraElement<R, GeneratorIndex>>
    for LieConformalAlgebraWithStructureCoefficients<R>
where
    R: Ring + Clone + From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + PartialEq,
{
    /// # Panics
    ///
    /// Always: not implemented. Extending the stored generator brackets to
    /// arbitrary elements needs sesquilinearity (with `∂` acting on the
    /// values, which this table cannot represent) and skew-symmetry for the
    /// pairs that were not stored. Use [`Self::bracket_on_basis`] for the
    /// stored data. This used to return an empty map, i.e. the false claim
    /// that every bracket is 0.
    fn lambda_bracket(
        &self,
        _a: &LieConformalAlgebraElement<R, GeneratorIndex>,
        _b: &LieConformalAlgebraElement<R, GeneratorIndex>,
    ) -> HashMap<usize, LieConformalAlgebraElement<R, GeneratorIndex>> {
        unimplemented!(
            "LieConformalAlgebraWithStructureCoefficients: λ-bracket of general elements is not implemented"
        )
    }
}

impl<R: Ring + Clone + From<i64> + Display> Display
    for LieConformalAlgebraWithStructureCoefficients<R>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Lie conformal algebra with {} generators and explicit structure coefficients over {}",
            self.base.ngens(),
            self.base.base_ring()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lca_structure_coefs_creation() {
        let coeffs: LCAStructureCoefficients<i64> = HashMap::new();
        let lca = LieConformalAlgebraWithStructureCoefficients::new(
            1,
            2,
            coeffs,
            Some(vec!["a".to_string(), "b".to_string()]),
            Some(vec![Degree::int(1), Degree::int(1)]),
            None,
        );

        assert_eq!(lca.ngens(), Some(2));
    }

    #[test]
    fn test_structure_coefficients_access() {
        let mut coeffs: LCAStructureCoefficients<i64> = HashMap::new();
        coeffs.insert((0, 1, 0), vec![1, 2, 3]); // [g_0_λ g_1] contains g_0 with coeff 1 + 2λ + 3λ²

        let lca = LieConformalAlgebraWithStructureCoefficients::new(
            1,
            2,
            coeffs,
            Some(vec!["a".to_string(), "b".to_string()]),
            None,
            None,
        );

        let coeff = lca.get_coefficient(0, 1, 0);
        assert!(coeff.is_some());
        assert_eq!(coeff.unwrap(), &vec![1, 2, 3]);

        let no_coeff = lca.get_coefficient(1, 0, 0);
        assert!(no_coeff.is_none());
    }

    #[test]
    fn test_bracket_on_basis() {
        let mut coeffs: LCAStructureCoefficients<i64> = HashMap::new();
        coeffs.insert((0, 1, 0), vec![1]);
        coeffs.insert((0, 1, 1), vec![2, 3]);

        let lca = LieConformalAlgebraWithStructureCoefficients::new(
            1,
            2,
            coeffs,
            None,
            None,
            None,
        );

        let bracket = lca.bracket_on_basis(0, 1);
        assert_eq!(bracket.len(), 2);
        assert_eq!(bracket.get(&0), Some(&vec![1]));
        assert_eq!(bracket.get(&1), Some(&vec![2, 3]));
    }

    #[test]
    fn test_set_coefficient() {
        let coeffs: LCAStructureCoefficients<i64> = HashMap::new();
        let mut lca = LieConformalAlgebraWithStructureCoefficients::new(
            1,
            2,
            coeffs,
            None,
            None,
            None,
        );

        lca.set_coefficient(0, 1, 0, vec![1, 2]);
        let coeff = lca.get_coefficient(0, 1, 0);
        assert_eq!(coeff, Some(&vec![1, 2]));
    }

    #[test]
    fn test_is_central() {
        let mut coeffs: LCAStructureCoefficients<i64> = HashMap::new();
        // [g_0_λ g_1] = g_0: neither g_0 nor g_1 is central
        coeffs.insert((0, 1, 0), vec![1]);
        // g_2 occurs in no bracket (central)

        let lca = LieConformalAlgebraWithStructureCoefficients::new(
            1,
            3,
            coeffs,
            None,
            None,
            None,
        );

        assert!(!lca.is_central(0)); // g_0 is not central
        // g_1 is not central: [g_0_λ g_1] = g_0 != 0, and by skew-symmetry
        // [g_1_λ g_0] = -[g_0_{-λ-∂} g_1] = -g_0 != 0. (The old expectation
        // `is_central(1)` only looked at the first slot.)
        assert!(!lca.is_central(1));
        assert!(lca.is_central(2)); // g_2 is central
    }

    #[test]
    fn test_is_central_ignores_occurrence_as_value() {
        // Virasoro-like table: [g_0_λ g_0] contains g_1 (like C in [L_λ L]);
        // g_1 is still central.
        let mut coeffs: LCAStructureCoefficients<i64> = HashMap::new();
        coeffs.insert((0, 0, 1), vec![0, 0, 0, 1]);
        let lca = LieConformalAlgebraWithStructureCoefficients::new(1, 2, coeffs, None, None, None);
        assert!(!lca.is_central(0));
        assert!(lca.is_central(1));
    }

    #[test]
    #[should_panic(expected = "λ-bracket of general elements is not implemented")]
    fn test_lambda_bracket_is_unimplemented() {
        let mut coeffs: LCAStructureCoefficients<i64> = HashMap::new();
        coeffs.insert((0, 1, 0), vec![1]);
        let lca = LieConformalAlgebraWithStructureCoefficients::new(1, 2, coeffs, None, None, None);
        let a = lca.generator(0).unwrap();
        let b = lca.generator(1).unwrap();
        let _ = lca.lambda_bracket(&a, &b);
    }

    #[test]
    fn test_display() {
        let coeffs: LCAStructureCoefficients<i64> = HashMap::new();
        let lca = LieConformalAlgebraWithStructureCoefficients::new(
            1,
            3,
            coeffs,
            Some(vec!["L".to_string(), "G".to_string(), "C".to_string()]),
            None,
            None,
        );

        let display = format!("{}", lca);
        assert!(display.contains("3 generators"));
        assert!(display.contains("structure coefficients"));
    }

    #[test]
    fn test_generator_access() {
        let coeffs: LCAStructureCoefficients<i64> = HashMap::new();
        let lca = LieConformalAlgebraWithStructureCoefficients::new(
            1,
            2,
            coeffs,
            Some(vec!["a".to_string(), "b".to_string()]),
            None,
            None,
        );

        assert!(lca.generator(0).is_some());
        assert!(lca.generator(1).is_some());
        assert!(lca.generator(2).is_none());
    }

    #[test]
    fn test_zero_element() {
        let coeffs: LCAStructureCoefficients<i64> = HashMap::new();
        let lca = LieConformalAlgebraWithStructureCoefficients::new(
            1,
            2,
            coeffs,
            None,
            None,
            None,
        );

        let zero = lca.zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_degrees() {
        let coeffs: LCAStructureCoefficients<i64> = HashMap::new();
        let lca = LieConformalAlgebraWithStructureCoefficients::new(
            1,
            2,
            coeffs,
            None,
            Some(vec![Degree::int(2), Degree::rational(3, 2)]),
            None,
        );

        assert_eq!(lca.generator_degree(0), Some(Degree::int(2)));
        assert_eq!(lca.generator_degree(1), Some(Degree::rational(3, 2)));
        assert_eq!(lca.generator_degree(2), None);
    }
}
