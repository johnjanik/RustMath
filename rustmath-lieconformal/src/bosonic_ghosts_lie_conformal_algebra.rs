//! Bosonic Ghosts Lie Conformal Algebra (β-γ system)
//!
//! A Lie conformal algebra arising from bosonic ghost fields in conformal field theory.
//!
//! # Mathematical Background
//!
//! The bosonic ghosts Lie conformal algebra (also known as the β-γ system) is an H-graded
//! Lie conformal algebra that arises from bosonic ghost fields in conformal field theory.
//! It is fundamental in the BRST quantization of string theory and in the study of
//! W-algebras.
//!
//! ## Structure
//!
//! The algebra has 2n generators (where n is a positive integer):
//! - n β-generators: β₀, β₁, ..., β_{n-1} with degree 1
//! - n γ-generators: γ₀, γ₁, ..., γ_{n-1} with degree 0
//! - One central element K
//!
//! ## λ-Bracket
//!
//! The non-vanishing λ-brackets are:
//!
//! [βᵢ_λ γⱼ] = δᵢⱼ K
//!
//! where δᵢⱼ is the Kronecker delta. All other brackets vanish.
//!
//! ## Properties
//!
//! - All generators are bosonic (even parity)
//! - The central element K has degree 0
//! - The algebra is H-graded (conformal weight grading)
//! - The structure is isomorphic to the Heisenberg-Virasoro-like algebra
//!
//! # Applications
//!
//! - BRST quantization of bosonic string theory
//! - W-algebra constructions
//! - Conformal field theory
//! - Algebraic quantum field theory
//!
//! # References
//!
//! - Kac, V. "Vertex Algebras for Beginners" (1998)
//! - Frenkel, E. & Ben-Zvi, D. "Vertex Algebras and Algebraic Curves" (2004)
//! - Kac, V. & Raina, A. "Bombay Lectures on Highest Weight Representations" (1987)
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.bosonic_ghosts_lie_conformal_algebra

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use crate::lie_conformal_algebra::{LieConformalAlgebra, LambdaBracket, GeneratorIndex};
use crate::graded_lie_conformal_algebra::{GradedLieConformalAlgebra, Degree, GradedLCA};
use crate::lie_conformal_algebra_element::LieConformalAlgebraElement;

/// Bosonic Ghosts (β-γ system) Lie conformal algebra
///
/// The algebra of bosonic ghost fields with β-generators (degree 1) and
/// γ-generators (degree 0), with the λ-bracket [βᵢ_λ γⱼ] = δᵢⱼ K.
///
/// # Type Parameters
///
/// * `R` - The base ring (must be a commutative ring)
///
/// # Examples
///
/// ```
/// use rustmath_lieconformal::BosonicGhostsLieConformalAlgebra;
///
/// // Create bosonic ghosts algebra with 1 pair (β₀, γ₀) + central K
/// let lca = BosonicGhostsLieConformalAlgebra::new(1i64, 2, None);
/// assert_eq!(lca.ngens(), Some(3)); // β₀, γ₀, K
/// assert_eq!(lca.npairs(), 1); // One (β, γ) pair
///
/// // Create algebra with 2 pairs
/// let lca2 = BosonicGhostsLieConformalAlgebra::new(1i64, 4, None);
/// assert_eq!(lca2.ngens(), Some(5)); // β₀, β₁, γ₀, γ₁, K
/// assert_eq!(lca2.npairs(), 2); // Two (β, γ) pairs
/// ```
#[derive(Clone)]
pub struct BosonicGhostsLieConformalAlgebra<R: Ring> {
    /// The underlying graded structure
    graded: GradedLCA<R>,
    /// Number of (β, γ) pairs
    npairs: usize,
    /// Index of the central element K
    central_index: usize,
}

impl<R: Ring + Clone + From<i64> + PartialEq> BosonicGhostsLieConformalAlgebra<R> {
    /// Create a new bosonic ghosts Lie conformal algebra
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring
    /// * `ngens` - Total number of non-central generators (must be even, ≥ 2)
    /// * `names` - Optional generator names (defaults to beta0, ..., gamma0, ..., K)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `ngens` is not a positive even integer
    /// - The number of names doesn't match `ngens + 1`
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::BosonicGhostsLieConformalAlgebra;
    ///
    /// // Default names: beta0, gamma0, K
    /// let lca1 = BosonicGhostsLieConformalAlgebra::new(1i64, 2, None);
    ///
    /// // Custom names
    /// let names = vec!["b".to_string(), "c".to_string(), "K".to_string()];
    /// let lca2 = BosonicGhostsLieConformalAlgebra::new(1i64, 2, Some(names));
    /// ```
    pub fn new(base_ring: R, ngens: usize, names: Option<Vec<String>>) -> Self {
        assert!(ngens > 0, "Number of generators must be positive");
        assert!(ngens % 2 == 0, "Number of generators must be even");

        let npairs = ngens / 2;
        let central_index = ngens; // Central element is last
        let total_gens = ngens + 1;

        // Generate default names if not provided
        // Convention: beta0, beta1, ..., gamma0, gamma1, ..., K
        let gen_names = names.unwrap_or_else(|| {
            let mut names = Vec::new();

            // β-generators
            for i in 0..npairs {
                names.push(format!("beta{}", i));
            }

            // γ-generators
            for i in 0..npairs {
                names.push(format!("gamma{}", i));
            }

            // Central element
            names.push("K".to_string());
            names
        });

        assert_eq!(
            gen_names.len(),
            total_gens,
            "Number of names must match total generators (ngens + 1)"
        );

        // Degrees:
        // - β-generators (first npairs): degree 1
        // - γ-generators (next npairs): degree 0
        // - K (central): degree 0
        let mut gen_weights = vec![Degree::int(1); npairs]; // β-generators
        gen_weights.extend(vec![Degree::int(0); npairs]); // γ-generators
        gen_weights.push(Degree::int(0)); // Central element

        // All generators are bosonic (even parity = 0)
        let parities = vec![0u8; total_gens];

        let graded = GradedLCA::new(base_ring, gen_names, gen_weights, Some(parities));

        BosonicGhostsLieConformalAlgebra {
            graded,
            npairs,
            central_index,
        }
    }

    /// Get the number of (β, γ) pairs
    pub fn npairs(&self) -> usize {
        self.npairs
    }

    /// Get the index of the i-th β-generator
    pub fn beta_index(&self, i: usize) -> Option<usize> {
        if i < self.npairs {
            Some(i)
        } else {
            None
        }
    }

    /// Get the index of the i-th γ-generator
    pub fn gamma_index(&self, i: usize) -> Option<usize> {
        if i < self.npairs {
            Some(self.npairs + i)
        } else {
            None
        }
    }

    /// Get the central element index
    pub fn central_element_index(&self) -> usize {
        self.central_index
    }

    /// Get the underlying graded structure
    pub fn graded_structure(&self) -> &GradedLCA<R> {
        &self.graded
    }

    /// Get total number of generators (including central element)
    pub fn ngens(&self) -> usize {
        self.graded.ngens()
    }

    /// Get generator names
    pub fn generator_names(&self) -> &[String] {
        self.graded.generator_names()
    }

    /// Check if generator i is a β-generator
    pub fn is_beta(&self, i: usize) -> bool {
        i < self.npairs
    }

    /// Check if generator i is a γ-generator
    pub fn is_gamma(&self, i: usize) -> bool {
        i >= self.npairs && i < self.central_index
    }

    /// Check if generator i is the central element
    pub fn is_central(&self, i: usize) -> bool {
        i == self.central_index
    }
}

/// Element type for bosonic ghosts Lie conformal algebras
pub type BosonicGhostsLCAElement<R> = LieConformalAlgebraElement<R, GeneratorIndex>;

impl<R: Ring + Clone + From<i64> + PartialEq> LieConformalAlgebra<R>
    for BosonicGhostsLieConformalAlgebra<R>
{
    type Element = BosonicGhostsLCAElement<R>;

    fn base_ring(&self) -> &R {
        self.graded.base_ring()
    }

    fn ngens(&self) -> Option<usize> {
        Some(self.graded.ngens())
    }

    fn generator(&self, i: usize) -> Option<Self::Element> {
        if i < self.graded.ngens() {
            Some(LieConformalAlgebraElement::from_basis(
                GeneratorIndex::finite(i),
            ))
        } else {
            None
        }
    }

    fn is_abelian(&self) -> bool {
        false // Has non-trivial brackets
    }

    fn zero(&self) -> Self::Element {
        LieConformalAlgebraElement::zero()
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq> GradedLieConformalAlgebra<R>
    for BosonicGhostsLieConformalAlgebra<R>
{
    fn generator_degree(&self, index: usize) -> Option<Degree> {
        self.graded.generator_degree(index)
    }

    fn degree(&self, element: &Self::Element) -> Option<Degree> {
        if element.is_zero() {
            return Some(Degree::int(0));
        }

        // Return None for non-homogeneous elements
        // A proper implementation would check all terms have the same degree
        None
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq + std::ops::Add<Output = R> + std::ops::Mul<Output = R>>
    LambdaBracket<R, BosonicGhostsLCAElement<R>> for BosonicGhostsLieConformalAlgebra<R>
{
    /// Compute the λ-bracket [a_λ b]
    ///
    /// The non-vanishing brackets are:
    /// - [βᵢ_λ γⱼ] = δᵢⱼ K (returned as the λ⁰ term, i.e., constant K)
    /// - All other brackets are zero
    ///
    /// Note: The bracket returns K directly (not λK) as the constant term.
    fn lambda_bracket(
        &self,
        a: &BosonicGhostsLCAElement<R>,
        b: &BosonicGhostsLCAElement<R>,
    ) -> HashMap<usize, BosonicGhostsLCAElement<R>> {
        let mut result = HashMap::new();

        // For each pair of basis elements in a and b
        for (basis_a, _poly_a) in a.terms() {
            for (basis_b, _poly_b) in b.terms() {
                if let (Some(i), Some(j)) = (basis_a.as_finite(), basis_b.as_finite()) {
                    // Skip if either is the central element (central element brackets to 0)
                    if i == self.central_index || j == self.central_index {
                        continue;
                    }

                    // Check if we have [βᵢ_λ γⱼ] or [γⱼ_λ βᵢ]
                    let (beta_idx, gamma_idx) = if self.is_beta(i) && self.is_gamma(j) {
                        // [βᵢ_λ γⱼ]
                        (i, j - self.npairs) // Convert gamma index to pair index
                    } else if self.is_gamma(i) && self.is_beta(j) {
                        // [γᵢ_λ βⱼ] = -[βⱼ_λ γᵢ] (antisymmetry)
                        // For Lie conformal algebras, we need to be careful about signs
                        // In the β-γ system: [γᵢ_λ βⱼ] = 0 (not -δᵢⱼ K)
                        // Only [βᵢ_λ γⱼ] = δᵢⱼ K is non-zero
                        continue;
                    } else {
                        // [βᵢ_λ βⱼ] = 0, [γᵢ_λ γⱼ] = 0
                        continue;
                    };

                    // Check if the pair indices match (Kronecker delta)
                    if beta_idx == gamma_idx {
                        // [βᵢ_λ γᵢ] = K (constant term, i.e., λ⁰)
                        let central_term = LieConformalAlgebraElement::from_basis(
                            GeneratorIndex::finite(self.central_index),
                        );

                        let existing = result
                            .entry(0) // Constant term (λ⁰)
                            .or_insert_with(|| LieConformalAlgebraElement::zero());
                        *existing = existing.add(&central_term);
                    }
                }
            }
        }

        result
    }

    fn n_product(
        &self,
        a: &BosonicGhostsLCAElement<R>,
        b: &BosonicGhostsLCAElement<R>,
        n: usize,
    ) -> BosonicGhostsLCAElement<R> {
        self.lambda_bracket(a, b)
            .get(&n)
            .cloned()
            .unwrap_or_else(|| LieConformalAlgebraElement::zero())
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq + Display> Display
    for BosonicGhostsLieConformalAlgebra<R>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "The Bosonic ghosts Lie conformal algebra with {} generator pair(s) over {}",
            self.npairs,
            self.graded.base_ring()
        )?;

        if self.npairs <= 3 {
            write!(f, ": (")?;
            for (i, name) in self.graded.generator_names().iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", name)?;
            }
            write!(f, ")")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bosonic_ghosts_creation() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 2, None);

        assert_eq!(lca.npairs(), 1);
        assert_eq!(lca.ngens(), Some(3)); // β₀, γ₀, K
        assert!(!lca.is_abelian());
    }

    #[test]
    fn test_bosonic_ghosts_multiple_pairs() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 4, None);

        assert_eq!(lca.npairs(), 2);
        assert_eq!(lca.ngens(), Some(5)); // β₀, β₁, γ₀, γ₁, K
    }

    #[test]
    fn test_bosonic_ghosts_generator_indices() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 4, None);

        // β-generators are at indices 0, 1
        assert_eq!(lca.beta_index(0), Some(0));
        assert_eq!(lca.beta_index(1), Some(1));
        assert_eq!(lca.beta_index(2), None);

        // γ-generators are at indices 2, 3
        assert_eq!(lca.gamma_index(0), Some(2));
        assert_eq!(lca.gamma_index(1), Some(3));
        assert_eq!(lca.gamma_index(2), None);

        // K is at index 4
        assert_eq!(lca.central_element_index(), 4);
    }

    #[test]
    fn test_bosonic_ghosts_generator_types() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 4, None);

        // β-generators
        assert!(lca.is_beta(0));
        assert!(lca.is_beta(1));
        assert!(!lca.is_beta(2));

        // γ-generators
        assert!(lca.is_gamma(2));
        assert!(lca.is_gamma(3));
        assert!(!lca.is_gamma(0));

        // Central element
        assert!(lca.is_central(4));
        assert!(!lca.is_central(0));
    }

    #[test]
    fn test_bosonic_ghosts_degrees() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 4, None);

        // β-generators have degree 1
        assert_eq!(lca.generator_degree(0), Some(Degree::int(1)));
        assert_eq!(lca.generator_degree(1), Some(Degree::int(1)));

        // γ-generators have degree 0
        assert_eq!(lca.generator_degree(2), Some(Degree::int(0)));
        assert_eq!(lca.generator_degree(3), Some(Degree::int(0)));

        // K has degree 0
        assert_eq!(lca.generator_degree(4), Some(Degree::int(0)));
    }

    #[test]
    fn test_bosonic_ghosts_generators() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 2, None);

        assert!(lca.generator(0).is_some()); // β₀
        assert!(lca.generator(1).is_some()); // γ₀
        assert!(lca.generator(2).is_some()); // K
        assert!(lca.generator(3).is_none());
    }

    #[test]
    fn test_bosonic_ghosts_lambda_bracket_matching_pair() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 2, None);

        let beta0 = lca.generator(0).unwrap(); // β₀
        let gamma0 = lca.generator(1).unwrap(); // γ₀

        // [β₀_λ γ₀] = K (constant term)
        let bracket = lca.lambda_bracket(&beta0, &gamma0);
        assert!(bracket.contains_key(&0)); // λ⁰ term (constant)

        let constant_term = bracket.get(&0).unwrap();
        assert!(!constant_term.is_zero());
    }

    #[test]
    fn test_bosonic_ghosts_lambda_bracket_different_pairs() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 4, None);

        let beta0 = lca.generator(0).unwrap(); // β₀
        let gamma1 = lca.generator(3).unwrap(); // γ₁

        // [β₀_λ γ₁] = 0 (different indices)
        let bracket = lca.lambda_bracket(&beta0, &gamma1);
        assert!(bracket.is_empty() || bracket.values().all(|v| v.is_zero()));
    }

    #[test]
    fn test_bosonic_ghosts_lambda_bracket_beta_beta() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 4, None);

        let beta0 = lca.generator(0).unwrap();
        let beta1 = lca.generator(1).unwrap();

        // [β₀_λ β₁] = 0
        let bracket = lca.lambda_bracket(&beta0, &beta1);
        assert!(bracket.is_empty() || bracket.values().all(|v| v.is_zero()));
    }

    #[test]
    fn test_bosonic_ghosts_lambda_bracket_gamma_gamma() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 4, None);

        let gamma0 = lca.generator(2).unwrap();
        let gamma1 = lca.generator(3).unwrap();

        // [γ₀_λ γ₁] = 0
        let bracket = lca.lambda_bracket(&gamma0, &gamma1);
        assert!(bracket.is_empty() || bracket.values().all(|v| v.is_zero()));
    }

    #[test]
    fn test_bosonic_ghosts_lambda_bracket_gamma_beta() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 2, None);

        let gamma0 = lca.generator(1).unwrap();
        let beta0 = lca.generator(0).unwrap();

        // [γ₀_λ β₀] = 0 (order matters in β-γ system)
        let bracket = lca.lambda_bracket(&gamma0, &beta0);
        assert!(bracket.is_empty() || bracket.values().all(|v| v.is_zero()));
    }

    #[test]
    fn test_bosonic_ghosts_central_element_bracket() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 2, None);

        let k = lca.generator(2).unwrap(); // Central element
        let beta0 = lca.generator(0).unwrap();

        // [K_λ β₀] = 0 (central element)
        let bracket = lca.lambda_bracket(&k, &beta0);
        assert!(bracket.is_empty() || bracket.values().all(|v| v.is_zero()));
    }

    #[test]
    fn test_bosonic_ghosts_zero() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 2, None);

        let zero = lca.zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_bosonic_ghosts_with_custom_names() {
        let names = vec!["b".to_string(), "c".to_string(), "K".to_string()];
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 2, Some(names.clone()));

        assert_eq!(lca.generator_names()[0], "b");
        assert_eq!(lca.generator_names()[1], "c");
        assert_eq!(lca.generator_names()[2], "K");
    }

    #[test]
    #[should_panic(expected = "Number of generators must be even")]
    fn test_bosonic_ghosts_odd_ngens() {
        let _lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 3, None);
    }

    #[test]
    #[should_panic(expected = "Number of names must match total generators")]
    fn test_bosonic_ghosts_wrong_name_count() {
        let names = vec!["a".to_string(), "b".to_string()]; // Only 2 names but need 3
        let _lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 2, Some(names));
    }

    #[test]
    fn test_bosonic_ghosts_display() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 2, None);

        let display = format!("{}", lca);
        assert!(display.contains("Bosonic ghosts"));
        assert!(display.contains("1 generator pair"));
    }

    #[test]
    fn test_bosonic_ghosts_n_product() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 2, None);

        let beta0 = lca.generator(0).unwrap();
        let gamma0 = lca.generator(1).unwrap();

        // Get the 0-product (constant term)
        let prod = lca.n_product(&beta0, &gamma0, 0);
        assert!(!prod.is_zero()); // Should be K

        // Get the 1-product (should be zero for bosonic ghosts)
        let prod1 = lca.n_product(&beta0, &gamma0, 1);
        assert!(prod1.is_zero());
    }

    #[test]
    fn test_bosonic_ghosts_parity() {
        let lca: BosonicGhostsLieConformalAlgebra<i64> =
            BosonicGhostsLieConformalAlgebra::new(1, 2, None);

        // All generators should be bosonic (even parity = 0)
        assert_eq!(lca.graded.generator_parity(0), Some(0)); // β₀
        assert_eq!(lca.graded.generator_parity(1), Some(0)); // γ₀
        assert_eq!(lca.graded.generator_parity(2), Some(0)); // K
    }
}
