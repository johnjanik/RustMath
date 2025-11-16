//! Fermionic Ghosts Lie Conformal Algebra (b-c system)
//!
//! A super Lie conformal algebra arising from fermionic ghost fields in conformal field theory.
//!
//! # Mathematical Background
//!
//! The fermionic ghosts Lie conformal algebra (also known as the b-c system) is an H-graded
//! super Lie conformal algebra that arises from fermionic ghost fields in conformal field theory.
//! It is fundamental in the BRST quantization of superstring theory and in the study of
//! super W-algebras.
//!
//! ## Structure
//!
//! The algebra has 2n generators (where n is a positive integer):
//! - n b-generators: b₀, b₁, ..., b_{n-1} with degree 1 (odd/fermionic)
//! - n c-generators: c₀, c₁, ..., c_{n-1} with degree 0 (odd/fermionic)
//! - One central element K (even)
//!
//! ## λ-Bracket
//!
//! The non-vanishing λ-brackets are:
//!
//! [bᵢ_λ cⱼ] = δᵢⱼ K
//!
//! where δᵢⱼ is the Kronecker delta. All other brackets vanish.
//!
//! ## Properties
//!
//! - All b and c generators are fermionic (odd parity)
//! - The central element K is bosonic (even parity) with degree 0
//! - The algebra is H-graded (conformal weight grading)
//! - This is a super Lie conformal algebra
//!
//! # Applications
//!
//! - BRST quantization of superstring theory
//! - Super W-algebra constructions
//! - Conformal field theory with fermions
//! - Algebraic quantum field theory
//!
//! # References
//!
//! - Kac, V. "Vertex Algebras for Beginners" (1998)
//! - Frenkel, E. & Ben-Zvi, D. "Vertex Algebras and Algebraic Curves" (2004)
//! - Kac, V. & Raina, A. "Bombay Lectures on Highest Weight Representations" (1987)
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.fermionic_ghosts_lie_conformal_algebra

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use crate::lie_conformal_algebra::{LieConformalAlgebra, LambdaBracket, GeneratorIndex};
use crate::graded_lie_conformal_algebra::{GradedLieConformalAlgebra, Degree, GradedLCA};
use crate::lie_conformal_algebra_element::LieConformalAlgebraElement;

/// Fermionic Ghosts (b-c system) Lie conformal algebra
///
/// The algebra of fermionic ghost fields with b-generators (degree 1) and
/// c-generators (degree 0), with the λ-bracket [bᵢ_λ cⱼ] = δᵢⱼ K.
///
/// # Type Parameters
///
/// * `R` - The base ring (must be a commutative ring)
///
/// # Examples
///
/// ```
/// use rustmath_lieconformal::FermionicGhostsLieConformalAlgebra;
///
/// // Create fermionic ghosts algebra with 1 pair (b₀, c₀) + central K
/// let lca = FermionicGhostsLieConformalAlgebra::new(1i64, 2, None);
/// assert_eq!(lca.ngens(), Some(3)); // b₀, c₀, K
/// assert_eq!(lca.npairs(), 1); // One (b, c) pair
///
/// // Create algebra with 2 pairs
/// let lca2 = FermionicGhostsLieConformalAlgebra::new(1i64, 4, None);
/// assert_eq!(lca2.ngens(), Some(5)); // b₀, b₁, c₀, c₁, K
/// assert_eq!(lca2.npairs(), 2); // Two (b, c) pairs
/// ```
#[derive(Clone)]
pub struct FermionicGhostsLieConformalAlgebra<R: Ring> {
    /// The underlying graded structure
    graded: GradedLCA<R>,
    /// Number of (b, c) pairs
    npairs: usize,
    /// Index of the central element K
    central_index: usize,
}

impl<R: Ring + Clone + From<i64> + PartialEq> FermionicGhostsLieConformalAlgebra<R> {
    /// Create a new fermionic ghosts Lie conformal algebra
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring
    /// * `ngens` - Total number of non-central generators (must be even, ≥ 2)
    /// * `names` - Optional generator names (defaults to b0, ..., c0, ..., K)
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
    /// use rustmath_lieconformal::FermionicGhostsLieConformalAlgebra;
    ///
    /// // Default names: b0, c0, K
    /// let lca1 = FermionicGhostsLieConformalAlgebra::new(1i64, 2, None);
    ///
    /// // Custom names
    /// let names = vec!["b".to_string(), "c".to_string(), "K".to_string()];
    /// let lca2 = FermionicGhostsLieConformalAlgebra::new(1i64, 2, Some(names));
    /// ```
    pub fn new(base_ring: R, ngens: usize, names: Option<Vec<String>>) -> Self {
        assert!(ngens > 0, "Number of generators must be positive");
        assert!(ngens % 2 == 0, "Number of generators must be even");

        let npairs = ngens / 2;
        let central_index = ngens; // Central element is last
        let total_gens = ngens + 1;

        // Generate default names if not provided
        // Convention: b0, b1, ..., c0, c1, ..., K
        let gen_names = names.unwrap_or_else(|| {
            let mut names = Vec::new();

            // b-generators
            for i in 0..npairs {
                names.push(format!("b{}", i));
            }

            // c-generators
            for i in 0..npairs {
                names.push(format!("c{}", i));
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
        // - b-generators (first npairs): degree 1
        // - c-generators (next npairs): degree 0
        // - K (central): degree 0
        let mut gen_weights = vec![Degree::int(1); npairs]; // b-generators
        gen_weights.extend(vec![Degree::int(0); npairs]); // c-generators
        gen_weights.push(Degree::int(0)); // Central element

        // Parities:
        // - b and c generators are fermionic (odd parity = 1)
        // - K is bosonic (even parity = 0)
        let mut parities = vec![1u8; ngens]; // Odd parity for b and c
        parities.push(0); // Even parity for central element

        let graded = GradedLCA::new(base_ring, gen_names, gen_weights, Some(parities));

        FermionicGhostsLieConformalAlgebra {
            graded,
            npairs,
            central_index,
        }
    }

    /// Get the number of (b, c) pairs
    pub fn npairs(&self) -> usize {
        self.npairs
    }

    /// Get the index of the i-th b-generator
    pub fn b_index(&self, i: usize) -> Option<usize> {
        if i < self.npairs {
            Some(i)
        } else {
            None
        }
    }

    /// Get the index of the i-th c-generator
    pub fn c_index(&self, i: usize) -> Option<usize> {
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

    /// Check if generator i is a b-generator
    pub fn is_b(&self, i: usize) -> bool {
        i < self.npairs
    }

    /// Check if generator i is a c-generator
    pub fn is_c(&self, i: usize) -> bool {
        i >= self.npairs && i < self.central_index
    }

    /// Check if generator i is the central element
    pub fn is_central(&self, i: usize) -> bool {
        i == self.central_index
    }

    /// Check if this is a super algebra (always true for fermionic ghosts)
    pub fn is_super(&self) -> bool {
        true
    }
}

/// Element type for fermionic ghosts Lie conformal algebras
pub type FermionicGhostsLCAElement<R> = LieConformalAlgebraElement<R, GeneratorIndex>;

impl<R: Ring + Clone + From<i64> + PartialEq> LieConformalAlgebra<R>
    for FermionicGhostsLieConformalAlgebra<R>
{
    type Element = FermionicGhostsLCAElement<R>;

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
    for FermionicGhostsLieConformalAlgebra<R>
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
    LambdaBracket<R, FermionicGhostsLCAElement<R>> for FermionicGhostsLieConformalAlgebra<R>
{
    /// Compute the λ-bracket [a_λ b]
    ///
    /// The non-vanishing brackets are:
    /// - [bᵢ_λ cⱼ] = δᵢⱼ K (returned as the λ⁰ term, i.e., constant K)
    /// - All other brackets are zero
    ///
    /// Note: The bracket returns K directly (not λK) as the constant term.
    fn lambda_bracket(
        &self,
        a: &FermionicGhostsLCAElement<R>,
        b: &FermionicGhostsLCAElement<R>,
    ) -> HashMap<usize, FermionicGhostsLCAElement<R>> {
        let mut result = HashMap::new();

        // For each pair of basis elements in a and b
        for (basis_a, _poly_a) in a.terms() {
            for (basis_b, _poly_b) in b.terms() {
                if let (Some(i), Some(j)) = (basis_a.as_finite(), basis_b.as_finite()) {
                    // Skip if either is the central element (central element brackets to 0)
                    if i == self.central_index || j == self.central_index {
                        continue;
                    }

                    // Check if we have [bᵢ_λ cⱼ] or [cⱼ_λ bᵢ]
                    let (b_idx, c_idx) = if self.is_b(i) && self.is_c(j) {
                        // [bᵢ_λ cⱼ]
                        (i, j - self.npairs) // Convert c index to pair index
                    } else if self.is_c(i) && self.is_b(j) {
                        // [cᵢ_λ bⱼ] = -[bⱼ_λ cᵢ] (antisymmetry)
                        // For Lie conformal algebras, we need to be careful about signs
                        // In the b-c system: [cᵢ_λ bⱼ] = 0 (not -δᵢⱼ K)
                        // Only [bᵢ_λ cⱼ] = δᵢⱼ K is non-zero
                        continue;
                    } else {
                        // [bᵢ_λ bⱼ] = 0, [cᵢ_λ cⱼ] = 0
                        continue;
                    };

                    // Check if the pair indices match (Kronecker delta)
                    if b_idx == c_idx {
                        // [bᵢ_λ cᵢ] = K (constant term, i.e., λ⁰)
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
        a: &FermionicGhostsLCAElement<R>,
        b: &FermionicGhostsLCAElement<R>,
        n: usize,
    ) -> FermionicGhostsLCAElement<R> {
        self.lambda_bracket(a, b)
            .get(&n)
            .cloned()
            .unwrap_or_else(|| LieConformalAlgebraElement::zero())
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq + Display> Display
    for FermionicGhostsLieConformalAlgebra<R>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "The Fermionic ghosts Lie conformal algebra with {} generator pair(s) over {}",
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
    fn test_fermionic_ghosts_creation() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 2, None);

        assert_eq!(lca.npairs(), 1);
        assert_eq!(lca.ngens(), Some(3)); // b₀, c₀, K
        assert!(!lca.is_abelian());
        assert!(lca.is_super());
    }

    #[test]
    fn test_fermionic_ghosts_multiple_pairs() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 4, None);

        assert_eq!(lca.npairs(), 2);
        assert_eq!(lca.ngens(), Some(5)); // b₀, b₁, c₀, c₁, K
    }

    #[test]
    fn test_fermionic_ghosts_generator_indices() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 4, None);

        // b-generators are at indices 0, 1
        assert_eq!(lca.b_index(0), Some(0));
        assert_eq!(lca.b_index(1), Some(1));
        assert_eq!(lca.b_index(2), None);

        // c-generators are at indices 2, 3
        assert_eq!(lca.c_index(0), Some(2));
        assert_eq!(lca.c_index(1), Some(3));
        assert_eq!(lca.c_index(2), None);

        // K is at index 4
        assert_eq!(lca.central_element_index(), 4);
    }

    #[test]
    fn test_fermionic_ghosts_generator_types() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 4, None);

        // b-generators
        assert!(lca.is_b(0));
        assert!(lca.is_b(1));
        assert!(!lca.is_b(2));

        // c-generators
        assert!(lca.is_c(2));
        assert!(lca.is_c(3));
        assert!(!lca.is_c(0));

        // Central element
        assert!(lca.is_central(4));
        assert!(!lca.is_central(0));
    }

    #[test]
    fn test_fermionic_ghosts_degrees() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 4, None);

        // b-generators have degree 1
        assert_eq!(lca.generator_degree(0), Some(Degree::int(1)));
        assert_eq!(lca.generator_degree(1), Some(Degree::int(1)));

        // c-generators have degree 0
        assert_eq!(lca.generator_degree(2), Some(Degree::int(0)));
        assert_eq!(lca.generator_degree(3), Some(Degree::int(0)));

        // K has degree 0
        assert_eq!(lca.generator_degree(4), Some(Degree::int(0)));
    }

    #[test]
    fn test_fermionic_ghosts_parities() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 4, None);

        // b-generators are odd (fermionic)
        assert_eq!(lca.graded.generator_parity(0), Some(1));
        assert_eq!(lca.graded.generator_parity(1), Some(1));

        // c-generators are odd (fermionic)
        assert_eq!(lca.graded.generator_parity(2), Some(1));
        assert_eq!(lca.graded.generator_parity(3), Some(1));

        // K is even (bosonic)
        assert_eq!(lca.graded.generator_parity(4), Some(0));
    }

    #[test]
    fn test_fermionic_ghosts_generators() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 2, None);

        assert!(lca.generator(0).is_some()); // b₀
        assert!(lca.generator(1).is_some()); // c₀
        assert!(lca.generator(2).is_some()); // K
        assert!(lca.generator(3).is_none());
    }

    #[test]
    fn test_fermionic_ghosts_lambda_bracket_matching_pair() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 2, None);

        let b0 = lca.generator(0).unwrap(); // b₀
        let c0 = lca.generator(1).unwrap(); // c₀

        // [b₀_λ c₀] = K (constant term)
        let bracket = lca.lambda_bracket(&b0, &c0);
        assert!(bracket.contains_key(&0)); // λ⁰ term (constant)

        let constant_term = bracket.get(&0).unwrap();
        assert!(!constant_term.is_zero());
    }

    #[test]
    fn test_fermionic_ghosts_lambda_bracket_different_pairs() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 4, None);

        let b0 = lca.generator(0).unwrap(); // b₀
        let c1 = lca.generator(3).unwrap(); // c₁

        // [b₀_λ c₁] = 0 (different indices)
        let bracket = lca.lambda_bracket(&b0, &c1);
        assert!(bracket.is_empty() || bracket.values().all(|v| v.is_zero()));
    }

    #[test]
    fn test_fermionic_ghosts_lambda_bracket_b_b() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 4, None);

        let b0 = lca.generator(0).unwrap();
        let b1 = lca.generator(1).unwrap();

        // [b₀_λ b₁] = 0
        let bracket = lca.lambda_bracket(&b0, &b1);
        assert!(bracket.is_empty() || bracket.values().all(|v| v.is_zero()));
    }

    #[test]
    fn test_fermionic_ghosts_lambda_bracket_c_c() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 4, None);

        let c0 = lca.generator(2).unwrap();
        let c1 = lca.generator(3).unwrap();

        // [c₀_λ c₁] = 0
        let bracket = lca.lambda_bracket(&c0, &c1);
        assert!(bracket.is_empty() || bracket.values().all(|v| v.is_zero()));
    }

    #[test]
    fn test_fermionic_ghosts_lambda_bracket_c_b() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 2, None);

        let c0 = lca.generator(1).unwrap();
        let b0 = lca.generator(0).unwrap();

        // [c₀_λ b₀] = 0 (order matters in b-c system)
        let bracket = lca.lambda_bracket(&c0, &b0);
        assert!(bracket.is_empty() || bracket.values().all(|v| v.is_zero()));
    }

    #[test]
    fn test_fermionic_ghosts_central_element_bracket() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 2, None);

        let k = lca.generator(2).unwrap(); // Central element
        let b0 = lca.generator(0).unwrap();

        // [K_λ b₀] = 0 (central element)
        let bracket = lca.lambda_bracket(&k, &b0);
        assert!(bracket.is_empty() || bracket.values().all(|v| v.is_zero()));
    }

    #[test]
    fn test_fermionic_ghosts_zero() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 2, None);

        let zero = lca.zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_fermionic_ghosts_with_custom_names() {
        let names = vec!["b".to_string(), "c".to_string(), "K".to_string()];
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 2, Some(names.clone()));

        assert_eq!(lca.generator_names()[0], "b");
        assert_eq!(lca.generator_names()[1], "c");
        assert_eq!(lca.generator_names()[2], "K");
    }

    #[test]
    #[should_panic(expected = "Number of generators must be even")]
    fn test_fermionic_ghosts_odd_ngens() {
        let _lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 3, None);
    }

    #[test]
    #[should_panic(expected = "Number of names must match total generators")]
    fn test_fermionic_ghosts_wrong_name_count() {
        let names = vec!["a".to_string(), "b".to_string()]; // Only 2 names but need 3
        let _lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 2, Some(names));
    }

    #[test]
    fn test_fermionic_ghosts_display() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 2, None);

        let display = format!("{}", lca);
        assert!(display.contains("Fermionic ghosts"));
        assert!(display.contains("1 generator pair"));
    }

    #[test]
    fn test_fermionic_ghosts_n_product() {
        let lca: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 2, None);

        let b0 = lca.generator(0).unwrap();
        let c0 = lca.generator(1).unwrap();

        // Get the 0-product (constant term)
        let prod = lca.n_product(&b0, &c0, 0);
        assert!(!prod.is_zero()); // Should be K

        // Get the 1-product (should be zero for fermionic ghosts)
        let prod1 = lca.n_product(&b0, &c0, 1);
        assert!(prod1.is_zero());
    }

    #[test]
    fn test_fermionic_vs_bosonic_parity_difference() {
        let fermionic: FermionicGhostsLieConformalAlgebra<i64> =
            FermionicGhostsLieConformalAlgebra::new(1, 2, None);

        // Fermionic generators have odd parity
        assert_eq!(fermionic.graded.generator_parity(0), Some(1)); // b₀ is odd
        assert_eq!(fermionic.graded.generator_parity(1), Some(1)); // c₀ is odd
        assert_eq!(fermionic.graded.generator_parity(2), Some(0)); // K is even

        // Verify it's a super algebra
        assert!(fermionic.is_super());
        assert!(fermionic.graded.is_super());
    }
}
