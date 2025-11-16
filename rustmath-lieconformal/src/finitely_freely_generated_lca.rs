//! Finitely Freely Generated Lie Conformal Algebras
//!
//! Provides an abstract base class for finitely generated Lie conformal algebras.
//! This is a specialization of FreelyGeneratedLCA that explicitly requires a finite
//! set of generators and provides additional convenience methods.
//!
//! # Mathematical Background
//!
//! A finitely freely generated Lie conformal algebra is a free R[∂]-module with
//! a finite number of generators, equipped with a λ-bracket operation satisfying
//! the axioms of a Lie conformal algebra.
//!
//! The key distinction from FreelyGeneratedLCA is:
//! - **FreelyGeneratedLCA**: Can theoretically have infinite generators
//! - **FinitelyFreelyGeneratedLCA**: Explicitly guarantees finite generators
//!
//! This class provides:
//! - Guaranteed finite generator count
//! - Convenience method `ngens()` returning `usize` (not `Option<usize>`)
//! - Efficient access to all generators via `gens()`
//! - Central elements tracking
//! - Sample element generation
//!
//! # Examples
//!
//! ```
//! use rustmath_lieconformal::{FinitelyFreelyGeneratedLCA, Degree};
//!
//! // Create a finitely generated LCA with 3 generators
//! let lca = FinitelyFreelyGeneratedLCA::new(
//!     1i64,
//!     3,
//!     Some(vec!["a".to_string(), "b".to_string(), "K".to_string()]),
//!     Some(vec![Degree::int(1), Degree::int(1), Degree::int(0)]),
//!     None,
//!     Some(vec![2]) // K is central (index 2)
//! );
//!
//! assert_eq!(lca.ngens(), 3);
//! assert_eq!(lca.central_elements_indices(), &[2]);
//! ```
//!
//! # References
//!
//! - Kac, V. "Vertex Algebras for Beginners" (1998)
//! - SageMath: sage.algebras.lie_conformal_algebras.finitely_freely_generated_lca
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.finitely_freely_generated_lca

use rustmath_core::Ring;
use std::fmt::{self, Display};
use crate::lie_conformal_algebra::{LieConformalAlgebra, GeneratorIndex};
use crate::graded_lie_conformal_algebra::{GradedLieConformalAlgebra, Degree};
use crate::freely_generated_lie_conformal_algebra::FreelyGeneratedLCA;
use crate::lie_conformal_algebra_element::LieConformalAlgebraElement;

/// Finitely freely generated Lie conformal algebra
///
/// This is an abstract base class for finitely generated Lie conformal algebras.
/// It wraps a `FreelyGeneratedLCA` and adds the explicit guarantee that the
/// number of generators is finite, along with convenience methods.
///
/// # Type Parameters
///
/// * `R` - The base ring (must implement `Ring`)
///
/// # Examples
///
/// ```
/// use rustmath_lieconformal::{FinitelyFreelyGeneratedLCA, Degree};
///
/// // Create an algebra with 2 generators
/// let lca = FinitelyFreelyGeneratedLCA::new(
///     1i64,
///     2,
///     Some(vec!["L".to_string(), "C".to_string()]),
///     Some(vec![Degree::int(2), Degree::int(0)]),
///     None,
///     Some(vec![1]) // C is central
/// );
///
/// assert_eq!(lca.ngens(), 2);
/// assert!(lca.is_finitely_generated());
/// ```
#[derive(Clone)]
pub struct FinitelyFreelyGeneratedLCA<R: Ring> {
    /// The underlying freely generated structure
    freely_generated: FreelyGeneratedLCA<R>,
    /// Number of generators (guaranteed finite)
    ngens: usize,
    /// Indices of central elements
    central_indices: Vec<usize>,
}

impl<R: Ring + Clone + From<i64>> FinitelyFreelyGeneratedLCA<R> {
    /// Create a new finitely freely generated Lie conformal algebra
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring
    /// * `ngens` - Number of generators (must be finite)
    /// * `names` - Optional generator names (defaults to a0, a1, ...)
    /// * `degrees` - Optional generator degrees (defaults to all degree 1)
    /// * `parities` - Optional generator parities for super-algebras (all even by default)
    /// * `central_indices` - Optional list of indices of central elements
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `ngens` is 0
    /// - Names, degrees, or parities have wrong length
    /// - Central indices are out of range
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::{FinitelyFreelyGeneratedLCA, Degree};
    ///
    /// // Simple case: 2 generators, default settings
    /// let lca1 = FinitelyFreelyGeneratedLCA::new(1i64, 2, None, None, None, None);
    ///
    /// // With names and degrees
    /// let lca2 = FinitelyFreelyGeneratedLCA::new(
    ///     1i64,
    ///     3,
    ///     Some(vec!["a".to_string(), "b".to_string(), "K".to_string()]),
    ///     Some(vec![Degree::int(1), Degree::int(1), Degree::int(0)]),
    ///     None,
    ///     Some(vec![2])
    /// );
    /// ```
    pub fn new(
        base_ring: R,
        ngens: usize,
        names: Option<Vec<String>>,
        degrees: Option<Vec<Degree>>,
        parities: Option<Vec<u8>>,
        central_indices: Option<Vec<usize>>,
    ) -> Self {
        assert!(ngens > 0, "Number of generators must be positive");

        // Validate names length if provided
        if let Some(ref n) = names {
            assert_eq!(n.len(), ngens, "Number of names must match ngens");
        }

        // Validate degrees length if provided
        if let Some(ref d) = degrees {
            assert_eq!(d.len(), ngens, "Number of degrees must match ngens");
        }

        // Validate parities length if provided
        if let Some(ref p) = parities {
            assert_eq!(p.len(), ngens, "Number of parities must match ngens");
        }

        // Validate central indices
        let central = central_indices.unwrap_or_default();
        for &idx in &central {
            assert!(idx < ngens, "Central element index {} out of range", idx);
        }

        let freely_generated = FreelyGeneratedLCA::new(
            base_ring,
            ngens,
            names,
            degrees,
            parities,
        );

        FinitelyFreelyGeneratedLCA {
            freely_generated,
            ngens,
            central_indices: central,
        }
    }

    /// Get the number of generators
    ///
    /// Unlike `LieConformalAlgebra::ngens()` which returns `Option<usize>`,
    /// this method returns `usize` directly since the number is guaranteed finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::FinitelyFreelyGeneratedLCA;
    ///
    /// let lca = FinitelyFreelyGeneratedLCA::new(1i64, 5, None, None, None, None);
    /// assert_eq!(lca.ngens(), 5);
    /// ```
    pub fn ngens(&self) -> usize {
        self.ngens
    }

    /// Get all generators as a vector
    ///
    /// Returns a vector containing all generator elements of this algebra.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::FinitelyFreelyGeneratedLCA;
    ///
    /// let lca = FinitelyFreelyGeneratedLCA::new(1i64, 3, None, None, None, None);
    /// let generators = lca.gens();
    /// assert_eq!(generators.len(), 3);
    /// ```
    pub fn gens(&self) -> Vec<FinitelyFreelyGeneratedElement<R>> {
        (0..self.ngens)
            .map(|i| self.generator_unchecked(i))
            .collect()
    }

    /// Get a generator by index without checking bounds
    ///
    /// # Panics
    ///
    /// Panics if index >= ngens
    fn generator_unchecked(&self, i: usize) -> FinitelyFreelyGeneratedElement<R> {
        assert!(i < self.ngens, "Generator index {} out of range", i);
        LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(i))
    }

    /// Get the indices of central elements
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::FinitelyFreelyGeneratedLCA;
    ///
    /// let lca = FinitelyFreelyGeneratedLCA::new(
    ///     1i64, 3, None, None, None,
    ///     Some(vec![2])
    /// );
    /// assert_eq!(lca.central_elements_indices(), &[2]);
    /// ```
    pub fn central_elements_indices(&self) -> &[usize] {
        &self.central_indices
    }

    /// Get the central elements as a vector
    ///
    /// Returns a vector of all central element generators.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::FinitelyFreelyGeneratedLCA;
    ///
    /// let lca = FinitelyFreelyGeneratedLCA::new(
    ///     1i64, 3,
    ///     Some(vec!["a".to_string(), "b".to_string(), "K".to_string()]),
    ///     None, None,
    ///     Some(vec![2])
    /// );
    /// let central = lca.central_elements();
    /// assert_eq!(central.len(), 1);
    /// ```
    pub fn central_elements(&self) -> Vec<FinitelyFreelyGeneratedElement<R>> {
        self.central_indices
            .iter()
            .map(|&i| self.generator_unchecked(i))
            .collect()
    }

    /// Get a sample element (first generator)
    ///
    /// Returns a sample element (the first generator) useful for
    /// testing and demonstration purposes.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::FinitelyFreelyGeneratedLCA;
    ///
    /// let lca = FinitelyFreelyGeneratedLCA::new(1i64, 2, None, None, None, None);
    /// let element = lca.an_element();
    /// // element is the first generator
    /// ```
    pub fn an_element(&self) -> FinitelyFreelyGeneratedElement<R> {
        if self.ngens > 0 {
            self.generator_unchecked(0)
        } else {
            self.zero()
        }
    }

    /// Check if this is a finitely generated algebra
    ///
    /// Always returns `true` for this type.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::FinitelyFreelyGeneratedLCA;
    ///
    /// let lca = FinitelyFreelyGeneratedLCA::new(1i64, 2, None, None, None, None);
    /// assert!(lca.is_finitely_generated());
    /// ```
    pub fn is_finitely_generated(&self) -> bool {
        true
    }

    /// Get the underlying freely generated structure
    ///
    /// Returns a reference to the internal `FreelyGeneratedLCA`.
    pub fn freely_generated_structure(&self) -> &FreelyGeneratedLCA<R> {
        &self.freely_generated
    }

    /// Get generator names
    ///
    /// Returns a slice of generator name strings.
    pub fn generator_names(&self) -> &[String] {
        self.freely_generated.graded_structure().generator_names()
    }

    /// Check if a generator is central
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::FinitelyFreelyGeneratedLCA;
    ///
    /// let lca = FinitelyFreelyGeneratedLCA::new(
    ///     1i64, 3, None, None, None,
    ///     Some(vec![2])
    /// );
    /// assert!(!lca.is_central(0));
    /// assert!(!lca.is_central(1));
    /// assert!(lca.is_central(2));
    /// ```
    pub fn is_central(&self, index: usize) -> bool {
        self.central_indices.contains(&index)
    }
}

/// Element type for finitely freely generated Lie conformal algebras
pub type FinitelyFreelyGeneratedElement<R> = LieConformalAlgebraElement<R, GeneratorIndex>;

impl<R: Ring + Clone + From<i64>> LieConformalAlgebra<R> for FinitelyFreelyGeneratedLCA<R> {
    type Element = FinitelyFreelyGeneratedElement<R>;

    fn base_ring(&self) -> &R {
        self.freely_generated.base_ring()
    }

    fn ngens(&self) -> Option<usize> {
        Some(self.ngens)
    }

    fn generator(&self, i: usize) -> Option<Self::Element> {
        if i < self.ngens {
            Some(LieConformalAlgebraElement::from_basis(
                GeneratorIndex::finite(i)
            ))
        } else {
            None
        }
    }

    fn zero(&self) -> Self::Element {
        LieConformalAlgebraElement::zero()
    }

    fn is_abelian(&self) -> bool {
        // Default: not abelian unless overridden by specific implementations
        false
    }
}

impl<R: Ring + Clone + From<i64>> GradedLieConformalAlgebra<R> for FinitelyFreelyGeneratedLCA<R> {
    fn generator_degree(&self, index: usize) -> Option<Degree> {
        self.freely_generated.generator_degree(index)
    }

    fn degree(&self, element: &Self::Element) -> Option<Degree> {
        self.freely_generated.degree(element)
    }
}

impl<R: Ring + Clone + From<i64> + Display> Display for FinitelyFreelyGeneratedLCA<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.ngens == 1 {
            write!(
                f,
                "Lie conformal algebra generated by {} over {}",
                self.generator_names()[0],
                self.base_ring()
            )
        } else {
            write!(f, "Lie conformal algebra with generators (")?;
            for (i, name) in self.generator_names().iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", name)?;
            }
            write!(f, ") over {}", self.base_ring())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finitely_generated_creation() {
        let lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 3, None, None, None, None);

        assert_eq!(lca.ngens(), 3);
        assert!(lca.is_finitely_generated());
    }

    #[test]
    fn test_finitely_generated_with_names() {
        let names = vec!["L".to_string(), "G".to_string(), "C".to_string()];
        let degrees = vec![Degree::int(2), Degree::rational(3, 2), Degree::int(0)];

        let lca: FinitelyFreelyGeneratedLCA<i64> = FinitelyFreelyGeneratedLCA::new(
            1,
            3,
            Some(names.clone()),
            Some(degrees),
            None,
            Some(vec![2]), // C is central
        );

        assert_eq!(lca.ngens(), 3);
        assert_eq!(lca.generator_names(), &names[..]);
        assert_eq!(lca.central_elements_indices(), &[2]);
        assert!(lca.is_central(2));
        assert!(!lca.is_central(0));
    }

    #[test]
    fn test_finitely_generated_gens() {
        let lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 4, None, None, None, None);

        let generators = lca.gens();
        assert_eq!(generators.len(), 4);
    }

    #[test]
    fn test_finitely_generated_central_elements() {
        let lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 5, None, None, None, Some(vec![3, 4]));

        let central = lca.central_elements();
        assert_eq!(central.len(), 2);
        assert_eq!(lca.central_elements_indices(), &[3, 4]);
    }

    #[test]
    fn test_finitely_generated_an_element() {
        let lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 2, None, None, None, None);

        let element = lca.an_element();
        // Should be the first generator
        assert!(!element.is_zero());

        // For an algebra with 0 generators (hypothetically), should return zero
        // But we can't test that since ngens > 0 is enforced in new()
    }

    #[test]
    fn test_finitely_generated_trait_methods() {
        let lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 3, None, None, None, None);

        // Test LieConformalAlgebra trait methods
        assert_eq!(lca.ngens(), 3); // Our method
        assert_eq!(
            <FinitelyFreelyGeneratedLCA<i64> as LieConformalAlgebra<i64>>::ngens(&lca),
            Some(3)
        ); // Trait method

        assert!(lca.generator(0).is_some());
        assert!(lca.generator(2).is_some());
        assert!(lca.generator(3).is_none());

        // Test GradedLieConformalAlgebra trait
        assert_eq!(lca.generator_degree(0), Some(Degree::int(1)));
    }

    #[test]
    fn test_finitely_generated_degrees() {
        let degrees = vec![Degree::int(2), Degree::rational(1, 2), Degree::int(0)];
        let lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 3, None, Some(degrees.clone()), None, None);

        assert_eq!(lca.generator_degree(0), Some(Degree::int(2)));
        assert_eq!(lca.generator_degree(1), Some(Degree::rational(1, 2)));
        assert_eq!(lca.generator_degree(2), Some(Degree::int(0)));
    }

    #[test]
    fn test_finitely_generated_display() {
        let lca: FinitelyFreelyGeneratedLCA<i64> = FinitelyFreelyGeneratedLCA::new(
            1,
            2,
            Some(vec!["L".to_string(), "C".to_string()]),
            None,
            None,
            None,
        );

        let display = format!("{}", lca);
        assert!(display.contains("Lie conformal algebra"));
        assert!(display.contains("L"));
        assert!(display.contains("C"));
    }

    #[test]
    fn test_finitely_generated_single_generator() {
        let lca: FinitelyFreelyGeneratedLCA<i64> = FinitelyFreelyGeneratedLCA::new(
            1,
            1,
            Some(vec!["V".to_string()]),
            None,
            None,
            None,
        );

        let display = format!("{}", lca);
        assert!(display.contains("generated by V"));
    }

    #[test]
    fn test_finitely_generated_with_parities() {
        let parities = vec![0, 1, 0]; // even, odd, even
        let lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 3, None, None, Some(parities), None);

        assert!(lca.freely_generated_structure().graded_structure().is_super());
    }

    #[test]
    #[should_panic(expected = "Number of generators must be positive")]
    fn test_finitely_generated_zero_gens() {
        let _lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 0, None, None, None, None);
    }

    #[test]
    #[should_panic(expected = "Number of names must match ngens")]
    fn test_finitely_generated_wrong_name_count() {
        let names = vec!["a".to_string()]; // Only 1 name for 2 gens
        let _lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 2, Some(names), None, None, None);
    }

    #[test]
    #[should_panic(expected = "Number of degrees must match ngens")]
    fn test_finitely_generated_wrong_degree_count() {
        let degrees = vec![Degree::int(1)]; // Only 1 degree for 2 gens
        let _lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 2, None, Some(degrees), None, None);
    }

    #[test]
    #[should_panic(expected = "Central element index")]
    fn test_finitely_generated_invalid_central_index() {
        let _lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 2, None, None, None, Some(vec![5]));
    }

    #[test]
    fn test_finitely_generated_no_central_elements() {
        let lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 3, None, None, None, None);

        assert!(lca.central_elements().is_empty());
        assert!(lca.central_elements_indices().is_empty());
        assert!(!lca.is_central(0));
        assert!(!lca.is_central(1));
        assert!(!lca.is_central(2));
    }

    #[test]
    fn test_finitely_generated_multiple_central() {
        let lca: FinitelyFreelyGeneratedLCA<i64> =
            FinitelyFreelyGeneratedLCA::new(1, 5, None, None, None, Some(vec![2, 3, 4]));

        assert_eq!(lca.central_elements().len(), 3);
        assert!(!lca.is_central(0));
        assert!(!lca.is_central(1));
        assert!(lca.is_central(2));
        assert!(lca.is_central(3));
        assert!(lca.is_central(4));
    }
}
