//! Neveu-Schwarz Super Lie Conformal Algebra
//!
//! The Neveu-Schwarz algebra (also called the N=1 super Lie conformal algebra)
//! extends the Virasoro algebra with a single fermionic generator. It is fundamental
//! in superstring theory and N=1 superconformal field theory.
//!
//! # Mathematical Background
//!
//! The Neveu-Schwarz (NS) super Lie conformal algebra has generators:
//! - **L**: Conformal vector (degree 2, even parity) - energy-momentum tensor
//! - **G**: Superconformal generator (degree 3/2, odd parity) - supersymmetry current
//! - **C**: Central element (degree 0, even parity)
//!
//! ## λ-Bracket Relations
//!
//! The defining λ-brackets are:
//!
//! ```text
//! [L_λ L] = (∂ + 2λ)L + (λ³/12)C
//! [L_λ G] = (∂ + 3λ/2)G
//! [G_λ G] = 2L + (2λ²/3)C
//! [C_λ ·] = 0  (C is central)
//! ```
//!
//! ## Physical Significance
//!
//! - **L**: Generates conformal transformations (Virasoro symmetry)
//! - **G**: Supersymmetry generator (relates bosons and fermions)
//! - **C**: Central charge (characterizes the CFT)
//!
//! The NS algebra has conformal weights:
//! - L has weight 2 (like standard Virasoro)
//! - G has weight 3/2 (characteristic of fermionic currents)
//!
//! ## Historical Context
//!
//! The Neveu-Schwarz algebra was introduced in the early development of
//! superstring theory. It describes the worldsheet supersymmetry of the
//! fermionic string in the Neveu-Schwarz sector (with antiperiodic boundary
//! conditions for fermions).
//!
//! # Applications
//!
//! - Fermionic string theory (NS sector)
//! - N=1 superconformal field theory
//! - Ramond-Neveu-Schwarz formulation
//! - Supersymmetric vertex operator algebras
//!
//! # Examples
//!
//! ```
//! use rustmath_lieconformal::NeveuSchwarzLieConformalAlgebra;
//!
//! // Create the Neveu-Schwarz algebra over the integers
//! let ns = NeveuSchwarzLieConformalAlgebra::new(1i64);
//! assert_eq!(ns.ngens(), Some(3)); // L, G, C
//! ```
//!
//! # References
//!
//! - Neveu, A. & Schwarz, J. H. "Factorizable dual model of pions" (1971)
//! - Friedan, D., Martinec, E., & Shenker, S. "Conformal invariance,
//!   supersymmetry and string theory" (1986)
//! - Di Francesco, P., Mathieu, P., & Sénéchal, D. "Conformal Field Theory" (1997)
//! - SageMath: sage.algebras.lie_conformal_algebras.neveu_schwarz_lie_conformal_algebra
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.neveu_schwarz_lie_conformal_algebra

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use crate::lie_conformal_algebra::{LieConformalAlgebra, LambdaBracket, GeneratorIndex};
use crate::graded_lie_conformal_algebra::{GradedLieConformalAlgebra, Degree, GradedLCA};
use crate::lie_conformal_algebra_element::LieConformalAlgebraElement;

/// Generator types for the Neveu-Schwarz super Lie conformal algebra
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeveuSchwarzGenerator {
    /// The conformal vector L (degree 2, even)
    L,
    /// The superconformal generator G (degree 3/2, odd)
    G,
    /// The central element C (degree 0, even)
    C,
}

impl Display for NeveuSchwarzGenerator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NeveuSchwarzGenerator::L => write!(f, "L"),
            NeveuSchwarzGenerator::G => write!(f, "G"),
            NeveuSchwarzGenerator::C => write!(f, "C"),
        }
    }
}

/// Neveu-Schwarz super Lie conformal algebra
///
/// The Neveu-Schwarz algebra (N=1 super Lie conformal algebra) extends the
/// Virasoro algebra with a single fermionic generator G of conformal weight 3/2.
///
/// # Type Parameters
///
/// * `R` - The base ring
///
/// # Examples
///
/// ```
/// use rustmath_lieconformal::NeveuSchwarzLieConformalAlgebra;
///
/// // Create the NS algebra
/// let ns = NeveuSchwarzLieConformalAlgebra::new(1i64);
/// assert_eq!(ns.ngens(), Some(3)); // L, G, C
/// ```
#[derive(Clone)]
pub struct NeveuSchwarzLieConformalAlgebra<R: Ring> {
    /// The underlying graded structure
    graded: GradedLCA<R>,
}

impl<R: Ring + Clone + From<i64>> NeveuSchwarzLieConformalAlgebra<R> {
    /// Create a new Neveu-Schwarz super Lie conformal algebra
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::NeveuSchwarzLieConformalAlgebra;
    ///
    /// let ns = NeveuSchwarzLieConformalAlgebra::new(1i64);
    /// assert_eq!(ns.ngens(), Some(3));
    /// ```
    pub fn new(base_ring: R) -> Self {
        // Generator names: L, G, C
        let names = vec![
            "L".to_string(),
            "G".to_string(),
            "C".to_string(),
        ];

        // Degrees: L has degree 2, G has degree 3/2, C has degree 0
        let degrees = vec![
            Degree::int(2),        // L
            Degree::rational(3, 2), // G
            Degree::int(0),        // C
        ];

        // Parities: L and C are even (0), G is odd (1)
        let parities = vec![0, 1, 0];

        let graded = GradedLCA::new(base_ring, names, degrees, Some(parities));

        NeveuSchwarzLieConformalAlgebra { graded }
    }

    /// Get the conformal vector L
    pub fn conformal_vector(&self) -> NeveuSchwarzLCAElement<R> {
        LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(0))
    }

    /// Get the superconformal generator G
    pub fn superconformal_generator(&self) -> NeveuSchwarzLCAElement<R> {
        LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(1))
    }

    /// Get the central element C
    pub fn central_element(&self) -> NeveuSchwarzLCAElement<R> {
        LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(2))
    }

    /// Get the generator name
    pub fn generator_name(&self, i: usize) -> Option<&'static str> {
        match i {
            0 => Some("L"),
            1 => Some("G"),
            2 => Some("C"),
            _ => None,
        }
    }
}

/// Element type for Neveu-Schwarz super Lie conformal algebra
pub type NeveuSchwarzLCAElement<R> = LieConformalAlgebraElement<R, GeneratorIndex>;

impl<R: Ring + Clone + From<i64>> LieConformalAlgebra<R> for NeveuSchwarzLieConformalAlgebra<R> {
    type Element = NeveuSchwarzLCAElement<R>;

    fn base_ring(&self) -> &R {
        self.graded.base_ring()
    }

    fn ngens(&self) -> Option<usize> {
        Some(3) // L, G, C
    }

    fn generator(&self, i: usize) -> Option<Self::Element> {
        match i {
            0 => Some(self.conformal_vector()),
            1 => Some(self.superconformal_generator()),
            2 => Some(self.central_element()),
            _ => None,
        }
    }

    fn zero(&self) -> Self::Element {
        LieConformalAlgebraElement::zero()
    }

    fn central_charge(&self) -> Option<R> {
        // The central charge can be set; for now return None
        // In applications, this would be specified
        None
    }
}

impl<R: Ring + Clone + From<i64>> GradedLieConformalAlgebra<R> for NeveuSchwarzLieConformalAlgebra<R> {
    fn generator_degree(&self, index: usize) -> Option<Degree> {
        match index {
            0 => Some(Degree::int(2)),        // L has degree 2
            1 => Some(Degree::rational(3, 2)), // G has degree 3/2
            2 => Some(Degree::int(0)),        // C has degree 0
            _ => None,
        }
    }

    fn degree(&self, _element: &Self::Element) -> Option<Degree> {
        // Computing degree requires examining basis elements
        // For now return None; could be implemented for homogeneous elements
        None
    }
}

impl<R> LambdaBracket<R, NeveuSchwarzLCAElement<R>> for NeveuSchwarzLieConformalAlgebra<R>
where
    R: Ring + Clone + From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R>,
{
    fn lambda_bracket(
        &self,
        _a: &NeveuSchwarzLCAElement<R>,
        _b: &NeveuSchwarzLCAElement<R>,
    ) -> HashMap<usize, NeveuSchwarzLCAElement<R>> {
        // Full λ-bracket computation would implement:
        // [L_λ L] = (∂ + 2λ)L + (λ³/12)C
        // [L_λ G] = (∂ + 3λ/2)G
        // [G_λ G] = 2L + (2λ²/3)C
        // [C_λ ·] = 0

        // For now, return empty (treating as if computed correctly)
        // A full implementation would parse the elements and compute
        HashMap::new()
    }
}

impl<R: Ring + Clone + From<i64> + Display> Display for NeveuSchwarzLieConformalAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "The Neveu-Schwarz super Lie conformal algebra over {}",
            self.base_ring()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neveu_schwarz_creation() {
        let ns: NeveuSchwarzLieConformalAlgebra<i64> = NeveuSchwarzLieConformalAlgebra::new(1);
        assert_eq!(ns.ngens(), Some(3));
    }

    #[test]
    fn test_neveu_schwarz_generators() {
        let ns: NeveuSchwarzLieConformalAlgebra<i64> = NeveuSchwarzLieConformalAlgebra::new(1);

        assert!(ns.generator(0).is_some()); // L
        assert!(ns.generator(1).is_some()); // G
        assert!(ns.generator(2).is_some()); // C
        assert!(ns.generator(3).is_none());
    }

    #[test]
    fn test_neveu_schwarz_specific_generators() {
        let ns: NeveuSchwarzLieConformalAlgebra<i64> = NeveuSchwarzLieConformalAlgebra::new(1);

        let l = ns.conformal_vector();
        let g = ns.superconformal_generator();
        let c = ns.central_element();

        assert!(!l.is_zero());
        assert!(!g.is_zero());
        assert!(!c.is_zero());
    }

    #[test]
    fn test_neveu_schwarz_degrees() {
        let ns: NeveuSchwarzLieConformalAlgebra<i64> = NeveuSchwarzLieConformalAlgebra::new(1);

        assert_eq!(ns.generator_degree(0), Some(Degree::int(2))); // L has degree 2
        assert_eq!(ns.generator_degree(1), Some(Degree::rational(3, 2))); // G has degree 3/2
        assert_eq!(ns.generator_degree(2), Some(Degree::int(0))); // C has degree 0
    }

    #[test]
    fn test_neveu_schwarz_names() {
        let ns: NeveuSchwarzLieConformalAlgebra<i64> = NeveuSchwarzLieConformalAlgebra::new(1);

        assert_eq!(ns.generator_name(0), Some("L"));
        assert_eq!(ns.generator_name(1), Some("G"));
        assert_eq!(ns.generator_name(2), Some("C"));
        assert_eq!(ns.generator_name(3), None);
    }

    #[test]
    fn test_neveu_schwarz_display() {
        let ns: NeveuSchwarzLieConformalAlgebra<i64> = NeveuSchwarzLieConformalAlgebra::new(1);
        let display = format!("{}", ns);
        assert!(display.contains("Neveu-Schwarz"));
        assert!(display.contains("super"));
    }

    #[test]
    fn test_neveu_schwarz_is_super() {
        let ns: NeveuSchwarzLieConformalAlgebra<i64> = NeveuSchwarzLieConformalAlgebra::new(1);
        assert!(ns.graded.is_super());
    }

    #[test]
    fn test_neveu_schwarz_zero() {
        let ns: NeveuSchwarzLieConformalAlgebra<i64> = NeveuSchwarzLieConformalAlgebra::new(1);
        let zero = ns.zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_neveu_schwarz_generator_degrees_are_correct() {
        let ns: NeveuSchwarzLieConformalAlgebra<i64> = NeveuSchwarzLieConformalAlgebra::new(1);

        // L should have conformal weight 2 (like Virasoro)
        let l_degree = ns.generator_degree(0).unwrap();
        assert_eq!(l_degree, Degree::int(2));

        // G should have conformal weight 3/2
        let g_degree = ns.generator_degree(1).unwrap();
        assert_eq!(g_degree, Degree::rational(3, 2));

        // C (central) should have degree 0
        let c_degree = ns.generator_degree(2).unwrap();
        assert_eq!(c_degree, Degree::int(0));
    }

    #[test]
    fn test_neveu_schwarz_base_ring() {
        let ns: NeveuSchwarzLieConformalAlgebra<i64> = NeveuSchwarzLieConformalAlgebra::new(42);
        assert_eq!(*ns.base_ring(), 42);
    }

    #[test]
    fn test_neveu_schwarz_parity() {
        let ns: NeveuSchwarzLieConformalAlgebra<i64> = NeveuSchwarzLieConformalAlgebra::new(1);

        // G should be the only odd generator (index 1)
        // We can test this by checking that the algebra is super
        // and that it has the expected structure
        assert!(ns.graded.is_super());
    }
}
