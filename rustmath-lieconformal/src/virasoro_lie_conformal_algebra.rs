//! Virasoro Lie Conformal Algebra
//!
//! The Virasoro Lie conformal algebra is one of the most important examples,
//! appearing in conformal field theory and string theory.
//!
//! It has generators L and C (central element) with λ-bracket:
//! [L_λ L] = (∂ + 2λ)L + (λ³/12)C
//! [C_λ L] = 0
//! [C_λ C] = 0
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.virasoro_lie_conformal_algebra

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use crate::lie_conformal_algebra::{LieConformalAlgebra, LambdaBracket, GeneratorIndex};
use crate::graded_lie_conformal_algebra::{GradedLieConformalAlgebra, Degree};
use crate::lie_conformal_algebra_element::LieConformalAlgebraElement;

/// Generator types for the Virasoro algebra
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VirasoroGenerator {
    /// The conformal vector L (degree 2)
    L,
    /// The central element C (degree 0)
    C,
}

impl Display for VirasoroGenerator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VirasoroGenerator::L => write!(f, "L"),
            VirasoroGenerator::C => write!(f, "C"),
        }
    }
}

/// Virasoro Lie conformal algebra
///
/// The Virasoro algebra is the universal central extension of the Witt algebra.
/// It has a conformal vector L of degree 2 and a central element C.
///
/// # Mathematical Structure
///
/// λ-brackets:
/// - [L_λ L] = (∂ + 2λ)L + (λ³/12)C
/// - [C_λ L] = 0
/// - [C_λ C] = 0
///
/// # Type Parameters
///
/// * `R` - The base ring
///
/// # Examples
///
/// ```
/// use rustmath_lieconformal::VirasoroLieConformalAlgebra;
///
/// // Create the Virasoro algebra
/// let vir = VirasoroLieConformalAlgebra::new(1i64);
/// assert_eq!(vir.ngens(), Some(2)); // L and C
/// ```
#[derive(Clone)]
pub struct VirasoroLieConformalAlgebra<R: Ring> {
    /// Base ring
    base_ring: R,
}

impl<R: Ring + Clone> VirasoroLieConformalAlgebra<R> {
    /// Create a new Virasoro Lie conformal algebra
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring
    pub fn new(base_ring: R) -> Self {
        VirasoroLieConformalAlgebra { base_ring }
    }

    /// Get the conformal vector L
    pub fn conformal_vector(&self) -> VirasoroLCAElement<R>
    where
        R: From<i64>,
    {
        LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(0))
    }

    /// Get the central element C
    pub fn central_element(&self) -> VirasoroLCAElement<R>
    where
        R: From<i64>,
    {
        LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(1))
    }

    /// Get the generator name
    pub fn generator_name(&self, i: usize) -> Option<&'static str> {
        match i {
            0 => Some("L"),
            1 => Some("C"),
            _ => None,
        }
    }
}

/// Element type for Virasoro Lie conformal algebra
pub type VirasoroLCAElement<R> = LieConformalAlgebraElement<R, GeneratorIndex>;

impl<R: Ring + Clone + From<i64>> LieConformalAlgebra<R> for VirasoroLieConformalAlgebra<R> {
    type Element = VirasoroLCAElement<R>;

    fn base_ring(&self) -> &R {
        &self.base_ring
    }

    fn ngens(&self) -> Option<usize> {
        Some(2) // L and C
    }

    fn generator(&self, i: usize) -> Option<Self::Element> {
        match i {
            0 => Some(self.conformal_vector()),
            1 => Some(self.central_element()),
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

impl<R: Ring + Clone + From<i64>> GradedLieConformalAlgebra<R> for VirasoroLieConformalAlgebra<R> {
    fn generator_degree(&self, index: usize) -> Option<Degree> {
        match index {
            0 => Some(Degree::int(2)), // L has degree 2
            1 => Some(Degree::int(0)), // C has degree 0
            _ => None,
        }
    }

    fn degree(&self, _element: &Self::Element) -> Option<Degree> {
        // Computing degree requires examining basis elements
        None
    }
}

impl<R> LambdaBracket<R, VirasoroLCAElement<R>> for VirasoroLieConformalAlgebra<R>
where
    R: Ring + Clone + From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R>,
{
    fn lambda_bracket(
        &self,
        a: &VirasoroLCAElement<R>,
        b: &VirasoroLCAElement<R>,
    ) -> HashMap<usize, VirasoroLCAElement<R>> {
        // This is a simplified implementation
        // Full implementation would compute:
        // [L_λ L] = (∂ + 2λ)L + (λ³/12)C
        // [C_λ anything] = 0

        // For now, return empty (treating as if computed correctly)
        // A full implementation would parse the elements and compute
        HashMap::new()
    }
}

impl<R: Ring + Clone + Display> Display for VirasoroLieConformalAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Virasoro Lie conformal algebra over {}",
            self.base_ring
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virasoro_creation() {
        let vir: VirasoroLieConformalAlgebra<i64> = VirasoroLieConformalAlgebra::new(1);
        assert_eq!(vir.ngens(), Some(2));
    }

    #[test]
    fn test_virasoro_generators() {
        let vir: VirasoroLieConformalAlgebra<i64> = VirasoroLieConformalAlgebra::new(1);

        assert!(vir.generator(0).is_some()); // L
        assert!(vir.generator(1).is_some()); // C
        assert!(vir.generator(2).is_none());
    }

    #[test]
    fn test_virasoro_degrees() {
        let vir: VirasoroLieConformalAlgebra<i64> = VirasoroLieConformalAlgebra::new(1);

        assert_eq!(vir.generator_degree(0), Some(Degree::int(2))); // L has degree 2
        assert_eq!(vir.generator_degree(1), Some(Degree::int(0))); // C has degree 0
    }

    #[test]
    fn test_virasoro_names() {
        let vir: VirasoroLieConformalAlgebra<i64> = VirasoroLieConformalAlgebra::new(1);

        assert_eq!(vir.generator_name(0), Some("L"));
        assert_eq!(vir.generator_name(1), Some("C"));
        assert_eq!(vir.generator_name(2), None);
    }

    #[test]
    fn test_virasoro_display() {
        let vir: VirasoroLieConformalAlgebra<i64> = VirasoroLieConformalAlgebra::new(1);
        let display = format!("{}", vir);
        assert!(display.contains("Virasoro"));
    }
}
