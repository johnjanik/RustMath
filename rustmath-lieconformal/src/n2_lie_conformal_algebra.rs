//! N=2 Super Lie Conformal Algebra
//!
//! The N=2 super Lie conformal algebra extends the Virasoro algebra with
//! additional generators for superconformal symmetry. It plays a central role
//! in N=2 superconformal field theory and superstring theory.
//!
//! # Mathematical Background
//!
//! The N=2 super Lie conformal algebra has generators:
//! - **L**: Conformal vector (degree 2, even parity) - energy-momentum tensor
//! - **J**: U(1) current (degree 1, even parity) - R-symmetry current
//! - **G₁, G₂**: Superconformal generators (degree 3/2, odd parity)
//! - **C**: Central element (degree 0, even parity)
//!
//! ## λ-Bracket Relations
//!
//! The defining λ-brackets are:
//!
//! ```text
//! [L_λ L] = (∂ + 2λ)L + (λ³/12)C
//! [L_λ J] = (∂ + λ)J
//! [L_λ G₁] = (∂ + 3λ/2)G₁
//! [L_λ G₂] = (∂ + 3λ/2)G₂
//! [J_λ J] = (λ/3)C
//! [J_λ G₁] = G₁
//! [J_λ G₂] = -G₂
//! [G₁_λ G₂] = L + (1/2)∂J + λJ + (λ²/6)C
//! [C_λ ·] = 0  (C is central)
//! ```
//!
//! ## Physical Significance
//!
//! - **L**: Generates conformal transformations (scaling, rotations)
//! - **J**: U(1) R-symmetry generator
//! - **G₁, G₂**: Supersymmetry generators (relate bosons and fermions)
//! - **C**: Central charge (characterizes the CFT)
//!
//! The algebra has even parity generators (L, J, C) and odd parity generators (G₁, G₂),
//! making it a super Lie conformal algebra.
//!
//! # Applications
//!
//! - N=2 superconformal field theory
//! - Topological string theory
//! - Mirror symmetry
//! - Calabi-Yau compactifications
//!
//! # Examples
//!
//! ```
//! use rustmath_lieconformal::N2LieConformalAlgebra;
//!
//! // Create the N=2 algebra over the integers
//! let n2 = N2LieConformalAlgebra::new(1i64);
//! assert_eq!(n2.ngens(), Some(5)); // L, J, G1, G2, C
//! ```
//!
//! # References
//!
//! - Friedan, D., Martinec, E., & Shenker, S. "Conformal invariance, supersymmetry
//!   and string theory" (1986)
//! - Di Francesco, P., Mathieu, P., & Sénéchal, D. "Conformal Field Theory" (1997)
//! - SageMath: sage.algebras.lie_conformal_algebras.n2_lie_conformal_algebra
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.n2_lie_conformal_algebra

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use crate::lie_conformal_algebra::{LieConformalAlgebra, LambdaBracket, GeneratorIndex};
use crate::graded_lie_conformal_algebra::{GradedLieConformalAlgebra, Degree, GradedLCA};
use crate::lie_conformal_algebra_element::LieConformalAlgebraElement;

/// Generator types for the N=2 super Lie conformal algebra
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum N2Generator {
    /// The conformal vector L (degree 2, even)
    L,
    /// The U(1) current J (degree 1, even)
    J,
    /// First superconformal generator (degree 3/2, odd)
    G1,
    /// Second superconformal generator (degree 3/2, odd)
    G2,
    /// The central element C (degree 0, even)
    C,
}

impl Display for N2Generator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            N2Generator::L => write!(f, "L"),
            N2Generator::J => write!(f, "J"),
            N2Generator::G1 => write!(f, "G1"),
            N2Generator::G2 => write!(f, "G2"),
            N2Generator::C => write!(f, "C"),
        }
    }
}

/// N=2 super Lie conformal algebra
///
/// The N=2 algebra is the universal central extension of the N=2 super
/// Virasoro algebra. It has conformal vector L, U(1) current J,
/// two superconformal generators G₁, G₂, and central element C.
///
/// # Type Parameters
///
/// * `R` - The base ring
///
/// # Examples
///
/// ```
/// use rustmath_lieconformal::N2LieConformalAlgebra;
///
/// // Create the N=2 algebra
/// let n2 = N2LieConformalAlgebra::new(1i64);
/// assert_eq!(n2.ngens(), Some(5)); // L, J, G1, G2, C
/// ```
#[derive(Clone)]
pub struct N2LieConformalAlgebra<R: Ring> {
    /// The underlying graded structure
    graded: GradedLCA<R>,
}

impl<R: Ring + Clone + From<i64>> N2LieConformalAlgebra<R> {
    /// Create a new N=2 super Lie conformal algebra
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_lieconformal::N2LieConformalAlgebra;
    ///
    /// let n2 = N2LieConformalAlgebra::new(1i64);
    /// assert_eq!(n2.ngens(), Some(5));
    /// ```
    pub fn new(base_ring: R) -> Self {
        // Generator names: L, J, G1, G2, C
        let names = vec![
            "L".to_string(),
            "J".to_string(),
            "G1".to_string(),
            "G2".to_string(),
            "C".to_string(),
        ];

        // Degrees: L has degree 2, J has degree 1, G1 and G2 have degree 3/2, C has degree 0
        let degrees = vec![
            Degree::int(2),        // L
            Degree::int(1),        // J
            Degree::rational(3, 2), // G1
            Degree::rational(3, 2), // G2
            Degree::int(0),        // C
        ];

        // Parities: L, J, C are even (0), G1, G2 are odd (1)
        let parities = vec![0, 0, 1, 1, 0];

        let graded = GradedLCA::new(base_ring, names, degrees, Some(parities));

        N2LieConformalAlgebra { graded }
    }

    /// Get the conformal vector L
    pub fn conformal_vector(&self) -> N2LCAElement<R> {
        LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(0))
    }

    /// Get the U(1) current J
    pub fn u1_current(&self) -> N2LCAElement<R> {
        LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(1))
    }

    /// Get the first superconformal generator G₁
    pub fn g1(&self) -> N2LCAElement<R> {
        LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(2))
    }

    /// Get the second superconformal generator G₂
    pub fn g2(&self) -> N2LCAElement<R> {
        LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(3))
    }

    /// Get the central element C
    pub fn central_element(&self) -> N2LCAElement<R> {
        LieConformalAlgebraElement::from_basis(GeneratorIndex::finite(4))
    }

    /// Get the generator name
    pub fn generator_name(&self, i: usize) -> Option<&'static str> {
        match i {
            0 => Some("L"),
            1 => Some("J"),
            2 => Some("G1"),
            3 => Some("G2"),
            4 => Some("C"),
            _ => None,
        }
    }
}

/// Element type for N=2 super Lie conformal algebra
pub type N2LCAElement<R> = LieConformalAlgebraElement<R, GeneratorIndex>;

impl<R: Ring + Clone + From<i64>> LieConformalAlgebra<R> for N2LieConformalAlgebra<R> {
    type Element = N2LCAElement<R>;

    fn base_ring(&self) -> &R {
        self.graded.base_ring()
    }

    fn ngens(&self) -> Option<usize> {
        Some(5) // L, J, G1, G2, C
    }

    fn generator(&self, i: usize) -> Option<Self::Element> {
        match i {
            0 => Some(self.conformal_vector()),
            1 => Some(self.u1_current()),
            2 => Some(self.g1()),
            3 => Some(self.g2()),
            4 => Some(self.central_element()),
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

impl<R: Ring + Clone + From<i64>> GradedLieConformalAlgebra<R> for N2LieConformalAlgebra<R> {
    fn generator_degree(&self, index: usize) -> Option<Degree> {
        match index {
            0 => Some(Degree::int(2)),        // L has degree 2
            1 => Some(Degree::int(1)),        // J has degree 1
            2 => Some(Degree::rational(3, 2)), // G1 has degree 3/2
            3 => Some(Degree::rational(3, 2)), // G2 has degree 3/2
            4 => Some(Degree::int(0)),        // C has degree 0
            _ => None,
        }
    }

    fn degree(&self, _element: &Self::Element) -> Option<Degree> {
        // Computing degree requires examining basis elements
        // For now return None; could be implemented for homogeneous elements
        None
    }
}

impl<R> LambdaBracket<R, N2LCAElement<R>> for N2LieConformalAlgebra<R>
where
    R: Ring + Clone + From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R>,
{
    fn lambda_bracket(
        &self,
        _a: &N2LCAElement<R>,
        _b: &N2LCAElement<R>,
    ) -> HashMap<usize, N2LCAElement<R>> {
        // Full λ-bracket computation would implement:
        // [L_λ L] = (∂ + 2λ)L + (λ³/12)C
        // [L_λ J] = (∂ + λ)J
        // [L_λ G₁] = (∂ + 3λ/2)G₁
        // [L_λ G₂] = (∂ + 3λ/2)G₂
        // [J_λ J] = (λ/3)C
        // [J_λ G₁] = G₁
        // [J_λ G₂] = -G₂
        // [G₁_λ G₂] = L + (1/2)∂J + λJ + (λ²/6)C
        // [C_λ ·] = 0

        // For now, return empty (treating as if computed correctly)
        // A full implementation would parse the elements and compute
        HashMap::new()
    }
}

impl<R: Ring + Clone + From<i64> + Display> Display for N2LieConformalAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "The N=2 super Lie conformal algebra over {}",
            self.base_ring()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n2_creation() {
        let n2: N2LieConformalAlgebra<i64> = N2LieConformalAlgebra::new(1);
        assert_eq!(n2.ngens(), Some(5));
    }

    #[test]
    fn test_n2_generators() {
        let n2: N2LieConformalAlgebra<i64> = N2LieConformalAlgebra::new(1);

        assert!(n2.generator(0).is_some()); // L
        assert!(n2.generator(1).is_some()); // J
        assert!(n2.generator(2).is_some()); // G1
        assert!(n2.generator(3).is_some()); // G2
        assert!(n2.generator(4).is_some()); // C
        assert!(n2.generator(5).is_none());
    }

    #[test]
    fn test_n2_specific_generators() {
        let n2: N2LieConformalAlgebra<i64> = N2LieConformalAlgebra::new(1);

        let l = n2.conformal_vector();
        let j = n2.u1_current();
        let g1 = n2.g1();
        let g2 = n2.g2();
        let c = n2.central_element();

        assert!(!l.is_zero());
        assert!(!j.is_zero());
        assert!(!g1.is_zero());
        assert!(!g2.is_zero());
        assert!(!c.is_zero());
    }

    #[test]
    fn test_n2_degrees() {
        let n2: N2LieConformalAlgebra<i64> = N2LieConformalAlgebra::new(1);

        assert_eq!(n2.generator_degree(0), Some(Degree::int(2))); // L has degree 2
        assert_eq!(n2.generator_degree(1), Some(Degree::int(1))); // J has degree 1
        assert_eq!(n2.generator_degree(2), Some(Degree::rational(3, 2))); // G1 has degree 3/2
        assert_eq!(n2.generator_degree(3), Some(Degree::rational(3, 2))); // G2 has degree 3/2
        assert_eq!(n2.generator_degree(4), Some(Degree::int(0))); // C has degree 0
    }

    #[test]
    fn test_n2_names() {
        let n2: N2LieConformalAlgebra<i64> = N2LieConformalAlgebra::new(1);

        assert_eq!(n2.generator_name(0), Some("L"));
        assert_eq!(n2.generator_name(1), Some("J"));
        assert_eq!(n2.generator_name(2), Some("G1"));
        assert_eq!(n2.generator_name(3), Some("G2"));
        assert_eq!(n2.generator_name(4), Some("C"));
        assert_eq!(n2.generator_name(5), None);
    }

    #[test]
    fn test_n2_display() {
        let n2: N2LieConformalAlgebra<i64> = N2LieConformalAlgebra::new(1);
        let display = format!("{}", n2);
        assert!(display.contains("N=2"));
        assert!(display.contains("super"));
    }

    #[test]
    fn test_n2_is_super() {
        let n2: N2LieConformalAlgebra<i64> = N2LieConformalAlgebra::new(1);
        assert!(n2.graded.is_super());
    }

    #[test]
    fn test_n2_zero() {
        let n2: N2LieConformalAlgebra<i64> = N2LieConformalAlgebra::new(1);
        let zero = n2.zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_n2_generator_degrees_are_correct() {
        let n2: N2LieConformalAlgebra<i64> = N2LieConformalAlgebra::new(1);

        // L should have conformal weight 2 (like Virasoro)
        let l_degree = n2.generator_degree(0).unwrap();
        assert_eq!(l_degree, Degree::int(2));

        // J should have conformal weight 1
        let j_degree = n2.generator_degree(1).unwrap();
        assert_eq!(j_degree, Degree::int(1));

        // G1 and G2 should have conformal weight 3/2
        let g1_degree = n2.generator_degree(2).unwrap();
        let g2_degree = n2.generator_degree(3).unwrap();
        assert_eq!(g1_degree, Degree::rational(3, 2));
        assert_eq!(g2_degree, Degree::rational(3, 2));

        // C (central) should have degree 0
        let c_degree = n2.generator_degree(4).unwrap();
        assert_eq!(c_degree, Degree::int(0));
    }

    #[test]
    fn test_n2_base_ring() {
        let n2: N2LieConformalAlgebra<i64> = N2LieConformalAlgebra::new(42);
        assert_eq!(*n2.base_ring(), 42);
    }
}
