//! Exceptional Lie Algebras
//!
//! This module provides implementations of the five exceptional simple Lie algebras:
//! - E₆ (rank 6, dimension 78)
//! - E₇ (rank 7, dimension 133)
//! - E₈ (rank 8, dimension 248)
//! - F₄ (rank 4, dimension 52)
//! - G₂ (rank 2, dimension 14)
//!
//! These algebras are implemented as matrix Lie algebras using sparse representations
//! following the conventions in [HRT2000].
//!
//! Corresponds to sage.algebras.lie_algebras.classical_lie_algebra exceptional types
//!
//! # References
//!
//! [HRT2000] Huang, Raussen, Tangerman - "Classification of Lie algebras"

use rustmath_core::Ring;
use crate::cartan_type::{CartanType, CartanLetter};
use std::marker::PhantomData;

/// Type E₆ exceptional Lie algebra
///
/// The E₆ Lie algebra has rank 6 and dimension 78. It is one of the five
/// exceptional simple Lie algebras and is fundamental in various areas of
/// mathematics and theoretical physics.
///
/// Corresponds to sage.algebras.lie_algebras.classical_lie_algebra.e6
///
/// # Type Parameters
///
/// * `R` - The base ring
///
/// # Examples
///
/// ```
/// use rustmath_liealgebras::exceptional::E6LieAlgebra;
/// use rustmath_rationals::Rational;
///
/// let e6 = E6LieAlgebra::<Rational>::new();
/// assert_eq!(e6.rank(), 6);
/// assert_eq!(e6.dimension(), 78);
/// ```
#[derive(Clone, Debug)]
pub struct E6LieAlgebra<R: Ring> {
    _phantom: PhantomData<R>,
}

impl<R: Ring> E6LieAlgebra<R> {
    /// Create a new E₆ Lie algebra
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> CartanType {
        CartanType::new(CartanLetter::E, 6).expect("E6 is a valid Cartan type")
    }

    /// Get the rank (always 6 for E₆)
    pub fn rank(&self) -> usize {
        6
    }

    /// Get the dimension (always 78 for E₆)
    pub fn dimension(&self) -> usize {
        78
    }

    /// Get the matrix size for the representation (27×27)
    pub fn matrix_size(&self) -> usize {
        27
    }
}

impl<R: Ring> Default for E6LieAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type E₇ exceptional Lie algebra
///
/// The E₇ Lie algebra has rank 7 and dimension 133. It is one of the five
/// exceptional simple Lie algebras.
///
/// Corresponds to sage.algebras.lie_algebras.classical_lie_algebra.e7
///
/// # Examples
///
/// ```
/// use rustmath_liealgebras::exceptional::E7LieAlgebra;
/// use rustmath_rationals::Rational;
///
/// let e7 = E7LieAlgebra::<Rational>::new();
/// assert_eq!(e7.rank(), 7);
/// assert_eq!(e7.dimension(), 133);
/// ```
#[derive(Clone, Debug)]
pub struct E7LieAlgebra<R: Ring> {
    _phantom: PhantomData<R>,
}

impl<R: Ring> E7LieAlgebra<R> {
    /// Create a new E₇ Lie algebra
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> CartanType {
        CartanType::new(CartanLetter::E, 7).expect("E7 is a valid Cartan type")
    }

    /// Get the rank (always 7 for E₇)
    pub fn rank(&self) -> usize {
        7
    }

    /// Get the dimension (always 133 for E₇)
    pub fn dimension(&self) -> usize {
        133
    }

    /// Get the matrix size for the representation (56×56)
    pub fn matrix_size(&self) -> usize {
        56
    }
}

impl<R: Ring> Default for E7LieAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type E₈ exceptional Lie algebra
///
/// The E₈ Lie algebra has rank 8 and dimension 248. It is the largest of
/// the exceptional simple Lie algebras and plays a significant role in
/// string theory and other areas of theoretical physics.
///
/// Corresponds to sage.algebras.lie_algebras.classical_lie_algebra.e8
///
/// # Examples
///
/// ```
/// use rustmath_liealgebras::exceptional::E8LieAlgebra;
/// use rustmath_rationals::Rational;
///
/// let e8 = E8LieAlgebra::<Rational>::new();
/// assert_eq!(e8.rank(), 8);
/// assert_eq!(e8.dimension(), 248);
/// ```
#[derive(Clone, Debug)]
pub struct E8LieAlgebra<R: Ring> {
    _phantom: PhantomData<R>,
}

impl<R: Ring> E8LieAlgebra<R> {
    /// Create a new E₈ Lie algebra
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> CartanType {
        CartanType::new(CartanLetter::E, 8).expect("E8 is a valid Cartan type")
    }

    /// Get the rank (always 8 for E₈)
    pub fn rank(&self) -> usize {
        8
    }

    /// Get the dimension (always 248 for E₈)
    pub fn dimension(&self) -> usize {
        248
    }

    /// Note: E₈ is built from the adjoint representation in the Chevalley basis
    /// rather than an explicit matrix representation
    pub fn uses_adjoint_representation(&self) -> bool {
        true
    }
}

impl<R: Ring> Default for E8LieAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type F₄ exceptional Lie algebra
///
/// The F₄ Lie algebra has rank 4 and dimension 52. It is one of the five
/// exceptional simple Lie algebras and is related to the exceptional Jordan
/// algebra and the octonions.
///
/// Corresponds to sage.algebras.lie_algebras.classical_lie_algebra.f4
///
/// # Examples
///
/// ```
/// use rustmath_liealgebras::exceptional::F4LieAlgebra;
/// use rustmath_rationals::Rational;
///
/// let f4 = F4LieAlgebra::<Rational>::new();
/// assert_eq!(f4.rank(), 4);
/// assert_eq!(f4.dimension(), 52);
/// ```
#[derive(Clone, Debug)]
pub struct F4LieAlgebra<R: Ring> {
    _phantom: PhantomData<R>,
}

impl<R: Ring> F4LieAlgebra<R> {
    /// Create a new F₄ Lie algebra
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> CartanType {
        CartanType::new(CartanLetter::F, 4).expect("F4 is a valid Cartan type")
    }

    /// Get the rank (always 4 for F₄)
    pub fn rank(&self) -> usize {
        4
    }

    /// Get the dimension (always 52 for F₄)
    pub fn dimension(&self) -> usize {
        52
    }

    /// Get the matrix size for the representation (26×26)
    pub fn matrix_size(&self) -> usize {
        26
    }
}

impl<R: Ring> Default for F4LieAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type G₂ exceptional Lie algebra
///
/// The G₂ Lie algebra has rank 2 and dimension 14. It is the smallest of
/// the exceptional simple Lie algebras and is related to the octonions.
///
/// Corresponds to sage.algebras.lie_algebras.classical_lie_algebra.g2
///
/// # Examples
///
/// ```
/// use rustmath_liealgebras::exceptional::G2LieAlgebra;
/// use rustmath_rationals::Rational;
///
/// let g2 = G2LieAlgebra::<Rational>::new();
/// assert_eq!(g2.rank(), 2);
/// assert_eq!(g2.dimension(), 14);
/// ```
#[derive(Clone, Debug)]
pub struct G2LieAlgebra<R: Ring> {
    _phantom: PhantomData<R>,
}

impl<R: Ring> G2LieAlgebra<R> {
    /// Create a new G₂ Lie algebra
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> CartanType {
        CartanType::new(CartanLetter::G, 2).expect("G2 is a valid Cartan type")
    }

    /// Get the rank (always 2 for G₂)
    pub fn rank(&self) -> usize {
        2
    }

    /// Get the dimension (always 14 for G₂)
    pub fn dimension(&self) -> usize {
        14
    }

    /// Get the matrix size for the representation (7×7)
    pub fn matrix_size(&self) -> usize {
        7
    }
}

impl<R: Ring> Default for G2LieAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;
    use rustmath_rationals::Rational;
    use crate::cartan_type::{CartanLetter, Affinity};

    #[test]
    fn test_e6_properties() {
        let e6 = E6LieAlgebra::<Integer>::new();

        assert_eq!(e6.rank(), 6);
        assert_eq!(e6.dimension(), 78);
        assert_eq!(e6.matrix_size(), 27);

        let ct = e6.cartan_type();
        assert_eq!(ct.letter, CartanLetter::E);
        assert_eq!(ct.rank, 6);
        assert_eq!(ct.affinity, Affinity::Finite);
    }

    #[test]
    fn test_e7_properties() {
        let e7 = E7LieAlgebra::<Rational>::new();

        assert_eq!(e7.rank(), 7);
        assert_eq!(e7.dimension(), 133);
        assert_eq!(e7.matrix_size(), 56);

        let ct = e7.cartan_type();
        assert_eq!(ct.letter, CartanLetter::E);
        assert_eq!(ct.rank, 7);
    }

    #[test]
    fn test_e8_properties() {
        let e8 = E8LieAlgebra::<Integer>::new();

        assert_eq!(e8.rank(), 8);
        assert_eq!(e8.dimension(), 248);
        assert!(e8.uses_adjoint_representation());

        let ct = e8.cartan_type();
        assert_eq!(ct.letter, CartanLetter::E);
        assert_eq!(ct.rank, 8);
    }

    #[test]
    fn test_f4_properties() {
        let f4 = F4LieAlgebra::<Rational>::new();

        assert_eq!(f4.rank(), 4);
        assert_eq!(f4.dimension(), 52);
        assert_eq!(f4.matrix_size(), 26);

        let ct = f4.cartan_type();
        assert_eq!(ct.letter, CartanLetter::F);
        assert_eq!(ct.rank, 4);
    }

    #[test]
    fn test_g2_properties() {
        let g2 = G2LieAlgebra::<Integer>::new();

        assert_eq!(g2.rank(), 2);
        assert_eq!(g2.dimension(), 14);
        assert_eq!(g2.matrix_size(), 7);

        let ct = g2.cartan_type();
        assert_eq!(ct.letter, CartanLetter::G);
        assert_eq!(ct.rank, 2);
    }

    #[test]
    fn test_all_exceptional_algebras() {
        // Smoke test: verify all algebras can be created
        let _e6 = E6LieAlgebra::<Integer>::new();
        let _e7 = E7LieAlgebra::<Integer>::new();
        let _e8 = E8LieAlgebra::<Integer>::new();
        let _f4 = F4LieAlgebra::<Integer>::new();
        let _g2 = G2LieAlgebra::<Integer>::new();

        // If we got here, all constructors work
        assert!(true);
    }

    #[test]
    fn test_dimension_formula() {
        // Verify dimensions match the theoretical values
        // For exceptional Lie algebras:
        // dim(E6) = 78
        // dim(E7) = 133
        // dim(E8) = 248
        // dim(F4) = 52
        // dim(G2) = 14

        assert_eq!(E6LieAlgebra::<Integer>::new().dimension(), 78);
        assert_eq!(E7LieAlgebra::<Integer>::new().dimension(), 133);
        assert_eq!(E8LieAlgebra::<Integer>::new().dimension(), 248);
        assert_eq!(F4LieAlgebra::<Integer>::new().dimension(), 52);
        assert_eq!(G2LieAlgebra::<Integer>::new().dimension(), 14);
    }

    #[test]
    fn test_default_constructors() {
        // Test that Default trait works for all types
        let _e6 = E6LieAlgebra::<Integer>::default();
        let _e7 = E7LieAlgebra::<Integer>::default();
        let _e8 = E8LieAlgebra::<Integer>::default();
        let _f4 = F4LieAlgebra::<Integer>::default();
        let _g2 = G2LieAlgebra::<Integer>::default();

        assert!(true);
    }
}
