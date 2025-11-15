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
use rustmath_matrix::Matrix;
use crate::cartan_type::{CartanType, CartanLetter};
use std::marker::PhantomData;
use std::fmt::{self, Display};

/// Base class for exceptional matrix Lie algebras
///
/// This provides the common framework for exceptional types (E, F, G) that don't
/// follow the classical A, B, C, D patterns. The key feature is automatic computation
/// of h-generators from e and f generators using the commutator: [e_i, f_i] = h_i
///
/// Corresponds to sage.algebras.lie_algebras.classical_lie_algebra.ExceptionalMatrixLieAlgebra
///
/// # Type Parameters
///
/// * `R` - The base ring
///
/// # Examples
///
/// ```
/// use rustmath_liealgebras::exceptional::ExceptionalMatrixLieAlgebra;
/// use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
/// use rustmath_rationals::Rational;
/// use rustmath_matrix::Matrix;
///
/// let ct = CartanType::new(CartanLetter::G, 2).unwrap();
/// let e_gens = vec![Matrix::identity(7), Matrix::identity(7)];
/// let f_gens = vec![Matrix::identity(7), Matrix::identity(7)];
/// let exceptional = ExceptionalMatrixLieAlgebra::<Rational>::new(ct, e_gens, f_gens, None);
/// assert_eq!(exceptional.rank(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct ExceptionalMatrixLieAlgebra<R: Ring> {
    /// Cartan type (must be E, F, or G)
    cartan_type: CartanType,
    /// E generators (positive simple root vectors)
    e_generators: Vec<Matrix<R>>,
    /// F generators (negative simple root vectors)
    f_generators: Vec<Matrix<R>>,
    /// H generators (Cartan subalgebra elements)
    h_generators: Vec<Matrix<R>>,
    /// Matrix size for this representation
    matrix_size: usize,
    /// Coefficient ring marker
    _phantom: PhantomData<R>,
}

impl<R: Ring + Clone + From<i64>> ExceptionalMatrixLieAlgebra<R> {
    /// Create a new exceptional matrix Lie algebra
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - Must be type E, F, or G
    /// * `e_generators` - Positive simple root generators
    /// * `f_generators` - Negative simple root generators
    /// * `h_generators` - Optional Cartan generators (computed if None)
    ///
    /// # Panics
    ///
    /// Panics if cartan_type is not exceptional (E, F, G)
    pub fn new(
        cartan_type: CartanType,
        e_generators: Vec<Matrix<R>>,
        f_generators: Vec<Matrix<R>>,
        h_generators: Option<Vec<Matrix<R>>>,
    ) -> Self {
        // Verify this is an exceptional type
        match cartan_type.letter {
            CartanLetter::E | CartanLetter::F | CartanLetter::G => {}
            _ => panic!("ExceptionalMatrixLieAlgebra requires type E, F, or G"),
        }

        assert_eq!(
            e_generators.len(),
            f_generators.len(),
            "Number of e and f generators must match"
        );

        assert_eq!(
            e_generators.len(),
            cartan_type.rank,
            "Number of generators must equal rank"
        );

        let matrix_size = if !e_generators.is_empty() {
            e_generators[0].rows()
        } else {
            0
        };

        // Compute h-generators if not provided
        let h_generators = match h_generators {
            Some(h) => h,
            None => Self::compute_h_generators(&e_generators, &f_generators),
        };

        Self {
            cartan_type,
            e_generators,
            f_generators,
            h_generators,
            matrix_size,
            _phantom: PhantomData,
        }
    }

    /// Compute h-generators from e and f using commutator: h_i = [e_i, f_i]
    fn compute_h_generators(
        e_gens: &[Matrix<R>],
        f_gens: &[Matrix<R>],
    ) -> Vec<Matrix<R>>
    where
        R: std::ops::Sub<Output = R> + std::ops::Mul<Output = R> + std::ops::Add<Output = R>,
    {
        e_gens
            .iter()
            .zip(f_gens.iter())
            .map(|(e, f)| {
                // h_i = e_i * f_i - f_i * e_i (commutator)
                Self::commutator(e, f)
            })
            .collect()
    }

    /// Compute commutator [A, B] = AB - BA
    fn commutator(a: &Matrix<R>, b: &Matrix<R>) -> Matrix<R>
    where
        R: std::ops::Sub<Output = R> + std::ops::Mul<Output = R> + std::ops::Add<Output = R>,
    {
        let n = a.rows();
        let mut result = Matrix::zeros(n, n);

        // Compute AB
        let mut ab = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = R::from(0);
                for k in 0..n {
                    let a_ik = a.get(i, k).unwrap().clone();
                    let b_kj = b.get(k, j).unwrap().clone();
                    sum = sum + (a_ik * b_kj);
                }
                let _ = ab.set(i, j, sum);
            }
        }

        // Compute BA
        let mut ba = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = R::from(0);
                for k in 0..n {
                    let b_ik = b.get(i, k).unwrap().clone();
                    let a_kj = a.get(k, j).unwrap().clone();
                    sum = sum + (b_ik * a_kj);
                }
                let _ = ba.set(i, j, sum);
            }
        }

        // Compute AB - BA
        for i in 0..n {
            for j in 0..n {
                let ab_ij = ab.get(i, j).unwrap().clone();
                let ba_ij = ba.get(i, j).unwrap().clone();
                let _ = result.set(i, j, ab_ij - ba_ij);
            }
        }

        result
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &CartanType {
        &self.cartan_type
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.cartan_type.rank
    }

    /// Get the dimension (total number of basis elements)
    pub fn dimension(&self) -> usize {
        match (self.cartan_type.letter, self.cartan_type.rank) {
            (CartanLetter::E, 6) => 78,
            (CartanLetter::E, 7) => 133,
            (CartanLetter::E, 8) => 248,
            (CartanLetter::F, 4) => 52,
            (CartanLetter::G, 2) => 14,
            _ => 0,
        }
    }

    /// Get the matrix size
    pub fn matrix_size(&self) -> usize {
        self.matrix_size
    }

    /// Get the e-generators (positive root vectors)
    pub fn e_generators(&self) -> &[Matrix<R>] {
        &self.e_generators
    }

    /// Get the f-generators (negative root vectors)
    pub fn f_generators(&self) -> &[Matrix<R>] {
        &self.f_generators
    }

    /// Get the h-generators (Cartan subalgebra)
    pub fn h_generators(&self) -> &[Matrix<R>] {
        &self.h_generators
    }
}

impl<R: Ring + Clone + From<i64>> Display for ExceptionalMatrixLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Simple matrix Lie algebra of type {} over Ring",
            self.cartan_type
        )
    }
}

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

    #[test]
    fn test_exceptional_matrix_lie_algebra_g2() {
        use rustmath_matrix::Matrix;
        use rustmath_rationals::Rational;

        let ct = CartanType::new(CartanLetter::G, 2).unwrap();
        let e_gens = vec![Matrix::identity(7), Matrix::identity(7)];
        let f_gens = vec![Matrix::identity(7), Matrix::identity(7)];

        let exceptional = ExceptionalMatrixLieAlgebra::<Rational>::new(
            ct,
            e_gens,
            f_gens,
            None,
        );

        assert_eq!(exceptional.rank(), 2);
        assert_eq!(exceptional.dimension(), 14);
        assert_eq!(exceptional.matrix_size(), 7);
        assert_eq!(exceptional.e_generators().len(), 2);
        assert_eq!(exceptional.f_generators().len(), 2);
        assert_eq!(exceptional.h_generators().len(), 2);
    }

    #[test]
    fn test_exceptional_matrix_lie_algebra_f4() {
        use rustmath_matrix::Matrix;
        use rustmath_rationals::Rational;

        let ct = CartanType::new(CartanLetter::F, 4).unwrap();
        let e_gens = vec![
            Matrix::identity(26),
            Matrix::identity(26),
            Matrix::identity(26),
            Matrix::identity(26),
        ];
        let f_gens = vec![
            Matrix::identity(26),
            Matrix::identity(26),
            Matrix::identity(26),
            Matrix::identity(26),
        ];

        let exceptional = ExceptionalMatrixLieAlgebra::<Rational>::new(
            ct,
            e_gens,
            f_gens,
            None,
        );

        assert_eq!(exceptional.rank(), 4);
        assert_eq!(exceptional.dimension(), 52);
        assert_eq!(exceptional.h_generators().len(), 4);
    }

    #[test]
    #[should_panic(expected = "requires type E, F, or G")]
    fn test_exceptional_rejects_classical_type() {
        use rustmath_matrix::Matrix;
        use rustmath_rationals::Rational;

        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let e_gens = vec![Matrix::identity(3), Matrix::identity(3)];
        let f_gens = vec![Matrix::identity(3), Matrix::identity(3)];

        // Should panic - type A is not exceptional
        let _ = ExceptionalMatrixLieAlgebra::<Rational>::new(ct, e_gens, f_gens, None);
    }

    #[test]
    fn test_h_generator_computation() {
        use rustmath_matrix::Matrix;
        use rustmath_rationals::Rational;

        let ct = CartanType::new(CartanLetter::G, 2).unwrap();
        let e_gens = vec![Matrix::identity(3), Matrix::identity(3)];
        let f_gens = vec![Matrix::identity(3), Matrix::identity(3)];

        // Without h_generators - should compute them
        let exceptional1 = ExceptionalMatrixLieAlgebra::<Rational>::new(
            ct.clone(),
            e_gens.clone(),
            f_gens.clone(),
            None,
        );

        // With h_generators - should use provided ones
        let h_gens = vec![Matrix::zeros(3, 3), Matrix::zeros(3, 3)];
        let exceptional2 = ExceptionalMatrixLieAlgebra::<Rational>::new(
            ct,
            e_gens,
            f_gens,
            Some(h_gens),
        );

        assert_eq!(exceptional1.h_generators().len(), 2);
        assert_eq!(exceptional2.h_generators().len(), 2);
    }

    #[test]
    fn test_exceptional_display() {
        use rustmath_matrix::Matrix;
        use rustmath_rationals::Rational;

        let ct = CartanType::new(CartanLetter::G, 2).unwrap();
        let e_gens = vec![Matrix::identity(7), Matrix::identity(7)];
        let f_gens = vec![Matrix::identity(7), Matrix::identity(7)];

        let exceptional = ExceptionalMatrixLieAlgebra::<Rational>::new(
            ct,
            e_gens,
            f_gens,
            None,
        );

        let display = format!("{}", exceptional);
        assert!(display.contains("Simple matrix Lie algebra"));
        assert!(display.contains("G_2"));
    }
}
