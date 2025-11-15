//! Compact Real Forms of Lie Algebras
//!
//! This module implements compact real forms of classical Lie algebras through
//! the Cartan decomposition. A compact real form is a real Lie algebra with a
//! negative-definite Killing form, which makes it "compact" in the sense of
//! Lie group theory.
//!
//! The construction uses the Cartan decomposition L = K ⊕ S where:
//! - K: skew-symmetric matrices
//! - S: symmetric matrices
//! - Compact real form U = K ⊕ iS (skew-hermitian matrices)
//!
//! Corresponds to sage.algebras.lie_algebras.classical_lie_algebra.MatrixCompactRealForm
//!
//! # Mathematical Background
//!
//! For a classical Lie algebra L over ℝ that is closed under matrix transpose,
//! the Cartan decomposition splits L into symmetric and skew-symmetric parts.
//! The compact real form U = K ⊕ iS consists of skew-hermitian matrices and
//! has a negative-definite Killing form, ensuring compactness.
//!
//! # References
//!
//! - Helgason, S. "Differential Geometry, Lie Groups, and Symmetric Spaces" (1978)
//! - Knapp, A. "Lie Groups Beyond an Introduction" (2002)

use crate::cartan_type::CartanType;
use rustmath_core::Ring;
use rustmath_matrix::Matrix;
use rustmath_complex::Complex;
use rustmath_rationals::Rational;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Element of a compact real form Lie algebra
///
/// Represented as A + iB where:
/// - A is skew-symmetric (the "real" part in K)
/// - B is symmetric (the "imaginary" part in S)
///
/// This corresponds to a skew-hermitian matrix A + iB.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactRealFormElement<R: Ring> {
    /// Skew-symmetric part (real component)
    pub skew_symmetric: Matrix<R>,
    /// Symmetric part (imaginary component)
    pub symmetric: Matrix<R>,
}

impl<R: Ring + Clone> CompactRealFormElement<R> {
    /// Create a new compact real form element
    ///
    /// # Arguments
    ///
    /// * `skew_symmetric` - The skew-symmetric (K) part
    /// * `symmetric` - The symmetric (S) part
    ///
    /// # Panics
    ///
    /// Panics if matrices have different dimensions
    pub fn new(skew_symmetric: Matrix<R>, symmetric: Matrix<R>) -> Self {
        assert_eq!(
            (skew_symmetric.rows(), skew_symmetric.cols()),
            (symmetric.rows(), symmetric.cols()),
            "Skew-symmetric and symmetric parts must have same dimensions"
        );

        Self {
            skew_symmetric,
            symmetric,
        }
    }

    /// Get the matrix size
    pub fn size(&self) -> usize {
        self.skew_symmetric.rows()
    }

    /// Convert to a complex matrix (for visualization)
    ///
    /// Returns A + iB as a matrix over ℂ
    pub fn to_complex_matrix(&self) -> Matrix<Complex<R>>
    where
        R: From<i64> + std::ops::Add<Output = R> + std::ops::Sub<Output = R> + std::ops::Mul<Output = R>,
    {
        let n = self.size();
        let mut result = Matrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let real_part = self.skew_symmetric.get(i, j).unwrap().clone();
                let imag_part = self.symmetric.get(i, j).unwrap().clone();
                let _ = result.set(i, j, Complex::new(real_part, imag_part));
            }
        }

        result
    }
}

impl<R: Ring + Clone + Display> Display for CompactRealFormElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CompactElement(skew={}, sym={})",
               self.skew_symmetric, self.symmetric)
    }
}

/// Compact Real Form of a Classical Lie Algebra
///
/// This represents the compact real form U = K ⊕ iS where:
/// - K consists of skew-symmetric matrices
/// - S consists of symmetric matrices
///
/// Elements are skew-hermitian matrices with a negative-definite Killing form.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring (typically ℚ or ℝ)
///
/// # Examples
///
/// ```
/// use rustmath_liealgebras::compact_real_form::MatrixCompactRealForm;
/// use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
/// use rustmath_rationals::Rational;
///
/// // Create compact real form of sl(3)
/// let ct = CartanType::new(CartanLetter::A, 2).unwrap();
/// let compact = MatrixCompactRealForm::<Rational>::new(ct);
/// assert_eq!(compact.rank(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct MatrixCompactRealForm<R: Ring> {
    /// Cartan type of the underlying Lie algebra
    cartan_type: CartanType,
    /// Dimension of matrices (n for n×n matrices)
    matrix_size: usize,
    /// Dimension of the compact real form as a real Lie algebra
    dimension: usize,
    /// Coefficient ring marker
    _phantom: PhantomData<R>,
}

impl<R: Ring + Clone + From<i64>> MatrixCompactRealForm<R> {
    /// Create a new compact real form
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The Cartan type determining the Lie algebra structure
    pub fn new(cartan_type: CartanType) -> Self {
        use crate::cartan_type::CartanLetter;

        let (matrix_size, dimension) = match cartan_type.letter {
            CartanLetter::A => {
                let n = cartan_type.rank + 1;
                // su(n): dimension = n² - 1
                (n, n * n - 1)
            }
            CartanLetter::B => {
                let n = 2 * cartan_type.rank + 1;
                // so(2n+1): dimension = n(2n+1)
                (n, cartan_type.rank * (2 * cartan_type.rank + 1))
            }
            CartanLetter::C => {
                let n = 2 * cartan_type.rank;
                // sp(n): dimension = n(2n+1)
                (n, cartan_type.rank * (2 * cartan_type.rank + 1))
            }
            CartanLetter::D => {
                let n = 2 * cartan_type.rank;
                // so(2n): dimension = n(2n-1)
                (n, cartan_type.rank * (2 * cartan_type.rank - 1))
            }
            _ => {
                // For exceptional types, use default matrix representation
                let n = cartan_type.rank + 2;
                (n, n * n)
            }
        };

        Self {
            cartan_type,
            matrix_size,
            dimension,
            _phantom: PhantomData,
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &CartanType {
        &self.cartan_type
    }

    /// Get the rank (dimension of Cartan subalgebra)
    pub fn rank(&self) -> usize {
        self.cartan_type.rank
    }

    /// Get the dimension of the Lie algebra
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the size of matrices
    pub fn matrix_size(&self) -> usize {
        self.matrix_size
    }

    /// Compute the Lie bracket [X, Y] for compact real form elements
    ///
    /// For X = A + iB and Y = C + iD:
    /// [X, Y] = (AC - CA - BD + DB) + i(AD - DA + BC - CB)
    ///
    /// # Arguments
    ///
    /// * `x` - First element
    /// * `y` - Second element
    pub fn bracket(
        &self,
        x: &CompactRealFormElement<R>,
        y: &CompactRealFormElement<R>,
    ) -> CompactRealFormElement<R>
    where
        R: std::ops::Sub<Output = R> + std::ops::Add<Output = R> + std::ops::Mul<Output = R>,
    {
        let n = self.matrix_size;

        // X = A + iB, Y = C + iD
        let a = &x.skew_symmetric;
        let b = &x.symmetric;
        let c = &y.skew_symmetric;
        let d = &y.symmetric;

        // Real part: AC - CA - BD + DB
        let ac = Self::matrix_multiply(a, c, n);
        let ca = Self::matrix_multiply(c, a, n);
        let bd = Self::matrix_multiply(b, d, n);
        let db = Self::matrix_multiply(d, b, n);

        let real_part = Self::matrix_sub(&Self::matrix_sub(&ac, &ca), &Self::matrix_sub(&bd, &db));

        // Imaginary part: AD - DA + BC - CB
        let ad = Self::matrix_multiply(a, d, n);
        let da = Self::matrix_multiply(d, a, n);
        let bc = Self::matrix_multiply(b, c, n);
        let cb = Self::matrix_multiply(c, b, n);

        let imag_part = Self::matrix_add(&Self::matrix_sub(&ad, &da), &Self::matrix_sub(&bc, &cb));

        CompactRealFormElement::new(real_part, imag_part)
    }

    /// Multiply two matrices (simplified implementation)
    fn matrix_multiply(a: &Matrix<R>, b: &Matrix<R>, n: usize) -> Matrix<R>
    where
        R: std::ops::Mul<Output = R> + std::ops::Add<Output = R>,
    {
        let mut result = Matrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let mut sum = R::from(0);
                for k in 0..n {
                    let a_ik = a.get(i, k).unwrap().clone();
                    let b_kj = b.get(k, j).unwrap().clone();
                    sum = sum + (a_ik * b_kj);
                }
                let _ = result.set(i, j, sum);
            }
        }

        result
    }

    /// Subtract two matrices
    fn matrix_sub(a: &Matrix<R>, b: &Matrix<R>) -> Matrix<R>
    where
        R: std::ops::Sub<Output = R>,
    {
        let n = a.rows();
        let mut result = Matrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let a_ij = a.get(i, j).unwrap().clone();
                let b_ij = b.get(i, j).unwrap().clone();
                let _ = result.set(i, j, a_ij - b_ij);
            }
        }

        result
    }

    /// Add two matrices
    fn matrix_add(a: &Matrix<R>, b: &Matrix<R>) -> Matrix<R>
    where
        R: std::ops::Add<Output = R>,
    {
        let n = a.rows();
        let mut result = Matrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let a_ij = a.get(i, j).unwrap().clone();
                let b_ij = b.get(i, j).unwrap().clone();
                let _ = result.set(i, j, a_ij + b_ij);
            }
        }

        result
    }

    /// Generate a simple basis element for testing
    ///
    /// Creates skew-symmetric and symmetric matrices with entries at specified positions
    pub fn basis_element(&self, skew_i: usize, skew_j: usize, sym_i: usize, sym_j: usize) -> CompactRealFormElement<R> {
        let n = self.matrix_size;
        let mut skew = Matrix::zeros(n, n);
        let mut sym = Matrix::zeros(n, n);

        // Skew-symmetric part: M[i,j] = 1, M[j,i] = -1
        if skew_i < n && skew_j < n && skew_i != skew_j {
            let _ = skew.set(skew_i, skew_j, R::from(1));
            let _ = skew.set(skew_j, skew_i, R::from(-1));
        }

        // Symmetric part: M[i,j] = M[j,i] = 1
        if sym_i < n && sym_j < n {
            let _ = sym.set(sym_i, sym_j, R::from(1));
            let _ = sym.set(sym_j, sym_i, R::from(1));
        }

        CompactRealFormElement::new(skew, sym)
    }
}

impl<R: Ring + Clone + From<i64>> Display for MatrixCompactRealForm<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Compact real form of Lie algebra of type {} ({}×{} matrices)",
               self.cartan_type, self.matrix_size, self.matrix_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartan_type::{CartanLetter, CartanType};

    #[test]
    fn test_compact_real_form_type_a() {
        // Compact form of sl(3) is su(3)
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let compact = MatrixCompactRealForm::<Rational>::new(ct);

        assert_eq!(compact.rank(), 2);
        assert_eq!(compact.matrix_size(), 3);
        assert_eq!(compact.dimension(), 8); // su(3) has dimension 8
    }

    #[test]
    fn test_compact_real_form_type_b() {
        // Compact form of so(5)
        let ct = CartanType::new(CartanLetter::B, 2).unwrap();
        let compact = MatrixCompactRealForm::<Rational>::new(ct);

        assert_eq!(compact.rank(), 2);
        assert_eq!(compact.matrix_size(), 5);
        assert_eq!(compact.dimension(), 10); // so(5) has dimension 10
    }

    #[test]
    fn test_compact_real_form_type_c() {
        // Compact form of sp(4)
        let ct = CartanType::new(CartanLetter::C, 2).unwrap();
        let compact = MatrixCompactRealForm::<Rational>::new(ct);

        assert_eq!(compact.rank(), 2);
        assert_eq!(compact.matrix_size(), 4);
        assert_eq!(compact.dimension(), 10); // sp(4) has dimension 10
    }

    #[test]
    fn test_compact_real_form_type_d() {
        // Compact form of so(6)
        let ct = CartanType::new(CartanLetter::D, 3).unwrap();
        let compact = MatrixCompactRealForm::<Rational>::new(ct);

        assert_eq!(compact.rank(), 3);
        assert_eq!(compact.matrix_size(), 6);
        assert_eq!(compact.dimension(), 15); // so(6) has dimension 15
    }

    #[test]
    fn test_element_creation() {
        let ct = CartanType::new(CartanLetter::A, 1).unwrap();
        let compact = MatrixCompactRealForm::<Rational>::new(ct);

        let elem = compact.basis_element(0, 1, 0, 0);
        assert_eq!(elem.size(), 2);
    }

    #[test]
    fn test_bracket_operation() {
        let ct = CartanType::new(CartanLetter::A, 1).unwrap();
        let compact = MatrixCompactRealForm::<Rational>::new(ct);

        let x = compact.basis_element(0, 1, 0, 0);
        let y = compact.basis_element(0, 1, 1, 1);

        let bracket = compact.bracket(&x, &y);
        assert_eq!(bracket.size(), 2);
    }

    #[test]
    fn test_display() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let compact = MatrixCompactRealForm::<Rational>::new(ct);

        let display = format!("{}", compact);
        assert!(display.contains("Compact real form"));
        assert!(display.contains("A_2"));
    }
}
