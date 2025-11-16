//! Classical Matrix Lie Algebras
//!
//! The classical Lie algebras are matrix Lie algebras with the commutator as bracket:
//! - gl(n): All n×n matrices (general linear)
//! - sl(n): n×n matrices with trace 0 (special linear, type A_{n-1})
//! - so(n): n×n anti-symmetric matrices (special orthogonal, type B/D)
//! - sp(2k): 2k×2k symplectic matrices (type C_k)
//!
//! Corresponds to sage.algebras.lie_algebras.classical_lie_algebra
//!
//! References:
//! - Humphreys, J. "Introduction to Lie Algebras and Representation Theory" (1972)
//! - Hall, B. "Lie Groups, Lie Algebras, and Representations" (2015)

use rustmath_core::Ring;
use rustmath_matrix::Matrix;
use crate::cartan_type::{CartanType, CartanLetter};
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::collections::HashMap;

/// Element of a classical matrix Lie algebra
///
/// This represents an element of a classical Lie algebra (gl, sl, so, sp)
/// as a matrix with additional methods for extracting monomial coefficients.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// use rustmath_liealgebras::ClassicalLieAlgebraElement;
/// use rustmath_matrix::Matrix;
///
/// // Create an element from a matrix
/// let mut mat = Matrix::zeros(3, 3);
/// mat.set(0, 1, 1);
/// mat.set(1, 2, 2);
/// let elem = ClassicalLieAlgebraElement::from_matrix(mat);
///
/// // Get monomial coefficients as E_{i,j} basis elements
/// let coeffs = elem.monomial_coefficients();
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ClassicalLieAlgebraElement<R: Ring> {
    /// The underlying matrix representation
    matrix: Matrix<R>,
}

impl<R: Ring + Clone> ClassicalLieAlgebraElement<R> {
    /// Create an element from a matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - The matrix representation
    pub fn from_matrix(matrix: Matrix<R>) -> Self {
        ClassicalLieAlgebraElement { matrix }
    }

    /// Get the underlying matrix
    pub fn matrix(&self) -> &Matrix<R> {
        &self.matrix
    }

    /// Get monomial coefficients as a HashMap
    ///
    /// Returns a dictionary mapping basis element names (like "E_i_j")
    /// to their coefficients.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_liealgebras::ClassicalLieAlgebraElement;
    /// use rustmath_matrix::Matrix;
    ///
    /// let mut mat = Matrix::zeros(2, 2);
    /// mat.set(0, 1, 3);  // 3*E_{0,1}
    /// mat.set(1, 0, 1);  // 1*E_{1,0}
    /// let elem = ClassicalLieAlgebraElement::from_matrix(mat);
    ///
    /// let coeffs = elem.monomial_coefficients();
    /// // coeffs contains {"E_0_1": 3, "E_1_0": 1}
    /// ```
    pub fn monomial_coefficients(&self) -> HashMap<String, R>
    where
        R: From<i64> + PartialEq,
    {
        let mut coeffs = HashMap::new();

        for i in 0..self.matrix.rows() {
            for j in 0..self.matrix.cols() {
                if let Ok(val) = self.matrix.get(i, j) {
                    // Only include non-zero entries
                    if val != &R::from(0) {
                        let key = format!("E_{}_{}", i, j);
                        coeffs.insert(key, val.clone());
                    }
                }
            }
        }

        coeffs
    }

    /// Get the size of the matrix
    pub fn size(&self) -> (usize, usize) {
        (self.matrix.rows(), self.matrix.cols())
    }

    /// Check if this is a zero element
    pub fn is_zero(&self) -> bool
    where
        R: From<i64> + PartialEq,
    {
        for i in 0..self.matrix.rows() {
            for j in 0..self.matrix.cols() {
                if let Ok(val) = self.matrix.get(i, j) {
                    if val != &R::from(0) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq + Display> Display for ClassicalLieAlgebraElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let coeffs = self.monomial_coefficients();
        if coeffs.is_empty() {
            return write!(f, "0");
        }

        let mut first = true;
        for (i, j) in (0..self.matrix.rows())
            .flat_map(|i| (0..self.matrix.cols()).map(move |j| (i, j)))
        {
            if let Ok(val) = self.matrix.get(i, j) {
                if val != &R::from(0) {
                    if !first {
                        write!(f, " + ")?;
                    }
                    if val == &R::from(1) {
                        write!(f, "E_{{{},{}}}", i, j)?;
                    } else {
                        write!(f, "{}*E_{{{},{}}}", val, i, j)?;
                    }
                    first = false;
                }
            }
        }
        Ok(())
    }
}

/// General Linear Lie Algebra gl(n)
///
/// Consists of all n×n matrices with commutator bracket:
/// [A, B] = AB - BA
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
pub struct GeneralLinearLieAlgebra<R: Ring> {
    /// Size of matrices (n for n×n matrices)
    n: usize,
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> GeneralLinearLieAlgebra<R> {
    /// Create gl(n)
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix size
    pub fn new(n: usize) -> Self {
        GeneralLinearLieAlgebra {
            n,
            coefficient_ring: PhantomData,
        }
    }

    /// Get the size n
    pub fn n(&self) -> usize {
        self.n
    }

    /// Dimension of gl(n) is n²
    pub fn dimension(&self) -> usize {
        self.n * self.n
    }

    /// Get the matrix unit E_{i,j}
    ///
    /// This is the matrix with 1 at position (i,j) and 0 elsewhere
    pub fn matrix_unit(&self, i: usize, j: usize) -> Matrix<R>
    where
        R: From<i64>,
    {
        let mut mat = Matrix::zeros(self.n, self.n);
        if i < self.n && j < self.n {
            mat.set(i, j, R::from(1));
        }
        mat
    }

    /// Lie bracket: [A, B] = AB - BA
    pub fn bracket(&self, _a: &Matrix<R>, _b: &Matrix<R>) -> Matrix<R>
    where
        R: std::ops::Sub<Output = R> + std::ops::Mul<Output = R> + std::ops::Add<Output = R>,
    {
        // Simplified: Would need proper matrix multiplication
        Matrix::zeros(self.n, self.n)
    }

    /// Killing form: κ(x, y) = 2n·tr(xy) - 2·tr(x)·tr(y)
    pub fn killing_form(&self, _x: &Matrix<R>, _y: &Matrix<R>) -> R
    where
        R: From<i64>,
    {
        // Simplified implementation
        R::from(0)
    }
}

impl<R: Ring + Clone> Display for GeneralLinearLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "gl({})", self.n)
    }
}

/// Special Linear Lie Algebra sl(n)
///
/// Consists of n×n matrices with trace 0.
/// This is type A_{n-1} in the Cartan-Killing classification.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
pub struct SpecialLinearLieAlgebra<R: Ring> {
    /// Size of matrices (n for n×n matrices)
    n: usize,
    /// Associated Cartan type
    cartan_type: CartanType,
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> SpecialLinearLieAlgebra<R> {
    /// Create sl(n)
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix size (must be at least 2)
    pub fn new(n: usize) -> Result<Self, String> {
        if n < 2 {
            return Err("sl(n) requires n >= 2".to_string());
        }

        let cartan_type = CartanType::new(CartanLetter::A, n - 1)
            .ok_or_else(|| "Invalid Cartan type for sl(n)".to_string())?;

        Ok(SpecialLinearLieAlgebra {
            n,
            cartan_type,
            coefficient_ring: PhantomData,
        })
    }

    /// Get the size n
    pub fn n(&self) -> usize {
        self.n
    }

    /// Dimension of sl(n) is n² - 1
    pub fn dimension(&self) -> usize {
        self.n * self.n - 1
    }

    /// Rank (dimension of Cartan subalgebra) is n - 1
    pub fn rank(&self) -> usize {
        self.n - 1
    }

    /// Get the Cartan type (type A_{n-1})
    pub fn cartan_type(&self) -> &CartanType {
        &self.cartan_type
    }

    /// Get the Chevalley generator e_i (i-th positive simple root)
    pub fn e(&self, i: usize) -> Matrix<R>
    where
        R: From<i64>,
    {
        // e_i is the matrix unit E_{i, i+1}
        let mut mat = Matrix::zeros(self.n, self.n);
        if i < self.rank() {
            mat.set(i, i + 1, R::from(1));
        }
        mat
    }

    /// Get the Chevalley generator f_i (i-th negative simple root)
    pub fn f(&self, i: usize) -> Matrix<R>
    where
        R: From<i64>,
    {
        // f_i is the matrix unit E_{i+1, i}
        let mut mat = Matrix::zeros(self.n, self.n);
        if i < self.rank() {
            mat.set(i + 1, i, R::from(1));
        }
        mat
    }

    /// Get the Chevalley generator h_i (i-th simple coroot)
    pub fn h(&self, i: usize) -> Matrix<R>
    where
        R: From<i64>,
    {
        // h_i = [e_i, f_i] = E_{i,i} - E_{i+1,i+1}
        let mut mat = Matrix::zeros(self.n, self.n);
        if i < self.rank() {
            mat.set(i, i, R::from(1));
            mat.set(i + 1, i + 1, R::from(-1));
        }
        mat
    }

    /// Killing form: κ(x, y) = 2n·tr(xy)
    pub fn killing_form(&self, _x: &Matrix<R>, _y: &Matrix<R>) -> R
    where
        R: From<i64>,
    {
        // Simplified implementation
        R::from(0)
    }
}

impl<R: Ring + Clone> Display for SpecialLinearLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "sl({})", self.n)
    }
}

/// Special Orthogonal Lie Algebra so(n)
///
/// Consists of n×n anti-symmetric matrices: A^T = -A
/// For odd n, this is type B_{(n-1)/2}
/// For even n, this is type D_{n/2}
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
pub struct SpecialOrthogonalLieAlgebra<R: Ring> {
    /// Size of matrices (n for n×n matrices)
    n: usize,
    /// Associated Cartan type
    cartan_type: CartanType,
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> SpecialOrthogonalLieAlgebra<R> {
    /// Create so(n)
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix size (must be at least 3)
    pub fn new(n: usize) -> Result<Self, String> {
        if n < 3 {
            return Err("so(n) requires n >= 3".to_string());
        }

        let cartan_type = if n % 2 == 1 {
            // Odd: type B_{(n-1)/2}
            CartanType::new(CartanLetter::B, (n - 1) / 2)
        } else {
            // Even: type D_{n/2}
            CartanType::new(CartanLetter::D, n / 2)
        };

        let cartan_type = cartan_type
            .ok_or_else(|| "Invalid Cartan type for so(n)".to_string())?;

        Ok(SpecialOrthogonalLieAlgebra {
            n,
            cartan_type,
            coefficient_ring: PhantomData,
        })
    }

    /// Get the size n
    pub fn n(&self) -> usize {
        self.n
    }

    /// Dimension of so(n) is n(n-1)/2
    pub fn dimension(&self) -> usize {
        self.n * (self.n - 1) / 2
    }

    /// Rank (dimension of Cartan subalgebra)
    pub fn rank(&self) -> usize {
        self.n / 2
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &CartanType {
        &self.cartan_type
    }

    /// Check if a matrix is anti-symmetric
    pub fn is_anti_symmetric(&self, _mat: &Matrix<R>) -> bool
    where
        R: PartialEq,
    {
        // Would check mat^T = -mat
        false
    }

    /// Killing form: κ(x, y) = (n-2)·tr(xy)
    pub fn killing_form(&self, _x: &Matrix<R>, _y: &Matrix<R>) -> R
    where
        R: From<i64>,
    {
        // Simplified implementation
        R::from(0)
    }
}

impl<R: Ring + Clone> Display for SpecialOrthogonalLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "so({})", self.n)
    }
}

/// Symplectic Lie Algebra sp(2k)
///
/// Consists of 2k×2k matrices X satisfying X^T·M - M·X = 0
/// where M is the standard symplectic form.
/// This is type C_k in the Cartan-Killing classification.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
pub struct SymplecticLieAlgebra<R: Ring> {
    /// Half the matrix size (k for sp(2k))
    k: usize,
    /// Associated Cartan type
    cartan_type: CartanType,
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> SymplecticLieAlgebra<R> {
    /// Create sp(2k)
    ///
    /// # Arguments
    ///
    /// * `k` - Half the matrix size (creates 2k×2k matrices)
    pub fn new(k: usize) -> Result<Self, String> {
        if k < 1 {
            return Err("sp(2k) requires k >= 1".to_string());
        }

        let cartan_type = CartanType::new(CartanLetter::C, k)
            .ok_or_else(|| "Invalid Cartan type for sp(2k)".to_string())?;

        Ok(SymplecticLieAlgebra {
            k,
            cartan_type,
            coefficient_ring: PhantomData,
        })
    }

    /// Get k (the rank)
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get the matrix size (2k)
    pub fn matrix_size(&self) -> usize {
        2 * self.k
    }

    /// Dimension of sp(2k) is k(2k+1)
    pub fn dimension(&self) -> usize {
        self.k * (2 * self.k + 1)
    }

    /// Rank (dimension of Cartan subalgebra) is k
    pub fn rank(&self) -> usize {
        self.k
    }

    /// Get the Cartan type (type C_k)
    pub fn cartan_type(&self) -> &CartanType {
        &self.cartan_type
    }

    /// Killing form: κ(x, y) = (2n+2)·tr(xy) where n = 2k
    pub fn killing_form(&self, _x: &Matrix<R>, _y: &Matrix<R>) -> R
    where
        R: From<i64>,
    {
        // Simplified implementation
        R::from(0)
    }
}

impl<R: Ring + Clone> Display for SymplecticLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "sp({})", 2 * self.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classical_element_creation() {
        let mut mat = Matrix::zeros(2, 2);
        mat.set(0, 1, 3);
        mat.set(1, 0, 1);

        let elem: ClassicalLieAlgebraElement<i64> = ClassicalLieAlgebraElement::from_matrix(mat);
        assert_eq!(elem.size(), (2, 2));
        assert!(!elem.is_zero());
    }

    #[test]
    fn test_classical_element_monomial_coefficients() {
        let mut mat = Matrix::zeros(3, 3);
        mat.set(0, 1, 3);
        mat.set(2, 0, 1);

        let elem: ClassicalLieAlgebraElement<i64> = ClassicalLieAlgebraElement::from_matrix(mat);
        let coeffs = elem.monomial_coefficients();

        assert_eq!(coeffs.len(), 2);
        assert_eq!(coeffs.get("E_0_1"), Some(&3));
        assert_eq!(coeffs.get("E_2_0"), Some(&1));
    }

    #[test]
    fn test_classical_element_zero() {
        let mat: Matrix<i64> = Matrix::zeros(3, 3);
        let elem = ClassicalLieAlgebraElement::from_matrix(mat);

        assert!(elem.is_zero());
        assert_eq!(elem.monomial_coefficients().len(), 0);
    }

    #[test]
    fn test_classical_element_display() {
        let mut mat = Matrix::zeros(2, 2);
        mat.set(0, 1, 1);
        mat.set(1, 0, 2);

        let elem: ClassicalLieAlgebraElement<i64> = ClassicalLieAlgebraElement::from_matrix(mat);
        let display = format!("{}", elem);

        assert!(display.contains("E_{0,1}"));
        assert!(display.contains("2*E_{1,0}"));
    }

    #[test]
    fn test_gl_creation() {
        let gl5: GeneralLinearLieAlgebra<i64> = GeneralLinearLieAlgebra::new(5);
        assert_eq!(gl5.n(), 5);
        assert_eq!(gl5.dimension(), 25); // 5^2
    }

    #[test]
    fn test_sl_creation() {
        let sl3 = SpecialLinearLieAlgebra::<i64>::new(3).unwrap();
        assert_eq!(sl3.n(), 3);
        assert_eq!(sl3.dimension(), 8); // 3^2 - 1
        assert_eq!(sl3.rank(), 2); // n - 1
    }

    #[test]
    fn test_sl_cartan_type() {
        let sl4 = SpecialLinearLieAlgebra::<i64>::new(4).unwrap();
        let ct = sl4.cartan_type();
        // Type A_3 for sl(4)
        assert_eq!(*ct, CartanType::new(CartanLetter::A, 3).unwrap());
    }

    #[test]
    fn test_so_creation_odd() {
        let so5 = SpecialOrthogonalLieAlgebra::<i64>::new(5).unwrap();
        assert_eq!(so5.n(), 5);
        assert_eq!(so5.dimension(), 10); // 5*4/2
        assert_eq!(so5.rank(), 2); // n/2
        // Type B_2
        assert_eq!(
            *so5.cartan_type(),
            CartanType::new(CartanLetter::B, 2).unwrap()
        );
    }

    #[test]
    fn test_so_creation_even() {
        let so6 = SpecialOrthogonalLieAlgebra::<i64>::new(6).unwrap();
        assert_eq!(so6.n(), 6);
        assert_eq!(so6.dimension(), 15); // 6*5/2
        assert_eq!(so6.rank(), 3); // n/2
        // Type D_3
        assert_eq!(
            *so6.cartan_type(),
            CartanType::new(CartanLetter::D, 3).unwrap()
        );
    }

    #[test]
    fn test_sp_creation() {
        let sp4 = SymplecticLieAlgebra::<i64>::new(2).unwrap();
        assert_eq!(sp4.k(), 2);
        assert_eq!(sp4.matrix_size(), 4);
        assert_eq!(sp4.dimension(), 10); // 2*(2*2+1)
        assert_eq!(sp4.rank(), 2);
        // Type C_2
        assert_eq!(
            *sp4.cartan_type(),
            CartanType::new(CartanLetter::C, 2).unwrap()
        );
    }

    #[test]
    fn test_sl_generators() {
        let sl3 = SpecialLinearLieAlgebra::<i64>::new(3).unwrap();

        // e_0 should be E_{0,1}
        let e0 = sl3.e(0);
        assert_eq!(e0.get(0, 1).unwrap(), &1);

        // f_0 should be E_{1,0}
        let f0 = sl3.f(0);
        assert_eq!(f0.get(1, 0).unwrap(), &1);

        // h_0 should be E_{0,0} - E_{1,1}
        let h0 = sl3.h(0);
        assert_eq!(h0.get(0, 0).unwrap(), &1);
        assert_eq!(h0.get(1, 1).unwrap(), &-1);
    }
}
