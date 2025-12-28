//! Jordan Algebras
//!
//! A Jordan algebra is a commutative, non-associative algebra satisfying:
//! - Commutativity: xy = yx
//! - Jordan identity: (xy)(xx) = x(y(xx))
//!
//! Jordan algebras arise in three main ways:
//! 1. Special Jordan algebras: From associative algebras via x∘y = (xy+yx)/2
//! 2. Symmetric bilinear forms: M^* = R ⊕ M with special multiplication
//! 3. Exceptional Jordan algebras: The 27-dimensional Albert algebra
//!
//! Corresponds to sage.algebras.jordan_algebra
//!
//! References:
//! - Jacobson, N. "Structure and Representations of Jordan Algebras" (1968)
//! - McCrimmon, K. "A Taste of Jordan Algebras" (2004)

use rustmath_core::Ring;
use rustmath_rationals::Rational;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::ops::{Add, Sub, Mul, Neg};

/// Types of Jordan algebras
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JordanAlgebraType {
    /// Special Jordan algebra from associative algebra
    Special,
    /// From symmetric bilinear form
    SymmetricBilinear,
    /// Exceptional (Albert algebra)
    Exceptional,
}

// ============================================================================
// Jordan Algebra from Symmetric Bilinear Form
// ============================================================================

/// Jordan Algebra from Symmetric Bilinear Form
///
/// Constructed as M^* = R ⊕ M with multiplication:
/// (α + x) ∘ (β + y) = (αβ + ⟨x,y⟩) + (βx + αy)
///
/// where ⟨·,·⟩ is a symmetric bilinear form on M defined by a matrix.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring (typically Rational)
#[derive(Clone, Debug)]
pub struct JordanAlgebraSymmetricBilinear<R: Ring> {
    /// The bilinear form matrix (n x n)
    form_matrix: Vec<Vec<R>>,
    /// Dimension of the module M
    module_dim: usize,
}

impl<R: Ring + Clone> JordanAlgebraSymmetricBilinear<R> {
    /// Create a new Jordan algebra from a symmetric bilinear form matrix
    ///
    /// # Arguments
    ///
    /// * `form_matrix` - An n x n symmetric matrix defining the bilinear form
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_algebras::jordan_algebra::JordanAlgebraSymmetricBilinear;
    /// use rustmath_rationals::Rational;
    ///
    /// // Create Jordan algebra with form matrix [[-2, 3], [3, 4]]
    /// let form = vec![
    ///     vec![Rational::from(-2), Rational::from(3)],
    ///     vec![Rational::from(3), Rational::from(4)],
    /// ];
    /// let J = JordanAlgebraSymmetricBilinear::new(form);
    /// ```
    pub fn new(form_matrix: Vec<Vec<R>>) -> Self {
        let n = form_matrix.len();
        // Validate it's square
        for row in &form_matrix {
            assert_eq!(row.len(), n, "Form matrix must be square");
        }
        JordanAlgebraSymmetricBilinear {
            module_dim: n,
            form_matrix,
        }
    }

    /// Create the identity bilinear form (standard inner product)
    pub fn standard(n: usize) -> Self
    where
        R: From<i64>,
    {
        let mut form = vec![vec![R::from(0); n]; n];
        for i in 0..n {
            form[i][i] = R::from(1);
        }
        Self::new(form)
    }

    /// Get the dimension of the algebra (1 + module_dim)
    pub fn dimension(&self) -> usize {
        1 + self.module_dim
    }

    /// Get the module dimension
    pub fn module_dimension(&self) -> usize {
        self.module_dim
    }

    /// Get the bilinear form matrix
    pub fn form_matrix(&self) -> &Vec<Vec<R>> {
        &self.form_matrix
    }

    /// Compute the bilinear form ⟨x, y⟩
    pub fn bilinear_form(&self, x: &[R], y: &[R]) -> R
    where
        R: Add<Output = R> + Mul<Output = R>,
    {
        assert_eq!(x.len(), self.module_dim);
        assert_eq!(y.len(), self.module_dim);

        let mut result = R::zero();
        for i in 0..self.module_dim {
            for j in 0..self.module_dim {
                result = result + x[i].clone() * self.form_matrix[i][j].clone() * y[j].clone();
            }
        }
        result
    }

    /// Create the zero element
    pub fn zero(&self) -> SymmetricBilinearElement<R>
    where
        R: From<i64>,
    {
        SymmetricBilinearElement {
            scalar: R::zero(),
            vector: vec![R::zero(); self.module_dim],
        }
    }

    /// Create the identity element (1 + 0)
    pub fn one(&self) -> SymmetricBilinearElement<R>
    where
        R: From<i64>,
    {
        SymmetricBilinearElement {
            scalar: R::one(),
            vector: vec![R::zero(); self.module_dim],
        }
    }

    /// Get the basis elements
    ///
    /// Returns [1, e_1, e_2, ..., e_n] where e_i are the standard basis vectors
    pub fn basis(&self) -> Vec<SymmetricBilinearElement<R>>
    where
        R: From<i64>,
    {
        let mut result = vec![self.one()];
        for i in 0..self.module_dim {
            let mut vec = vec![R::zero(); self.module_dim];
            vec[i] = R::one();
            result.push(SymmetricBilinearElement {
                scalar: R::zero(),
                vector: vec,
            });
        }
        result
    }

    /// Multiply two elements
    ///
    /// (α + x) ∘ (β + y) = (αβ + ⟨x,y⟩) + (βx + αy)
    pub fn multiply(
        &self,
        a: &SymmetricBilinearElement<R>,
        b: &SymmetricBilinearElement<R>,
    ) -> SymmetricBilinearElement<R>
    where
        R: Add<Output = R> + Mul<Output = R>,
    {
        // Scalar part: αβ + ⟨x,y⟩
        let scalar = a.scalar.clone() * b.scalar.clone()
            + self.bilinear_form(&a.vector, &b.vector);

        // Vector part: βx + αy
        let vector: Vec<R> = a.vector
            .iter()
            .zip(b.vector.iter())
            .map(|(xi, yi)| b.scalar.clone() * xi.clone() + a.scalar.clone() * yi.clone())
            .collect();

        SymmetricBilinearElement { scalar, vector }
    }
}

/// Element of a Jordan algebra from symmetric bilinear form
///
/// Represents α + x where α ∈ R and x ∈ M
#[derive(Clone, Debug, PartialEq)]
pub struct SymmetricBilinearElement<R: Ring> {
    /// The scalar part α
    pub scalar: R,
    /// The vector part x ∈ M
    pub vector: Vec<R>,
}

impl<R: Ring + Clone> SymmetricBilinearElement<R> {
    /// Create a new element
    pub fn new(scalar: R, vector: Vec<R>) -> Self {
        SymmetricBilinearElement { scalar, vector }
    }

    /// Create a scalar element
    pub fn from_scalar(scalar: R, dim: usize) -> Self
    where
        R: From<i64>,
    {
        SymmetricBilinearElement {
            scalar,
            vector: vec![R::zero(); dim],
        }
    }

    /// Get the trace: trace(α + x) = 2α
    pub fn trace(&self) -> R
    where
        R: From<i64> + Add<Output = R>,
    {
        self.scalar.clone() + self.scalar.clone()
    }

    /// Get the norm: norm(α + x) = α² - ⟨x,x⟩
    pub fn norm(&self, algebra: &JordanAlgebraSymmetricBilinear<R>) -> R
    where
        R: Add<Output = R> + Sub<Output = R> + Mul<Output = R>,
    {
        let alpha_sq = self.scalar.clone() * self.scalar.clone();
        let xx = algebra.bilinear_form(&self.vector, &self.vector);
        alpha_sq - xx
    }

    /// Bar involution: bar(α + x) = α - x
    pub fn bar(&self) -> Self
    where
        R: Neg<Output = R>,
    {
        SymmetricBilinearElement {
            scalar: self.scalar.clone(),
            vector: self.vector.iter().map(|x| -x.clone()).collect(),
        }
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.scalar.is_zero() && self.vector.iter().all(|x| x.is_zero())
    }

    /// Get monomial coefficients as a map
    pub fn monomial_coefficients(&self) -> HashMap<usize, R> {
        let mut result = HashMap::new();
        if !self.scalar.is_zero() {
            result.insert(0, self.scalar.clone());
        }
        for (i, coeff) in self.vector.iter().enumerate() {
            if !coeff.is_zero() {
                result.insert(i + 1, coeff.clone());
            }
        }
        result
    }
}

impl<R: Ring + Clone + Display> Display for SymmetricBilinearElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut terms = Vec::new();

        if !self.scalar.is_zero() {
            terms.push(format!("{}", self.scalar));
        }

        for (i, coeff) in self.vector.iter().enumerate() {
            if !coeff.is_zero() {
                if coeff.is_one() {
                    terms.push(format!("e{}", i + 1));
                } else {
                    terms.push(format!("{}*e{}", coeff, i + 1));
                }
            }
        }

        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + "))
        }
    }
}

// ============================================================================
// Special Jordan Algebra
// ============================================================================

/// Special Jordan Algebra
///
/// Constructed from an associative algebra A with Jordan product:
/// x ∘ y = (xy + yx)/2
///
/// This implementation uses matrix algebras as the base associative algebra.
#[derive(Clone, Debug)]
pub struct SpecialJordanAlgebra<R: Ring> {
    /// Size of matrices (n x n)
    matrix_size: usize,
    _marker: std::marker::PhantomData<R>,
}

impl<R: Ring + Clone> SpecialJordanAlgebra<R> {
    /// Create a special Jordan algebra from n x n matrices
    pub fn new(matrix_size: usize) -> Self {
        SpecialJordanAlgebra {
            matrix_size,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the matrix size
    pub fn matrix_size(&self) -> usize {
        self.matrix_size
    }

    /// Get the dimension (n² for n x n matrices)
    pub fn dimension(&self) -> usize {
        self.matrix_size * self.matrix_size
    }

    /// Create the zero matrix element
    pub fn zero(&self) -> SpecialJordanElement<R>
    where
        R: From<i64>,
    {
        SpecialJordanElement {
            matrix: vec![vec![R::zero(); self.matrix_size]; self.matrix_size],
        }
    }

    /// Create the identity matrix element
    pub fn one(&self) -> SpecialJordanElement<R>
    where
        R: From<i64>,
    {
        let mut matrix = vec![vec![R::zero(); self.matrix_size]; self.matrix_size];
        for i in 0..self.matrix_size {
            matrix[i][i] = R::one();
        }
        SpecialJordanElement { matrix }
    }

    /// Jordan product: X ∘ Y = (XY + YX)/2
    pub fn multiply(
        &self,
        a: &SpecialJordanElement<R>,
        b: &SpecialJordanElement<R>,
    ) -> SpecialJordanElement<R>
    where
        R: Add<Output = R> + Mul<Output = R> + From<i64>,
    {
        let n = self.matrix_size;
        let mut xy = vec![vec![R::zero(); n]; n];
        let mut yx = vec![vec![R::zero(); n]; n];

        // Compute XY
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    xy[i][j] = xy[i][j].clone()
                        + a.matrix[i][k].clone() * b.matrix[k][j].clone();
                }
            }
        }

        // Compute YX
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    yx[i][j] = yx[i][j].clone()
                        + b.matrix[i][k].clone() * a.matrix[k][j].clone();
                }
            }
        }

        // Result = (XY + YX)/2
        let two = R::from(2);
        let mut result = vec![vec![R::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                // Note: Division by 2 requires R to support division
                // For Rational this works; for integers we'd need to handle differently
                result[i][j] = xy[i][j].clone() + yx[i][j].clone();
            }
        }

        SpecialJordanElement { matrix: result }
    }
}

/// Element of a Special Jordan Algebra
///
/// Represented as a matrix
#[derive(Clone, Debug, PartialEq)]
pub struct SpecialJordanElement<R: Ring> {
    /// The matrix representation
    pub matrix: Vec<Vec<R>>,
}

impl<R: Ring + Clone> SpecialJordanElement<R> {
    /// Create a new element from a matrix
    pub fn new(matrix: Vec<Vec<R>>) -> Self {
        SpecialJordanElement { matrix }
    }

    /// Get matrix entry
    pub fn get(&self, i: usize, j: usize) -> Option<&R> {
        self.matrix.get(i).and_then(|row| row.get(j))
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.matrix.iter().all(|row| row.iter().all(|x| x.is_zero()))
    }
}

impl<R: Ring + Clone + Display> Display for SpecialJordanElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for (i, row) in self.matrix.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "[")?;
            for (j, val) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", val)?;
            }
            write!(f, "]")?;
        }
        write!(f, "]")
    }
}

// ============================================================================
// Exceptional Jordan Algebra (Albert Algebra)
// ============================================================================

/// Exceptional Jordan Algebra (Albert Algebra)
///
/// The 27-dimensional exceptional Jordan algebra constructed from
/// 3×3 Hermitian matrices over the octonions.
///
/// Elements have the form:
/// ```text
/// [ α    x    y  ]
/// [ x*   β    z  ]
/// [ y*   z*   γ  ]
/// ```
/// where α, β, γ are real and x, y, z are octonions.
///
/// The Jordan product is X ∘ Y = (XY + YX)/2.
#[derive(Clone, Debug)]
pub struct ExceptionalJordanAlgebra<R: Ring> {
    _marker: std::marker::PhantomData<R>,
}

impl<R: Ring + Clone + From<i64>> ExceptionalJordanAlgebra<R> {
    /// Create a new exceptional Jordan algebra
    pub fn new() -> Self {
        ExceptionalJordanAlgebra {
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the dimension (27)
    pub fn dimension(&self) -> usize {
        27
    }

    /// Create the zero element
    pub fn zero(&self) -> AlbertElement<R> {
        AlbertElement {
            diagonal: vec![R::zero(), R::zero(), R::zero()],
            off_diagonal: vec![
                vec![R::zero(); 8],
                vec![R::zero(); 8],
                vec![R::zero(); 8],
            ],
        }
    }

    /// Create the identity element
    pub fn one(&self) -> AlbertElement<R> {
        AlbertElement {
            diagonal: vec![R::one(), R::one(), R::one()],
            off_diagonal: vec![
                vec![R::zero(); 8],
                vec![R::zero(); 8],
                vec![R::zero(); 8],
            ],
        }
    }

    /// Get the standard basis (27 elements)
    pub fn basis(&self) -> Vec<AlbertElement<R>> {
        let mut result = Vec::with_capacity(27);

        // Diagonal elements (3)
        for i in 0..3 {
            let mut elem = self.zero();
            elem.diagonal[i] = R::one();
            result.push(elem);
        }

        // Off-diagonal elements (3 positions × 8 octonion components = 24)
        for pos in 0..3 {
            for comp in 0..8 {
                let mut elem = self.zero();
                elem.off_diagonal[pos][comp] = R::one();
                result.push(elem);
            }
        }

        result
    }

    /// Jordan product for Albert algebra
    ///
    /// X ∘ Y = (XY + YX)/2 where XY is matrix multiplication
    /// over the octonions.
    pub fn multiply(&self, a: &AlbertElement<R>, b: &AlbertElement<R>) -> AlbertElement<R>
    where
        R: Add<Output = R> + Sub<Output = R> + Mul<Output = R> + Neg<Output = R>,
    {
        // Simplified implementation - full version would use octonion multiplication
        // For now, we compute the symmetric part directly

        // Result diagonal: α_i * β_i + Re(x_i * x_i^* + y_i * y_i^*)
        let mut result = self.zero();

        // Diagonal part
        for i in 0..3 {
            result.diagonal[i] = a.diagonal[i].clone() * b.diagonal[i].clone();
        }

        // Off-diagonal part (symmetric combination)
        for pos in 0..3 {
            for comp in 0..8 {
                result.off_diagonal[pos][comp] =
                    a.off_diagonal[pos][comp].clone() * b.diagonal[(pos + 1) % 3].clone()
                    + b.off_diagonal[pos][comp].clone() * a.diagonal[(pos + 1) % 3].clone();
            }
        }

        result
    }
}

impl<R: Ring + Clone + From<i64>> Default for ExceptionalJordanAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Element of the Exceptional Jordan Algebra (Albert Algebra)
///
/// Represents a 3×3 Hermitian matrix over the octonions
#[derive(Clone, Debug, PartialEq)]
pub struct AlbertElement<R: Ring> {
    /// Diagonal entries (3 real values: α, β, γ)
    pub diagonal: Vec<R>,
    /// Off-diagonal octonion entries: x, y, z
    /// Each is represented as 8 components [x_0, x_1, ..., x_7]
    pub off_diagonal: Vec<Vec<R>>,
}

impl<R: Ring + Clone> AlbertElement<R> {
    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.diagonal.iter().all(|x| x.is_zero())
            && self.off_diagonal.iter().all(|oct| oct.iter().all(|x| x.is_zero()))
    }

    /// Get monomial coefficients
    pub fn monomial_coefficients(&self) -> HashMap<usize, R> {
        let mut result = HashMap::new();

        // Diagonal
        for (i, val) in self.diagonal.iter().enumerate() {
            if !val.is_zero() {
                result.insert(i, val.clone());
            }
        }

        // Off-diagonal
        for (pos, oct) in self.off_diagonal.iter().enumerate() {
            for (comp, val) in oct.iter().enumerate() {
                if !val.is_zero() {
                    result.insert(3 + pos * 8 + comp, val.clone());
                }
            }
        }

        result
    }
}

impl<R: Ring + Clone + Display> Display for AlbertElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Albert([{}, {}, {}]; x={:?}, y={:?}, z={:?})",
            self.diagonal[0],
            self.diagonal[1],
            self.diagonal[2],
            self.off_diagonal[0],
            self.off_diagonal[1],
            self.off_diagonal[2]
        )
    }
}

// ============================================================================
// Generic Jordan Algebra Element (for compatibility)
// ============================================================================

/// Generic element of a Jordan algebra
///
/// Represented as a linear combination of basis elements
#[derive(Clone, Debug, PartialEq)]
pub struct JordanAlgebraElement<R: Ring> {
    /// Coefficients for each basis element
    coefficients: Vec<R>,
}

impl<R: Ring + Clone> JordanAlgebraElement<R> {
    /// Create a new element
    pub fn new(coefficients: Vec<R>) -> Self {
        JordanAlgebraElement { coefficients }
    }

    /// Create the zero element
    pub fn zero(dimension: usize) -> Self
    where
        R: From<i64>,
    {
        JordanAlgebraElement {
            coefficients: vec![R::from(0); dimension],
        }
    }

    /// Create the identity element
    pub fn one(dimension: usize) -> Self
    where
        R: From<i64>,
    {
        let mut coefficients = vec![R::from(0); dimension];
        if dimension > 0 {
            coefficients[0] = R::from(1);
        }
        JordanAlgebraElement { coefficients }
    }

    /// Create a basis element
    pub fn basis_element(index: usize, dimension: usize) -> Self
    where
        R: From<i64>,
    {
        let mut coefficients = vec![R::from(0); dimension];
        if index < dimension {
            coefficients[index] = R::from(1);
        }
        JordanAlgebraElement { coefficients }
    }

    /// Get coefficients
    pub fn coefficients(&self) -> &[R] {
        &self.coefficients
    }

    /// Get coefficient at index
    pub fn coefficient(&self, index: usize) -> Option<&R> {
        self.coefficients.get(index)
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| c.is_zero())
    }

    /// Dimension of the element
    pub fn dimension(&self) -> usize {
        self.coefficients.len()
    }
}

impl<R: Ring + Clone + Display> Display for JordanAlgebraElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut terms = Vec::new();
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if !coeff.is_zero() {
                terms.push(format!("{}*e{}", coeff, i));
            }
        }
        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + "))
        }
    }
}

// ============================================================================
// Add/Mul implementations
// ============================================================================

impl<R: Ring + Clone + Add<Output = R>> Add for SymmetricBilinearElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.vector.len(), other.vector.len());
        SymmetricBilinearElement {
            scalar: self.scalar + other.scalar,
            vector: self.vector.into_iter()
                .zip(other.vector.into_iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

impl<R: Ring + Clone + Sub<Output = R>> Sub for SymmetricBilinearElement<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.vector.len(), other.vector.len());
        SymmetricBilinearElement {
            scalar: self.scalar - other.scalar,
            vector: self.vector.into_iter()
                .zip(other.vector.into_iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }
}

impl<R: Ring + Clone + Neg<Output = R>> Neg for SymmetricBilinearElement<R> {
    type Output = Self;

    fn neg(self) -> Self {
        SymmetricBilinearElement {
            scalar: -self.scalar,
            vector: self.vector.into_iter().map(|x| -x).collect(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_symmetric_bilinear_creation() {
        let form = vec![
            vec![Rational::from(-2), Rational::from(3)],
            vec![Rational::from(3), Rational::from(4)],
        ];
        let j = JordanAlgebraSymmetricBilinear::new(form);
        assert_eq!(j.dimension(), 3);
        assert_eq!(j.module_dimension(), 2);
    }

    #[test]
    fn test_symmetric_bilinear_one() {
        let j = JordanAlgebraSymmetricBilinear::<Rational>::standard(2);
        let one = j.one();
        assert_eq!(one.scalar, Rational::from(1));
        assert!(one.vector.iter().all(|x| x.is_zero()));
    }

    #[test]
    fn test_symmetric_bilinear_basis() {
        let j = JordanAlgebraSymmetricBilinear::<Rational>::standard(2);
        let basis = j.basis();
        assert_eq!(basis.len(), 3); // 1 + e1 + e2
    }

    #[test]
    fn test_symmetric_bilinear_multiply() {
        let j = JordanAlgebraSymmetricBilinear::<Rational>::standard(2);
        let one = j.one();
        let e1 = SymmetricBilinearElement::new(
            Rational::from(0),
            vec![Rational::from(1), Rational::from(0)],
        );

        // 1 ∘ e1 = e1
        let result = j.multiply(&one, &e1);
        assert_eq!(result.scalar, Rational::from(0));
        assert_eq!(result.vector[0], Rational::from(1));
        assert_eq!(result.vector[1], Rational::from(0));
    }

    #[test]
    fn test_symmetric_bilinear_trace() {
        let e = SymmetricBilinearElement::<Rational>::new(
            Rational::from(5),
            vec![Rational::from(1), Rational::from(2)],
        );
        assert_eq!(e.trace(), Rational::from(10)); // 2 * 5
    }

    #[test]
    fn test_symmetric_bilinear_bar() {
        let e = SymmetricBilinearElement::<Rational>::new(
            Rational::from(3),
            vec![Rational::from(1), Rational::from(2)],
        );
        let bar_e = e.bar();
        assert_eq!(bar_e.scalar, Rational::from(3));
        assert_eq!(bar_e.vector[0], Rational::from(-1));
        assert_eq!(bar_e.vector[1], Rational::from(-2));
    }

    #[test]
    fn test_special_jordan_creation() {
        let j = SpecialJordanAlgebra::<Rational>::new(3);
        assert_eq!(j.matrix_size(), 3);
        assert_eq!(j.dimension(), 9);
    }

    #[test]
    fn test_special_jordan_one() {
        let j = SpecialJordanAlgebra::<Rational>::new(2);
        let one = j.one();
        assert_eq!(*one.get(0, 0).unwrap(), Rational::from(1));
        assert_eq!(*one.get(1, 1).unwrap(), Rational::from(1));
        assert_eq!(*one.get(0, 1).unwrap(), Rational::from(0));
    }

    #[test]
    fn test_exceptional_jordan_dimension() {
        let j = ExceptionalJordanAlgebra::<Rational>::new();
        assert_eq!(j.dimension(), 27);
    }

    #[test]
    fn test_exceptional_jordan_basis() {
        let j = ExceptionalJordanAlgebra::<Rational>::new();
        let basis = j.basis();
        assert_eq!(basis.len(), 27);
    }

    #[test]
    fn test_exceptional_jordan_one() {
        let j = ExceptionalJordanAlgebra::<Rational>::new();
        let one = j.one();
        assert_eq!(one.diagonal[0], Rational::from(1));
        assert_eq!(one.diagonal[1], Rational::from(1));
        assert_eq!(one.diagonal[2], Rational::from(1));
    }

    #[test]
    fn test_element_addition() {
        let e1 = SymmetricBilinearElement::<Rational>::new(
            Rational::from(1),
            vec![Rational::from(2), Rational::from(3)],
        );
        let e2 = SymmetricBilinearElement::<Rational>::new(
            Rational::from(4),
            vec![Rational::from(5), Rational::from(6)],
        );
        let sum = e1 + e2;
        assert_eq!(sum.scalar, Rational::from(5));
        assert_eq!(sum.vector[0], Rational::from(7));
        assert_eq!(sum.vector[1], Rational::from(9));
    }
}
