//! Two-Dimensional Lie Algebras
//!
//! Provides a general implementation of 2-dimensional Lie algebras
//! parameterized by structure coefficients.
//!
//! A 2-dimensional Lie algebra is determined by the bracket [e_0, e_1] = c*e_0 + d*e_1

use rustmath_core::Ring;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// A two-dimensional Lie algebra with structure coefficients
///
/// The Lie algebra has basis {e_0, e_1} with bracket:
/// - [e_0, e_1] = c*e_0 + d*e_1
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::two_dimensional::TwoDimensionalLieAlgebra;
/// # use rustmath_integers::Integer;
/// // Create affine transformations algebra: [X, Y] = Y
/// let affine = TwoDimensionalLieAlgebra::new(
///     Integer::from(0),  // c: no e_0 component
///     Integer::from(1),  // d: [e_0, e_1] = e_1
/// );
/// ```
pub struct TwoDimensionalLieAlgebra<R: Ring> {
    /// Structure coefficient: [e_0, e_1] = c*e_0 + d*e_1
    c: R,
    /// Structure coefficient: [e_0, e_1] = c*e_0 + d*e_1
    d: R,
    /// Base ring marker
    _phantom: PhantomData<R>,
}

impl<R: Ring + Clone> TwoDimensionalLieAlgebra<R> {
    /// Create a new two-dimensional Lie algebra with given structure coefficients
    ///
    /// # Arguments
    ///
    /// * `c` - Coefficient for [e_0, e_1] = c*e_0 + d*e_1
    /// * `d` - Coefficient for [e_0, e_1] = c*e_0 + d*e_1
    pub fn new(c: R, d: R) -> Self {
        TwoDimensionalLieAlgebra {
            c,
            d,
            _phantom: PhantomData,
        }
    }

    /// Get the dimension (always 2)
    pub fn dimension(&self) -> usize {
        2
    }

    /// Compute the Lie bracket [e_i, e_j] for basis elements
    ///
    /// Returns the result as coefficients [c_0, c_1] representing
    /// c_0*e_0 + c_1*e_1
    ///
    /// # Arguments
    ///
    /// * `i` - First basis element index (0 or 1)
    /// * `j` - Second basis element index (0 or 1)
    ///
    /// # Returns
    ///
    /// Vector of 2 coefficients [c_0, c_1]
    pub fn bracket_on_basis(&self, i: usize, j: usize) -> [R; 2] {
        match (i, j) {
            // [e_i, e_i] = 0
            (0, 0) | (1, 1) => [R::zero(), R::zero()],

            // [e_0, e_1] = c*e_0 + d*e_1
            (0, 1) => [self.c.clone(), self.d.clone()],

            // [e_1, e_0] = -c*e_0 - d*e_1 (antisymmetry)
            (1, 0) => [-self.c.clone(), -self.d.clone()],

            _ => panic!("Invalid basis indices: must be 0 or 1"),
        }
    }

    /// Get the structure coefficients
    pub fn structure_coefficients(&self) -> (R, R) {
        (self.c.clone(), self.d.clone())
    }

    /// Create a Lie algebra element from coefficients
    pub fn element(&self, coeffs: [R; 2]) -> TwoDimensionalLieAlgebraElement<R> {
        TwoDimensionalLieAlgebraElement::new(coeffs, self.c.clone(), self.d.clone())
    }

    /// Get basis elements
    pub fn basis(&self) -> [TwoDimensionalLieAlgebraElement<R>; 2]
    where
        R: From<i64>,
    {
        [
            self.element([R::from(1), R::from(0)]),
            self.element([R::from(0), R::from(1)]),
        ]
    }

    /// Check if this is abelian
    pub fn is_abelian(&self) -> bool {
        self.c.is_zero() && self.d.is_zero()
    }

    /// Check if this is solvable (always true for 2D algebras)
    pub fn is_solvable(&self) -> bool {
        true
    }
}

impl<R: Ring + Clone> Display for TwoDimensionalLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "2-dimensional Lie algebra")
    }
}

/// Element of a two-dimensional Lie algebra
///
/// Represented as a linear combination of basis elements
#[derive(Clone, Debug, PartialEq)]
pub struct TwoDimensionalLieAlgebraElement<R: Ring> {
    /// Coefficients [c_0, c_1] representing c_0*e_0 + c_1*e_1
    coeffs: [R; 2],
    /// Structure coefficients of the parent algebra
    c: R,
    d: R,
}

impl<R: Ring + Clone> TwoDimensionalLieAlgebraElement<R> {
    /// Create a new element
    pub fn new(coeffs: [R; 2], c: R, d: R) -> Self {
        TwoDimensionalLieAlgebraElement { coeffs, c, d }
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[R; 2] {
        &self.coeffs
    }

    /// Compute the Lie bracket [self, other]
    pub fn bracket(&self, other: &Self) -> Self {
        let [x0, x1] = &self.coeffs;
        let [y0, y1] = &other.coeffs;

        // Expand [x0*e_0 + x1*e_1, y0*e_0 + y1*e_1]
        // Only non-zero bracket is [e_0, e_1] = c*e_0 + d*e_1

        let term_01 = x0.clone() * y1.clone() - x1.clone() * y0.clone();

        let result = [
            term_01.clone() * self.c.clone(),
            term_01 * self.d.clone(),
        ];

        TwoDimensionalLieAlgebraElement::new(result, self.c.clone(), self.d.clone())
    }
}

impl<R: Ring + Clone> Display for TwoDimensionalLieAlgebraElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({:?}, {:?})", self.coeffs[0], self.coeffs[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_affine_transformations() {
        // Affine transformations: [e_0, e_1] = e_1
        let affine = TwoDimensionalLieAlgebra::new(Integer::from(0), Integer::from(1));

        assert_eq!(affine.dimension(), 2);

        // Test [e_0, e_1] = e_1
        let result = affine.bracket_on_basis(0, 1);
        assert_eq!(result[0], Integer::from(0));
        assert_eq!(result[1], Integer::from(1));

        // Test [e_1, e_0] = -e_1 (antisymmetry)
        let result = affine.bracket_on_basis(1, 0);
        assert_eq!(result[0], Integer::from(0));
        assert_eq!(result[1], Integer::from(-1));
    }

    #[test]
    fn test_abelian_case() {
        // Abelian case: all brackets zero
        let abelian = TwoDimensionalLieAlgebra::new(Integer::from(0), Integer::from(0));

        assert!(abelian.is_abelian());

        let result = abelian.bracket_on_basis(0, 1);
        assert_eq!(result[0], Integer::from(0));
        assert_eq!(result[1], Integer::from(0));
    }

    #[test]
    fn test_element_bracket() {
        let affine = TwoDimensionalLieAlgebra::new(Integer::from(0), Integer::from(1));

        let e0 = affine.element([Integer::from(1), Integer::from(0)]);
        let e1 = affine.element([Integer::from(0), Integer::from(1)]);

        // [e_0, e_1] = e_1
        let result = e0.bracket(&e1);
        assert_eq!(result.coefficients()[0], Integer::from(0));
        assert_eq!(result.coefficients()[1], Integer::from(1));
    }

    #[test]
    fn test_antisymmetry() {
        let algebra = TwoDimensionalLieAlgebra::new(Integer::from(2), Integer::from(3));

        let x = algebra.element([Integer::from(1), Integer::from(2)]);
        let y = algebra.element([Integer::from(3), Integer::from(4)]);

        let xy = x.bracket(&y);
        let yx = y.bracket(&x);

        // [x, y] = -[y, x]
        for i in 0..2 {
            assert_eq!(
                xy.coefficients()[i].clone() + yx.coefficients()[i].clone(),
                Integer::from(0)
            );
        }
    }
}
