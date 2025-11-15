//! Three-Dimensional Lie Algebras
//!
//! Provides a general implementation of 3-dimensional Lie algebras
//! parameterized by structure coefficients.
//!
//! A 3-dimensional Lie algebra is determined by structure coefficients
//! satisfying the Jacobi identity. This module provides several standard
//! 3-dimensional Lie algebras.
//!
//! Corresponds to sage.algebras.lie_algebras.lie_algebra.LieAlgebraFromAssociative

use rustmath_core::Ring;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// A three-dimensional Lie algebra with structure coefficients
///
/// The Lie algebra has basis {e_0, e_1, e_2} with brackets determined by
/// parameters a, b, c, d:
///
/// - [e_0, e_1] = a*e_2
/// - [e_0, e_2] = b*e_1
/// - [e_1, e_2] = c*e_0 + d*e_2
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::three_dimensional::ThreeDimensionalLieAlgebra;
/// # use rustmath_integers::Integer;
/// // Create R^3 with cross product
/// let cross = ThreeDimensionalLieAlgebra::new(
///     Integer::from(1),  // a
///     Integer::from(1),  // b
///     Integer::from(1),  // c
///     Integer::from(0),  // d
/// );
/// ```
pub struct ThreeDimensionalLieAlgebra<R: Ring> {
    /// Structure coefficient: [e_0, e_1] = a*e_2
    a: R,
    /// Structure coefficient: [e_0, e_2] = b*e_1
    b: R,
    /// Structure coefficient: [e_1, e_2] = c*e_0 + d*e_2
    c: R,
    /// Structure coefficient: [e_1, e_2] = c*e_0 + d*e_2
    d: R,
    /// Base ring marker
    _phantom: PhantomData<R>,
}

impl<R: Ring + Clone> ThreeDimensionalLieAlgebra<R> {
    /// Create a new three-dimensional Lie algebra with given structure coefficients
    ///
    /// # Arguments
    ///
    /// * `a` - Coefficient for [e_0, e_1] = a*e_2
    /// * `b` - Coefficient for [e_0, e_2] = b*e_1
    /// * `c` - Coefficient for [e_1, e_2] = c*e_0 + d*e_2
    /// * `d` - Coefficient for [e_1, e_2] = c*e_0 + d*e_2
    pub fn new(a: R, b: R, c: R, d: R) -> Self {
        ThreeDimensionalLieAlgebra {
            a,
            b,
            c,
            d,
            _phantom: PhantomData,
        }
    }

    /// Get the dimension (always 3)
    pub fn dimension(&self) -> usize {
        3
    }

    /// Compute the Lie bracket [e_i, e_j] for basis elements
    ///
    /// Returns the result as coefficients [c_0, c_1, c_2] representing
    /// c_0*e_0 + c_1*e_1 + c_2*e_2
    ///
    /// # Arguments
    ///
    /// * `i` - First basis element index (0, 1, or 2)
    /// * `j` - Second basis element index (0, 1, or 2)
    ///
    /// # Returns
    ///
    /// Vector of 3 coefficients [c_0, c_1, c_2]
    pub fn bracket_on_basis(&self, i: usize, j: usize) -> [R; 3] {
        match (i, j) {
            // [e_i, e_i] = 0
            (0, 0) | (1, 1) | (2, 2) => [R::zero(), R::zero(), R::zero()],

            // [e_0, e_1] = a*e_2
            (0, 1) => [R::zero(), R::zero(), self.a.clone()],

            // [e_1, e_0] = -a*e_2 (antisymmetry)
            (1, 0) => [R::zero(), R::zero(), -self.a.clone()],

            // [e_0, e_2] = b*e_1
            (0, 2) => [R::zero(), self.b.clone(), R::zero()],

            // [e_2, e_0] = -b*e_1 (antisymmetry)
            (2, 0) => [R::zero(), -self.b.clone(), R::zero()],

            // [e_1, e_2] = c*e_0 + d*e_2
            (1, 2) => [self.c.clone(), R::zero(), self.d.clone()],

            // [e_2, e_1] = -c*e_0 - d*e_2 (antisymmetry)
            (2, 1) => [-self.c.clone(), R::zero(), -self.d.clone()],

            _ => panic!("Invalid basis indices: must be 0, 1, or 2"),
        }
    }

    /// Get the structure coefficients
    pub fn structure_coefficients(&self) -> (R, R, R, R) {
        (self.a.clone(), self.b.clone(), self.c.clone(), self.d.clone())
    }

    /// Create a Lie algebra element from coefficients
    pub fn element(&self, coeffs: [R; 3]) -> ThreeDimensionalLieAlgebraElement<R> {
        ThreeDimensionalLieAlgebraElement::new(
            coeffs,
            self.a.clone(),
            self.b.clone(),
            self.c.clone(),
            self.d.clone(),
        )
    }

    /// Get basis elements
    pub fn basis(&self) -> [ThreeDimensionalLieAlgebraElement<R>; 3]
    where
        R: From<i64>,
    {
        [
            self.element([R::from(1), R::from(0), R::from(0)]),
            self.element([R::from(0), R::from(1), R::from(0)]),
            self.element([R::from(0), R::from(0), R::from(1)]),
        ]
    }
}

impl<R: Ring + Clone> Display for ThreeDimensionalLieAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "3-dimensional Lie algebra")
    }
}

/// Element of a three-dimensional Lie algebra
///
/// Represented as a linear combination of basis elements
#[derive(Clone, Debug, PartialEq)]
pub struct ThreeDimensionalLieAlgebraElement<R: Ring> {
    /// Coefficients [c_0, c_1, c_2] representing c_0*e_0 + c_1*e_1 + c_2*e_2
    coeffs: [R; 3],
    /// Structure coefficients of the parent algebra
    a: R,
    b: R,
    c: R,
    d: R,
}

impl<R: Ring + Clone> ThreeDimensionalLieAlgebraElement<R> {
    /// Create a new element
    pub fn new(coeffs: [R; 3], a: R, b: R, c: R, d: R) -> Self {
        ThreeDimensionalLieAlgebraElement { coeffs, a, b, c, d }
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[R; 3] {
        &self.coeffs
    }

    /// Compute the Lie bracket [self, other]
    pub fn bracket(&self, other: &Self) -> Self {
        let [x0, x1, x2] = &self.coeffs;
        let [y0, y1, y2] = &other.coeffs;

        // Expand [x0*e_0 + x1*e_1 + x2*e_2, y0*e_0 + y1*e_1 + y2*e_2]
        // using bilinearity and the structure coefficients

        let mut result = [R::zero(), R::zero(), R::zero()];

        // [e_0, e_1] = a*e_2
        let term_01 = x0.clone() * y1.clone() - x1.clone() * y0.clone();
        result[2] = result[2].clone() + term_01 * self.a.clone();

        // [e_0, e_2] = b*e_1
        let term_02 = x0.clone() * y2.clone() - x2.clone() * y0.clone();
        result[1] = result[1].clone() + term_02 * self.b.clone();

        // [e_1, e_2] = c*e_0 + d*e_2
        let term_12 = x1.clone() * y2.clone() - x2.clone() * y1.clone();
        result[0] = result[0].clone() + term_12.clone() * self.c.clone();
        result[2] = result[2].clone() + term_12 * self.d.clone();

        ThreeDimensionalLieAlgebraElement::new(
            result,
            self.a.clone(),
            self.b.clone(),
            self.c.clone(),
            self.d.clone(),
        )
    }
}

impl<R: Ring + Clone> Display for ThreeDimensionalLieAlgebraElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({:?}, {:?}, {:?})", self.coeffs[0], self.coeffs[1], self.coeffs[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_cross_product_algebra() {
        // R^3 with cross product: a=1, b=1, c=1, d=0
        let cross = ThreeDimensionalLieAlgebra::new(
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
            Integer::from(0),
        );

        assert_eq!(cross.dimension(), 3);

        // Test [e_0, e_1] = e_2
        let result = cross.bracket_on_basis(0, 1);
        assert_eq!(result[0], Integer::from(0));
        assert_eq!(result[1], Integer::from(0));
        assert_eq!(result[2], Integer::from(1));

        // Test [e_1, e_2] = e_0
        let result = cross.bracket_on_basis(1, 2);
        assert_eq!(result[0], Integer::from(1));
        assert_eq!(result[1], Integer::from(0));
        assert_eq!(result[2], Integer::from(0));

        // Test [e_2, e_0] = -e_1
        let result = cross.bracket_on_basis(2, 0);
        assert_eq!(result[0], Integer::from(0));
        assert_eq!(result[1], Integer::from(-1));
        assert_eq!(result[2], Integer::from(0));
    }

    #[test]
    fn test_element_bracket() {
        let cross = ThreeDimensionalLieAlgebra::new(
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
            Integer::from(0),
        );

        let e0 = cross.element([Integer::from(1), Integer::from(0), Integer::from(0)]);
        let e1 = cross.element([Integer::from(0), Integer::from(1), Integer::from(0)]);

        // [e_0, e_1] = e_2
        let result = e0.bracket(&e1);
        assert_eq!(result.coefficients()[0], Integer::from(0));
        assert_eq!(result.coefficients()[1], Integer::from(0));
        assert_eq!(result.coefficients()[2], Integer::from(1));
    }

    #[test]
    fn test_antisymmetry() {
        let algebra = ThreeDimensionalLieAlgebra::new(
            Integer::from(2),
            Integer::from(3),
            Integer::from(5),
            Integer::from(7),
        );

        let x = algebra.element([Integer::from(1), Integer::from(2), Integer::from(3)]);
        let y = algebra.element([Integer::from(4), Integer::from(5), Integer::from(6)]);

        let xy = x.bracket(&y);
        let yx = y.bracket(&x);

        // [x, y] = -[y, x]
        for i in 0..3 {
            assert_eq!(xy.coefficients()[i].clone() + yx.coefficients()[i].clone(), Integer::from(0));
        }
    }
}
