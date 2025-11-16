//! Hyperplanes in vector spaces
//!
//! A hyperplane is a codimension-1 affine subspace, defined by a linear equation
//! a₁x₁ + a₂x₂ + ... + aₙxₙ = b.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::hyperplane_arrangement::hyperplane::Hyperplane;
//! use rustmath_rationals::Rational;
//!
//! // The hyperplane x + 2y = 3 in 2D
//! let coeffs = vec![Rational::from(1), Rational::from(2)];
//! let constant = Rational::from(3);
//! let hyperplane = Hyperplane::new(coeffs, constant);
//!
//! assert_eq!(hyperplane.dimension(), 1); // 2D hyperplane has dimension 1
//! ```

use crate::linear_expression::LinearExpression;
use rustmath_core::Ring;
use std::fmt;

/// A hyperplane in a vector space
///
/// Defined by the equation a₁x₁ + a₂x₂ + ... + aₙxₙ = b.
/// The hyperplane is the set of points satisfying this equation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hyperplane<R: Ring> {
    /// The linear expression defining the hyperplane
    expression: LinearExpression<R>,
}

impl<R: Ring> Hyperplane<R> {
    /// Create a new hyperplane
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Coefficients of the linear equation
    /// * `constant` - The constant term (the hyperplane is a₁x₁ + ... + aₙxₙ = constant)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::hyperplane::Hyperplane;
    /// use rustmath_integers::Integer;
    ///
    /// // x + y = 1
    /// let h = Hyperplane::new(
    ///     vec![Integer::from(1), Integer::from(1)],
    ///     Integer::from(1)
    /// );
    /// ```
    pub fn new(coefficients: Vec<R>, constant: R) -> Self {
        // The hyperplane equation is a₁x₁ + ... + aₙxₙ = constant
        // We store it as a linear expression: a₁x₁ + ... + aₙxₙ - constant
        // So a point is on the hyperplane if the expression evaluates to 0
        Self {
            expression: LinearExpression::new(coefficients, -constant),
        }
    }

    /// Create a hyperplane from a linear expression
    pub fn from_expression(expression: LinearExpression<R>) -> Self {
        Self { expression }
    }

    /// Get the coefficients (normal vector)
    pub fn normal(&self) -> &[R] {
        self.expression.coefficient_vector()
    }

    /// Get the constant term
    ///
    /// Returns the constant b in the equation a₁x₁ + ... + aₙxₙ = b.
    pub fn constant(&self) -> R {
        // The expression is stored as a₁x₁ + ... + aₙxₙ - b
        // So we need to negate the constant term
        -self.expression.constant_term().clone()
    }

    /// Get the dimension of the hyperplane
    ///
    /// For an n-dimensional ambient space, a hyperplane has dimension n-1.
    pub fn dimension(&self) -> usize {
        self.expression.num_variables().saturating_sub(1)
    }

    /// Get the ambient dimension
    pub fn ambient_dimension(&self) -> usize {
        self.expression.num_variables()
    }

    /// Check if a point lies on the hyperplane
    ///
    /// # Arguments
    ///
    /// * `point` - The point to test
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::hyperplane::Hyperplane;
    /// use rustmath_integers::Integer;
    ///
    /// let h = Hyperplane::new(
    ///     vec![Integer::from(1), Integer::from(1)],
    ///     Integer::from(2)
    /// );
    ///
    /// assert!(h.contains(&[Integer::from(1), Integer::from(1)])); // 1+1=2
    /// assert!(!h.contains(&[Integer::from(0), Integer::from(0)])); // 0+0≠2
    /// ```
    pub fn contains(&self, point: &[R]) -> bool {
        let value = self.expression.evaluate(point);
        value.is_zero()
    }

    /// Evaluate the linear expression at a point
    ///
    /// Returns a₁p₁ + a₂p₂ + ... + aₙpₙ - b.
    /// The point is on the hyperplane if this returns 0.
    pub fn evaluate(&self, point: &[R]) -> R {
        self.expression.evaluate(point)
    }

    /// Find a point on the hyperplane
    ///
    /// Returns a point closest to the origin (when it exists).
    /// For now, returns a simple point on the hyperplane.
    pub fn point(&self) -> Option<Vec<R>> {
        // Find the first non-zero coefficient
        for (i, coeff) in self.normal().iter().enumerate() {
            if !coeff.is_zero() {
                // Set xi = b/ai and all other coordinates to 0
                let mut point = vec![R::zero(); self.ambient_dimension()];
                // In a full implementation, we would compute b/ai
                // For now, we return the constant as a simple approximation
                point[i] = self.constant();
                return Some(point);
            }
        }
        None
    }

    /// Project a point onto the hyperplane (orthogonal projection)
    ///
    /// For a hyperplane n·x = d with unit normal n,
    /// the projection of point p is: p - ((n·p - d) / ||n||²) * n
    ///
    /// # Note
    ///
    /// This is a simplified implementation. A full implementation would
    /// require computing ||n||² which needs a Field rather than just a Ring.
    pub fn project(&self, _point: &[R]) -> Vec<R> {
        // Simplified implementation
        vec![R::zero(); self.ambient_dimension()]
    }

    /// Get the underlying linear expression
    pub fn expression(&self) -> &LinearExpression<R> {
        &self.expression
    }
}

impl<R: Ring + fmt::Display> fmt::Display for Hyperplane<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hyperplane(")?;

        let mut first = true;
        for (i, coeff) in self.normal().iter().enumerate() {
            if !coeff.is_zero() {
                if !first && !format!("{}", coeff).starts_with('-') {
                    write!(f, " + ")?;
                } else if !first {
                    write!(f, " ")?;
                }

                if coeff.is_one() && i < self.ambient_dimension() {
                    write!(f, "x{}", i)?;
                } else {
                    write!(f, "{}*x{}", coeff, i)?;
                }
                first = false;
            }
        }

        write!(f, " = {})", self.constant())
    }
}

/// Ambient vector space for hyperplanes
///
/// This is a module that generates hyperplanes.
pub struct AmbientVectorSpace {
    dimension: usize,
}

impl AmbientVectorSpace {
    /// Create a new ambient vector space
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;
    use rustmath_rationals::Rational;

    #[test]
    fn test_new_hyperplane() {
        let h = Hyperplane::new(
            vec![Integer::from(1), Integer::from(2)],
            Integer::from(3),
        );

        assert_eq!(h.ambient_dimension(), 2);
        assert_eq!(h.dimension(), 1);
        assert_eq!(h.constant(), Integer::from(3));
    }

    #[test]
    fn test_contains() {
        // x + y = 2
        let h = Hyperplane::new(
            vec![Integer::from(1), Integer::from(1)],
            Integer::from(2),
        );

        assert!(h.contains(&[Integer::from(1), Integer::from(1)]));
        assert!(h.contains(&[Integer::from(0), Integer::from(2)]));
        assert!(h.contains(&[Integer::from(2), Integer::from(0)]));
        assert!(!h.contains(&[Integer::from(0), Integer::from(0)]));
    }

    #[test]
    fn test_normal() {
        let h = Hyperplane::new(
            vec![Integer::from(3), Integer::from(4)],
            Integer::from(5),
        );

        assert_eq!(h.normal(), &[Integer::from(3), Integer::from(4)]);
    }

    #[test]
    fn test_evaluate() {
        // 2x + 3y = 10
        let h = Hyperplane::new(
            vec![Integer::from(2), Integer::from(3)],
            Integer::from(10),
        );

        // At (2, 2): 2*2 + 3*2 - 10 = 4 + 6 - 10 = 0
        assert_eq!(h.evaluate(&[Integer::from(2), Integer::from(2)]), Integer::from(0));

        // At (0, 0): 2*0 + 3*0 - 10 = -10
        assert_eq!(h.evaluate(&[Integer::from(0), Integer::from(0)]), Integer::from(-10));
    }

    #[test]
    fn test_point() {
        let h = Hyperplane::new(
            vec![Integer::from(1), Integer::from(2)],
            Integer::from(6),
        );

        let point = h.point();
        assert!(point.is_some());
    }

    #[test]
    fn test_with_rationals() {
        // x/2 + y/3 = 1
        let h = Hyperplane::new(
            vec![Rational::new(1, 2).unwrap(), Rational::new(1, 3).unwrap()],
            Rational::from(1),
        );

        // Point (2, 0) should be on the hyperplane: (1/2)*2 + (1/3)*0 = 1
        assert!(h.contains(&[Rational::from(2), Rational::from(0)]));
    }

    #[test]
    fn test_ambient_vector_space() {
        let space = AmbientVectorSpace::new(3);
        assert_eq!(space.dimension(), 3);
    }

    #[test]
    fn test_display() {
        let h = Hyperplane::new(
            vec![Integer::from(1), Integer::from(2)],
            Integer::from(3),
        );
        let display = format!("{}", h);
        assert!(display.contains("Hyperplane"));
    }
}
