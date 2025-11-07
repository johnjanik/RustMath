//! General polynomial trait

use rustmath_core::Ring;

/// Trait for polynomial types
pub trait Polynomial: Sized + Clone {
    /// The coefficient ring
    type Coeff: Ring;

    /// The variable type (String for named variables)
    type Var;

    /// Create a polynomial from coefficients
    fn from_coeffs(coeffs: Vec<Self::Coeff>) -> Self;

    /// Get the degree (highest power with non-zero coefficient)
    fn degree(&self) -> Option<usize>;

    /// Evaluate the polynomial at a given point
    fn eval(&self, point: &Self::Coeff) -> Self::Coeff;

    /// Get the coefficient of a specific term
    fn coeff(&self, degree: usize) -> &Self::Coeff;

    /// Get the leading coefficient (coefficient of highest degree term)
    fn leading_coeff(&self) -> Option<&Self::Coeff> {
        self.degree().map(|d| self.coeff(d))
    }

    /// Check if the polynomial is zero
    fn is_zero(&self) -> bool {
        self.degree().is_none()
    }

    /// Check if the polynomial is constant
    fn is_constant(&self) -> bool {
        matches!(self.degree(), None | Some(0))
    }
}
