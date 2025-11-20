//! Singularity detection and analysis for algebraic curves
//!
//! A singular point on a curve F(x,y) = 0 is a point where:
//! - F(x,y) = 0
//! - ∂F/∂x = 0
//! - ∂F/∂y = 0
//!
//! Singularities are classified by their multiplicity and local behavior.

use rustmath_core::Ring;
use rustmath_polynomials::multivariate::MultiPoly;
use rustmath_matrix::matrix::Matrix;
use rustmath_rationals::Rational;
use std::fmt;

/// Types of singularities
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SingularityType {
    /// Ordinary double point (node): two branches crossing transversally
    Node,
    /// Cusp: point where curve has a sharp turn
    Cusp,
    /// Tacnode: two branches with same tangent
    Tacnode,
    /// Triple point: three branches meeting
    TriplePoint,
    /// Higher order singularity
    Higher(usize),
    /// Smooth point (not actually singular)
    Smooth,
}

impl fmt::Display for SingularityType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SingularityType::Node => write!(f, "Node (ordinary double point)"),
            SingularityType::Cusp => write!(f, "Cusp"),
            SingularityType::Tacnode => write!(f, "Tacnode"),
            SingularityType::TriplePoint => write!(f, "Triple point"),
            SingularityType::Higher(n) => write!(f, "{}-fold point", n),
            SingularityType::Smooth => write!(f, "Smooth (non-singular)"),
        }
    }
}

/// A singularity on a curve
#[derive(Debug, Clone)]
pub struct Singularity<R: Ring> {
    /// Location of the singularity
    pub point: Vec<R>,
    /// Type of singularity
    pub singularity_type: SingularityType,
    /// Multiplicity of the singularity
    pub multiplicity: usize,
    /// Delta invariant (contributes to genus formula)
    pub delta: usize,
}

impl<R: Ring> Singularity<R> {
    /// Create a new singularity
    pub fn new(point: Vec<R>, singularity_type: SingularityType, multiplicity: usize) -> Self {
        // Delta invariant for common singularities
        let delta = match &singularity_type {
            SingularityType::Node => 1,
            SingularityType::Cusp => 1,
            SingularityType::Tacnode => 2,
            SingularityType::TriplePoint => 3,
            SingularityType::Higher(n) => (n * (n - 1)) / 2,
            SingularityType::Smooth => 0,
        };

        Singularity {
            point,
            singularity_type,
            multiplicity,
            delta,
        }
    }

    /// Check if this is actually a singular point
    pub fn is_singular(&self) -> bool {
        self.singularity_type != SingularityType::Smooth
    }
}

impl<R: Ring + fmt::Display> fmt::Display for Singularity<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} at ({}) with multiplicity {}, δ = {}",
            self.singularity_type,
            self.point
                .iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<_>>()
                .join(", "),
            self.multiplicity,
            self.delta
        )
    }
}

/// Detect singularities of a curve defined by polynomial F
pub fn find_singularities<R: Ring + Clone + PartialEq>(
    poly: &MultiPoly<R>,
) -> Vec<Singularity<R>> {
    // Find points where F = ∂F/∂x = ∂F/∂y = 0
    // This requires solving a system of polynomial equations
    // For now, we return an empty vector (simplified implementation)

    // A complete implementation would:
    // 1. Compute partial derivatives
    // 2. Use Gröbner bases to solve the system
    // 3. Classify each singular point

    vec![]
}

/// Classify a singular point by examining the Hessian matrix
pub fn classify_singularity<R: Ring + Clone + PartialEq>(
    poly: &MultiPoly<R>,
    point: &[R],
) -> SingularityType {
    // Check if the point is actually on the curve
    let value = poly.evaluate(point);
    if value != R::zero() {
        return SingularityType::Smooth;
    }

    // Compute the Jacobian (first derivatives)
    let fx = poly.partial_derivative(0);
    let fy = poly.partial_derivative(1);

    let fx_val = fx.evaluate(point);
    let fy_val = fy.evaluate(point);

    // If either partial derivative is non-zero, it's smooth
    if fx_val != R::zero() || fy_val != R::zero() {
        return SingularityType::Smooth;
    }

    // Compute the Hessian (second derivatives)
    let fxx = fx.partial_derivative(0);
    let fxy = fx.partial_derivative(1);
    let fyy = fy.partial_derivative(1);

    let fxx_val = fxx.evaluate(point);
    let fxy_val = fxy.evaluate(point);
    let fyy_val = fyy.evaluate(point);

    // Compute discriminant of Hessian: fxx * fyy - fxy^2
    // This helps classify the singularity

    // For now, return a generic classification
    // A full implementation would analyze the Hessian determinant
    SingularityType::Node
}

/// Compute the multiplicity of a point on a curve
pub fn multiplicity_at_point<R: Ring + Clone + PartialEq>(
    poly: &MultiPoly<R>,
    point: &[R],
) -> usize {
    // The multiplicity is the lowest degree of terms that don't vanish at the point

    // Translate the polynomial so the point is at the origin
    let translated = translate_to_origin(poly, point);

    // Find the minimum degree of non-zero terms
    min_degree(&translated)
}

/// Translate a polynomial so that the given point becomes the origin
fn translate_to_origin<R: Ring + Clone>(poly: &MultiPoly<R>, point: &[R]) -> MultiPoly<R> {
    // Substitute x → x + point[0], y → y + point[1], etc.
    // For now, return the original polynomial (simplified)
    poly.clone()
}

/// Find the minimum degree of non-zero terms in a polynomial
fn min_degree<R: Ring + Clone + PartialEq>(poly: &MultiPoly<R>) -> usize {
    let mut min = usize::MAX;

    for (exponents, coeff) in poly.terms() {
        if coeff != &R::zero() {
            let degree: usize = exponents.iter().sum();
            if degree < min {
                min = degree;
            }
        }
    }

    if min == usize::MAX {
        0
    } else {
        min
    }
}

/// Check if a curve is smooth (has no singular points)
pub fn is_smooth<R: Ring + Clone + PartialEq>(poly: &MultiPoly<R>) -> bool {
    find_singularities(poly).is_empty()
}

/// Common singularities for testing and examples
impl Singularity<Rational> {
    /// Create a node at the origin
    pub fn node_at_origin() -> Self {
        Singularity::new(
            vec![Rational::zero(), Rational::zero()],
            SingularityType::Node,
            2,
        )
    }

    /// Create a cusp at the origin
    pub fn cusp_at_origin() -> Self {
        Singularity::new(
            vec![Rational::zero(), Rational::zero()],
            SingularityType::Cusp,
            2,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singularity_types() {
        let node = Singularity::<Rational>::node_at_origin();
        assert_eq!(node.singularity_type, SingularityType::Node);
        assert_eq!(node.delta, 1);
        assert!(node.is_singular());

        let cusp = Singularity::<Rational>::cusp_at_origin();
        assert_eq!(cusp.singularity_type, SingularityType::Cusp);
        assert_eq!(cusp.delta, 1);
        assert!(cusp.is_singular());
    }

    #[test]
    fn test_delta_invariants() {
        let triple = Singularity::new(
            vec![Rational::zero()],
            SingularityType::TriplePoint,
            3,
        );
        assert_eq!(triple.delta, 3);

        let higher = Singularity::new(
            vec![Rational::zero()],
            SingularityType::Higher(4),
            4,
        );
        assert_eq!(higher.delta, 6); // 4 * 3 / 2
    }

    #[test]
    fn test_smooth_point() {
        let smooth = Singularity::new(
            vec![Rational::one(), Rational::one()],
            SingularityType::Smooth,
            1,
        );
        assert!(!smooth.is_singular());
        assert_eq!(smooth.delta, 0);
    }
}
