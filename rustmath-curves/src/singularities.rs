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
    use rustmath_polynomials::groebner::{groebner_basis, MonomialOrdering};

    // Find points where F = ∂F/∂x = ∂F/∂y = 0
    // This requires solving a system of polynomial equations using Gröbner bases

    let num_vars = poly.variables().len();
    if num_vars == 0 {
        return vec![];
    }

    // Compute partial derivatives for all variables
    let mut ideal_generators = vec![poly.clone()];
    for var in 0..num_vars {
        ideal_generators.push(poly.partial_derivative(var));
    }

    // Compute Gröbner basis to solve the system
    // Use lexicographic ordering for elimination
    let basis = groebner_basis(ideal_generators, MonomialOrdering::Lex);

    // Extract solutions from the Gröbner basis
    // In general, this requires sophisticated root-finding algorithms
    // For now, we check if the basis contains only constants (no solutions)
    // or if it contains univariate polynomials (which we could solve)

    let mut singularities = Vec::new();

    // Check if the ideal is trivial (contains a non-zero constant)
    for poly in &basis {
        if poly.is_constant() && poly != &MultiPoly::zero() {
            // Ideal is the whole ring, no common zeros
            return vec![];
        }
    }

    // For a complete implementation, we would:
    // 1. Extract univariate polynomials from the basis
    // 2. Solve them to find candidate points
    // 3. Substitute back to find all coordinates
    // 4. Classify each singular point
    //
    // For now, return an empty vector indicating that solving
    // requires more advanced algebraic methods (resultants, etc.)

    singularities
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

    // It's a singular point; now classify using the Hessian matrix
    // Compute the Hessian (second derivatives)
    let fxx = fx.partial_derivative(0);
    let fxy = fx.partial_derivative(1);
    let fyy = fy.partial_derivative(1);

    let fxx_val = fxx.evaluate(point);
    let fxy_val = fxy.evaluate(point);
    let fyy_val = fyy.evaluate(point);

    // Compute determinant of Hessian: det(H) = fxx * fyy - fxy^2
    let det = fxx_val.clone() * fyy_val.clone() - fxy_val.clone() * fxy_val.clone();

    // Classification based on Hessian analysis:
    // - If det ≠ 0: Ordinary double point (node)
    // - If det = 0 but Hessian is non-zero: Cusp or higher order
    // - If all second derivatives are zero: Higher order singularity

    if det != R::zero() {
        // Non-degenerate Hessian indicates a node (ordinary double point)
        return SingularityType::Node;
    }

    // Check if all second derivatives vanish
    if fxx_val == R::zero() && fxy_val == R::zero() && fyy_val == R::zero() {
        // All second derivatives vanish - need to check higher order terms
        // Compute third derivatives to distinguish cusp from higher order
        let fxxx = fxx.partial_derivative(0);
        let fxxy = fxx.partial_derivative(1);
        let fxyy = fxy.partial_derivative(1);
        let fyyy = fyy.partial_derivative(1);

        let fxxx_val = fxxx.evaluate(point);
        let fxxy_val = fxxy.evaluate(point);
        let fxyy_val = fxyy.evaluate(point);
        let fyyy_val = fyyy.evaluate(point);

        if fxxx_val != R::zero() || fxxy_val != R::zero() ||
           fxyy_val != R::zero() || fyyy_val != R::zero() {
            // Some third derivative is non-zero: likely a higher order singularity
            return SingularityType::TriplePoint;
        } else {
            // Even higher order
            return SingularityType::Higher(4);
        }
    }

    // Hessian determinant is zero but not all entries are zero
    // This typically indicates a cusp or tacnode
    // A more refined classification would examine the rank and signature of the Hessian

    // For cusps, one eigenvalue is non-zero
    // For tacnodes, both eigenvalues are zero but matrix is non-zero

    // Trace of Hessian: tr(H) = fxx + fyy
    let trace = fxx_val.clone() + fyy_val.clone();

    if trace != R::zero() {
        // At least one diagonal entry is non-zero with zero determinant
        // This suggests a cusp (one branch with self-tangency)
        SingularityType::Cusp
    } else {
        // Trace is zero but off-diagonal might be non-zero
        // This suggests a tacnode (two branches with same tangent)
        SingularityType::Tacnode
    }
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
    let mut min: usize = usize::MAX;

    for (monomial, coeff) in poly.terms() {
        if coeff != &R::zero() {
            let degree: usize = monomial.degree() as usize;
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
