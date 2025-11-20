//! # RustMath Curves
//!
//! This crate provides comprehensive support for algebraic curves including:
//! - Plane curves (affine and projective)
//! - Singularity detection and analysis
//! - Genus computation
//! - Hyperelliptic curves
//! - Curve parameterization
//! - Weierstrass form transformations
//!
//! ## Examples
//!
//! ```
//! use rustmath_curves::plane_curve::PlaneCurve;
//! use rustmath_polynomials::multivariate::MultiPoly;
//! use rustmath_rationals::Rational;
//!
//! // Create a plane curve y^2 = x^3 + x
//! // This is represented by the polynomial y^2 - x^3 - x = 0
//! ```

pub mod plane_curve;
pub mod singularities;
pub mod genus;
pub mod hyperelliptic;
pub mod parameterization;
pub mod weierstrass;

pub use plane_curve::{PlaneCurve, AffineCurve, ProjectiveCurve};
pub use singularities::{Singularity, SingularityType};
pub use genus::compute_genus;
pub use hyperelliptic::HyperellipticCurve;
pub use parameterization::{CurveParameterization, RationalParameterization};
pub use weierstrass::{WeierstrassForm, weierstrass_transform};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        // Basic test to ensure the module compiles
        assert!(true);
    }
}
