//! # RustMath Curves
//!
//! This crate provides comprehensive support for algebraic curves including:
//! - Plane curves (affine and projective)
//! - Singularity detection and analysis
//! - Genus computation
//! - Hyperelliptic curves
//! - Curve parameterization
//! - Weierstrass form transformations
//! - Jacobian of hyperelliptic curves (Picard group)
//! - Mumford divisor representation
//! - Cantor's algorithm for divisor arithmetic
//!
//! ## Examples
//!
//! ### Basic Curve Creation
//!
//! ```
//! use rustmath_curves::plane_curve::PlaneCurve;
//! use rustmath_polynomials::multivariate::MultiPoly;
//! use rustmath_rationals::Rational;
//!
//! // Create a plane curve y^2 = x^3 + x
//! // This is represented by the polynomial y^2 - x^3 - x = 0
//! ```
//!
//! ### Jacobian and Divisor Arithmetic
//!
//! ```ignore
//! use rustmath_curves::{HyperellipticCurve, Jacobian};
//! use rustmath_rationals::Rational;
//!
//! // Create a genus 2 hyperelliptic curve: y^2 = x^5 - x
//! let curve = HyperellipticCurve::simple_genus_2().unwrap();
//!
//! // Create the Jacobian (Picard group)
//! let jac = Jacobian::new(curve);
//!
//! // Create divisors from points on the curve
//! let p1 = jac.point(Rational::zero(), Rational::zero());
//! let p2 = jac.point(Rational::one(), Rational::zero());
//!
//! // Add divisors using Cantor's algorithm
//! let sum = p1.add(&p2);
//!
//! // Scalar multiplication
//! let doubled = p1.scalar_multiply(2);
//! ```

pub mod plane_curve;
pub mod singularities;
pub mod genus;
pub mod hyperelliptic;
pub mod parameterization;
pub mod weierstrass;
pub mod divisor;
pub mod cantor;
pub mod jacobian;

pub use plane_curve::{PlaneCurve, AffineCurve, ProjectiveCurve};
pub use singularities::{Singularity, SingularityType};
pub use genus::compute_genus;
pub use hyperelliptic::HyperellipticCurve;
pub use parameterization::{CurveParameterization, RationalParameterization};
pub use weierstrass::{WeierstrassForm, weierstrass_transform};
pub use divisor::MumfordDivisor;
pub use cantor::CantorAlgorithm;
pub use jacobian::{Jacobian, JacobianElement};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        // Basic test to ensure the module compiles
        assert!(true);
    }
}
