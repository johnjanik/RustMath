//! Elliptic Curves in Algebraic Geometry
//!
//! This module provides algebraic geometry perspectives on elliptic curves,
//! including isogenies, moduli spaces, and geometric properties.

pub mod isogeny;

pub use isogeny::{Isogeny, IsogenyGraph, KernelPolynomial};
// Elliptic Curves over Schemes
//
// This module provides elliptic curve functionality from an algebraic geometry
// perspective, implementing generic elliptic curves over arbitrary fields.
//
// This corresponds to `sage.schemes.elliptic_curves` in SageMath.
//
// # Modules
//
// - `generic`: Generic elliptic curves over any field (EllipticCurve_generic base class)
//
// # Examples
//
// ## Working with elliptic curves over different fields
//
// ```
// use rustmath_schemes::elliptic_curves::generic::{EllipticCurve, Point};
// use rustmath_rationals::Rational;
//
// // Curve over the rationals
// let curve = EllipticCurve::short_weierstrass(
//     Rational::from_integer(-1),
//     Rational::from_integer(0),
// );
// ```

pub mod generic;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use generic::{EllipticCurve, Point};
// Elliptic Curves as Schemes
//
// This module provides the scheme-theoretic perspective on elliptic curves,
// complementing the arithmetic approach in `rustmath-ellipticcurves`.
//
// # Contents
//
// - **Heegner Points**: Construction via complex multiplication, Gross-Zagier
//   formula, and applications to BSD conjecture
//
// # Overview
//
// An elliptic curve is a smooth projective curve of genus 1 with a specified
// rational point (serving as the identity for the group law). From the
// scheme-theoretic perspective, we can study:
//
// - Moduli spaces of elliptic curves
// - Modular curves and modular parametrizations
// - Heegner points and complex multiplication
// - The arithmetic of elliptic curves over number fields
//
// This module focuses on the construction and analysis of Heegner points,
// which are special points on elliptic curves constructed via the theory
// of complex multiplication.

pub mod heegner;

// Re-export main types
pub use heegner::{
    ImaginaryQuadraticField,
    HeegnerDiscriminant,
    HeegnerPoint,
    CanonicalHeight,
    HeightPairing,
    GrossZagierFormula,
    BSDHeegner,
    BSDVerificationResult,
};
// Elliptic Curves
//
// This module provides comprehensive support for elliptic curves as schemes.
//
// # Overview
//
// An elliptic curve is a smooth projective curve of genus 1 with a specified base point.
// In the affine plane, an elliptic curve over a field k is typically given by a
// Weierstrass equation:
//
// y² = x³ + ax + b  (characteristic ≠ 2, 3)
//
// or more generally:
//
// y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆  (general Weierstrass form)
//
// # Key Features
//
// - **Weierstrass Models**: Standard and short Weierstrass forms
// - **Group Law**: The group structure on rational points
// - **Discriminant and j-invariant**: Isomorphism invariants
// - **Torsion Points**: Points of finite order
// - **Isogenies**: Morphisms of elliptic curves
// - **Modular Forms**: Connection to modular curves (future)
//
// # Mathematical Background
//
// ## The Group Law
//
// The set of rational points E(k) forms an abelian group under a geometric addition law:
// - Identity: The point at infinity O
// - Addition: Three collinear points sum to O
// - Inversion: Reflection across the x-axis
//
// ## Invariants
//
// - **Discriminant Δ**: Measures singularity (Δ ≠ 0 for smooth curves)
// - **j-invariant**: Classifies curves up to isomorphism over algebraically closed fields
//
// # Examples
//
// ## Creating an Elliptic Curve
//
// ```rust
// use rustmath_schemes::elliptic_curves::EllipticCurve;
//
// // Create y² = x³ - x (short Weierstrass form)
// // let e = EllipticCurve::short_weierstrass(0, -1);
// // assert!(e.is_smooth());
// ```
//
// ## Working with Points
//
// ```rust
// use rustmath_schemes::elliptic_curves::{EllipticCurve, EllipticCurvePoint};
//
// // Create a curve and points
// // let e = EllipticCurve::short_weierstrass(0, -1);
// // let p = EllipticCurvePoint::affine(0, 0);
// // let q = EllipticCurvePoint::affine(1, 0);
//
// // Point addition
// // let r = e.add(&p, &q);
// ```
//
// ## Computing Invariants
//
// ```rust
// use rustmath_schemes::elliptic_curves::EllipticCurve;
//
// // let e = EllipticCurve::short_weierstrass(-1, 0);
// // let disc = e.discriminant();
// // let j = e.j_invariant();
// ```

// Note: EllipticCurve, Point, Isogeny and related types are re-exported from submodules above.
// Additional helper types specific to this module:

use rustmath_core::{Field, Ring};

/// Torsion subgroup E[n] of n-torsion points
///
/// E[n] = {P ∈ E(k̄) | [n]P = O}
///
/// Over an algebraically closed field:
/// - E[n] ≅ ℤ/nℤ × ℤ/nℤ for n coprime to char(k)
/// - Structure depends on reduction type otherwise
#[derive(Debug, Clone)]
pub struct TorsionSubgroup<F: Field> {
    curve: EllipticCurve<F>,
    n: usize,
}

impl<F: Field> TorsionSubgroup<F> {
    /// Create the n-torsion subgroup E[n]
    pub fn new(curve: EllipticCurve<F>, n: usize) -> Self {
        TorsionSubgroup { curve, n }
    }

    /// Get the order n
    pub fn order(&self) -> usize {
        self.n
    }

    /// Compute the structure of the torsion subgroup
    ///
    /// Returns the invariant factors (for n coprime to characteristic)
    pub fn invariant_factors(&self) -> Vec<usize> {
        vec![self.n, self.n] // Placeholder: E[n] ≅ ℤ/nℤ × ℤ/nℤ
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elliptic_curve_point() {
        // Test point creation (would use concrete field in practice)
        // let p = EllipticCurvePoint::infinity();
        // assert!(p.is_infinity());
    }

    #[test]
    fn test_elliptic_curve_properties() {
        // Would test with concrete fields
        // let e = EllipticCurve::short_weierstrass(a, b, field);
        // assert_eq!(e.dimension(), Some(1));
        // assert_eq!(e.genus(), Some(1));
    }

    #[test]
    fn test_isogeny() {
        // Would test with concrete curves
        // let phi = Isogeny::new(e1, e2);
        // assert_eq!(phi.degree(), Some(deg));
    }
}
