//! Elliptic Curves over Schemes
//!
//! This module provides elliptic curve functionality from an algebraic geometry
//! perspective, implementing generic elliptic curves over arbitrary fields.
//!
//! This corresponds to `sage.schemes.elliptic_curves` in SageMath.
//!
//! # Modules
//!
//! - `generic`: Generic elliptic curves over any field (EllipticCurve_generic base class)
//!
//! # Examples
//!
//! ## Working with elliptic curves over different fields
//!
//! ```
//! use rustmath_schemes::elliptic_curves::generic::{EllipticCurve, Point};
//! use rustmath_rationals::Rational;
//!
//! // Curve over the rationals
//! let curve = EllipticCurve::short_weierstrass(
//!     Rational::from_integer(-1),
//!     Rational::from_integer(0),
//! );
//! ```

pub mod generic;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use generic::{EllipticCurve, Point};
//! Elliptic Curves as Schemes
//!
//! This module provides the scheme-theoretic perspective on elliptic curves,
//! complementing the arithmetic approach in `rustmath-ellipticcurves`.
//!
//! # Contents
//!
//! - **Heegner Points**: Construction via complex multiplication, Gross-Zagier
//!   formula, and applications to BSD conjecture
//!
//! # Overview
//!
//! An elliptic curve is a smooth projective curve of genus 1 with a specified
//! rational point (serving as the identity for the group law). From the
//! scheme-theoretic perspective, we can study:
//!
//! - Moduli spaces of elliptic curves
//! - Modular curves and modular parametrizations
//! - Heegner points and complex multiplication
//! - The arithmetic of elliptic curves over number fields
//!
//! This module focuses on the construction and analysis of Heegner points,
//! which are special points on elliptic curves constructed via the theory
//! of complex multiplication.

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
//! Elliptic Curves
//!
//! This module provides comprehensive support for elliptic curves as schemes.
//!
//! # Overview
//!
//! An elliptic curve is a smooth projective curve of genus 1 with a specified base point.
//! In the affine plane, an elliptic curve over a field k is typically given by a
//! Weierstrass equation:
//!
//! y² = x³ + ax + b  (characteristic ≠ 2, 3)
//!
//! or more generally:
//!
//! y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆  (general Weierstrass form)
//!
//! # Key Features
//!
//! - **Weierstrass Models**: Standard and short Weierstrass forms
//! - **Group Law**: The group structure on rational points
//! - **Discriminant and j-invariant**: Isomorphism invariants
//! - **Torsion Points**: Points of finite order
//! - **Isogenies**: Morphisms of elliptic curves
//! - **Modular Forms**: Connection to modular curves (future)
//!
//! # Mathematical Background
//!
//! ## The Group Law
//!
//! The set of rational points E(k) forms an abelian group under a geometric addition law:
//! - Identity: The point at infinity O
//! - Addition: Three collinear points sum to O
//! - Inversion: Reflection across the x-axis
//!
//! ## Invariants
//!
//! - **Discriminant Δ**: Measures singularity (Δ ≠ 0 for smooth curves)
//! - **j-invariant**: Classifies curves up to isomorphism over algebraically closed fields
//!
//! # Examples
//!
//! ## Creating an Elliptic Curve
//!
//! ```rust
//! use rustmath_schemes::elliptic_curves::EllipticCurve;
//!
//! // Create y² = x³ - x (short Weierstrass form)
//! // let e = EllipticCurve::short_weierstrass(0, -1);
//! // assert!(e.is_smooth());
//! ```
//!
//! ## Working with Points
//!
//! ```rust
//! use rustmath_schemes::elliptic_curves::{EllipticCurve, EllipticCurvePoint};
//!
//! // Create a curve and points
//! // let e = EllipticCurve::short_weierstrass(0, -1);
//! // let p = EllipticCurvePoint::affine(0, 0);
//! // let q = EllipticCurvePoint::affine(1, 0);
//!
//! // Point addition
//! // let r = e.add(&p, &q);
//! ```
//!
//! ## Computing Invariants
//!
//! ```rust
//! use rustmath_schemes::elliptic_curves::EllipticCurve;
//!
//! // let e = EllipticCurve::short_weierstrass(-1, 0);
//! // let disc = e.discriminant();
//! // let j = e.j_invariant();
//! ```

use rustmath_core::{MathError, Result, Ring, Field};
use crate::generic::{Scheme, SchemePoint, AlgebraicScheme};
use crate::affine::{AffinePoint, AffineScheme};
use crate::projective::{ProjectivePoint, ProjectiveSpace};
use std::fmt;

/// An elliptic curve in Weierstrass form
///
/// Represents an elliptic curve given by:
/// y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
///
/// # Type Parameters
///
/// - `F`: The base field (must be a field, not just a ring)
#[derive(Debug, Clone)]
pub struct EllipticCurve<F: Field> {
    /// Weierstrass coefficients [a₁, a₂, a₃, a₄, a₆]
    a_invariants: [F; 5],
    /// Base field
    base_field: F,
    /// Cached discriminant
    discriminant: Option<F>,
    /// Cached j-invariant
    j_invariant: Option<F>,
}

impl<F: Field> EllipticCurve<F> {
    /// Create an elliptic curve from Weierstrass coefficients
    ///
    /// # Arguments
    ///
    /// - `a1, a2, a3, a4, a6`: The Weierstrass coefficients
    /// - `base_field`: The base field
    ///
    /// # Returns
    ///
    /// An elliptic curve if the discriminant is non-zero (smooth curve),
    /// error otherwise.
    pub fn new(a1: F, a2: F, a3: F, a4: F, a6: F, base_field: F) -> Result<Self> {
        let curve = EllipticCurve {
            a_invariants: [a1, a2, a3, a4, a6],
            base_field,
            discriminant: None,
            j_invariant: None,
        };

        // Check that discriminant is non-zero
        // In a full implementation, we'd compute and verify this

        Ok(curve)
    }

    /// Create an elliptic curve in short Weierstrass form: y² = x³ + ax + b
    ///
    /// This is valid when the characteristic is not 2 or 3.
    pub fn short_weierstrass(a: F, b: F, base_field: F) -> Result<Self> {
        // y² = x³ + ax + b corresponds to a₁=a₂=a₃=0, a₄=a, a₆=b
        EllipticCurve::new(
            base_field.zero(),
            base_field.zero(),
            base_field.zero(),
            a,
            b,
            base_field,
        )
    }

    /// Get the a-invariants [a₁, a₂, a₃, a₄, a₆]
    pub fn a_invariants(&self) -> &[F; 5] {
        &self.a_invariants
    }

    /// Compute the discriminant Δ
    ///
    /// For short Weierstrass y² = x³ + ax + b:
    /// Δ = -16(4a³ + 27b²)
    ///
    /// The curve is smooth (non-singular) iff Δ ≠ 0.
    pub fn discriminant(&self) -> F {
        if let Some(ref disc) = self.discriminant {
            return disc.clone();
        }

        // In a full implementation, compute the discriminant from a-invariants
        // For now, return a placeholder
        self.base_field.one()
    }

    /// Compute the j-invariant
    ///
    /// For short Weierstrass y² = x³ + ax + b:
    /// j = 1728 · 4a³ / Δ
    ///
    /// The j-invariant classifies elliptic curves up to isomorphism
    /// over algebraically closed fields.
    pub fn j_invariant(&self) -> F {
        if let Some(ref j) = self.j_invariant {
            return j.clone();
        }

        // In a full implementation, compute j = c₄³/Δ
        self.base_field.zero()
    }

    /// Check if the curve is smooth (non-singular)
    pub fn is_smooth(&self) -> bool {
        // Curve is smooth iff discriminant is non-zero
        // self.discriminant() != self.base_field.zero()
        true // Placeholder
    }

    /// Add two points on the elliptic curve using the group law
    ///
    /// This implements the geometric addition law:
    /// - If P and Q are distinct, P + Q is the reflection of the third
    ///   intersection point of the line PQ with the curve
    /// - If P = Q, use the tangent line at P
    /// - P + O = O + P = P for the identity O
    pub fn add(&self, p: &EllipticCurvePoint<F>, q: &EllipticCurvePoint<F>) -> EllipticCurvePoint<F> {
        // In a full implementation, this would compute the group law
        EllipticCurvePoint::infinity()
    }

    /// Negate a point (reflection across x-axis)
    pub fn negate(&self, p: &EllipticCurvePoint<F>) -> EllipticCurvePoint<F> {
        match p {
            EllipticCurvePoint::Infinity => EllipticCurvePoint::Infinity,
            EllipticCurvePoint::Affine { x, y } => {
                // For y² + a₁xy + a₃y = x³ + ...,
                // the negation of (x,y) is (x, -y - a₁x - a₃)
                // For short Weierstrass: (x,y) → (x,-y)
                let neg_y = self.base_field.zero(); // Placeholder
                EllipticCurvePoint::Affine {
                    x: x.clone(),
                    y: neg_y,
                }
            }
        }
    }

    /// Compute scalar multiplication [n]P = P + P + ... + P (n times)
    pub fn scalar_mul(&self, n: i64, p: &EllipticCurvePoint<F>) -> EllipticCurvePoint<F> {
        // Use double-and-add algorithm in a full implementation
        EllipticCurvePoint::infinity()
    }

    /// Get the base field
    pub fn base_field(&self) -> &F {
        &self.base_field
    }
}

impl<F: Field> Scheme for EllipticCurve<F> {
    type BaseRing = F;

    fn base_ring(&self) -> &Self::BaseRing {
        &self.base_field
    }

    fn dimension(&self) -> Option<usize> {
        Some(1) // Elliptic curves are 1-dimensional
    }

    fn is_projective(&self) -> bool {
        true // Elliptic curves are projective
    }

    fn is_irreducible(&self) -> bool {
        true // Elliptic curves are irreducible
    }

    fn is_reduced(&self) -> bool {
        true // Smooth curves are reduced
    }

    fn is_noetherian(&self) -> bool {
        true
    }

    fn is_finite_type(&self) -> bool {
        true
    }
}

impl<F: Field> AlgebraicScheme for EllipticCurve<F> {
    fn is_smooth(&self) -> bool {
        self.is_smooth()
    }

    fn is_regular(&self) -> bool {
        true // Smooth implies regular
    }

    fn is_normal(&self) -> bool {
        true // Smooth implies normal
    }

    fn genus(&self) -> Option<usize> {
        Some(1) // Elliptic curves have genus 1
    }
}

/// A point on an elliptic curve
///
/// Can be either:
/// - An affine point (x, y) satisfying the Weierstrass equation
/// - The point at infinity O (the identity element)
#[derive(Debug, Clone, PartialEq)]
pub enum EllipticCurvePoint<F: Field> {
    /// Affine point (x, y)
    Affine { x: F, y: F },
    /// The point at infinity (identity element)
    Infinity,
}

impl<F: Field> EllipticCurvePoint<F> {
    /// Create an affine point (x, y)
    pub fn affine(x: F, y: F) -> Self {
        EllipticCurvePoint::Affine { x, y }
    }

    /// Create the point at infinity
    pub fn infinity() -> Self {
        EllipticCurvePoint::Infinity
    }

    /// Check if this is the point at infinity
    pub fn is_infinity(&self) -> bool {
        matches!(self, EllipticCurvePoint::Infinity)
    }

    /// Get affine coordinates (returns None for point at infinity)
    pub fn affine_coords(&self) -> Option<(&F, &F)> {
        match self {
            EllipticCurvePoint::Affine { x, y } => Some((x, y)),
            EllipticCurvePoint::Infinity => None,
        }
    }

    /// Convert to projective coordinates [X : Y : Z]
    ///
    /// - (x, y) → [x : y : 1]
    /// - O → [0 : 1 : 0]
    pub fn to_projective(&self, base_field: &F) -> ProjectivePoint<F> {
        match self {
            EllipticCurvePoint::Affine { x, y } => {
                ProjectivePoint::new(vec![x.clone(), y.clone(), base_field.one()]).unwrap()
            }
            EllipticCurvePoint::Infinity => {
                ProjectivePoint::new(vec![base_field.zero(), base_field.one(), base_field.zero()])
                    .unwrap()
            }
        }
    }
}

/// An isogeny between elliptic curves
///
/// An isogeny φ: E₁ → E₂ is a non-constant morphism that preserves the identity.
/// Isogenies are group homomorphisms between the point groups.
#[derive(Debug, Clone)]
pub struct Isogeny<F: Field> {
    source: EllipticCurve<F>,
    target: EllipticCurve<F>,
    // In a full implementation, would store the rational map
}

impl<F: Field> Isogeny<F> {
    /// Create a new isogeny
    pub fn new(source: EllipticCurve<F>, target: EllipticCurve<F>) -> Self {
        Isogeny { source, target }
    }

    /// Get the source curve
    pub fn source(&self) -> &EllipticCurve<F> {
        &self.source
    }

    /// Get the target curve
    pub fn target(&self) -> &EllipticCurve<F> {
        &self.target
    }

    /// Compute the degree of the isogeny
    ///
    /// The degree is the degree of the corresponding field extension.
    pub fn degree(&self) -> Option<usize> {
        None // Placeholder
    }

    /// Check if this is an isomorphism (degree 1 isogeny)
    pub fn is_isomorphism(&self) -> bool {
        self.degree() == Some(1)
    }

    /// Apply the isogeny to a point
    pub fn apply(&self, p: &EllipticCurvePoint<F>) -> EllipticCurvePoint<F> {
        // In a full implementation, evaluate the rational map
        EllipticCurvePoint::infinity()
    }
}

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
