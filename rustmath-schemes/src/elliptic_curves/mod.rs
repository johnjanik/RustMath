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
