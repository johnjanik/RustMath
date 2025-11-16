//! Special Functions - Elliptic Integrals and Spherical Harmonics
//!
//! This module implements various special functions including elliptic integrals
//! and spherical harmonics.
//!
//! Corresponds to sage.functions.special
//!
//! # Functions
//!
//! ## Elliptic Integrals
//!
//! - `elliptic_e(phi, m)`: Incomplete elliptic integral E(φ|m)
//! - `elliptic_ec(m)`: Complete elliptic integral E(m) = E(π/2|m)
//! - `elliptic_eu(u, m)`: Jacobi's form of elliptic integral E
//! - `elliptic_f(phi, m)`: Incomplete elliptic integral F(φ|m)
//! - `elliptic_kc(m)`: Complete elliptic integral K(m) = F(π/2|m)
//! - `elliptic_pi(n, phi, m)`: Incomplete elliptic integral Π(n;φ|m)
//! - `elliptic_j(z)`: Elliptic modular j-invariant
//!
//! ## Spherical Harmonics
//!
//! - `spherical_harmonic(m, n, theta, phi)`: Spherical harmonic Y_n^m(θ, φ)
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::special::*;
//! use rustmath_symbolic::{Expr, Symbol};
//!
//! let phi = Symbol::new("phi");
//! let m = Expr::from(1) / Expr::from(2);
//! let e = elliptic_e(&Expr::Symbol(phi), &m);
//! ```
//!
//! # Mathematical Background
//!
//! Elliptic integrals are functions that arise in the calculation of arc lengths
//! of ellipses and in many other applications. They cannot be expressed in terms
//! of elementary functions.
//!
//! The incomplete elliptic integral of the first kind is:
//! F(φ|m) = ∫₀^φ dθ/√(1 - m sin²θ)
//!
//! The incomplete elliptic integral of the second kind is:
//! E(φ|m) = ∫₀^φ √(1 - m sin²θ) dθ
//!
//! The incomplete elliptic integral of the third kind is:
//! Π(n;φ|m) = ∫₀^φ dθ/[(1 + n sin²θ)√(1 - m sin²θ)]

use crate::expression::Expr;
use std::sync::Arc;

/// Incomplete elliptic integral of the second kind E(φ|m)
///
/// # Arguments
///
/// * `phi` - The amplitude φ
/// * `m` - The parameter m (0 ≤ m ≤ 1)
///
/// # Returns
///
/// E(φ|m) = ∫₀^φ √(1 - m sin²θ) dθ
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::special::elliptic_e;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let phi = Symbol::new("phi");
/// let e = elliptic_e(&Expr::Symbol(phi), &Expr::from(0));
/// ```
///
/// # Properties
///
/// - E(0|m) = 0
/// - E(π/2|m) = E(m) (complete elliptic integral)
/// - E(φ|0) = φ
/// - E(φ|1) = sin(φ)
pub fn elliptic_e(phi: &Expr, m: &Expr) -> Expr {
    Expr::Function(
        "elliptic_e".to_string(),
        vec![Arc::new(phi.clone()), Arc::new(m.clone())],
    )
}

/// Complete elliptic integral of the second kind E(m)
///
/// This is E(π/2|m), the complete elliptic integral.
///
/// # Arguments
///
/// * `m` - The parameter m (0 ≤ m ≤ 1)
///
/// # Returns
///
/// E(m) = E(π/2|m) = ∫₀^(π/2) √(1 - m sin²θ) dθ
///
/// # Properties
///
/// - E(0) = π/2
/// - E(1) = 1
/// - As m → 0, E(m) → π/2
/// - As m → 1, E(m) → 1
pub fn elliptic_ec(m: &Expr) -> Expr {
    Expr::Function("elliptic_ec".to_string(), vec![Arc::new(m.clone())])
}

/// Jacobi's form of the elliptic integral E
///
/// # Arguments
///
/// * `u` - The argument
/// * `m` - The parameter
///
/// # Returns
///
/// E(u, m) in Jacobi's notation
pub fn elliptic_eu(u: &Expr, m: &Expr) -> Expr {
    Expr::Function(
        "elliptic_eu".to_string(),
        vec![Arc::new(u.clone()), Arc::new(m.clone())],
    )
}

/// Incomplete elliptic integral of the first kind F(φ|m)
///
/// # Arguments
///
/// * `phi` - The amplitude φ
/// * `m` - The parameter m (0 ≤ m ≤ 1)
///
/// # Returns
///
/// F(φ|m) = ∫₀^φ dθ/√(1 - m sin²θ)
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::special::elliptic_f;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let phi = Symbol::new("phi");
/// let f = elliptic_f(&Expr::Symbol(phi), &Expr::from(0));
/// ```
///
/// # Properties
///
/// - F(0|m) = 0
/// - F(π/2|m) = K(m) (complete elliptic integral)
/// - F(φ|0) = φ
/// - F(φ|1) = ln|sec φ + tan φ|
pub fn elliptic_f(phi: &Expr, m: &Expr) -> Expr {
    Expr::Function(
        "elliptic_f".to_string(),
        vec![Arc::new(phi.clone()), Arc::new(m.clone())],
    )
}

/// Complete elliptic integral of the first kind K(m)
///
/// This is F(π/2|m), the complete elliptic integral.
///
/// # Arguments
///
/// * `m` - The parameter m (0 ≤ m ≤ 1)
///
/// # Returns
///
/// K(m) = F(π/2|m) = ∫₀^(π/2) dθ/√(1 - m sin²θ)
///
/// # Properties
///
/// - K(0) = π/2
/// - K(1) = ∞
/// - K'(m) = K(1-m) (complementary integral)
/// - As m → 0, K(m) → π/2
/// - As m → 1⁻, K(m) → ∞
pub fn elliptic_kc(m: &Expr) -> Expr {
    Expr::Function("elliptic_kc".to_string(), vec![Arc::new(m.clone())])
}

/// Incomplete elliptic integral of the third kind Π(n;φ|m)
///
/// # Arguments
///
/// * `n` - The characteristic n
/// * `phi` - The amplitude φ
/// * `m` - The parameter m
///
/// # Returns
///
/// Π(n;φ|m) = ∫₀^φ dθ/[(1 + n sin²θ)√(1 - m sin²θ)]
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::special::elliptic_pi;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let phi = Symbol::new("phi");
/// let pi_integral = elliptic_pi(&Expr::from(1), &Expr::Symbol(phi), &Expr::from(0));
/// ```
///
/// # Properties
///
/// - Π(0;φ|m) = F(φ|m)
/// - Π(n;0|m) = 0
/// - Reduces to F when n = 0
pub fn elliptic_pi(n: &Expr, phi: &Expr, m: &Expr) -> Expr {
    Expr::Function(
        "elliptic_pi".to_string(),
        vec![
            Arc::new(n.clone()),
            Arc::new(phi.clone()),
            Arc::new(m.clone()),
        ],
    )
}

/// Elliptic modular j-invariant
///
/// The j-invariant is a modular function of weight 0 for SL₂(ℤ) in the theory
/// of elliptic functions.
///
/// # Arguments
///
/// * `tau` - The argument (typically in the upper half-plane)
///
/// # Returns
///
/// j(τ) = 1728 · g₂³/(g₂³ - 27g₃²)
///
/// where g₂ and g₃ are Eisenstein series
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::special::elliptic_j;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let tau = Symbol::new("tau");
/// let j = elliptic_j(&Expr::Symbol(tau));
/// ```
///
/// # Properties
///
/// - j(i) = 1728 (at τ = i)
/// - j(e^(2πi/3)) = 0
/// - j is invariant under the modular group SL₂(ℤ)
pub fn elliptic_j(tau: &Expr) -> Expr {
    Expr::Function("elliptic_j".to_string(), vec![Arc::new(tau.clone())])
}

/// Spherical harmonic Y_n^m(θ, φ)
///
/// Spherical harmonics are special functions defined on the surface of a sphere.
/// They arise naturally in solving Laplace's equation in spherical coordinates.
///
/// # Arguments
///
/// * `m` - The azimuthal (magnetic) quantum number (-n ≤ m ≤ n)
/// * `n` - The degree (principal quantum number, n ≥ 0)
/// * `theta` - The polar angle θ (0 ≤ θ ≤ π)
/// * `phi` - The azimuthal angle φ (0 ≤ φ < 2π)
///
/// # Returns
///
/// Y_n^m(θ, φ) = √[(2n+1)/(4π) · (n-m)!/(n+m)!] · P_n^m(cos θ) · e^(imφ)
///
/// where P_n^m is the associated Legendre polynomial
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::special::spherical_harmonic;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let theta = Symbol::new("theta");
/// let phi = Symbol::new("phi");
/// let y = spherical_harmonic(&Expr::from(0), &Expr::from(1),
///                            &Expr::Symbol(theta), &Expr::Symbol(phi));
/// ```
///
/// # Properties
///
/// - Y_n^m is an eigenfunction of the angular part of the Laplacian
/// - The spherical harmonics form an orthonormal basis for L²(S²)
/// - Y_n^(-m) = (-1)^m · conj(Y_n^m)
/// - Y_0^0 = 1/√(4π) (constant function)
///
/// # Applications
///
/// - Solutions to Schrödinger equation for hydrogen atom
/// - Multipole expansions in electromagnetism and gravitation
/// - Computer graphics and 3D modeling
pub fn spherical_harmonic(m: &Expr, n: &Expr, theta: &Expr, phi: &Expr) -> Expr {
    Expr::Function(
        "spherical_harmonic".to_string(),
        vec![
            Arc::new(m.clone()),
            Arc::new(n.clone()),
            Arc::new(theta.clone()),
            Arc::new(phi.clone()),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_elliptic_e_symbolic() {
        let phi = Symbol::new("phi");
        let e = elliptic_e(&Expr::Symbol(phi), &Expr::from(0));
        assert!(matches!(e, Expr::Function(name, _) if name == "elliptic_e"));
    }

    #[test]
    fn test_elliptic_ec_symbolic() {
        let e = elliptic_ec(&Expr::from(0));
        assert!(matches!(e, Expr::Function(name, args)
            if name == "elliptic_ec" && args.len() == 1));
    }

    #[test]
    fn test_elliptic_eu_symbolic() {
        let u = Symbol::new("u");
        let e = elliptic_eu(&Expr::Symbol(u), &Expr::from(0));
        assert!(matches!(e, Expr::Function(name, _) if name == "elliptic_eu"));
    }

    #[test]
    fn test_elliptic_f_symbolic() {
        let phi = Symbol::new("phi");
        let f = elliptic_f(&Expr::Symbol(phi), &Expr::from(0));
        assert!(matches!(f, Expr::Function(name, _) if name == "elliptic_f"));
    }

    #[test]
    fn test_elliptic_kc_symbolic() {
        let k = elliptic_kc(&Expr::from(0));
        assert!(matches!(k, Expr::Function(name, args)
            if name == "elliptic_kc" && args.len() == 1));
    }

    #[test]
    fn test_elliptic_pi_symbolic() {
        let phi = Symbol::new("phi");
        let pi_integral = elliptic_pi(&Expr::from(1), &Expr::Symbol(phi), &Expr::from(0));
        assert!(matches!(pi_integral, Expr::Function(name, args)
            if name == "elliptic_pi" && args.len() == 3));
    }

    #[test]
    fn test_elliptic_j_symbolic() {
        let tau = Symbol::new("tau");
        let j = elliptic_j(&Expr::Symbol(tau));
        assert!(matches!(j, Expr::Function(name, _) if name == "elliptic_j"));
    }

    #[test]
    fn test_spherical_harmonic_symbolic() {
        let theta = Symbol::new("theta");
        let phi = Symbol::new("phi");
        let y = spherical_harmonic(&Expr::from(0), &Expr::from(1),
                                   &Expr::Symbol(theta), &Expr::Symbol(phi));
        assert!(matches!(y, Expr::Function(name, args)
            if name == "spherical_harmonic" && args.len() == 4));
    }

    #[test]
    fn test_spherical_harmonic_params() {
        let theta = Symbol::new("theta");
        let phi = Symbol::new("phi");

        let y1 = spherical_harmonic(&Expr::from(0), &Expr::from(0),
                                    &Expr::Symbol(theta.clone()), &Expr::Symbol(phi.clone()));
        let y2 = spherical_harmonic(&Expr::from(1), &Expr::from(1),
                                    &Expr::Symbol(theta), &Expr::Symbol(phi));

        assert_ne!(y1, y2);
    }

    #[test]
    fn test_elliptic_integrals_different() {
        let phi = Symbol::new("phi");

        let e = elliptic_e(&Expr::Symbol(phi.clone()), &Expr::from(0));
        let f = elliptic_f(&Expr::Symbol(phi), &Expr::from(0));

        assert_ne!(e, f);
    }
}
