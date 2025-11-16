//! Jacobi Elliptic Functions
//!
//! This module implements Jacobi elliptic functions, which generalize
//! trigonometric functions and arise in the theory of elliptic integrals.
//!
//! Corresponds to sage.functions.jacobi
//!
//! # Functions
//!
//! - `jacobi_sn(u, m)`: Jacobi elliptic function sn(u|m)
//! - `jacobi_cn(u, m)`: Jacobi elliptic function cn(u|m)
//! - `jacobi_dn(u, m)`: Jacobi elliptic function dn(u|m)
//! - `jacobi_am(u, m)`: Jacobi amplitude function am(u|m)
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::jacobi::*;
//! use rustmath_symbolic::{Expr, Symbol};
//!
//! let u = Symbol::new("u");
//! let m = Expr::from(1) / Expr::from(2);
//! let sn = jacobi_sn(&Expr::Symbol(u), &m);
//! ```
//!
//! # Mathematical Background
//!
//! Jacobi elliptic functions are doubly periodic functions that generalize
//! the trigonometric functions. They are defined using elliptic integrals:
//!
//! u = ∫₀^φ dθ/√(1 - m sin²θ)
//!
//! Then:
//! - sn(u|m) = sin(φ)
//! - cn(u|m) = cos(φ)
//! - dn(u|m) = √(1 - m sin²φ)
//!
//! When m = 0: sn(u|0) = sin(u), cn(u|0) = cos(u), dn(u|0) = 1
//! When m = 1: sn(u|1) = tanh(u), cn(u|1) = sech(u), dn(u|1) = sech(u)

use crate::expression::Expr;
use std::sync::Arc;

/// Jacobi elliptic function sn(u|m)
///
/// Generalizes the sine function.
///
/// # Arguments
///
/// * `u` - The argument
/// * `m` - The parameter (0 ≤ m ≤ 1)
///
/// # Returns
///
/// sn(u|m)
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::jacobi::jacobi_sn;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let u = Symbol::new("u");
/// let sn = jacobi_sn(&Expr::Symbol(u), &Expr::from(0));
/// ```
///
/// # Properties
///
/// - sn(0|m) = 0
/// - sn(u|0) = sin(u)
/// - sn(u|1) = tanh(u)
/// - sn²(u) + cn²(u) = 1
/// - dn²(u) + m·sn²(u) = 1
pub fn jacobi_sn(u: &Expr, m: &Expr) -> Expr {
    Expr::Function(
        "jacobi_sn".to_string(),
        vec![Arc::new(u.clone()), Arc::new(m.clone())],
    )
}

/// Jacobi elliptic function cn(u|m)
///
/// Generalizes the cosine function.
///
/// # Arguments
///
/// * `u` - The argument
/// * `m` - The parameter (0 ≤ m ≤ 1)
///
/// # Returns
///
/// cn(u|m)
///
/// # Properties
///
/// - cn(0|m) = 1
/// - cn(u|0) = cos(u)
/// - cn(u|1) = sech(u)
/// - sn²(u) + cn²(u) = 1
pub fn jacobi_cn(u: &Expr, m: &Expr) -> Expr {
    Expr::Function(
        "jacobi_cn".to_string(),
        vec![Arc::new(u.clone()), Arc::new(m.clone())],
    )
}

/// Jacobi elliptic function dn(u|m)
///
/// Delta amplitude function.
///
/// # Arguments
///
/// * `u` - The argument
/// * `m` - The parameter (0 ≤ m ≤ 1)
///
/// # Returns
///
/// dn(u|m)
///
/// # Properties
///
/// - dn(0|m) = 1
/// - dn(u|0) = 1
/// - dn(u|1) = sech(u)
/// - dn²(u) + m·sn²(u) = 1
pub fn jacobi_dn(u: &Expr, m: &Expr) -> Expr {
    Expr::Function(
        "jacobi_dn".to_string(),
        vec![Arc::new(u.clone()), Arc::new(m.clone())],
    )
}

/// Jacobi amplitude function am(u|m)
///
/// The amplitude φ such that u = ∫₀^φ dθ/√(1 - m sin²θ)
///
/// # Arguments
///
/// * `u` - The argument
/// * `m` - The parameter
///
/// # Returns
///
/// am(u|m) = φ
///
/// # Properties
///
/// - am(0|m) = 0
/// - sn(u|m) = sin(am(u|m))
/// - cn(u|m) = cos(am(u|m))
pub fn jacobi_am(u: &Expr, m: &Expr) -> Expr {
    Expr::Function(
        "jacobi_am".to_string(),
        vec![Arc::new(u.clone()), Arc::new(m.clone())],
    )
}

/// Inverse Jacobi elliptic function
///
/// Computes the inverse of a Jacobi elliptic function.
///
/// # Arguments
///
/// * `func_name` - Name of the function ("sn", "cn", or "dn")
/// * `x` - The value
/// * `m` - The parameter
///
/// # Returns
///
/// Inverse function value
pub fn inverse_jacobi(func_name: &str, x: &Expr, m: &Expr) -> Expr {
    let name = format!("jacobi_{}_inverse", func_name);
    Expr::Function(name, vec![Arc::new(x.clone()), Arc::new(m.clone())])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_jacobi_sn_symbolic() {
        let u = Symbol::new("u");
        let sn = jacobi_sn(&Expr::Symbol(u), &Expr::from(0));
        assert!(matches!(sn, Expr::Function(name, _) if name == "jacobi_sn"));
    }

    #[test]
    fn test_jacobi_cn_symbolic() {
        let u = Symbol::new("u");
        let cn = jacobi_cn(&Expr::Symbol(u), &Expr::from(0));
        assert!(matches!(cn, Expr::Function(name, _) if name == "jacobi_cn"));
    }

    #[test]
    fn test_jacobi_dn_symbolic() {
        let u = Symbol::new("u");
        let dn = jacobi_dn(&Expr::Symbol(u), &Expr::from(0));
        assert!(matches!(dn, Expr::Function(name, _) if name == "jacobi_dn"));
    }

    #[test]
    fn test_jacobi_am_symbolic() {
        let u = Symbol::new("u");
        let am = jacobi_am(&Expr::Symbol(u), &Expr::from(0));
        assert!(matches!(am, Expr::Function(name, _) if name == "jacobi_am"));
    }

    #[test]
    fn test_inverse_jacobi() {
        let x = Symbol::new("x");
        let inv = inverse_jacobi("sn", &Expr::Symbol(x), &Expr::from(0));
        assert!(matches!(inv, Expr::Function(name, _) if name == "jacobi_sn_inverse"));
    }

    #[test]
    fn test_jacobi_different_params() {
        let u = Symbol::new("u");
        let sn1 = jacobi_sn(&Expr::Symbol(u.clone()), &Expr::from(0));
        let sn2 = jacobi_sn(&Expr::Symbol(u), &Expr::from(1));
        assert_ne!(sn1, sn2);
    }
}
