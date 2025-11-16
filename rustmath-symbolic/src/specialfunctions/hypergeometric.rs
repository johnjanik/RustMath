//! Hypergeometric Functions
//!
//! This module implements hypergeometric functions, which are solutions to the
//! hypergeometric differential equation and generalize many elementary and special functions.
//!
//! Corresponds to sage.functions.hypergeometric
//!
//! # Functions
//!
//! - `hypergeometric(a, b, z)`: Generalized hypergeometric function ₚFₑ
//! - `hypergeometric_m(a, b, z)`: Confluent hypergeometric function M(a,b,z) = ₁F₁(a;b;z)
//! - `hypergeometric_u(a, b, z)`: Confluent hypergeometric function U(a,b,z)
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::hypergeometric::*;
//! use rustmath_symbolic::{Expr, Symbol};
//!
//! let z = Symbol::new("z");
//! let hyp = hypergeometric_m(&Expr::from(1), &Expr::from(2), &Expr::Symbol(z));
//! ```
//!
//! # Mathematical Background
//!
//! The generalized hypergeometric function ₚFₑ is defined by:
//! ₚFₑ(a₁,...,aₚ; b₁,...,bₑ; z) = Σ(n=0 to ∞) [(a₁)ₙ...(aₚ)ₙ / (b₁)ₙ...(bₑ)ₙ] zⁿ/n!
//!
//! where (a)ₙ = a(a+1)...(a+n-1) is the Pochhammer symbol (rising factorial).
//!
//! Many special functions are special cases:
//! - eˣ = ₀F₀(;;x)
//! - (1-x)⁻ᵃ = ₁F₀(a;;x)
//! - Bessel functions can be expressed using ₀F₁

use crate::expression::Expr;
use std::sync::Arc;

/// Generalized hypergeometric function ₚFₑ(a; b; z)
///
/// Returns a symbolic representation of the hypergeometric function.
///
/// # Arguments
///
/// * `a` - Vector of upper parameters [a₁, a₂, ..., aₚ]
/// * `b` - Vector of lower parameters [b₁, b₂, ..., bₑ]
/// * `z` - The argument
///
/// # Returns
///
/// Symbolic hypergeometric function
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::hypergeometric::hypergeometric;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let x = Symbol::new("x");
/// // ₂F₁(1, 2; 3; x)
/// let hyp = hypergeometric(
///     &[Expr::from(1), Expr::from(2)],
///     &[Expr::from(3)],
///     &Expr::Symbol(x)
/// );
/// ```
///
/// # Special Cases
///
/// - ₀F₀(;;z) = eᶻ
/// - ₁F₀(a;;z) = (1-z)⁻ᵃ
/// - ₁F₁(1;2;z) = (eᶻ - 1)/z
pub fn hypergeometric(a: &[Expr], b: &[Expr], z: &Expr) -> Expr {
    // Create nested list structures to preserve parameter grouping
    // This ensures ₚFᵩ(a₁,...,aₚ; b₁,...,bᵩ; z) maintains distinction between p and q
    let a_list = Expr::Function(
        "list".to_string(),
        a.iter().map(|e| Arc::new(e.clone())).collect(),
    );
    let b_list = Expr::Function(
        "list".to_string(),
        b.iter().map(|e| Arc::new(e.clone())).collect(),
    );

    Expr::Function(
        "hypergeometric".to_string(),
        vec![Arc::new(a_list), Arc::new(b_list), Arc::new(z.clone())],
    )
}

/// Confluent hypergeometric function M(a, b, z) = ₁F₁(a; b; z)
///
/// Also known as Kummer's function. This is a solution to Kummer's differential equation:
/// z·d²w/dz² + (b - z)·dw/dz - a·w = 0
///
/// # Arguments
///
/// * `a` - Upper parameter
/// * `b` - Lower parameter
/// * `z` - The argument
///
/// # Returns
///
/// Symbolic confluent hypergeometric function M
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::hypergeometric::hypergeometric_m;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let z = Symbol::new("z");
/// let m = hypergeometric_m(&Expr::from(1), &Expr::from(2), &Expr::Symbol(z));
/// ```
///
/// # Properties
///
/// - M(a, b, 0) = 1
/// - M(0, b, z) = 1
/// - M(a, a, z) = eᶻ
/// - M(1, 2, z) = (eᶻ - 1)/z
///
/// # Relation to Other Functions
///
/// - eᶻ = M(a, a, z)
/// - Bessel functions: Iᵥ(z) ∝ M(ν+1/2, 2ν+1, 2z)
/// - Error function: erf(z) ∝ z·M(1/2, 3/2, -z²)
pub fn hypergeometric_m(a: &Expr, b: &Expr, z: &Expr) -> Expr {
    Expr::Function(
        "hypergeometric_m".to_string(),
        vec![Arc::new(a.clone()), Arc::new(b.clone()), Arc::new(z.clone())],
    )
}

/// Confluent hypergeometric function U(a, b, z)
///
/// Also known as Tricomi's function. This is a second linearly independent
/// solution to Kummer's differential equation.
///
/// # Arguments
///
/// * `a` - Upper parameter
/// * `b` - Lower parameter
/// * `z` - The argument
///
/// # Returns
///
/// Symbolic confluent hypergeometric function U
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::hypergeometric::hypergeometric_u;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let z = Symbol::new("z");
/// let u = hypergeometric_u(&Expr::from(1), &Expr::from(2), &Expr::Symbol(z));
/// ```
///
/// # Properties
///
/// - U(a, b, z) ~ z⁻ᵃ as z → ∞
/// - W(M, U) = Γ(b)/Γ(a) where W is the Wronskian
///
/// # Relation to M
///
/// U and M are related by:
/// U(a,b,z) = π/sin(πb) · [M(a,b,z)/Γ(1+a-b)Γ(b) - z^(1-b)M(1+a-b,2-b,z)/Γ(a)Γ(2-b)]
pub fn hypergeometric_u(a: &Expr, b: &Expr, z: &Expr) -> Expr {
    Expr::Function(
        "hypergeometric_u".to_string(),
        vec![Arc::new(a.clone()), Arc::new(b.clone()), Arc::new(z.clone())],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_hypergeometric_symbolic() {
        let x = Symbol::new("x");

        let hyp = hypergeometric(
            &[Expr::from(1), Expr::from(2)],
            &[Expr::from(3)],
            &Expr::Symbol(x),
        );

        assert!(matches!(hyp, Expr::Function(name, _) if name == "hypergeometric"));
    }

    #[test]
    fn test_hypergeometric_m_symbolic() {
        let z = Symbol::new("z");
        let m = hypergeometric_m(&Expr::from(1), &Expr::from(2), &Expr::Symbol(z));

        assert!(matches!(m, Expr::Function(name, args)
            if name == "hypergeometric_m" && args.len() == 3));
    }

    #[test]
    fn test_hypergeometric_u_symbolic() {
        let z = Symbol::new("z");
        let u = hypergeometric_u(&Expr::from(1), &Expr::from(2), &Expr::Symbol(z));

        assert!(matches!(u, Expr::Function(name, args)
            if name == "hypergeometric_u" && args.len() == 3));
    }

    #[test]
    fn test_hypergeometric_different_params() {
        let x = Symbol::new("x");

        let hyp1 = hypergeometric(&[Expr::from(1)], &[Expr::from(2)], &Expr::Symbol(x.clone()));
        let hyp2 = hypergeometric(&[Expr::from(1), Expr::from(2)], &[], &Expr::Symbol(x));

        assert_ne!(hyp1, hyp2);
    }
}
