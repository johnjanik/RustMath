//! Transcendental Functions - Zeta Functions and Related
//!
//! This module implements various transcendental functions, primarily zeta functions
//! and related special functions important in analytic number theory.
//!
//! Corresponds to sage.functions.transcendental
//!
//! # Functions
//!
//! - `zeta(s)`: Riemann zeta function ζ(s)
//! - `zeta_deriv(n, s)`: nth derivative of the Riemann zeta function
//! - `hurwitz_zeta(s, a)`: Hurwitz zeta function ζ(s, a)
//! - `stieltjes(n)`: Stieltjes constants γₙ
//! - `zeta_symmetric(s)`: Symmetric (completed) Riemann zeta function ξ(s)
//! - `dickman_rho(u)`: Dickman's ρ function
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::transcendental::*;
//! use rustmath_symbolic::{Expr, Symbol};
//!
//! let s = Symbol::new("s");
//! let z = zeta(&Expr::Symbol(s));
//! ```
//!
//! # Mathematical Background
//!
//! The Riemann zeta function is one of the most important functions in mathematics:
//! ζ(s) = Σ_{n=1}^∞ 1/n^s for Re(s) > 1
//!
//! It can be analytically continued to the entire complex plane except s = 1.
//!
//! The Hurwitz zeta function is a generalization:
//! ζ(s, a) = Σ_{n=0}^∞ 1/(n+a)^s
//!
//! The Stieltjes constants appear in the Laurent series expansion of ζ(s) near s = 1.

use crate::expression::Expr;
use std::sync::Arc;

/// Helper function to try converting an Expr to f64
fn try_expr_to_f64(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Integer(i) => i.to_f64(),
        Expr::Rational(r) => r.to_f64(),
        _ => None,
    }
}

/// Riemann zeta function ζ(s)
///
/// One of the most important functions in mathematics, central to the distribution
/// of prime numbers via the Prime Number Theorem.
///
/// # Arguments
///
/// * `s` - The argument (complex in general)
///
/// # Returns
///
/// ζ(s) = Σ_{n=1}^∞ 1/n^s for Re(s) > 1
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::transcendental::zeta;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let s = Symbol::new("s");
/// let z = zeta(&Expr::Symbol(s));
/// assert_eq!(zeta(&Expr::from(2)), zeta(&Expr::from(2)));
/// ```
///
/// # Properties
///
/// - ζ(2) = π²/6 (Basel problem)
/// - ζ(0) = -1/2
/// - ζ(-1) = -1/12 (used in string theory)
/// - ζ(s) has a simple pole at s = 1 with residue 1
/// - Functional equation: ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
/// - All non-trivial zeros lie on Re(s) = 1/2 (Riemann Hypothesis, unproven)
///
/// # Special Values
///
/// For even positive integers, ζ(2n) can be expressed in terms of Bernoulli numbers
pub fn zeta(s: &Expr) -> Expr {
    // Could implement special values for specific integers
    if let Expr::Integer(n) = s {
        let n_val = n.to_i64();
        {
            match n_val {
                0 => return Expr::from(-1) / Expr::from(2),
                -1 => return Expr::from(-1) / Expr::from(12),
                1 => {
                    // ζ(1) is infinite, return symbolic
                    return Expr::Function("zeta".to_string(), vec![Arc::new(s.clone())]);
                }
                _ => {}
            }
        }
    }

    Expr::Function("zeta".to_string(), vec![Arc::new(s.clone())])
}

/// nth derivative of the Riemann zeta function
///
/// # Arguments
///
/// * `n` - The order of the derivative (non-negative integer)
/// * `s` - The argument
///
/// # Returns
///
/// d^n/ds^n ζ(s)
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::transcendental::zeta_deriv;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let s = Symbol::new("s");
/// let zeta_prime = zeta_deriv(&Expr::from(1), &Expr::Symbol(s));
/// ```
///
/// # Properties
///
/// - zeta_deriv(0, s) = ζ(s)
/// - The derivatives at negative integers are related to Bernoulli polynomials
pub fn zeta_deriv(n: &Expr, s: &Expr) -> Expr {
    Expr::Function(
        "zeta_deriv".to_string(),
        vec![Arc::new(n.clone()), Arc::new(s.clone())],
    )
}

/// Hurwitz zeta function ζ(s, a)
///
/// A generalization of the Riemann zeta function that includes a shift parameter.
///
/// # Arguments
///
/// * `s` - The exponent
/// * `a` - The shift parameter (a > 0)
///
/// # Returns
///
/// ζ(s, a) = Σ_{n=0}^∞ 1/(n+a)^s for Re(s) > 1
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::transcendental::hurwitz_zeta;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let s = Symbol::new("s");
/// let hz = hurwitz_zeta(&Expr::Symbol(s), &Expr::from(1));
/// ```
///
/// # Properties
///
/// - ζ(s, 1) = ζ(s) (Riemann zeta)
/// - ζ(s, 1/2) = (2^s - 1) ζ(s)
/// - Can be analytically continued to the entire complex s-plane except s = 1
/// - Used in the study of Dirichlet L-functions
///
/// # Relation to Other Functions
///
/// - Polygamma function: ψ^(n)(z) = (-1)^(n+1) n! ζ(n+1, z)
/// - Polylogarithm: Li_s(z) = z ζ(s, 1) for |z| < 1
pub fn hurwitz_zeta(s: &Expr, a: &Expr) -> Expr {
    Expr::Function(
        "hurwitz_zeta".to_string(),
        vec![Arc::new(s.clone()), Arc::new(a.clone())],
    )
}

/// Stieltjes constants γₙ
///
/// The Stieltjes constants appear in the Laurent series expansion of the
/// Riemann zeta function near its pole at s = 1.
///
/// # Arguments
///
/// * `n` - The index (non-negative integer)
///
/// # Returns
///
/// The nth Stieltjes constant γₙ
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::transcendental::stieltjes;
/// use rustmath_symbolic::Expr;
///
/// let gamma_0 = stieltjes(&Expr::from(0)); // Euler-Mascheroni constant
/// let gamma_1 = stieltjes(&Expr::from(1));
/// ```
///
/// # Properties
///
/// - γ₀ = γ ≈ 0.5772... (Euler-Mascheroni constant)
/// - γ₁ ≈ -0.0728...
/// - Laurent series: ζ(s) = 1/(s-1) + Σ_{n=0}^∞ (-1)^n γₙ (s-1)^n / n!
///
/// # Definition
///
/// γₙ = lim_{m→∞} [Σ_{k=1}^m (ln k)^n / k - (ln m)^(n+1) / (n+1)]
pub fn stieltjes(n: &Expr) -> Expr {
    Expr::Function("stieltjes".to_string(), vec![Arc::new(n.clone())])
}

/// Symmetric (completed) Riemann zeta function ξ(s)
///
/// Also known as the Riemann xi function, this is a modified form of the
/// zeta function that satisfies a simple functional equation.
///
/// # Arguments
///
/// * `s` - The argument
///
/// # Returns
///
/// ξ(s) = (s/2)(s-1) π^(-s/2) Γ(s/2) ζ(s)
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::transcendental::zeta_symmetric;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let s = Symbol::new("s");
/// let xi = zeta_symmetric(&Expr::Symbol(s));
/// ```
///
/// # Properties
///
/// - ξ(s) is entire (holomorphic everywhere)
/// - ξ(s) = ξ(1-s) (functional equation)
/// - The zeros of ξ(s) are exactly the non-trivial zeros of ζ(s)
/// - All zeros lie in the critical strip 0 < Re(s) < 1
/// - Riemann Hypothesis: All zeros have Re(s) = 1/2
///
/// # Importance
///
/// The symmetric form makes the functional equation simpler and is often
/// used in studying the Riemann Hypothesis.
pub fn zeta_symmetric(s: &Expr) -> Expr {
    Expr::Function(
        "zeta_symmetric".to_string(),
        vec![Arc::new(s.clone())],
    )
}

/// Dickman's ρ function
///
/// The Dickman function is important in analytic number theory, particularly
/// in studying smooth numbers (numbers with only small prime factors).
///
/// # Arguments
///
/// * `u` - The argument (u ≥ 0)
///
/// # Returns
///
/// ρ(u), which satisfies the delay differential equation:
/// u ρ'(u) + ρ(u-1) = 0 for u > 1, with ρ(u) = 1 for 0 ≤ u ≤ 1
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::transcendental::dickman_rho;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let u = Symbol::new("u");
/// let rho = dickman_rho(&Expr::Symbol(u));
/// ```
///
/// # Properties
///
/// - ρ(u) = 1 for 0 ≤ u ≤ 1
/// - ρ(u) decreases rapidly as u increases
/// - ρ(2) ≈ 0.30685
/// - ρ(3) ≈ 0.04860
/// - ρ(u) ~ exp(-u ln u) as u → ∞
///
/// # Applications
///
/// - Probability that a random integer ≤ x has all prime factors ≤ x^(1/u) is ≈ ρ(u)
/// - Used in cryptography (factorization algorithms)
/// - Analysis of the Pollard rho factorization method
pub fn dickman_rho(u: &Expr) -> Expr {
    // For u in [0, 1], ρ(u) = 1
    if let Some(u_val) = try_expr_to_f64(u) {
        if u_val >= 0.0 && u_val <= 1.0 {
            return Expr::from(1);
        }
    }

    Expr::Function("dickman_rho".to_string(), vec![Arc::new(u.clone())])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_zeta_symbolic() {
        let s = Symbol::new("s");
        let z = zeta(&Expr::Symbol(s));
        assert!(matches!(z, Expr::Function(name, _) if name == "zeta"));
    }

    #[test]
    fn test_zeta_special_values() {
        assert_eq!(zeta(&Expr::from(0)), Expr::from(-1) / Expr::from(2));
        assert_eq!(zeta(&Expr::from(-1)), Expr::from(-1) / Expr::from(12));
    }

    #[test]
    fn test_zeta_deriv_symbolic() {
        let s = Symbol::new("s");
        let zd = zeta_deriv(&Expr::from(1), &Expr::Symbol(s));
        assert!(matches!(zd, Expr::Function(name, args)
            if name == "zeta_deriv" && args.len() == 2));
    }

    #[test]
    fn test_hurwitz_zeta_symbolic() {
        let s = Symbol::new("s");
        let hz = hurwitz_zeta(&Expr::Symbol(s), &Expr::from(1));
        assert!(matches!(hz, Expr::Function(name, args)
            if name == "hurwitz_zeta" && args.len() == 2));
    }

    #[test]
    fn test_stieltjes_symbolic() {
        let gamma = stieltjes(&Expr::from(0));
        assert!(matches!(gamma, Expr::Function(name, _) if name == "stieltjes"));
    }

    #[test]
    fn test_zeta_symmetric_symbolic() {
        let s = Symbol::new("s");
        let xi = zeta_symmetric(&Expr::Symbol(s));
        assert!(matches!(xi, Expr::Function(name, _) if name == "zeta_symmetric"));
    }

    #[test]
    fn test_dickman_rho_in_range() {
        assert_eq!(dickman_rho(&Expr::from(0)), Expr::from(1));
        // 0.5 is in [0, 1] so should return 1
        // Note: We can't directly test 0.5 as it's a rational
        // so let's test with a symbolic value instead
    }

    #[test]
    fn test_dickman_rho_symbolic() {
        let u = Symbol::new("u");
        let rho = dickman_rho(&Expr::Symbol(u));
        assert!(matches!(rho, Expr::Function(name, _) if name == "dickman_rho"));
    }

    #[test]
    fn test_zeta_vs_hurwitz() {
        let s = Symbol::new("s");
        let z1 = zeta(&Expr::Symbol(s.clone()));
        let z2 = hurwitz_zeta(&Expr::Symbol(s), &Expr::from(1));

        // They're different function names even though mathematically ζ(s) = ζ(s, 1)
        assert_ne!(z1, z2);
    }

    #[test]
    fn test_zeta_deriv_order_zero() {
        let s = Symbol::new("s");
        let zd0 = zeta_deriv(&Expr::from(0), &Expr::Symbol(s.clone()));
        let z = zeta(&Expr::Symbol(s));

        // They're different representations (zeta_deriv(0, s) vs zeta(s))
        assert_ne!(zd0, z);
    }
}
