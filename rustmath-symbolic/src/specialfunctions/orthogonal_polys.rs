//! Orthogonal Polynomials
//!
//! This module implements classical orthogonal polynomials which arise in many
//! areas of mathematics, physics, and engineering.
//!
//! Corresponds to sage.functions.orthogonal_polys
//!
//! # Functions
//!
//! - `chebyshev_t(n, x)`: Chebyshev polynomial of the first kind Tₙ(x)
//! - `chebyshev_u(n, x)`: Chebyshev polynomial of the second kind Uₙ(x)
//! - `legendre_p(n, x)`: Legendre polynomial Pₙ(x)
//! - `legendre_q(n, x)`: Legendre function of the second kind Qₙ(x)
//! - `hermite(n, x)`: Hermite polynomial Hₙ(x)
//! - `laguerre(n, x)`: Laguerre polynomial Lₙ(x)
//! - `gen_laguerre(n, alpha, x)`: Generalized Laguerre polynomial L_n^α(x)
//! - `jacobi_p(n, alpha, beta, x)`: Jacobi polynomial P_n^(α,β)(x)
//! - `ultraspherical(n, lambda, x)`: Ultraspherical (Gegenbauer) polynomial C_n^λ(x)
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::orthogonal_polys::*;
//! use rustmath_symbolic::Expr;
//!
//! // T₃(x) = 4x³ - 3x
//! let t3 = chebyshev_t(&Expr::from(3), &Expr::from(0));
//! ```
//!
//! # Mathematical Background
//!
//! Orthogonal polynomials satisfy orthogonality relations with respect to
//! weight functions on specific intervals. They are solutions to
//! Sturm-Liouville differential equations.

use crate::expression::Expr;
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use std::sync::Arc;

/// Chebyshev polynomial of the first kind Tₙ(x)
///
/// Defined by: Tₙ(cos(θ)) = cos(nθ)
/// Recurrence: T₀(x) = 1, T₁(x) = x, Tₙ₊₁(x) = 2xTₙ(x) - Tₙ₋₁(x)
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `x` - The argument
///
/// # Returns
///
/// Tₙ(x)
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::orthogonal_polys::chebyshev_t;
/// use rustmath_symbolic::Expr;
///
/// // T₀(x) = 1
/// assert_eq!(chebyshev_t(&Expr::from(0), &Expr::from(5)), Expr::from(1));
/// // T₁(x) = x
/// assert_eq!(chebyshev_t(&Expr::from(1), &Expr::from(5)), Expr::from(5));
/// ```
///
/// # Properties
///
/// - Orthogonal on [-1, 1] with weight 1/√(1-x²)
/// - |Tₙ(x)| ≤ 1 for x ∈ [-1, 1]
/// - Tₙ(1) = 1, Tₙ(-1) = (-1)ⁿ
pub fn chebyshev_t(n: &Expr, x: &Expr) -> Expr {
    // For small integer n, compute explicitly
    if let Expr::Integer(n_int) = n {
        if let Some(n_val) = n_int.to_i64() {
            if n_val >= 0 && n_val <= 5 {
                return compute_chebyshev_t(n_val as usize, x);
            }
        }
    }

    // Return symbolic form
    Expr::Function(
        "chebyshev_t".to_string(),
        vec![Arc::new(n.clone()), Arc::new(x.clone())],
    )
}

/// Compute Chebyshev T polynomial explicitly for small n
fn compute_chebyshev_t(n: usize, x: &Expr) -> Expr {
    match n {
        0 => Expr::from(1),
        1 => x.clone(),
        2 => Expr::from(2) * x.clone() * x.clone() - Expr::from(1),
        3 => Expr::from(4) * x.clone().pow(Expr::from(3)) - Expr::from(3) * x.clone(),
        4 => {
            Expr::from(8) * x.clone().pow(Expr::from(4))
                - Expr::from(8) * x.clone().pow(Expr::from(2))
                + Expr::from(1)
        }
        5 => {
            Expr::from(16) * x.clone().pow(Expr::from(5))
                - Expr::from(20) * x.clone().pow(Expr::from(3))
                + Expr::from(5) * x.clone()
        }
        _ => Expr::Function(
            "chebyshev_t".to_string(),
            vec![Arc::new(Expr::from(n as i64)), Arc::new(x.clone())],
        ),
    }
}

/// Chebyshev polynomial of the second kind Uₙ(x)
///
/// Defined by: Uₙ(cos(θ)) = sin((n+1)θ)/sin(θ)
/// Recurrence: U₀(x) = 1, U₁(x) = 2x, Uₙ₊₁(x) = 2xUₙ(x) - Uₙ₋₁(x)
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `x` - The argument
///
/// # Returns
///
/// Uₙ(x)
///
/// # Properties
///
/// - Orthogonal on [-1, 1] with weight √(1-x²)
/// - Uₙ(1) = n + 1
/// - Uₙ(-1) = (-1)ⁿ(n + 1)
pub fn chebyshev_u(n: &Expr, x: &Expr) -> Expr {
    // For small integer n, compute explicitly
    if let Expr::Integer(n_int) = n {
        if let Some(n_val) = n_int.to_i64() {
            if n_val >= 0 && n_val <= 3 {
                return compute_chebyshev_u(n_val as usize, x);
            }
        }
    }

    Expr::Function(
        "chebyshev_u".to_string(),
        vec![Arc::new(n.clone()), Arc::new(x.clone())],
    )
}

fn compute_chebyshev_u(n: usize, x: &Expr) -> Expr {
    match n {
        0 => Expr::from(1),
        1 => Expr::from(2) * x.clone(),
        2 => Expr::from(4) * x.clone() * x.clone() - Expr::from(1),
        3 => Expr::from(8) * x.clone().pow(Expr::from(3)) - Expr::from(4) * x.clone(),
        _ => Expr::Function(
            "chebyshev_u".to_string(),
            vec![Arc::new(Expr::from(n as i64)), Arc::new(x.clone())],
        ),
    }
}

/// Legendre polynomial Pₙ(x)
///
/// Solutions to Legendre's differential equation.
/// Recurrence: P₀(x) = 1, P₁(x) = x, (n+1)Pₙ₊₁(x) = (2n+1)xPₙ(x) - nPₙ₋₁(x)
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `x` - The argument
///
/// # Returns
///
/// Pₙ(x)
///
/// # Properties
///
/// - Orthogonal on [-1, 1] with weight 1
/// - Pₙ(1) = 1, Pₙ(-1) = (-1)ⁿ
/// - ∫₋₁¹ Pₙ(x)Pₘ(x) dx = 2/(2n+1) δₙₘ
pub fn legendre_p(n: &Expr, x: &Expr) -> Expr {
    if let Expr::Integer(n_int) = n {
        if let Some(n_val) = n_int.to_i64() {
            if n_val >= 0 && n_val <= 3 {
                return compute_legendre_p(n_val as usize, x);
            }
        }
    }

    Expr::Function(
        "legendre_p".to_string(),
        vec![Arc::new(n.clone()), Arc::new(x.clone())],
    )
}

fn compute_legendre_p(n: usize, x: &Expr) -> Expr {
    match n {
        0 => Expr::from(1),
        1 => x.clone(),
        2 => (Expr::from(3) * x.clone() * x.clone() - Expr::from(1)) / Expr::from(2),
        3 => {
            (Expr::from(5) * x.clone().pow(Expr::from(3)) - Expr::from(3) * x.clone())
                / Expr::from(2)
        }
        _ => Expr::Function(
            "legendre_p".to_string(),
            vec![Arc::new(Expr::from(n as i64)), Arc::new(x.clone())],
        ),
    }
}

/// Legendre function of the second kind Qₙ(x)
///
/// Second linearly independent solution to Legendre's differential equation.
///
/// # Arguments
///
/// * `n` - Degree
/// * `x` - The argument
///
/// # Returns
///
/// Qₙ(x)
pub fn legendre_q(n: &Expr, x: &Expr) -> Expr {
    Expr::Function(
        "legendre_q".to_string(),
        vec![Arc::new(n.clone()), Arc::new(x.clone())],
    )
}

/// Hermite polynomial Hₙ(x)
///
/// Defined by: Hₙ(x) = (-1)ⁿ eˣ² dⁿ/dxⁿ(e⁻ˣ²)
/// Recurrence: H₀(x) = 1, H₁(x) = 2x, Hₙ₊₁(x) = 2xHₙ(x) - 2nHₙ₋₁(x)
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `x` - The argument
///
/// # Returns
///
/// Hₙ(x)
///
/// # Properties
///
/// - Orthogonal on (-∞, ∞) with weight e⁻ˣ²
/// - ∫₋∞^∞ Hₙ(x)Hₘ(x)e⁻ˣ² dx = √π 2ⁿ n! δₙₘ
pub fn hermite(n: &Expr, x: &Expr) -> Expr {
    if let Expr::Integer(n_int) = n {
        if let Some(n_val) = n_int.to_i64() {
            if n_val >= 0 && n_val <= 3 {
                return compute_hermite(n_val as usize, x);
            }
        }
    }

    Expr::Function(
        "hermite".to_string(),
        vec![Arc::new(n.clone()), Arc::new(x.clone())],
    )
}

fn compute_hermite(n: usize, x: &Expr) -> Expr {
    match n {
        0 => Expr::from(1),
        1 => Expr::from(2) * x.clone(),
        2 => Expr::from(4) * x.clone() * x.clone() - Expr::from(2),
        3 => Expr::from(8) * x.clone().pow(Expr::from(3)) - Expr::from(12) * x.clone(),
        _ => Expr::Function(
            "hermite".to_string(),
            vec![Arc::new(Expr::from(n as i64)), Arc::new(x.clone())],
        ),
    }
}

/// Laguerre polynomial Lₙ(x)
///
/// Defined by: Lₙ(x) = eˣ/n! dⁿ/dxⁿ(xⁿe⁻ˣ)
/// Recurrence: L₀(x) = 1, L₁(x) = 1-x, (n+1)Lₙ₊₁(x) = (2n+1-x)Lₙ(x) - nLₙ₋₁(x)
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `x` - The argument
///
/// # Returns
///
/// Lₙ(x)
///
/// # Properties
///
/// - Orthogonal on [0, ∞) with weight e⁻ˣ
/// - ∫₀^∞ Lₙ(x)Lₘ(x)e⁻ˣ dx = δₙₘ
pub fn laguerre(n: &Expr, x: &Expr) -> Expr {
    if let Expr::Integer(n_int) = n {
        if let Some(n_val) = n_int.to_i64() {
            if n_val >= 0 && n_val <= 2 {
                return compute_laguerre(n_val as usize, x);
            }
        }
    }

    Expr::Function(
        "laguerre".to_string(),
        vec![Arc::new(n.clone()), Arc::new(x.clone())],
    )
}

fn compute_laguerre(n: usize, x: &Expr) -> Expr {
    match n {
        0 => Expr::from(1),
        1 => Expr::from(1) - x.clone(),
        2 => (x.clone() * x.clone() - Expr::from(4) * x.clone() + Expr::from(2)) / Expr::from(2),
        _ => Expr::Function(
            "laguerre".to_string(),
            vec![Arc::new(Expr::from(n as i64)), Arc::new(x.clone())],
        ),
    }
}

/// Generalized Laguerre polynomial L_n^α(x)
///
/// # Arguments
///
/// * `n` - Degree
/// * `alpha` - Parameter
/// * `x` - The argument
///
/// # Returns
///
/// L_n^α(x)
///
/// # Properties
///
/// - L_n^0(x) = Lₙ(x) (ordinary Laguerre)
/// - Orthogonal on [0, ∞) with weight xᵅe⁻ˣ
pub fn gen_laguerre(n: &Expr, alpha: &Expr, x: &Expr) -> Expr {
    Expr::Function(
        "gen_laguerre".to_string(),
        vec![
            Arc::new(n.clone()),
            Arc::new(alpha.clone()),
            Arc::new(x.clone()),
        ],
    )
}

/// Jacobi polynomial P_n^(α,β)(x)
///
/// Most general of the classical orthogonal polynomials.
///
/// # Arguments
///
/// * `n` - Degree
/// * `alpha` - First parameter (α > -1)
/// * `beta` - Second parameter (β > -1)
/// * `x` - The argument
///
/// # Returns
///
/// P_n^(α,β)(x)
///
/// # Properties
///
/// - Orthogonal on [-1, 1] with weight (1-x)ᵅ(1+x)ᵝ
/// - Special cases:
///   - P_n^(0,0)(x) = Pₙ(x) (Legendre)
///   - P_n^(-1/2,-1/2)(x) ∝ Tₙ(x) (Chebyshev T)
///   - P_n^(1/2,1/2)(x) ∝ Uₙ(x) (Chebyshev U)
pub fn jacobi_p(n: &Expr, alpha: &Expr, beta: &Expr, x: &Expr) -> Expr {
    Expr::Function(
        "jacobi_p".to_string(),
        vec![
            Arc::new(n.clone()),
            Arc::new(alpha.clone()),
            Arc::new(beta.clone()),
            Arc::new(x.clone()),
        ],
    )
}

/// Ultraspherical (Gegenbauer) polynomial C_n^λ(x)
///
/// # Arguments
///
/// * `n` - Degree
/// * `lambda` - Parameter
/// * `x` - The argument
///
/// # Returns
///
/// C_n^λ(x)
///
/// # Properties
///
/// - Orthogonal on [-1, 1] with weight (1-x²)^(λ-1/2)
/// - C_n^(1/2)(x) = Pₙ(x) (Legendre)
/// - C_n^1(x) = Uₙ(x) (Chebyshev U)
pub fn ultraspherical(n: &Expr, lambda: &Expr, x: &Expr) -> Expr {
    Expr::Function(
        "ultraspherical".to_string(),
        vec![
            Arc::new(n.clone()),
            Arc::new(lambda.clone()),
            Arc::new(x.clone()),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_chebyshev_t_small() {
        // T₀(x) = 1
        assert_eq!(chebyshev_t(&Expr::from(0), &Expr::from(5)), Expr::from(1));
        // T₁(x) = x
        assert_eq!(chebyshev_t(&Expr::from(1), &Expr::from(5)), Expr::from(5));
    }

    #[test]
    fn test_chebyshev_t_symbolic() {
        let x = Symbol::new("x");
        let t = chebyshev_t(&Expr::from(10), &Expr::Symbol(x));
        assert!(matches!(t, Expr::Function(name, _) if name == "chebyshev_t"));
    }

    #[test]
    fn test_chebyshev_u_small() {
        // U₀(x) = 1
        assert_eq!(chebyshev_u(&Expr::from(0), &Expr::from(5)), Expr::from(1));
    }

    #[test]
    fn test_legendre_p_small() {
        // P₀(x) = 1
        assert_eq!(legendre_p(&Expr::from(0), &Expr::from(5)), Expr::from(1));
        // P₁(x) = x
        assert_eq!(legendre_p(&Expr::from(1), &Expr::from(5)), Expr::from(5));
    }

    #[test]
    fn test_hermite_small() {
        // H₀(x) = 1
        assert_eq!(hermite(&Expr::from(0), &Expr::from(5)), Expr::from(1));
    }

    #[test]
    fn test_laguerre_small() {
        // L₀(x) = 1
        assert_eq!(laguerre(&Expr::from(0), &Expr::from(5)), Expr::from(1));
    }

    #[test]
    fn test_gen_laguerre_symbolic() {
        let x = Symbol::new("x");
        let l = gen_laguerre(&Expr::from(5), &Expr::from(2), &Expr::Symbol(x));
        assert!(matches!(l, Expr::Function(name, _) if name == "gen_laguerre"));
    }

    #[test]
    fn test_jacobi_p_symbolic() {
        let x = Symbol::new("x");
        let j = jacobi_p(&Expr::from(5), &Expr::from(1), &Expr::from(2), &Expr::Symbol(x));
        assert!(matches!(j, Expr::Function(name, _) if name == "jacobi_p"));
    }

    #[test]
    fn test_ultraspherical_symbolic() {
        let x = Symbol::new("x");
        let u = ultraspherical(&Expr::from(5), &Expr::from(1), &Expr::Symbol(x));
        assert!(matches!(u, Expr::Function(name, _) if name == "ultraspherical"));
    }

    #[test]
    fn test_legendre_q_symbolic() {
        let x = Symbol::new("x");
        let q = legendre_q(&Expr::from(5), &Expr::Symbol(x));
        assert!(matches!(q, Expr::Function(name, _) if name == "legendre_q"));
    }
}
