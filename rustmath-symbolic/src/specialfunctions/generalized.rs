//! Generalized Functions
//!
//! This module implements generalized functions (also known as distributions)
//! commonly used in analysis, physics, and signal processing.
//!
//! Corresponds to sage.functions.generalized
//!
//! # Functions
//!
//! - `dirac_delta(x)`: Dirac delta function δ(x)
//! - `heaviside(x)`: Heaviside step function H(x)
//! - `unit_step(x)`: Unit step function (same as Heaviside)
//! - `signum(x)`: Sign function sgn(x)
//! - `kronecker_delta(i, j)`: Kronecker delta δᵢⱼ
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::generalized::*;
//! use rustmath_symbolic::{Expr, Symbol};
//!
//! let x = Symbol::new("x");
//! let delta = dirac_delta(&Expr::Symbol(x.clone()));
//! let h = heaviside(&Expr::Symbol(x.clone()));
//! let sgn = signum(&Expr::Symbol(x.clone()));
//!
//! // Kronecker delta
//! assert_eq!(kronecker_delta(&Expr::from(1), &Expr::from(1)), Expr::from(1));
//! assert_eq!(kronecker_delta(&Expr::from(1), &Expr::from(2)), Expr::from(0));
//! ```

use std::sync::Arc;
use crate::expression::Expr;
use rustmath_rationals::Rational;

/// Helper function to try converting an Expr to f64
fn try_expr_to_f64(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Integer(i) => i.to_f64(),
        Expr::Rational(r) => r.to_f64(),
        _ => None,
    }
}

/// Dirac delta function δ(x)
///
/// The Dirac delta is a generalized function satisfying:
/// - δ(x) = 0 for x ≠ 0
/// - ∫_{-∞}^{∞} δ(x) dx = 1
/// - ∫_{-∞}^{∞} f(x)δ(x) dx = f(0)
///
/// # Arguments
///
/// * `x` - The argument expression
///
/// # Returns
///
/// A symbolic expression representing δ(x)
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::generalized::dirac_delta;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let x = Symbol::new("x");
/// let delta = dirac_delta(&Expr::Symbol(x));
/// ```
///
/// # Properties
///
/// - δ(x) = 0 for x ≠ 0
/// - δ(-x) = δ(x) (even function)
/// - x·δ(x) = 0
/// - δ(ax) = δ(x)/|a|
/// - ∫ δ(x-a)f(x) dx = f(a)
pub fn dirac_delta(x: &Expr) -> Expr {
    // For constant evaluation
    if let Some(val) = try_expr_to_f64(x) {
        if val == 0.0 {
            // δ(0) is technically undefined/infinite, but we represent it symbolically
            return Expr::Function("dirac_delta".to_string(), vec![Arc::new(Expr::from(0))]);
        } else {
            return Expr::from(0);
        }
    }

    // Return symbolic form
    Expr::Function("dirac_delta".to_string(), vec![Arc::new(x.clone())])
}

/// Heaviside step function H(x)
///
/// Defined as:
/// - H(x) = 0 for x < 0
/// - H(x) = 1/2 for x = 0 (convention)
/// - H(x) = 1 for x > 0
///
/// # Arguments
///
/// * `x` - The argument expression
///
/// # Returns
///
/// An expression representing H(x), simplified if x is constant
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::generalized::heaviside;
/// use rustmath_symbolic::Expr;
///
/// assert_eq!(heaviside(&Expr::from(-1)), Expr::from(0));
/// assert_eq!(heaviside(&Expr::from(1)), Expr::from(1));
/// ```
///
/// # Properties
///
/// - H'(x) = δ(x) (derivative is Dirac delta)
/// - H(x) + H(-x) = 1
/// - H(x) · H(-x) = 0
pub fn heaviside(x: &Expr) -> Expr {
    // For constant evaluation
    if let Some(val) = try_expr_to_f64(x) {
        if val < 0.0 {
            return Expr::from(0);
        } else if val > 0.0 {
            return Expr::from(1);
        } else {
            // Convention: H(0) = 1/2
            return Expr::Rational(Rational::new(1, 2).unwrap());
        }
    }

    // Return symbolic form
    Expr::Function("heaviside".to_string(), vec![Arc::new(x.clone())])
}

/// Unit step function (same as Heaviside)
///
/// This is an alias for the Heaviside function, commonly used in
/// engineering and control theory.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::generalized::{unit_step, heaviside};
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::from(5);
/// assert_eq!(unit_step(&x), heaviside(&x));
/// ```
pub fn unit_step(x: &Expr) -> Expr {
    heaviside(x)
}

/// Sign function sgn(x)
///
/// Defined as:
/// - sgn(x) = -1 for x < 0
/// - sgn(x) = 0 for x = 0
/// - sgn(x) = 1 for x > 0
///
/// # Arguments
///
/// * `x` - The argument expression
///
/// # Returns
///
/// An expression representing sgn(x), simplified if x is constant
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::generalized::signum;
/// use rustmath_symbolic::Expr;
///
/// assert_eq!(signum(&Expr::from(-5)), Expr::from(-1));
/// assert_eq!(signum(&Expr::from(0)), Expr::from(0));
/// assert_eq!(signum(&Expr::from(5)), Expr::from(1));
/// ```
///
/// # Properties
///
/// - sgn(x) = 2H(x) - 1 (relation to Heaviside)
/// - sgn(-x) = -sgn(x) (odd function)
/// - sgn(x)² = 1 for x ≠ 0
/// - |x| = x · sgn(x)
pub fn signum(x: &Expr) -> Expr {
    // For constant evaluation
    if let Some(val) = try_expr_to_f64(x) {
        if val < 0.0 {
            return Expr::from(-1);
        } else if val > 0.0 {
            return Expr::from(1);
        } else {
            return Expr::from(0);
        }
    }

    // Return symbolic form
    Expr::Function("sgn".to_string(), vec![Arc::new(x.clone())])
}

/// Kronecker delta δᵢⱼ
///
/// Defined as:
/// - δᵢⱼ = 1 if i = j
/// - δᵢⱼ = 0 if i ≠ j
///
/// # Arguments
///
/// * `i` - First index
/// * `j` - Second index
///
/// # Returns
///
/// 1 if indices are equal, 0 otherwise (when computable)
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::generalized::kronecker_delta;
/// use rustmath_symbolic::Expr;
///
/// assert_eq!(kronecker_delta(&Expr::from(1), &Expr::from(1)), Expr::from(1));
/// assert_eq!(kronecker_delta(&Expr::from(1), &Expr::from(2)), Expr::from(0));
/// assert_eq!(kronecker_delta(&Expr::from(0), &Expr::from(0)), Expr::from(1));
/// ```
///
/// # Properties
///
/// - δᵢⱼ = δⱼᵢ (symmetric)
/// - Σⱼ δᵢⱼ aⱼ = aᵢ (sifting property)
/// - Σⱼ δᵢⱼ δⱼₖ = δᵢₖ (composition)
pub fn kronecker_delta(i: &Expr, j: &Expr) -> Expr {
    // Try to evaluate if both are constants
    if let (Some(i_val), Some(j_val)) = (try_expr_to_f64(i), try_expr_to_f64(j)) {
        if (i_val - j_val).abs() < 1e-10 {
            return Expr::from(1);
        } else {
            return Expr::from(0);
        }
    }

    // Check if expressions are symbolically equal
    if i == j {
        return Expr::from(1);
    }

    // Return symbolic form
    Expr::Function(
        "kronecker_delta".to_string(),
        vec![Arc::new(i.clone()), Arc::new(j.clone())],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_dirac_delta_constant() {
        assert_eq!(dirac_delta(&Expr::from(5)), Expr::from(0));
        assert_eq!(dirac_delta(&Expr::from(-3)), Expr::from(0));

        // δ(0) should return symbolic form
        let delta_zero = dirac_delta(&Expr::from(0));
        assert!(matches!(delta_zero, Expr::Function(name, _) if name == "dirac_delta"));
    }

    #[test]
    fn test_dirac_delta_symbolic() {
        let x = Symbol::new("x");
        let delta = dirac_delta(&Expr::Symbol(x));

        assert!(matches!(delta, Expr::Function(name, _) if name == "dirac_delta"));
    }

    #[test]
    fn test_heaviside_constants() {
        assert_eq!(heaviside(&Expr::from(-5)), Expr::from(0));
        assert_eq!(heaviside(&Expr::from(5)), Expr::from(1));
        assert_eq!(
            heaviside(&Expr::from(0)),
            Expr::Rational(Rational::new(1, 2).unwrap())
        );
    }

    #[test]
    fn test_heaviside_symbolic() {
        let x = Symbol::new("x");
        let h = heaviside(&Expr::Symbol(x));

        assert!(matches!(h, Expr::Function(name, _) if name == "heaviside"));
    }

    #[test]
    fn test_unit_step() {
        assert_eq!(unit_step(&Expr::from(-1)), Expr::from(0));
        assert_eq!(unit_step(&Expr::from(1)), Expr::from(1));
        assert_eq!(
            unit_step(&Expr::from(0)),
            Expr::Rational(Rational::new(1, 2).unwrap())
        );
    }

    #[test]
    fn test_signum_constants() {
        assert_eq!(signum(&Expr::from(-10)), Expr::from(-1));
        assert_eq!(signum(&Expr::from(0)), Expr::from(0));
        assert_eq!(signum(&Expr::from(10)), Expr::from(1));
    }

    #[test]
    fn test_signum_rationals() {
        assert_eq!(
            signum(&Expr::Rational(Rational::new(3, 4).unwrap())),
            Expr::from(1)
        );
        assert_eq!(
            signum(&Expr::Rational(Rational::new(-3, 4).unwrap())),
            Expr::from(-1)
        );
        assert_eq!(
            signum(&Expr::Rational(Rational::new(0, 1).unwrap())),
            Expr::from(0)
        );
    }

    #[test]
    fn test_signum_symbolic() {
        let x = Symbol::new("x");
        let sgn = signum(&Expr::Symbol(x));

        assert!(matches!(sgn, Expr::Function(name, _) if name == "sgn"));
    }

    #[test]
    fn test_kronecker_delta_integers() {
        assert_eq!(
            kronecker_delta(&Expr::from(0), &Expr::from(0)),
            Expr::from(1)
        );
        assert_eq!(
            kronecker_delta(&Expr::from(1), &Expr::from(1)),
            Expr::from(1)
        );
        assert_eq!(
            kronecker_delta(&Expr::from(1), &Expr::from(2)),
            Expr::from(0)
        );
        assert_eq!(
            kronecker_delta(&Expr::from(5), &Expr::from(3)),
            Expr::from(0)
        );
    }

    #[test]
    fn test_kronecker_delta_rationals() {
        // Equal fractions
        assert_eq!(
            kronecker_delta(
                &Expr::Rational(Rational::new(1, 2).unwrap()),
                &Expr::Rational(Rational::new(1, 2).unwrap())
            ),
            Expr::from(1)
        );
        assert_eq!(
            kronecker_delta(
                &Expr::Rational(Rational::new(2, 4).unwrap()),
                &Expr::Rational(Rational::new(1, 2).unwrap())
            ),
            Expr::from(1)
        );

        // Different fractions
        assert_eq!(
            kronecker_delta(
                &Expr::Rational(Rational::new(1, 2).unwrap()),
                &Expr::Rational(Rational::new(1, 3).unwrap())
            ),
            Expr::from(0)
        );
    }

    #[test]
    fn test_kronecker_delta_symbolic() {
        let i = Symbol::new("i");
        let j = Symbol::new("j");

        // Same symbol
        let delta_ii = kronecker_delta(&Expr::Symbol(i.clone()), &Expr::Symbol(i.clone()));
        assert_eq!(delta_ii, Expr::from(1));

        // Different symbols
        let delta_ij = kronecker_delta(&Expr::Symbol(i), &Expr::Symbol(j));
        assert!(matches!(delta_ij, Expr::Function(name, _) if name == "kronecker_delta"));
    }

    #[test]
    fn test_kronecker_delta_properties() {
        // Symmetry: δᵢⱼ = δⱼᵢ
        let i = Expr::from(3);
        let j = Expr::from(5);
        assert_eq!(kronecker_delta(&i, &j), kronecker_delta(&j, &i));

        // Identity: δᵢᵢ = 1
        assert_eq!(kronecker_delta(&i, &i), Expr::from(1));
    }

    #[test]
    fn test_heaviside_properties() {
        // H(x) + H(-x) = 1 for x ≠ 0
        let x = Expr::from(5);
        let neg_x = Expr::from(-5);

        let h_x = heaviside(&x);
        let h_neg_x = heaviside(&neg_x);

        if let (Expr::Integer(a), Expr::Integer(b)) = (&h_x, &h_neg_x) {
            assert_eq!(a.clone() + b.clone(), Integer::one());
        }
    }

    #[test]
    fn test_signum_properties() {
        // sgn(-x) = -sgn(x)
        let x = Expr::from(7);
        let neg_x = Expr::from(-7);

        let sgn_x = signum(&x);
        let sgn_neg_x = signum(&neg_x);

        if let (Expr::Integer(a), Expr::Integer(b)) = (&sgn_x, &sgn_neg_x) {
            assert_eq!(a.clone(), -b.clone());
        }

        // sgn(x)² = 1 for x ≠ 0
        if let Expr::Integer(s) = sgn_x {
            assert_eq!(s.clone() * s.clone(), Integer::one());
        }
    }

    #[test]
    fn test_signum_heaviside_relation() {
        // sgn(x) = 2H(x) - 1 for x ≠ 0
        let x = Expr::from(5);

        let sgn_x = signum(&x);
        let h_x = heaviside(&x);

        if let (Expr::Integer(s), Expr::Integer(h)) = (&sgn_x, &h_x) {
            assert_eq!(
                s.clone(),
                Integer::from(2) * h.clone() - Integer::one()
            );
        }
    }
}
