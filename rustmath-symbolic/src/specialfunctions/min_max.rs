//! Symbolic Min and Max Functions
//!
//! This module implements symbolic minimum and maximum functions that work with
//! both numeric and symbolic expressions.
//!
//! Corresponds to sage.functions.min_max
//!
//! # Functions
//!
//! - `min_symbolic(args...)`: Symbolic minimum of arguments
//! - `max_symbolic(args...)`: Symbolic maximum of arguments
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::min_max::{min_symbolic, max_symbolic};
//! use rustmath_symbolic::Expr;
//!
//! // Numeric evaluation
//! assert_eq!(min_symbolic(&[Expr::from(1), Expr::from(2), Expr::from(3)]), Expr::from(1));
//! assert_eq!(max_symbolic(&[Expr::from(1), Expr::from(2), Expr::from(3)]), Expr::from(3));
//! ```

use crate::expression::Expr;
use rustmath_core::NumericConversion;
use std::sync::Arc;

/// Helper function to try converting an Expr to f64
fn try_expr_to_f64(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Integer(i) => i.to_f64(),
        Expr::Rational(r) => r.to_f64(),
        _ => None,
    }
}

/// Symbolic minimum function
///
/// Returns the minimum of the given expressions. If all expressions are
/// numeric constants, returns the numeric minimum. Otherwise, returns
/// a symbolic min expression.
///
/// # Arguments
///
/// * `args` - Slice of expressions to find minimum of
///
/// # Returns
///
/// The minimum expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::min_max::min_symbolic;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// // Numeric case
/// assert_eq!(min_symbolic(&[Expr::from(5), Expr::from(2), Expr::from(8)]), Expr::from(2));
///
/// // Mixed case with symbols
/// let x = Symbol::new("x");
/// let result = min_symbolic(&[Expr::from(5), Expr::Symbol(x)]);
/// // Returns symbolic min(5, x)
/// ```
///
/// # Properties
///
/// - min(a, b) = min(b, a) (commutative)
/// - min(min(a, b), c) = min(a, min(b, c)) (associative)
/// - min(a, a) = a (idempotent)
/// - min(a, +∞) = a
/// - min(a, -∞) = -∞
pub fn min_symbolic(args: &[Expr]) -> Expr {
    if args.is_empty() {
        return Expr::Function("min".to_string(), vec![]);
    }

    if args.len() == 1 {
        return args[0].clone();
    }

    // Try to evaluate if all arguments are numeric
    let mut all_numeric = true;
    let mut min_val: Option<f64> = None;
    let mut min_idx: Option<usize> = None;

    for (idx, arg) in args.iter().enumerate() {
        if let Some(val) = try_expr_to_f64(arg) {
            match min_val {
                None => {
                    min_val = Some(val);
                    min_idx = Some(idx);
                }
                Some(current) if val < current => {
                    min_val = Some(val);
                    min_idx = Some(idx);
                }
                _ => {}
            }
        } else {
            all_numeric = false;
            break;
        }
    }

    if all_numeric {
        if let Some(idx) = min_idx {
            return args[idx].clone();
        }
    }

    // Return symbolic form
    Expr::Function("min".to_string(), args.iter().map(|e| Arc::new(e.clone())).collect())
}

/// Symbolic maximum function
///
/// Returns the maximum of the given expressions. If all expressions are
/// numeric constants, returns the numeric maximum. Otherwise, returns
/// a symbolic max expression.
///
/// # Arguments
///
/// * `args` - Slice of expressions to find maximum of
///
/// # Returns
///
/// The maximum expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::min_max::max_symbolic;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// // Numeric case
/// assert_eq!(max_symbolic(&[Expr::from(5), Expr::from(2), Expr::from(8)]), Expr::from(8));
///
/// // Mixed case with symbols
/// let x = Symbol::new("x");
/// let result = max_symbolic(&[Expr::from(5), Expr::Symbol(x)]);
/// // Returns symbolic max(5, x)
/// ```
///
/// # Properties
///
/// - max(a, b) = max(b, a) (commutative)
/// - max(max(a, b), c) = max(a, max(b, c)) (associative)
/// - max(a, a) = a (idempotent)
/// - max(a, +∞) = +∞
/// - max(a, -∞) = a
pub fn max_symbolic(args: &[Expr]) -> Expr {
    if args.is_empty() {
        return Expr::Function("max".to_string(), vec![]);
    }

    if args.len() == 1 {
        return args[0].clone();
    }

    // Try to evaluate if all arguments are numeric
    let mut all_numeric = true;
    let mut max_val: Option<f64> = None;
    let mut max_idx: Option<usize> = None;

    for (idx, arg) in args.iter().enumerate() {
        if let Some(val) = try_expr_to_f64(arg) {
            match max_val {
                None => {
                    max_val = Some(val);
                    max_idx = Some(idx);
                }
                Some(current) if val > current => {
                    max_val = Some(val);
                    max_idx = Some(idx);
                }
                _ => {}
            }
        } else {
            all_numeric = false;
            break;
        }
    }

    if all_numeric {
        if let Some(idx) = max_idx {
            return args[idx].clone();
        }
    }

    // Return symbolic form
    Expr::Function("max".to_string(), args.iter().map(|e| Arc::new(e.clone())).collect())
}

/// Pairwise minimum
///
/// Convenience function for finding the minimum of exactly two expressions.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::min_max::min2;
/// use rustmath_symbolic::Expr;
///
/// assert_eq!(min2(&Expr::from(5), &Expr::from(3)), Expr::from(3));
/// ```
pub fn min2(a: &Expr, b: &Expr) -> Expr {
    min_symbolic(&[a.clone(), b.clone()])
}

/// Pairwise maximum
///
/// Convenience function for finding the maximum of exactly two expressions.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::min_max::max2;
/// use rustmath_symbolic::Expr;
///
/// assert_eq!(max2(&Expr::from(5), &Expr::from(3)), Expr::from(5));
/// ```
pub fn max2(a: &Expr, b: &Expr) -> Expr {
    max_symbolic(&[a.clone(), b.clone()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;
    use rustmath_rationals::Rational;

    #[test]
    fn test_min_integers() {
        assert_eq!(
            min_symbolic(&[Expr::from(5), Expr::from(2), Expr::from(8)]),
            Expr::from(2)
        );
        assert_eq!(
            min_symbolic(&[Expr::from(-3), Expr::from(-1), Expr::from(-5)]),
            Expr::from(-5)
        );
    }

    #[test]
    fn test_min_single() {
        assert_eq!(min_symbolic(&[Expr::from(42)]), Expr::from(42));
    }

    #[test]
    fn test_min_empty() {
        let result = min_symbolic(&[]);
        assert!(matches!(result, Expr::Function(name, args) if name == "min" && args.is_empty()));
    }

    #[test]
    fn test_min_rationals() {
        assert_eq!(
            min_symbolic(&[
                Expr::Rational(Rational::new(1, 2).unwrap()),
                Expr::Rational(Rational::new(1, 3).unwrap())
            ]),
            Expr::Rational(Rational::new(1, 3).unwrap())
        );
    }

    #[test]
    fn test_min_symbolic() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        let result = min_symbolic(&[Expr::Symbol(x.clone()), Expr::Symbol(y.clone())]);
        assert!(matches!(result, Expr::Function(name, _) if name == "min"));

        // Mixed numeric and symbolic
        let result2 = min_symbolic(&[Expr::from(5), Expr::Symbol(x)]);
        assert!(matches!(result2, Expr::Function(name, _) if name == "min"));
    }

    #[test]
    fn test_max_integers() {
        assert_eq!(
            max_symbolic(&[Expr::from(5), Expr::from(2), Expr::from(8)]),
            Expr::from(8)
        );
        assert_eq!(
            max_symbolic(&[Expr::from(-3), Expr::from(-1), Expr::from(-5)]),
            Expr::from(-1)
        );
    }

    #[test]
    fn test_max_single() {
        assert_eq!(max_symbolic(&[Expr::from(42)]), Expr::from(42));
    }

    #[test]
    fn test_max_empty() {
        let result = max_symbolic(&[]);
        assert!(matches!(result, Expr::Function(name, args) if name == "max" && args.is_empty()));
    }

    #[test]
    fn test_max_rationals() {
        assert_eq!(
            max_symbolic(&[
                Expr::Rational(Rational::new(1, 2).unwrap()),
                Expr::Rational(Rational::new(1, 3).unwrap())
            ]),
            Expr::Rational(Rational::new(1, 2).unwrap())
        );
    }

    #[test]
    fn test_max_symbolic() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        let result = max_symbolic(&[Expr::Symbol(x.clone()), Expr::Symbol(y.clone())]);
        assert!(matches!(result, Expr::Function(name, _) if name == "max"));

        // Mixed numeric and symbolic
        let result2 = max_symbolic(&[Expr::from(5), Expr::Symbol(x)]);
        assert!(matches!(result2, Expr::Function(name, _) if name == "max"));
    }

    #[test]
    fn test_min2() {
        assert_eq!(min2(&Expr::from(5), &Expr::from(3)), Expr::from(3));
        assert_eq!(min2(&Expr::from(-2), &Expr::from(7)), Expr::from(-2));
    }

    #[test]
    fn test_max2() {
        assert_eq!(max2(&Expr::from(5), &Expr::from(3)), Expr::from(5));
        assert_eq!(max2(&Expr::from(-2), &Expr::from(7)), Expr::from(7));
    }

    #[test]
    fn test_min_max_properties() {
        // Commutativity
        let a = Expr::from(3);
        let b = Expr::from(7);
        assert_eq!(min2(&a, &b), min2(&b, &a));
        assert_eq!(max2(&a, &b), max2(&b, &a));

        // Idempotence
        assert_eq!(min2(&a, &a), a);
        assert_eq!(max2(&a, &a), a);
    }

    #[test]
    fn test_min_max_many_args() {
        let args = vec![
            Expr::from(10),
            Expr::from(3),
            Expr::from(7),
            Expr::from(1),
            Expr::from(15),
        ];

        assert_eq!(min_symbolic(&args), Expr::from(1));
        assert_eq!(max_symbolic(&args), Expr::from(15));
    }

    #[test]
    fn test_min_max_negative() {
        let args = vec![Expr::from(-10), Expr::from(-3), Expr::from(-7)];

        assert_eq!(min_symbolic(&args), Expr::from(-10));
        assert_eq!(max_symbolic(&args), Expr::from(-3));
    }

    #[test]
    fn test_min_max_zero() {
        let args = vec![Expr::from(-5), Expr::from(0), Expr::from(5)];

        assert_eq!(min_symbolic(&args), Expr::from(-5));
        assert_eq!(max_symbolic(&args), Expr::from(5));
    }
}
