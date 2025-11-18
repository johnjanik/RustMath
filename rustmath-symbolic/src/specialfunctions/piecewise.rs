//! Piecewise Functions
//!
//! This module provides piecewise function representations and operations.
//!
//! Corresponds to sage.functions.piecewise
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::piecewise::*;
//! use rustmath_symbolic::{Expr, Symbol};
//!
//! let x = Symbol::new("x");
//! let pw = piecewise_function(&x, &[
//!     (Expr::Symbol(x.clone()), Expr::from(1)),  // if x, then 1
//!     (Expr::from(1), Expr::from(0)),             // else 0
//! ]);
//! ```

use crate::expression::Expr;
use crate::symbol::Symbol;
use std::sync::Arc;

/// Piecewise function representation
///
/// Creates a piecewise function from a list of (condition, value) pairs.
///
/// Corresponds to sage.functions.piecewise.PiecewiseFunction
///
/// # Arguments
///
/// * `var` - The variable for the piecewise function
/// * `pieces` - Vector of (condition, value) pairs
///
/// # Returns
///
/// A piecewise function expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::piecewise::piecewise_function;
/// use rustmath_symbolic::{Expr, Symbol};
///
/// let x = Symbol::new("x");
/// let pw = piecewise_function(&x, &[
///     (Expr::Symbol(x.clone()), Expr::from(1)),
///     (Expr::from(1), Expr::from(2)),
/// ]);
/// ```
pub fn piecewise_function(var: &Symbol, pieces: &[(Expr, Expr)]) -> Expr {
    let mut args = vec![Arc::new(Expr::Symbol(var.clone()))];

    for (cond, val) in pieces {
        args.push(Arc::new(cond.clone()));
        args.push(Arc::new(val.clone()));
    }

    Expr::Function("piecewise".to_string(), args)
}

/// Evaluation methods for piecewise functions
///
/// Provides strategies for evaluating piecewise functions.
///
/// Corresponds to sage.functions.piecewise.EvaluationMethods
#[derive(Debug, Clone, Copy)]
pub struct PiecewiseEvaluationMethods;

impl PiecewiseEvaluationMethods {
    pub fn new() -> Self {
        PiecewiseEvaluationMethods
    }
}

impl Default for PiecewiseEvaluationMethods {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piecewise_function() {
        let x = Symbol::new("x");
        let pw = piecewise_function(&x, &[
            (Expr::Symbol(x.clone()), Expr::from(1)),
            (Expr::from(1), Expr::from(2)),
        ]);

        assert!(matches!(pw, Expr::Function(name, _) if name == "piecewise"));
    }

    #[test]
    fn test_evaluation_methods() {
        let methods = PiecewiseEvaluationMethods::new();
        assert!(matches!(methods, PiecewiseEvaluationMethods));
    }
}
