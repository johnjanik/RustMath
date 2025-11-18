//! Spike Functions
//!
//! This module provides spike function (triangular pulse) representations.
//!
//! Corresponds to sage.functions.spike_function
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::spike_function::*;
//! use rustmath_symbolic::Expr;
//!
//! let spike = spike_function(&Expr::from(0), &Expr::from(2), &Expr::from(1));
//! ```

use crate::expression::Expr;
use std::sync::Arc;

/// Spike function - a piecewise linear function with a single peak
///
/// Creates a triangular "spike" with specified center, width, and height.
///
/// Corresponds to sage.functions.spike_function.SpikeFunction
///
/// # Arguments
///
/// * `center` - The x-coordinate of the peak
/// * `width` - The full width of the base
/// * `height` - The height of the peak
///
/// # Returns
///
/// A spike function expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::spike_function::spike_function;
/// use rustmath_symbolic::Expr;
///
/// // Spike centered at 0, width 2, height 1
/// let spike = spike_function(&Expr::from(0), &Expr::from(2), &Expr::from(1));
/// ```
///
/// # Properties
///
/// - The function is 0 outside [center - width/2, center + width/2]
/// - The function reaches `height` at `center`
/// - The function is piecewise linear
pub fn spike_function(center: &Expr, width: &Expr, height: &Expr) -> Expr {
    Expr::Function(
        "spike_function".to_string(),
        vec![
            Arc::new(center.clone()),
            Arc::new(width.clone()),
            Arc::new(height.clone()),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_function() {
        let spike = spike_function(&Expr::from(0), &Expr::from(2), &Expr::from(1));
        assert!(matches!(spike, Expr::Function(name, args)
            if name == "spike_function" && args.len() == 3));
    }

    #[test]
    fn test_spike_function_symbolic() {
        use crate::symbol::Symbol;
        let x = Symbol::new("x");
        let spike = spike_function(&Expr::Symbol(x), &Expr::from(2), &Expr::from(1));
        assert!(matches!(spike, Expr::Function(..)));
    }
}
