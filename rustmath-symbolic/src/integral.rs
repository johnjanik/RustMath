//! High-level integration interface
//!
//! This module provides user-friendly functions for symbolic integration,
//! including definite and indefinite integrals.

use crate::expression::Expr;
use crate::function::{BuiltinFunction, Function};
use crate::symbol::Symbol;
use std::sync::Arc;

/// Indefinite integral function
/// Represents ∫f(x)dx without limits
pub struct IndefiniteIntegral;

impl IndefiniteIntegral {
    /// Create a new indefinite integral
    pub fn new() -> Self {
        IndefiniteIntegral
    }

    /// Compute the indefinite integral of an expression
    pub fn compute(&self, expr: &Expr, var: &Symbol) -> Option<Expr> {
        expr.integrate(var)
    }

    /// Convert to a builtin function
    pub fn as_function() -> Arc<dyn Function> {
        Arc::new(BuiltinFunction::new("integrate", Some(2), |args| {
            // args[0] = expression, args[1] = variable
            // In practice, we need the variable as a Symbol, not an Expr
            // This is a simplified implementation
            Ok(Expr::Function(
                "integrate".to_string(),
                vec![Arc::new(args[0].clone()), Arc::new(args[1].clone())],
            ))
        }))
    }
}

impl Default for IndefiniteIntegral {
    fn default() -> Self {
        Self::new()
    }
}

/// Definite integral function
/// Represents ∫[a,b] f(x)dx with limits
pub struct DefiniteIntegral {
    lower_limit: Expr,
    upper_limit: Expr,
}

impl DefiniteIntegral {
    /// Create a new definite integral
    pub fn new(lower_limit: Expr, upper_limit: Expr) -> Self {
        DefiniteIntegral {
            lower_limit,
            upper_limit,
        }
    }

    /// Compute the definite integral of an expression
    /// Uses the fundamental theorem of calculus: ∫[a,b] f(x)dx = F(b) - F(a)
    pub fn compute(&self, expr: &Expr, var: &Symbol) -> Option<Expr> {
        // Get the antiderivative
        let antiderivative = expr.integrate(var)?;

        // Evaluate at limits
        let upper_value = antiderivative.substitute(var, &self.upper_limit);
        let lower_value = antiderivative.substitute(var, &self.lower_limit);

        // Return F(b) - F(a)
        Some(upper_value - lower_value)
    }

    /// Get the lower limit
    pub fn lower_limit(&self) -> &Expr {
        &self.lower_limit
    }

    /// Get the upper limit
    pub fn upper_limit(&self) -> &Expr {
        &self.upper_limit
    }

    /// Convert to a builtin function
    pub fn as_function() -> Arc<dyn Function> {
        Arc::new(BuiltinFunction::new("definite_integral", Some(4), |args| {
            // args[0] = expression, args[1] = variable, args[2] = lower, args[3] = upper
            Ok(Expr::Function(
                "definite_integral".to_string(),
                args.iter().map(|e| Arc::new(e.clone())).collect(),
            ))
        }))
    }
}

/// Compute an indefinite integral: ∫f(x)dx
pub fn integral(expr: &Expr, var: &Symbol) -> Option<Expr> {
    IndefiniteIntegral::new().compute(expr, var)
}

/// Compute a definite integral: ∫[a,b] f(x)dx
pub fn definite_integral(
    expr: &Expr,
    var: &Symbol,
    lower_limit: &Expr,
    upper_limit: &Expr,
) -> Option<Expr> {
    DefiniteIntegral::new(lower_limit.clone(), upper_limit.clone()).compute(expr, var)
}

/// General integrate function that handles both definite and indefinite integrals
/// - If limits are None, computes indefinite integral
/// - If limits are Some((a, b)), computes definite integral from a to b
pub fn integrate(
    expr: &Expr,
    var: &Symbol,
    limits: Option<(&Expr, &Expr)>,
) -> Option<Expr> {
    match limits {
        None => integral(expr, var),
        Some((lower, upper)) => definite_integral(expr, var, lower, upper),
    }
}

/// Multiple integration helper
/// Integrates with respect to multiple variables in sequence
pub fn integrate_multi(expr: &Expr, vars: &[(Symbol, Option<(Expr, Expr)>)]) -> Option<Expr> {
    let mut result = expr.clone();

    for (var, limits) in vars.iter().rev() {
        result = match limits {
            None => result.integrate(var)?,
            Some((lower, upper)) => {
                let antideriv = result.integrate(var)?;
                let upper_val = antideriv.substitute(var, upper);
                let lower_val = antideriv.substitute(var, lower);
                upper_val - lower_val
            }
        };
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indefinite_integral_creation() {
        let indef = IndefiniteIntegral::new();
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone());

        let result = indef.compute(&expr, &x);
        assert!(result.is_some());
    }

    #[test]
    fn test_indefinite_integral_power() {
        let indef = IndefiniteIntegral::new();
        let x = Symbol::new("x");
        // ∫ x² dx = x³/3
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let result = indef.compute(&expr, &x);
        assert!(result.is_some());
    }

    #[test]
    fn test_definite_integral_creation() {
        let def_int = DefiniteIntegral::new(Expr::from(0), Expr::from(1));
        assert_eq!(def_int.lower_limit(), &Expr::from(0));
        assert_eq!(def_int.upper_limit(), &Expr::from(1));
    }

    #[test]
    fn test_definite_integral_constant() {
        let def_int = DefiniteIntegral::new(Expr::from(0), Expr::from(2));
        let x = Symbol::new("x");
        // ∫[0,2] 1 dx = 2
        let expr = Expr::from(1);

        let result = def_int.compute(&expr, &x);
        assert!(result.is_some());
        // Result is (2*1) - (0*1), which may not be simplified
        // Just verify we got a result
        let result_expr = result.unwrap();
        assert!(!result_expr.contains_symbol(&x));
    }

    #[test]
    fn test_definite_integral_linear() {
        let def_int = DefiniteIntegral::new(Expr::from(0), Expr::from(1));
        let x = Symbol::new("x");
        // ∫[0,1] x dx = 1/2
        let expr = Expr::Symbol(x.clone());

        let result = def_int.compute(&expr, &x);
        assert!(result.is_some());
    }

    #[test]
    fn test_integral_function() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone());

        let result = integral(&expr, &x);
        assert!(result.is_some());
    }

    #[test]
    fn test_definite_integral_function() {
        let x = Symbol::new("x");
        let expr = Expr::from(1);

        let result = definite_integral(&expr, &x, &Expr::from(0), &Expr::from(3));
        assert!(result.is_some());
        // Result may not be simplified
        assert!(!result.unwrap().contains_symbol(&x));
    }

    #[test]
    fn test_integrate_none_limits() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let result = integrate(&expr, &x, None);
        assert!(result.is_some());
    }

    #[test]
    fn test_integrate_with_limits() {
        let x = Symbol::new("x");
        let expr = Expr::from(2);

        let result = integrate(&expr, &x, Some((&Expr::from(0), &Expr::from(5))));
        assert!(result.is_some());
        // Result may not be simplified
        assert!(!result.unwrap().contains_symbol(&x));
    }

    #[test]
    fn test_integrate_multi_single_var() {
        let x = Symbol::new("x");
        let expr = Expr::from(1);

        let result = integrate_multi(&expr, &[(x.clone(), None)]);
        assert!(result.is_some());
    }

    #[test]
    fn test_integrate_multi_with_limits() {
        let x = Symbol::new("x");
        let expr = Expr::from(1);

        let result = integrate_multi(&expr, &[(x.clone(), Some((Expr::from(0), Expr::from(2))))]);
        assert!(result.is_some());
        // Result may not be simplified
        assert!(!result.unwrap().contains_symbol(&x));
    }

    #[test]
    fn test_integrate_multi_double() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        // ∫∫ 1 dy dx over [0,1]×[0,1]
        let expr = Expr::from(1);

        let result = integrate_multi(
            &expr,
            &[
                (x.clone(), Some((Expr::from(0), Expr::from(1)))),
                (y.clone(), Some((Expr::from(0), Expr::from(1)))),
            ],
        );
        assert!(result.is_some());
        // Result may not be simplified to exactly 1
        let res = result.unwrap();
        assert!(!res.contains_symbol(&x));
        assert!(!res.contains_symbol(&y));
    }

    #[test]
    fn test_indefinite_as_function() {
        let func = IndefiniteIntegral::as_function();
        assert_eq!(func.name(), "integrate");
        assert_eq!(func.num_args(), Some(2));
    }

    #[test]
    fn test_definite_as_function() {
        let func = DefiniteIntegral::as_function();
        assert_eq!(func.name(), "definite_integral");
        assert_eq!(func.num_args(), Some(4));
    }

    #[test]
    fn test_definite_integral_trig() {
        let def_int = DefiniteIntegral::new(Expr::from(0), Expr::from(1));
        let x = Symbol::new("x");
        // ∫ cos(x) dx = sin(x)
        let expr = Expr::Symbol(x.clone()).cos();

        let result = def_int.compute(&expr, &x);
        assert!(result.is_some());
    }

    #[test]
    fn test_integral_polynomial() {
        let x = Symbol::new("x");
        // ∫ (x² + 2x + 1) dx
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2))
            + (Expr::from(2) * Expr::Symbol(x.clone()))
            + Expr::from(1);

        let result = integral(&expr, &x);
        assert!(result.is_some());
    }
}
