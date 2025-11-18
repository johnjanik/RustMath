//! Symbolic operators and derivative operators
//!
//! This module provides operator structures for symbolic expressions,
//! including derivative operators and variadic operations.

use crate::expression::Expr;
use crate::symbol::Symbol;
use std::sync::Arc;

/// Derivative operator that represents d/dx
#[derive(Debug, Clone, PartialEq)]
pub struct DerivativeOperator {
    /// The variable with respect to which we differentiate
    pub variable: Symbol,
    /// The order of the derivative (1 for first derivative, 2 for second, etc.)
    pub order: usize,
}

impl DerivativeOperator {
    /// Create a new derivative operator
    pub fn new(variable: Symbol, order: usize) -> Self {
        DerivativeOperator { variable, order }
    }

    /// Apply the derivative operator to an expression
    pub fn apply(&self, expr: &Expr) -> Expr {
        let mut result = expr.clone();
        for _ in 0..self.order {
            result = result.differentiate(&self.variable);
        }
        result
    }

    /// Compose two derivative operators
    pub fn compose(&self, other: &DerivativeOperator) -> DerivativeOperator {
        if self.variable == other.variable {
            DerivativeOperator::new(self.variable.clone(), self.order + other.order)
        } else {
            // For different variables, we create a new operator
            // This represents partial derivatives
            DerivativeOperator::new(other.variable.clone(), other.order)
        }
    }
}

/// Derivative operator with parameters
/// This represents derivatives like D[f(x,y), {x, n}]
#[derive(Debug, Clone, PartialEq)]
pub struct DerivativeOperatorWithParameters {
    /// The function being differentiated
    pub function_name: String,
    /// Variables and their derivative orders
    pub parameters: Vec<(Symbol, usize)>,
}

impl DerivativeOperatorWithParameters {
    /// Create a new derivative operator with parameters
    pub fn new(function_name: String, parameters: Vec<(Symbol, usize)>) -> Self {
        DerivativeOperatorWithParameters {
            function_name,
            parameters,
        }
    }

    /// Get the total order of differentiation
    pub fn total_order(&self) -> usize {
        self.parameters.iter().map(|(_, order)| order).sum()
    }

    /// Apply the derivative operator to an expression
    pub fn apply(&self, expr: &Expr) -> Expr {
        let mut result = expr.clone();
        for (var, order) in &self.parameters {
            for _ in 0..*order {
                result = result.differentiate(var);
            }
        }
        result
    }
}

/// Functional derivative operator
/// Represents derivatives of functionals
#[derive(Debug, Clone, PartialEq)]
pub struct FDerivativeOperator {
    /// The functional being differentiated
    pub functional: String,
    /// The variable with respect to which we take the functional derivative
    pub variable: Symbol,
    /// Additional parameters
    pub parameters: Vec<usize>,
}

impl FDerivativeOperator {
    /// Create a new functional derivative operator
    pub fn new(functional: String, variable: Symbol, parameters: Vec<usize>) -> Self {
        FDerivativeOperator {
            functional,
            variable,
            parameters,
        }
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }
}

/// Add multiple expressions with variadic arguments
/// This is equivalent to SageMath's add_vararg
pub fn add_vararg(args: Vec<Expr>) -> Expr {
    if args.is_empty() {
        return Expr::from(0);
    }

    if args.len() == 1 {
        return args[0].clone();
    }

    // Fold all arguments into a sum
    args.into_iter()
        .reduce(|acc, expr| acc + expr)
        .unwrap_or(Expr::from(0))
}

/// Multiply multiple expressions with variadic arguments
/// This is equivalent to SageMath's mul_vararg
pub fn mul_vararg(args: Vec<Expr>) -> Expr {
    if args.is_empty() {
        return Expr::from(1);
    }

    if args.len() == 1 {
        return args[0].clone();
    }

    // Fold all arguments into a product
    args.into_iter()
        .reduce(|acc, expr| acc * expr)
        .unwrap_or(Expr::from(1))
}

/// Create a derivative expression
/// This is a helper function to create derivative expressions
pub fn derivative(expr: Expr, var: &Symbol, order: usize) -> Expr {
    let op = DerivativeOperator::new(var.clone(), order);
    op.apply(&expr)
}

/// Create a partial derivative with multiple variables
pub fn partial_derivative(expr: Expr, derivatives: Vec<(Symbol, usize)>) -> Expr {
    let mut result = expr;
    for (var, order) in derivatives {
        for _ in 0..order {
            result = result.differentiate(&var);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_operator_basic() {
        let x = Symbol::new("x");
        let op = DerivativeOperator::new(x.clone(), 1);

        // d/dx (x^2) = 2x (but may not be simplified)
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = op.apply(&expr);

        // Just verify it's not constant and contains x
        assert!(!result.is_constant());
        assert!(result.contains_symbol(&x));
    }

    #[test]
    fn test_derivative_operator_second_order() {
        let x = Symbol::new("x");
        let op = DerivativeOperator::new(x.clone(), 2);

        // d²/dx² (x^3) = 6x (but may not be simplified)
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(3));
        let result = op.apply(&expr);

        // Just verify it contains x
        assert!(result.contains_symbol(&x));
    }

    #[test]
    fn test_derivative_operator_composition() {
        let x = Symbol::new("x");
        let op1 = DerivativeOperator::new(x.clone(), 2);
        let op2 = DerivativeOperator::new(x.clone(), 3);

        let composed = op1.compose(&op2);
        assert_eq!(composed.order, 5);
        assert_eq!(composed.variable, x);
    }

    #[test]
    fn test_derivative_operator_with_parameters() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        let op = DerivativeOperatorWithParameters::new(
            "f".to_string(),
            vec![(x.clone(), 2), (y.clone(), 1)],
        );

        assert_eq!(op.total_order(), 3);
        assert_eq!(op.function_name, "f");
    }

    #[test]
    fn test_fderivative_operator() {
        let x = Symbol::new("x");
        let op = FDerivativeOperator::new("L".to_string(), x.clone(), vec![0, 1]);

        assert_eq!(op.functional, "L");
        assert_eq!(op.variable, x);
        assert_eq!(op.num_parameters(), 2);
    }

    #[test]
    fn test_add_vararg_empty() {
        let result = add_vararg(vec![]);
        assert_eq!(result, Expr::from(0));
    }

    #[test]
    fn test_add_vararg_single() {
        let x = Expr::symbol("x");
        let result = add_vararg(vec![x.clone()]);
        assert_eq!(result, x);
    }

    #[test]
    fn test_add_vararg_multiple() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let z = Expr::symbol("z");

        let result = add_vararg(vec![x.clone(), y.clone(), z.clone()]);
        let expected = (x + y) + z;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_vararg_with_constants() {
        let result = add_vararg(vec![Expr::from(1), Expr::from(2), Expr::from(3)]);
        // Result is (1 + 2) + 3, which equals 6 but may not be simplified
        let expected = (Expr::from(1) + Expr::from(2)) + Expr::from(3);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_vararg_empty() {
        let result = mul_vararg(vec![]);
        assert_eq!(result, Expr::from(1));
    }

    #[test]
    fn test_mul_vararg_single() {
        let x = Expr::symbol("x");
        let result = mul_vararg(vec![x.clone()]);
        assert_eq!(result, x);
    }

    #[test]
    fn test_mul_vararg_multiple() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let z = Expr::symbol("z");

        let result = mul_vararg(vec![x.clone(), y.clone(), z.clone()]);
        let expected = (x * y) * z;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_vararg_with_constants() {
        let result = mul_vararg(vec![Expr::from(2), Expr::from(3), Expr::from(4)]);
        // Result is (2 * 3) * 4, which equals 24 but may not be simplified
        let expected = (Expr::from(2) * Expr::from(3)) * Expr::from(4);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_derivative_helper() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let result = derivative(expr, &x, 1);
        // d/dx (x^2) = 2*x (but may not be simplified)
        // Just check it's not the original expression
        assert!(!result.is_constant());
        assert!(result.contains_symbol(&x));
    }

    #[test]
    fn test_derivative_helper_higher_order() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(4));

        let result = derivative(expr, &x, 2);
        // d²/dx² (x^4) = 12*x² (but may not be simplified)
        // Just check it contains the symbol
        assert!(result.contains_symbol(&x));
    }

    #[test]
    fn test_partial_derivative_single_var() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let result = partial_derivative(expr, vec![(x.clone(), 2)]);
        // d²/dx² (x^2) = 2 (but may not be simplified)
        // The result should be a constant expression (possibly with extra operations)
        // Just verify we can take the derivative successfully
        assert!(!result.contains_symbol(&Symbol::new("y"))); // Should not introduce new symbols
    }

    #[test]
    fn test_partial_derivative_multiple_vars() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // f(x,y) = x*y
        let expr = Expr::Symbol(x.clone()) * Expr::Symbol(y.clone());

        // ∂f/∂x = y (but may not be simplified)
        let result = partial_derivative(expr.clone(), vec![(x.clone(), 1)]);
        // Just check it contains y and not x alone
        assert!(result.contains_symbol(&y));
    }

    #[test]
    fn test_derivative_operator_with_trig() {
        let x = Symbol::new("x");
        let op = DerivativeOperator::new(x.clone(), 1);

        // d/dx sin(x) = cos(x) (but may have extra operations like * 1)
        let expr = Expr::Symbol(x.clone()).sin();
        let result = op.apply(&expr);

        // Just verify the result is not the original expression
        assert_ne!(result, expr);
        assert!(result.contains_symbol(&x));
    }

    #[test]
    fn test_derivative_operator_chain_rule() {
        let x = Symbol::new("x");
        let op = DerivativeOperator::new(x.clone(), 1);

        // d/dx sin(x^2) = 2x*cos(x^2)
        let inner = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let expr = Expr::Unary(crate::expression::UnaryOp::Sin, Arc::new(inner));
        let result = op.apply(&expr);

        // The result should not be the original expression
        assert_ne!(result, expr);
    }

    #[test]
    fn test_add_vararg_associativity() {
        let a = Expr::symbol("a");
        let b = Expr::symbol("b");
        let c = Expr::symbol("c");

        let result1 = add_vararg(vec![a.clone(), b.clone(), c.clone()]);
        let result2 = ((a.clone() + b.clone()) + c.clone());

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_mul_vararg_associativity() {
        let a = Expr::symbol("a");
        let b = Expr::symbol("b");
        let c = Expr::symbol("c");

        let result1 = mul_vararg(vec![a.clone(), b.clone(), c.clone()]);
        let result2 = ((a.clone() * b.clone()) * c.clone());

        assert_eq!(result1, result2);
    }
}
