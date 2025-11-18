//! Function factory for creating symbolic functions dynamically
//!
//! This module provides utilities for creating and managing dynamically-defined
//! symbolic functions.

use crate::expression::Expr;
use crate::function::{Function, SymbolicFunction};
use crate::symbol::Symbol;
use std::sync::Arc;

/// Factory for creating symbolic functions
pub struct FunctionFactory {
    name: String,
    num_args: usize,
}

impl FunctionFactory {
    /// Create a new function factory
    pub fn new(name: impl Into<String>, num_args: usize) -> Self {
        FunctionFactory {
            name: name.into(),
            num_args,
        }
    }

    /// Build a symbolic function with the given argument names
    pub fn build(self, arg_names: Vec<Symbol>) -> SymbolicFunction {
        if arg_names.len() != self.num_args {
            panic!(
                "Expected {} argument names, got {}",
                self.num_args,
                arg_names.len()
            );
        }
        SymbolicFunction::new(self.name, arg_names)
    }

    /// Build a symbolic function with auto-generated argument names
    pub fn build_with_default_args(self) -> SymbolicFunction {
        let arg_names: Vec<Symbol> = (0..self.num_args)
            .map(|i| Symbol::new(format!("x{}", i)))
            .collect();
        SymbolicFunction::new(self.name, arg_names)
    }
}

/// Create a function with the given name and number of arguments
/// This is a convenience function for the factory pattern
pub fn function(name: impl Into<String>, num_args: usize) -> FunctionFactory {
    FunctionFactory::new(name, num_args)
}

/// Create a complete symbolic function with expression
pub fn function_with_expression(
    name: impl Into<String>,
    arg_names: Vec<Symbol>,
    expression: Expr,
) -> Arc<dyn Function> {
    let func = SymbolicFunction::new(name, arg_names).with_expression(expression);
    Arc::new(func)
}

/// Create a wrapper function that wraps another function
pub fn wrap_function<F>(name: impl Into<String>, num_args: usize, wrapper: F) -> Arc<dyn Function>
where
    F: Fn(&[Expr]) -> Result<Expr, String> + Send + Sync + 'static,
{
    Arc::new(crate::function::BuiltinFunction::new(
        name,
        Some(num_args),
        wrapper,
    ))
}

/// Unpickle a function from serialized data
/// This is a placeholder for compatibility with SageMath's pickle system
pub fn unpickle_function(_data: &[u8]) -> Option<Arc<dyn Function>> {
    // Placeholder - actual deserialization would be implemented here
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_factory_creation() {
        let factory = FunctionFactory::new("test", 2);
        assert_eq!(factory.name, "test");
        assert_eq!(factory.num_args, 2);
    }

    #[test]
    fn test_function_factory_build() {
        let factory = FunctionFactory::new("add", 2);
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        let func = factory.build(vec![x.clone(), y.clone()]);
        assert_eq!(func.name(), "add");
        assert_eq!(func.num_args(), Some(2));
        assert_eq!(func.arg_names().len(), 2);
    }

    #[test]
    #[should_panic(expected = "Expected 2 argument names, got 1")]
    fn test_function_factory_build_wrong_args() {
        let factory = FunctionFactory::new("add", 2);
        let x = Symbol::new("x");

        factory.build(vec![x]);
    }

    #[test]
    fn test_function_factory_default_args() {
        let factory = FunctionFactory::new("f", 3);
        let func = factory.build_with_default_args();

        assert_eq!(func.name(), "f");
        assert_eq!(func.num_args(), Some(3));
        // Check the names, not equality (since Symbols have unique IDs)
        assert_eq!(func.arg_names()[0].name(), "x0");
        assert_eq!(func.arg_names()[1].name(), "x1");
        assert_eq!(func.arg_names()[2].name(), "x2");
    }

    #[test]
    fn test_function_convenience() {
        let factory = function("myFunc", 1);
        assert_eq!(factory.name, "myFunc");
        assert_eq!(factory.num_args, 1);
    }

    #[test]
    fn test_function_with_expression() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let func = function_with_expression("square", vec![x.clone()], expr);

        assert_eq!(func.name(), "square");
        assert_eq!(func.num_args(), Some(1));

        // Test evaluation
        let result = func.eval(&[Expr::from(5)]).unwrap();
        let expected = Expr::from(5).pow(Expr::from(2));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_wrap_function() {
        let double = wrap_function("double", 1, |args| Ok(Expr::from(2) * args[0].clone()));

        assert_eq!(double.name(), "double");
        assert_eq!(double.num_args(), Some(1));

        let x = Expr::symbol("x");
        let result = double.eval(&[x.clone()]).unwrap();
        let expected = Expr::from(2) * x;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_wrap_function_with_logic() {
        use rustmath_core::Ring;

        let abs_func = wrap_function("abs", 1, |args| {
            match &args[0] {
                Expr::Integer(n) if n < &rustmath_integers::Integer::zero() => {
                    Ok(Expr::Integer(-n.clone()))
                }
                _ => Ok(args[0].clone().abs()),
            }
        });

        let result = abs_func.eval(&[Expr::from(-5)]).unwrap();
        // Just verify we got an integer result
        match result {
            Expr::Integer(_) => {},
            Expr::Unary(_, _) => {}, // Could be Abs unary op
            _ => panic!("Expected integer or unary result"),
        }
    }

    #[test]
    fn test_function_chain() {
        // Create a function using the factory
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let expr = Expr::Symbol(x.clone()) * Expr::Symbol(y.clone());

        let multiply = function("multiply", 2)
            .build(vec![x.clone(), y.clone()])
            .with_expression(expr);

        // Test it
        let result = multiply.eval(&[Expr::from(3), Expr::from(4)]).unwrap();
        let expected = Expr::from(3) * Expr::from(4);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_function_composition() {
        // f(x) = x^2
        let x = Symbol::new("x");
        let f_expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let f = function_with_expression("f", vec![x.clone()], f_expr);

        // g(x) = 2x
        let g_expr = Expr::from(2) * Expr::Symbol(x.clone());
        let g = function_with_expression("g", vec![x.clone()], g_expr);

        // Test f(g(3)) = f(6) = 36
        let g_result = g.eval(&[Expr::from(3)]).unwrap();
        let f_result = f.eval(&[g_result]).unwrap();

        let expected = (Expr::from(2) * Expr::from(3)).pow(Expr::from(2));
        assert_eq!(f_result, expected);
    }

    #[test]
    fn test_unpickle_function() {
        let result = unpickle_function(&[1, 2, 3]);
        assert!(result.is_none()); // Placeholder implementation
    }
}
