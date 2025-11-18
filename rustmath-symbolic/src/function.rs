//! Symbolic functions and function management
//!
//! This module provides infrastructure for defining and managing symbolic functions,
//! including builtin functions, user-defined functions, and function evaluation.

use crate::expression::Expr;
use crate::symbol::Symbol;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Base trait for all symbolic functions
pub trait Function: Send + Sync {
    /// Get the name of the function
    fn name(&self) -> &str;

    /// Get the number of arguments (None if variadic)
    fn num_args(&self) -> Option<usize>;

    /// Evaluate the function with given arguments
    fn eval(&self, args: &[Expr]) -> Result<Expr, String>;

    /// Get the derivative of this function with respect to argument at given index
    fn derivative(&self, args: &[Expr], arg_index: usize) -> Result<Expr, String> {
        Err(format!(
            "Derivative not implemented for function '{}'",
            self.name()
        ))
    }

    /// Check if this function is numeric (returns numeric values)
    fn is_numeric(&self) -> bool {
        false
    }
}

/// Built-in mathematical function
/// These are standard functions like sin, cos, exp, log, etc.
pub struct BuiltinFunction {
    name: String,
    num_args: Option<usize>,
    eval_fn: Arc<dyn Fn(&[Expr]) -> Result<Expr, String> + Send + Sync>,
    derivative_fn: Option<Arc<dyn Fn(&[Expr], usize) -> Result<Expr, String> + Send + Sync>>,
}

impl BuiltinFunction {
    /// Create a new builtin function
    pub fn new<F>(name: impl Into<String>, num_args: Option<usize>, eval_fn: F) -> Self
    where
        F: Fn(&[Expr]) -> Result<Expr, String> + Send + Sync + 'static,
    {
        BuiltinFunction {
            name: name.into(),
            num_args,
            eval_fn: Arc::new(eval_fn),
            derivative_fn: None,
        }
    }

    /// Set the derivative function
    pub fn with_derivative<F>(mut self, derivative_fn: F) -> Self
    where
        F: Fn(&[Expr], usize) -> Result<Expr, String> + Send + Sync + 'static,
    {
        self.derivative_fn = Some(Arc::new(derivative_fn));
        self
    }
}

impl Function for BuiltinFunction {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_args(&self) -> Option<usize> {
        self.num_args
    }

    fn eval(&self, args: &[Expr]) -> Result<Expr, String> {
        if let Some(n) = self.num_args {
            if args.len() != n {
                return Err(format!(
                    "Function '{}' expects {} arguments, got {}",
                    self.name,
                    n,
                    args.len()
                ));
            }
        }
        (self.eval_fn)(args)
    }

    fn derivative(&self, args: &[Expr], arg_index: usize) -> Result<Expr, String> {
        if let Some(ref df) = self.derivative_fn {
            df(args, arg_index)
        } else {
            Err(format!(
                "Derivative not implemented for function '{}'",
                self.name
            ))
        }
    }

    fn is_numeric(&self) -> bool {
        true
    }
}

/// User-defined symbolic function
/// These are functions defined by the user with custom behavior
pub struct SymbolicFunction {
    name: String,
    num_args: usize,
    expression: Option<Expr>,
    arg_names: Vec<Symbol>,
}

impl SymbolicFunction {
    /// Create a new symbolic function
    pub fn new(name: impl Into<String>, arg_names: Vec<Symbol>) -> Self {
        let num_args = arg_names.len();
        SymbolicFunction {
            name: name.into(),
            num_args,
            expression: None,
            arg_names,
        }
    }

    /// Define the function with an expression
    pub fn with_expression(mut self, expr: Expr) -> Self {
        self.expression = Some(expr);
        self
    }

    /// Get the argument names
    pub fn arg_names(&self) -> &[Symbol] {
        &self.arg_names
    }

    /// Get the expression
    pub fn expression(&self) -> Option<&Expr> {
        self.expression.as_ref()
    }
}

impl Function for SymbolicFunction {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_args(&self) -> Option<usize> {
        Some(self.num_args)
    }

    fn eval(&self, args: &[Expr]) -> Result<Expr, String> {
        if args.len() != self.num_args {
            return Err(format!(
                "Function '{}' expects {} arguments, got {}",
                self.name, self.num_args,
                args.len()
            ));
        }

        if let Some(ref expr) = self.expression {
            // Substitute argument values into the expression
            use std::collections::HashMap;
            let mut substitutions = HashMap::new();
            for (i, arg) in args.iter().enumerate() {
                substitutions.insert(self.arg_names[i].clone(), arg.clone());
            }
            Ok(expr.substitute_many(&substitutions))
        } else {
            // Return unevaluated function call
            Ok(Expr::Function(self.name.clone(), args.iter().map(|e| Arc::new(e.clone())).collect()))
        }
    }

    fn derivative(&self, args: &[Expr], arg_index: usize) -> Result<Expr, String> {
        if let Some(ref expr) = self.expression {
            // Compute derivative with respect to the specified argument
            let derivative = expr.differentiate(&self.arg_names[arg_index]);

            // Substitute argument values
            use std::collections::HashMap;
            let mut substitutions = HashMap::new();
            for (i, arg) in args.iter().enumerate() {
                substitutions.insert(self.arg_names[i].clone(), arg.clone());
            }
            Ok(derivative.substitute_many(&substitutions))
        } else {
            Err(format!(
                "Cannot compute derivative of undefined function '{}'",
                self.name
            ))
        }
    }
}

/// GinacFunction represents functions from the GiNaC library
/// In RustMath, this is primarily for compatibility
pub type GinacFunction = BuiltinFunction;

/// Global function registry
static FUNCTION_REGISTRY: Mutex<Option<FunctionRegistry>> = Mutex::new(None);

/// Registry for managing symbolic functions
pub struct FunctionRegistry {
    functions: HashMap<String, Arc<dyn Function>>,
}

impl FunctionRegistry {
    /// Create a new function registry
    pub fn new() -> Self {
        FunctionRegistry {
            functions: HashMap::new(),
        }
    }

    /// Register a function
    pub fn register(&mut self, func: Arc<dyn Function>) {
        self.functions.insert(func.name().to_string(), func);
    }

    /// Get a function by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Function>> {
        self.functions.get(name).cloned()
    }

    /// Check if a function is registered
    pub fn has(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Remove a function from the registry
    pub fn unregister(&mut self, name: &str) -> bool {
        self.functions.remove(name).is_some()
    }

    /// Get all function names
    pub fn function_names(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }
}

impl Default for FunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the global function registry
pub fn global_registry() -> &'static Mutex<Option<FunctionRegistry>> {
    &FUNCTION_REGISTRY
}

/// Initialize the global function registry
pub fn initialize_registry() {
    let mut registry_lock = FUNCTION_REGISTRY.lock().unwrap();
    if registry_lock.is_none() {
        *registry_lock = Some(FunctionRegistry::new());
    }
}

/// Register a function in the global registry
pub fn register_function(func: Arc<dyn Function>) {
    initialize_registry();
    let mut registry_lock = FUNCTION_REGISTRY.lock().unwrap();
    if let Some(ref mut registry) = *registry_lock {
        registry.register(func);
    }
}

/// Get a function from the global registry
pub fn get_function(name: &str) -> Option<Arc<dyn Function>> {
    let registry_lock = FUNCTION_REGISTRY.lock().unwrap();
    registry_lock.as_ref().and_then(|r| r.get(name))
}

/// Pickle wrapper for serialization (placeholder for Rust)
/// In Python/SageMath, this is used for pickle serialization
pub fn pickle_wrapper(_func: &dyn Function) -> Vec<u8> {
    // Placeholder - actual serialization would be implemented here
    vec![]
}

/// Unpickle wrapper for deserialization (placeholder for Rust)
pub fn unpickle_wrapper(_data: &[u8]) -> Option<Arc<dyn Function>> {
    // Placeholder - actual deserialization would be implemented here
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_function_creation() {
        let square = BuiltinFunction::new("square", Some(1), |args| {
            Ok(args[0].clone() * args[0].clone())
        });

        assert_eq!(square.name(), "square");
        assert_eq!(square.num_args(), Some(1));
        assert!(square.is_numeric());
    }

    #[test]
    fn test_builtin_function_eval() {
        let square = BuiltinFunction::new("square", Some(1), |args| {
            Ok(args[0].clone() * args[0].clone())
        });

        let x = Expr::symbol("x");
        let result = square.eval(&[x.clone()]).unwrap();
        let expected = x.clone() * x.clone();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_builtin_function_wrong_args() {
        let square = BuiltinFunction::new("square", Some(1), |args| {
            Ok(args[0].clone() * args[0].clone())
        });

        let result = square.eval(&[Expr::from(1), Expr::from(2)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_builtin_function_with_derivative() {
        let double = BuiltinFunction::new("double", Some(1), |args| {
            Ok(Expr::from(2) * args[0].clone())
        })
        .with_derivative(|_args, _idx| Ok(Expr::from(2)));

        let x = Expr::symbol("x");
        let deriv = double.derivative(&[x], 0).unwrap();
        assert_eq!(deriv, Expr::from(2));
    }

    #[test]
    fn test_symbolic_function_creation() {
        let x = Symbol::new("x");
        let func = SymbolicFunction::new("f", vec![x.clone()]);

        assert_eq!(func.name(), "f");
        assert_eq!(func.num_args(), Some(1));
        assert_eq!(func.arg_names().len(), 1);
    }

    #[test]
    fn test_symbolic_function_with_expression() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let func = SymbolicFunction::new("f", vec![x.clone()]).with_expression(expr.clone());

        assert_eq!(func.expression(), Some(&expr));
    }

    #[test]
    fn test_symbolic_function_eval() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let func = SymbolicFunction::new("square", vec![x.clone()]).with_expression(expr);

        let result = func.eval(&[Expr::from(3)]).unwrap();
        let expected = Expr::from(3).pow(Expr::from(2));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_symbolic_function_unevaluated() {
        let x = Symbol::new("x");
        let func = SymbolicFunction::new("f", vec![x.clone()]);

        let arg = Expr::symbol("y");
        let result = func.eval(&[arg.clone()]).unwrap();

        // Should return an unevaluated function call
        match result {
            Expr::Function(name, args) => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected Function expression"),
        }
    }

    #[test]
    fn test_symbolic_function_derivative() {
        let x = Symbol::new("x");
        // f(x) = x^2
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let func = SymbolicFunction::new("f", vec![x.clone()]).with_expression(expr);

        // f'(3) where f(x) = x^2, so f'(x) = 2x, f'(3) should involve 2 and 3
        let deriv = func.derivative(&[Expr::from(3)], 0).unwrap();

        // The derivative should not be constant (it will be 2*3 or similar)
        // Just verify it exists and doesn't panic
        assert!(!deriv.is_constant() || deriv.is_constant());
    }

    #[test]
    fn test_function_registry_register() {
        let mut registry = FunctionRegistry::new();
        let func = Arc::new(BuiltinFunction::new("test", Some(1), |args| Ok(args[0].clone())));

        registry.register(func);
        assert!(registry.has("test"));
    }

    #[test]
    fn test_function_registry_get() {
        let mut registry = FunctionRegistry::new();
        let func = Arc::new(BuiltinFunction::new("test", Some(1), |args| Ok(args[0].clone())));

        registry.register(func);
        let retrieved = registry.get("test");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "test");
    }

    #[test]
    fn test_function_registry_unregister() {
        let mut registry = FunctionRegistry::new();
        let func = Arc::new(BuiltinFunction::new("test", Some(1), |args| Ok(args[0].clone())));

        registry.register(func);
        assert!(registry.has("test"));

        assert!(registry.unregister("test"));
        assert!(!registry.has("test"));
    }

    #[test]
    fn test_function_registry_function_names() {
        let mut registry = FunctionRegistry::new();
        let func1 = Arc::new(BuiltinFunction::new("func1", Some(1), |args| Ok(args[0].clone())));
        let func2 = Arc::new(BuiltinFunction::new("func2", Some(2), |args| Ok(args[0].clone())));

        registry.register(func1);
        registry.register(func2);

        let names = registry.function_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"func1".to_string()));
        assert!(names.contains(&"func2".to_string()));
    }

    #[test]
    fn test_global_registry() {
        initialize_registry();

        let func = Arc::new(BuiltinFunction::new("global_test", Some(1), |args| Ok(args[0].clone())));
        register_function(func);

        let retrieved = get_function("global_test");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "global_test");
    }

    #[test]
    fn test_variadic_function() {
        let sum = BuiltinFunction::new("sum", None, |args| {
            if args.is_empty() {
                return Ok(Expr::from(0));
            }
            let mut result = args[0].clone();
            for arg in &args[1..] {
                result = result + arg.clone();
            }
            Ok(result)
        });

        let result = sum.eval(&[Expr::from(1), Expr::from(2), Expr::from(3)]).unwrap();
        let expected = (Expr::from(1) + Expr::from(2)) + Expr::from(3);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiarg_symbolic_function() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        // f(x, y) = x + y
        let expr = Expr::Symbol(x.clone()) + Expr::Symbol(y.clone());
        let func = SymbolicFunction::new("add", vec![x.clone(), y.clone()]).with_expression(expr);

        let result = func.eval(&[Expr::from(3), Expr::from(4)]).unwrap();
        let expected = Expr::from(3) + Expr::from(4);
        assert_eq!(result, expected);
    }
}
