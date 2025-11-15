//! Variable and Function Creation for Symbolic Mathematics
//!
//! This module provides utilities for creating symbolic variables and functions,
//! mirroring SageMath's `sage.calculus.var` module functionality.
//!
//! # Examples
//!
//! ```
//! use rustmath_calculus::var::{var, vars};
//! use rustmath_symbolic::Expr;
//!
//! // Create a single variable
//! let x = var("x");
//!
//! // Create multiple variables
//! let (a, b, c) = vars(&["a", "b", "c"]);
//!
//! // Use in expressions
//! let expr = x.clone() * Expr::from(2) + Expr::from(1);
//! ```

use rustmath_symbolic::{Expr, Symbol};
use std::collections::HashMap;
use std::sync::Mutex;

// Global registry for tracking created variables
// This allows clear_vars to work properly
lazy_static::lazy_static! {
    static ref VAR_REGISTRY: Mutex<HashMap<String, Symbol>> = Mutex::new(HashMap::new());
}

/// Create a symbolic variable with the given name
///
/// This function creates a new symbolic variable that can be used in algebraic
/// expressions. The variable is registered globally for later cleanup.
///
/// # Arguments
///
/// * `name` - The name of the variable (must be a valid identifier)
///
/// # Returns
///
/// A symbolic expression representing the variable
///
/// # Examples
///
/// ```
/// use rustmath_calculus::var::var;
/// use rustmath_symbolic::Expr;
///
/// let x = var("x");
/// let y = var("y");
///
/// let expr = x + y * Expr::from(2);
/// ```
///
/// # Note
///
/// In SageMath, `var()` injects variables into the global namespace.
/// In Rust, we return the variable which the user must bind to a name.
pub fn var(name: &str) -> Expr {
    let symbol = Symbol::new(name);

    // Register the variable
    if let Ok(mut registry) = VAR_REGISTRY.lock() {
        registry.insert(name.to_string(), symbol.clone());
    }

    Expr::Symbol(symbol)
}

/// Create multiple symbolic variables from a slice of names
///
/// This is a convenience function for creating several variables at once.
///
/// # Arguments
///
/// * `names` - Slice of variable names
///
/// # Returns
///
/// Tuple of symbolic expressions (up to 12 variables supported)
///
/// # Examples
///
/// ```
/// use rustmath_calculus::var::vars;
///
/// let (x, y, z) = vars(&["x", "y", "z"]);
/// let (a, b) = vars(&["a", "b"]);
/// ```
pub fn vars<const N: usize>(names: &[&str; N]) -> Vec<Expr> {
    names.iter().map(|&name| var(name)).collect()
}

/// Create a symbolic function with the given name
///
/// Creates a function object that can be used for symbolic manipulation
/// without specifying the actual function implementation.
///
/// # Arguments
///
/// * `name` - The name of the function
/// * `num_args` - Optional number of arguments (None for variable arity)
///
/// # Returns
///
/// A function expression
///
/// # Examples
///
/// ```
/// use rustmath_calculus::var::{var, function};
/// use rustmath_symbolic::Expr;
///
/// let f = function("f", Some(1));
/// let x = var("x");
///
/// // Create f(x) - an unevaluated function call
/// let expr = Expr::FunctionCall(Box::new(f), vec![x]);
/// ```
///
/// # Note
///
/// This creates an abstract function placeholder. To define actual behavior,
/// use the rustmath-symbolic function definition mechanisms.
pub fn function(name: &str, _num_args: Option<usize>) -> Expr {
    // Create a function symbol
    // In a full implementation, we'd store metadata about num_args
    Expr::symbol(name)
}

/// Clear all single-letter symbolic variables
///
/// Removes all single-character variables (a-z, A-Z) from the variable registry.
/// This is useful for cleaning up the workspace between computations.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::var::{var, clear_vars};
///
/// let x = var("x");
/// let y = var("y");
/// let abc = var("abc"); // Multi-letter variable
///
/// clear_vars(); // Removes x and y, but not abc
/// ```
///
/// # Note
///
/// In SageMath, this clears predefined variables from the global namespace.
/// In Rust, this clears our internal registry and allows the user to
/// rebind the same variable names.
pub fn clear_vars() {
    if let Ok(mut registry) = VAR_REGISTRY.lock() {
        registry.retain(|name, _| name.len() != 1);
    }
}

/// Clear all variables from the registry
///
/// This clears all variables, not just single-letter ones.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::var::{var, clear_all_vars};
///
/// let x = var("x");
/// let foo = var("foo");
///
/// clear_all_vars(); // Removes both x and foo
/// ```
pub fn clear_all_vars() {
    if let Ok(mut registry) = VAR_REGISTRY.lock() {
        registry.clear();
    }
}

/// Get a list of all currently registered variables
///
/// # Returns
///
/// Vector of variable names currently in the registry
///
/// # Examples
///
/// ```
/// use rustmath_calculus::var::{var, get_vars};
///
/// let x = var("x");
/// let y = var("y");
///
/// let vars = get_vars();
/// assert!(vars.contains(&"x".to_string()));
/// assert!(vars.contains(&"y".to_string()));
/// ```
pub fn get_vars() -> Vec<String> {
    if let Ok(registry) = VAR_REGISTRY.lock() {
        registry.keys().cloned().collect()
    } else {
        vec![]
    }
}

/// Parse a space-separated string into variables
///
/// Convenience function to create variables from a string like "x y z"
///
/// # Arguments
///
/// * `s` - Space-separated variable names
///
/// # Returns
///
/// Vector of symbolic expressions
///
/// # Examples
///
/// ```
/// use rustmath_calculus::var::var_from_string;
///
/// let vars = var_from_string("x y z");
/// assert_eq!(vars.len(), 3);
/// ```
pub fn var_from_string(s: &str) -> Vec<Expr> {
    s.split_whitespace().map(var).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_creation() {
        let x = var("x");
        assert!(matches!(x, Expr::Symbol(_)));

        if let Expr::Symbol(sym) = &x {
            assert_eq!(sym.name(), "x");
        }
    }

    #[test]
    fn test_vars_multiple() {
        let vars_vec = vars(&["a", "b", "c"]);
        assert_eq!(vars_vec.len(), 3);

        if let Expr::Symbol(sym) = &vars_vec[0] {
            assert_eq!(sym.name(), "a");
        }

        if let Expr::Symbol(sym) = &vars_vec[2] {
            assert_eq!(sym.name(), "c");
        }
    }

    #[test]
    fn test_var_registry() {
        clear_all_vars(); // Start fresh

        let _x = var("test_var_x");
        let _y = var("test_var_y");

        let vars = get_vars();
        assert!(vars.contains(&"test_var_x".to_string()));
        assert!(vars.contains(&"test_var_y".to_string()));

        clear_all_vars();
    }

    #[test]
    fn test_clear_vars() {
        clear_all_vars();

        let _x = var("x");
        let _y = var("y");
        let _abc = var("abc");

        let before = get_vars();
        assert_eq!(before.len(), 3);

        clear_vars(); // Only clears single-letter vars

        let after = get_vars();
        assert_eq!(after.len(), 1);
        assert!(after.contains(&"abc".to_string()));

        clear_all_vars();
    }

    #[test]
    fn test_function_creation() {
        let f = function("f", Some(1));
        assert!(matches!(f, Expr::Symbol(_)));

        if let Expr::Symbol(sym) = &f {
            assert_eq!(sym.name(), "f");
        }
    }

    #[test]
    fn test_var_from_string() {
        clear_all_vars();

        let vars = var_from_string("x y z");
        assert_eq!(vars.len(), 3);

        if let Expr::Symbol(sym) = &vars[0] {
            assert_eq!(sym.name(), "x");
        }

        if let Expr::Symbol(sym) = &vars[1] {
            assert_eq!(sym.name(), "y");
        }

        if let Expr::Symbol(sym) = &vars[2] {
            assert_eq!(sym.name(), "z");
        }

        clear_all_vars();
    }

    #[test]
    fn test_var_in_expression() {
        let x = var("expr_test_x");
        let expr = x + Expr::from(1);

        // Should be a binary operation
        assert!(matches!(expr, Expr::Binary(..)));
    }
}
