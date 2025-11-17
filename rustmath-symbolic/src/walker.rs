//! Expression tree walking infrastructure
//!
//! This module provides visitor and mutator traits for traversing and transforming
//! symbolic expression trees. These are critical for differential geometry operations
//! like chart transitions, coordinate transformations, and symbolic differentiation.

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Trait for visiting expression trees (read-only traversal)
///
/// Implement this trait to analyze expressions without modifying them.
/// Common use cases: collecting symbols, counting operations, checking properties.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::{Expr, walker::{ExprVisitor, SymbolCollector}};
///
/// let x = Expr::symbol("x");
/// let y = Expr::symbol("y");
/// let expr = x.clone() + y.clone() * Expr::from(2);
///
/// let mut collector = SymbolCollector::new();
/// collector.visit(&expr);
/// assert_eq!(collector.symbols().len(), 2);
/// ```
pub trait ExprVisitor {
    /// Output type for visit operations
    type Output;

    /// Visit an integer constant
    fn visit_integer(&mut self, value: &Integer) -> Self::Output;

    /// Visit a rational constant
    fn visit_rational(&mut self, value: &Rational) -> Self::Output;

    /// Visit a symbol
    fn visit_symbol(&mut self, symbol: &Symbol) -> Self::Output;

    /// Visit a binary operation
    fn visit_binary(&mut self, op: BinaryOp, left: &Expr, right: &Expr) -> Self::Output;

    /// Visit a unary operation
    fn visit_unary(&mut self, op: UnaryOp, inner: &Expr) -> Self::Output;

    /// Visit a function call
    fn visit_function(&mut self, name: &str, args: &[Arc<Expr>]) -> Self::Output;

    /// Main visit dispatch method
    fn visit(&mut self, expr: &Expr) -> Self::Output {
        match expr {
            Expr::Integer(i) => self.visit_integer(i),
            Expr::Rational(r) => self.visit_rational(r),
            Expr::Symbol(s) => self.visit_symbol(s),
            Expr::Binary(op, left, right) => self.visit_binary(*op, left, right),
            Expr::Unary(op, inner) => self.visit_unary(*op, inner),
            Expr::Function(name, args) => self.visit_function(name, args),
        }
    }
}

/// Trait for mutating expression trees (transforming expressions)
///
/// Implement this trait to transform expressions into new expressions.
/// Common use cases: substitution, simplification, coordinate transformations.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::{Expr, walker::{ExprMutator, Substituter}};
/// use std::collections::HashMap;
///
/// let x = Expr::symbol("x");
/// let expr = x.clone() + Expr::from(1);
///
/// let mut substituter = Substituter::new();
/// substituter.add_replacement(&x.as_symbol().unwrap(), Expr::from(2));
///
/// let result = substituter.mutate(&expr);
/// // result is now 2 + 1 = 3
/// ```
pub trait ExprMutator {
    /// Transform an expression (default: traverse and rebuild)
    fn mutate(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Symbol(_) => expr.clone(),
            Expr::Binary(op, left, right) => {
                let new_left = self.mutate(left);
                let new_right = self.mutate(right);
                Expr::Binary(*op, Arc::new(new_left), Arc::new(new_right))
            }
            Expr::Unary(op, inner) => {
                let new_inner = self.mutate(inner);
                Expr::Unary(*op, Arc::new(new_inner))
            }
            Expr::Function(name, args) => {
                let new_args: Vec<Arc<Expr>> = args
                    .iter()
                    .map(|a| Arc::new(self.mutate(a)))
                    .collect();
                Expr::Function(name.clone(), new_args)
            }
        }
    }
}

// ============================================================================
// Concrete Visitor Implementations
// ============================================================================

/// Collects all free symbols in an expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::{Expr, walker::SymbolCollector};
///
/// let x = Expr::symbol("x");
/// let y = Expr::symbol("y");
/// let expr = x.clone().pow(Expr::from(2)) + y.clone().sin();
///
/// let mut collector = SymbolCollector::new();
/// collector.visit(&expr);
/// let symbols = collector.symbols();
/// assert_eq!(symbols.len(), 2);
/// ```
pub struct SymbolCollector {
    symbols: HashSet<Symbol>,
}

impl SymbolCollector {
    /// Create a new symbol collector
    pub fn new() -> Self {
        Self {
            symbols: HashSet::new(),
        }
    }

    /// Get the collected symbols
    pub fn symbols(&self) -> &HashSet<Symbol> {
        &self.symbols
    }

    /// Get symbols as a sorted vector
    pub fn symbols_sorted(&self) -> Vec<Symbol> {
        let mut vec: Vec<Symbol> = self.symbols.iter().cloned().collect();
        vec.sort_by(|a, b| a.name().cmp(b.name()));
        vec
    }

    /// Check if a specific symbol appears in the expression
    pub fn contains(&self, symbol: &Symbol) -> bool {
        self.symbols.contains(symbol)
    }

    /// Get the number of unique symbols
    pub fn count(&self) -> usize {
        self.symbols.len()
    }
}

impl Default for SymbolCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprVisitor for SymbolCollector {
    type Output = ();

    fn visit_integer(&mut self, _value: &Integer) -> Self::Output {}

    fn visit_rational(&mut self, _value: &Rational) -> Self::Output {}

    fn visit_symbol(&mut self, symbol: &Symbol) -> Self::Output {
        self.symbols.insert(symbol.clone());
    }

    fn visit_binary(&mut self, _op: BinaryOp, left: &Expr, right: &Expr) -> Self::Output {
        self.visit(left);
        self.visit(right);
    }

    fn visit_unary(&mut self, _op: UnaryOp, inner: &Expr) -> Self::Output {
        self.visit(inner);
    }

    fn visit_function(&mut self, _name: &str, args: &[Arc<Expr>]) -> Self::Output {
        for arg in args {
            self.visit(arg);
        }
    }
}

// ============================================================================
// Concrete Mutator Implementations
// ============================================================================

/// Substitutes symbols with expressions
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::{Expr, Symbol, walker::Substituter};
///
/// let x = Symbol::new("x");
/// let y = Symbol::new("y");
/// let expr = Expr::Symbol(x.clone()) + Expr::Symbol(y.clone());
///
/// let mut substituter = Substituter::new();
/// substituter.add_replacement(&x, Expr::from(2));
/// substituter.add_replacement(&y, Expr::from(3));
///
/// let result = substituter.mutate(&expr);
/// // result is 2 + 3
/// ```
pub struct Substituter {
    replacements: HashMap<Symbol, Expr>,
}

impl Substituter {
    /// Create a new substituter
    pub fn new() -> Self {
        Self {
            replacements: HashMap::new(),
        }
    }

    /// Add a replacement rule: symbol -> expression
    pub fn add_replacement(&mut self, symbol: &Symbol, expr: Expr) {
        self.replacements.insert(symbol.clone(), expr);
    }

    /// Create a substituter from a hashmap
    pub fn from_map(replacements: HashMap<Symbol, Expr>) -> Self {
        Self { replacements }
    }

    /// Get the number of replacement rules
    pub fn num_rules(&self) -> usize {
        self.replacements.len()
    }

    /// Check if a symbol has a replacement
    pub fn has_replacement(&self, symbol: &Symbol) -> bool {
        self.replacements.contains_key(symbol)
    }
}

impl Default for Substituter {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprMutator for Substituter {
    fn mutate(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Symbol(s) => {
                if let Some(replacement) = self.replacements.get(s) {
                    replacement.clone()
                } else {
                    expr.clone()
                }
            }
            Expr::Integer(_) | Expr::Rational(_) => expr.clone(),
            Expr::Binary(op, left, right) => {
                let new_left = self.mutate(left);
                let new_right = self.mutate(right);
                Expr::Binary(*op, Arc::new(new_left), Arc::new(new_right))
            }
            Expr::Unary(op, inner) => {
                let new_inner = self.mutate(inner);
                Expr::Unary(*op, Arc::new(new_inner))
            }
            Expr::Function(name, args) => {
                let new_args: Vec<Arc<Expr>> = args
                    .iter()
                    .map(|a| Arc::new(self.mutate(a)))
                    .collect();
                Expr::Function(name.clone(), new_args)
            }
        }
    }
}

/// Counts operations in an expression tree
///
/// Useful for complexity analysis and optimization decisions.
pub struct OperationCounter {
    binary_ops: usize,
    unary_ops: usize,
    function_calls: usize,
    total_nodes: usize,
}

impl OperationCounter {
    /// Create a new operation counter
    pub fn new() -> Self {
        Self {
            binary_ops: 0,
            unary_ops: 0,
            function_calls: 0,
            total_nodes: 0,
        }
    }

    /// Get total number of binary operations
    pub fn binary_ops(&self) -> usize {
        self.binary_ops
    }

    /// Get total number of unary operations
    pub fn unary_ops(&self) -> usize {
        self.unary_ops
    }

    /// Get total number of function calls
    pub fn function_calls(&self) -> usize {
        self.function_calls
    }

    /// Get total number of nodes in the tree
    pub fn total_nodes(&self) -> usize {
        self.total_nodes
    }

    /// Get total number of operations (binary + unary + functions)
    pub fn total_ops(&self) -> usize {
        self.binary_ops + self.unary_ops + self.function_calls
    }
}

impl Default for OperationCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprVisitor for OperationCounter {
    type Output = ();

    fn visit_integer(&mut self, _value: &Integer) -> Self::Output {
        self.total_nodes += 1;
    }

    fn visit_rational(&mut self, _value: &Rational) -> Self::Output {
        self.total_nodes += 1;
    }

    fn visit_symbol(&mut self, _symbol: &Symbol) -> Self::Output {
        self.total_nodes += 1;
    }

    fn visit_binary(&mut self, _op: BinaryOp, left: &Expr, right: &Expr) -> Self::Output {
        self.binary_ops += 1;
        self.total_nodes += 1;
        self.visit(left);
        self.visit(right);
    }

    fn visit_unary(&mut self, _op: UnaryOp, inner: &Expr) -> Self::Output {
        self.unary_ops += 1;
        self.total_nodes += 1;
        self.visit(inner);
    }

    fn visit_function(&mut self, _name: &str, args: &[Arc<Expr>]) -> Self::Output {
        self.function_calls += 1;
        self.total_nodes += 1;
        for arg in args {
            self.visit(arg);
        }
    }
}

/// Depth calculator for expression trees
///
/// Computes the maximum depth (nesting level) of an expression.
pub struct DepthCalculator {
    current_depth: usize,
    max_depth: usize,
}

impl DepthCalculator {
    /// Create a new depth calculator
    pub fn new() -> Self {
        Self {
            current_depth: 0,
            max_depth: 0,
        }
    }

    /// Get the maximum depth
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    fn update_depth(&mut self) {
        if self.current_depth > self.max_depth {
            self.max_depth = self.current_depth;
        }
    }
}

impl Default for DepthCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprVisitor for DepthCalculator {
    type Output = ();

    fn visit_integer(&mut self, _value: &Integer) -> Self::Output {
        self.current_depth += 1;
        self.update_depth();
        self.current_depth -= 1;
    }

    fn visit_rational(&mut self, _value: &Rational) -> Self::Output {
        self.current_depth += 1;
        self.update_depth();
        self.current_depth -= 1;
    }

    fn visit_symbol(&mut self, _symbol: &Symbol) -> Self::Output {
        self.current_depth += 1;
        self.update_depth();
        self.current_depth -= 1;
    }

    fn visit_binary(&mut self, _op: BinaryOp, left: &Expr, right: &Expr) -> Self::Output {
        self.current_depth += 1;
        self.update_depth();
        self.visit(left);
        self.visit(right);
        self.current_depth -= 1;
    }

    fn visit_unary(&mut self, _op: UnaryOp, inner: &Expr) -> Self::Output {
        self.current_depth += 1;
        self.update_depth();
        self.visit(inner);
        self.current_depth -= 1;
    }

    fn visit_function(&mut self, _name: &str, args: &[Arc<Expr>]) -> Self::Output {
        self.current_depth += 1;
        self.update_depth();
        for arg in args {
            self.visit(arg);
        }
        self.current_depth -= 1;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Collect all free symbols in an expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::{Expr, walker::collect_symbols};
///
/// let x = Expr::symbol("x");
/// let y = Expr::symbol("y");
/// let expr = x + y * Expr::from(2);
///
/// let symbols = collect_symbols(&expr);
/// assert_eq!(symbols.len(), 2);
/// ```
pub fn collect_symbols(expr: &Expr) -> Vec<Symbol> {
    let mut collector = SymbolCollector::new();
    collector.visit(expr);
    collector.symbols_sorted()
}

/// Substitute symbols in an expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::{Expr, Symbol, walker::substitute};
/// use std::collections::HashMap;
///
/// let x = Symbol::new("x");
/// let expr = Expr::Symbol(x.clone()) * Expr::from(2);
///
/// let mut map = HashMap::new();
/// map.insert(x, Expr::from(5));
///
/// let result = substitute(&expr, &map);
/// // result is 5 * 2
/// ```
pub fn substitute(expr: &Expr, replacements: &HashMap<Symbol, Expr>) -> Expr {
    let mut substituter = Substituter::from_map(replacements.clone());
    substituter.mutate(expr)
}

/// Count total operations in an expression
pub fn count_operations(expr: &Expr) -> usize {
    let mut counter = OperationCounter::new();
    counter.visit(expr);
    counter.total_ops()
}

/// Calculate the depth of an expression tree
pub fn calculate_depth(expr: &Expr) -> usize {
    let mut calculator = DepthCalculator::new();
    calculator.visit(expr);
    calculator.max_depth()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::Expr;

    #[test]
    fn test_symbol_collector() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let z = Expr::symbol("z");

        let expr = x.clone() + y.clone() * z.clone();

        let mut collector = SymbolCollector::new();
        collector.visit(&expr);

        assert_eq!(collector.count(), 3);

        // Check that the collected symbols have the right names
        let symbols = collector.symbols_sorted();
        let names: Vec<&str> = symbols.iter().map(|s| s.name()).collect();
        assert_eq!(names, vec!["x", "y", "z"]);
    }

    #[test]
    fn test_symbol_collector_duplicates() {
        let x = Expr::symbol("x");
        let expr = x.clone() + x.clone() * x.clone(); // x + x*x

        let mut collector = SymbolCollector::new();
        collector.visit(&expr);

        assert_eq!(collector.count(), 1); // Only one unique symbol
    }

    #[test]
    fn test_substituter() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        let expr = Expr::Symbol(x.clone()) + Expr::Symbol(y.clone());

        let mut substituter = Substituter::new();
        substituter.add_replacement(&x, Expr::from(2));
        substituter.add_replacement(&y, Expr::from(3));

        let result = substituter.mutate(&expr);

        // Result should be: 2 + 3
        assert!(matches!(result, Expr::Binary(BinaryOp::Add, _, _)));
    }

    #[test]
    fn test_substituter_partial() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        let expr = Expr::Symbol(x.clone()) + Expr::Symbol(y.clone());

        let mut substituter = Substituter::new();
        substituter.add_replacement(&x, Expr::from(5));

        let result = substituter.mutate(&expr);

        // Result should still have y as a symbol
        let symbols = collect_symbols(&result);
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].name(), "y");
    }

    #[test]
    fn test_collect_symbols_helper() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let expr = x.clone().pow(Expr::from(2)) + y.clone().sin();

        let symbols = collect_symbols(&expr);
        assert_eq!(symbols.len(), 2);
    }

    #[test]
    fn test_operation_counter() {
        let x = Expr::symbol("x");
        let expr = x.clone() + x.clone() * Expr::from(2); // x + (x * 2)

        let mut counter = OperationCounter::new();
        counter.visit(&expr);

        assert_eq!(counter.binary_ops(), 2); // + and *
        assert_eq!(counter.unary_ops(), 0);
        assert_eq!(counter.function_calls(), 0);
        assert!(counter.total_nodes() > 0);
    }

    #[test]
    fn test_operation_counter_with_functions() {
        let x = Expr::symbol("x");
        let expr = x.clone().sin() + x.clone().cos(); // sin(x) + cos(x)

        let mut counter = OperationCounter::new();
        counter.visit(&expr);

        assert_eq!(counter.binary_ops(), 1); // +
        assert_eq!(counter.unary_ops(), 2); // sin and cos
    }

    #[test]
    fn test_depth_calculator() {
        let x = Expr::symbol("x");
        let expr = x.clone() + Expr::from(1); // Depth 2

        let mut calc = DepthCalculator::new();
        calc.visit(&expr);

        assert_eq!(calc.max_depth(), 2);
    }

    #[test]
    fn test_depth_calculator_nested() {
        let x = Expr::symbol("x");
        let expr = (x.clone() + Expr::from(1)).pow(Expr::from(2)); // ((x + 1)^2) - Depth 3

        let mut calc = DepthCalculator::new();
        calc.visit(&expr);

        assert_eq!(calc.max_depth(), 3);
    }

    #[test]
    fn test_substitute_helper() {
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()) * Expr::from(3);

        let mut map = HashMap::new();
        map.insert(x, Expr::from(4));

        let result = substitute(&expr, &map);

        let symbols = collect_symbols(&result);
        assert_eq!(symbols.len(), 0); // All symbols replaced
    }

    #[test]
    fn test_substituter_in_function() {
        let x = Symbol::new("x");
        let expr = Expr::Function(
            "f".to_string(),
            vec![Arc::new(Expr::Symbol(x.clone()))],
        );

        let mut substituter = Substituter::new();
        substituter.add_replacement(&x, Expr::from(10));

        let result = substituter.mutate(&expr);

        // Should be f(10)
        if let Expr::Function(name, args) = result {
            assert_eq!(name, "f");
            assert_eq!(args.len(), 1);
            assert!(matches!(*args[0], Expr::Integer(_)));
        } else {
            panic!("Expected function expression");
        }
    }
}
