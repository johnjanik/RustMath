//! Pattern matching engine for symbolic expressions
//!
//! This module provides a pattern matching system that allows:
//! - Wildcards (matches any expression)
//! - Named wildcards (captures matched expressions)
//! - Commutative matching (a+b matches b+a)
//! - Conditional patterns (patterns that only match under certain conditions)
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::pattern::{Pattern, Matcher};
//! use rustmath_symbolic::Expr;
//!
//! let x = Expr::symbol("x");
//!
//! // Pattern: sin(x)^2 + cos(x)^2
//! // This is built through the pattern API
//! ```

pub mod matcher;
pub mod rules;

pub use matcher::{Matcher, MatchResult, Substitution};
pub use rules::{
    TrigRule, ExpLogRule, RuleDatabase,
    get_trig_rules, get_exp_log_rules, apply_rules,
};

use crate::expression::{Expr, BinaryOp, UnaryOp};
use std::sync::Arc;

/// A pattern that can match expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// Match any expression (anonymous wildcard)
    Wildcard,
    /// Named wildcard that captures the matched expression
    Named(String),
    /// Match a specific integer
    Integer(i64),
    /// Match a specific symbol
    Symbol(String),
    /// Match a binary operation
    Binary(BinaryOp, Box<Pattern>, Box<Pattern>),
    /// Match a unary operation
    Unary(UnaryOp, Box<Pattern>),
    /// Match a function call
    Function(String, Vec<Pattern>),
}

impl Pattern {
    /// Create a named wildcard pattern
    pub fn named(name: impl Into<String>) -> Self {
        Pattern::Named(name.into())
    }

    /// Create a pattern that matches a specific symbol
    pub fn symbol(name: impl Into<String>) -> Self {
        Pattern::Symbol(name.into())
    }

    /// Create a pattern for a binary operation
    pub fn binary(op: BinaryOp, left: Pattern, right: Pattern) -> Self {
        Pattern::Binary(op, Box::new(left), Box::new(right))
    }

    /// Create a pattern for a unary operation
    pub fn unary(op: UnaryOp, inner: Pattern) -> Self {
        Pattern::Unary(op, Box::new(inner))
    }

    /// Create a power pattern (syntactic sugar)
    pub fn pow(base: Pattern, exp: Pattern) -> Self {
        Pattern::binary(BinaryOp::Pow, base, exp)
    }

    /// Create an addition pattern (syntactic sugar)
    pub fn add(left: Pattern, right: Pattern) -> Self {
        Pattern::binary(BinaryOp::Add, left, right)
    }

    /// Create a multiplication pattern (syntactic sugar)
    pub fn mul(left: Pattern, right: Pattern) -> Self {
        Pattern::binary(BinaryOp::Mul, left, right)
    }

    /// Create a sine pattern
    pub fn sin(inner: Pattern) -> Self {
        Pattern::unary(UnaryOp::Sin, inner)
    }

    /// Create a cosine pattern
    pub fn cos(inner: Pattern) -> Self {
        Pattern::unary(UnaryOp::Cos, inner)
    }

    /// Create an exponential pattern
    pub fn exp(inner: Pattern) -> Self {
        Pattern::unary(UnaryOp::Exp, inner)
    }

    /// Create a logarithm pattern
    pub fn log(inner: Pattern) -> Self {
        Pattern::unary(UnaryOp::Log, inner)
    }

    /// Create a square root pattern
    pub fn sqrt(inner: Pattern) -> Self {
        Pattern::unary(UnaryOp::Sqrt, inner)
    }
}

/// A rewrite rule that transforms an expression matching a pattern
#[derive(Clone)]
pub struct RewriteRule {
    /// The pattern to match
    pub pattern: Pattern,
    /// The replacement expression builder
    /// Takes the matched substitutions and returns the replacement expression
    pub replacement: Arc<dyn Fn(&Substitution) -> Expr + Send + Sync>,
    /// Optional description of the rule
    pub description: Option<String>,
}

impl RewriteRule {
    /// Create a new rewrite rule
    pub fn new<F>(pattern: Pattern, replacement: F) -> Self
    where
        F: Fn(&Substitution) -> Expr + Send + Sync + 'static,
    {
        RewriteRule {
            pattern,
            replacement: Arc::new(replacement),
            description: None,
        }
    }

    /// Create a rewrite rule with a description
    pub fn with_description<F>(pattern: Pattern, replacement: F, description: impl Into<String>) -> Self
    where
        F: Fn(&Substitution) -> Expr + Send + Sync + 'static,
    {
        RewriteRule {
            pattern,
            replacement: Arc::new(replacement),
            description: Some(description.into()),
        }
    }

    /// Try to apply this rule to an expression
    pub fn apply(&self, expr: &Expr) -> Option<Expr> {
        let matcher = Matcher::new();
        if let Some(subst) = matcher.matches(&self.pattern, expr) {
            Some((self.replacement)(&subst))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Expr;

    #[test]
    fn test_pattern_creation() {
        // Create pattern for sin(x)^2
        let x = Pattern::named("x");
        let sin_x = Pattern::sin(x);
        let sin_x_squared = Pattern::pow(sin_x, Pattern::Integer(2));

        assert!(matches!(sin_x_squared, Pattern::Binary(BinaryOp::Pow, _, _)));
    }

    #[test]
    fn test_simple_matching() {
        // Pattern: ?x (any expression)
        let pattern = Pattern::named("x");
        let expr = Expr::symbol("a");

        let matcher = Matcher::new();
        let result = matcher.matches(&pattern, &expr);

        assert!(result.is_some());
    }
}
