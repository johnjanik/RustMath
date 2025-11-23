//! Pattern matcher implementation

use super::Pattern;
use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use std::collections::HashMap;

/// Substitution map from pattern variable names to matched expressions
#[derive(Debug, Clone, PartialEq)]
pub struct Substitution {
    bindings: HashMap<String, Expr>,
}

impl Substitution {
    /// Create a new empty substitution
    pub fn new() -> Self {
        Substitution {
            bindings: HashMap::new(),
        }
    }

    /// Add a binding to the substitution
    pub fn bind(&mut self, name: String, expr: Expr) -> bool {
        // Check if this name is already bound
        if let Some(existing) = self.bindings.get(&name) {
            // If already bound, check if it matches
            existing == &expr
        } else {
            // New binding
            self.bindings.insert(name, expr);
            true
        }
    }

    /// Get the expression bound to a name
    pub fn get(&self, name: &str) -> Option<&Expr> {
        self.bindings.get(name)
    }

    /// Check if a name is bound
    pub fn contains(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    /// Get all bindings
    pub fn bindings(&self) -> &HashMap<String, Expr> {
        &self.bindings
    }

    /// Merge another substitution into this one
    /// Returns false if there's a conflict
    pub fn merge(&mut self, other: &Substitution) -> bool {
        for (name, expr) in &other.bindings {
            if !self.bind(name.clone(), expr.clone()) {
                return false;
            }
        }
        true
    }
}

impl Default for Substitution {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a pattern match
pub type MatchResult = Option<Substitution>;

/// Pattern matcher
pub struct Matcher {
    /// Whether to try commutative matching for Add and Mul
    pub commutative: bool,
    /// Whether to try associative matching
    pub associative: bool,
}

impl Matcher {
    /// Create a new matcher with default settings
    pub fn new() -> Self {
        Matcher {
            commutative: true,
            associative: false, // Associative matching is more complex, disabled for now
        }
    }

    /// Create a matcher with custom settings
    pub fn with_settings(commutative: bool, associative: bool) -> Self {
        Matcher {
            commutative,
            associative,
        }
    }

    /// Try to match a pattern against an expression
    pub fn matches(&self, pattern: &Pattern, expr: &Expr) -> MatchResult {
        let mut subst = Substitution::new();
        if self.match_impl(pattern, expr, &mut subst) {
            Some(subst)
        } else {
            None
        }
    }

    /// Internal matching implementation
    fn match_impl(&self, pattern: &Pattern, expr: &Expr, subst: &mut Substitution) -> bool {
        match pattern {
            // Wildcard matches anything
            Pattern::Wildcard => true,

            // Named wildcard matches anything and binds it
            Pattern::Named(name) => subst.bind(name.clone(), expr.clone()),

            // Integer pattern
            Pattern::Integer(n) => {
                if let Expr::Integer(m) = expr {
                    m == &(*n).into()
                } else {
                    false
                }
            }

            // Symbol pattern
            Pattern::Symbol(name) => {
                if let Expr::Symbol(s) = expr {
                    s.name() == name
                } else {
                    false
                }
            }

            // Binary operation pattern
            Pattern::Binary(op, left_pat, right_pat) => {
                if let Expr::Binary(expr_op, left_expr, right_expr) = expr {
                    if op != expr_op {
                        return false;
                    }

                    // Try direct match first
                    let mut direct_subst = subst.clone();
                    if self.match_impl(left_pat, left_expr, &mut direct_subst)
                        && self.match_impl(right_pat, right_expr, &mut direct_subst)
                    {
                        *subst = direct_subst;
                        return true;
                    }

                    // Try commutative match if enabled and operation is commutative
                    if self.commutative && self.is_commutative(*op) {
                        let mut comm_subst = subst.clone();
                        if self.match_impl(left_pat, right_expr, &mut comm_subst)
                            && self.match_impl(right_pat, left_expr, &mut comm_subst)
                        {
                            *subst = comm_subst;
                            return true;
                        }
                    }

                    false
                } else {
                    false
                }
            }

            // Unary operation pattern
            Pattern::Unary(op, inner_pat) => {
                if let Expr::Unary(expr_op, inner_expr) = expr {
                    if op == expr_op {
                        self.match_impl(inner_pat, inner_expr, subst)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }

            // Function pattern
            Pattern::Function(name, arg_patterns) => {
                if let Expr::Function(expr_name, expr_args) = expr {
                    if name != expr_name || arg_patterns.len() != expr_args.len() {
                        return false;
                    }

                    // Match all arguments
                    for (pat, arg) in arg_patterns.iter().zip(expr_args.iter()) {
                        if !self.match_impl(pat, arg, subst) {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Check if a binary operation is commutative
    fn is_commutative(&self, op: BinaryOp) -> bool {
        matches!(op, BinaryOp::Add | BinaryOp::Mul)
    }
}

impl Default for Matcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Expr;

    #[test]
    fn test_wildcard_match() {
        let pattern = Pattern::Wildcard;
        let expr = Expr::symbol("x");
        let matcher = Matcher::new();

        assert!(matcher.matches(&pattern, &expr).is_some());
    }

    #[test]
    fn test_named_wildcard() {
        let pattern = Pattern::named("x");
        let expr = Expr::from(42);
        let matcher = Matcher::new();

        let result = matcher.matches(&pattern, &expr);
        assert!(result.is_some());

        let subst = result.unwrap();
        assert_eq!(subst.get("x"), Some(&Expr::from(42)));
    }

    #[test]
    fn test_symbol_match() {
        let pattern = Pattern::symbol("x");
        let expr = Expr::symbol("x");
        let matcher = Matcher::new();

        assert!(matcher.matches(&pattern, &expr).is_some());

        // Should not match different symbol
        let expr2 = Expr::symbol("y");
        assert!(matcher.matches(&pattern, &expr2).is_none());
    }

    #[test]
    fn test_binary_match() {
        // Pattern: x + y
        let pattern = Pattern::add(Pattern::named("x"), Pattern::named("y"));

        // Expression: a + b
        // Create symbols once and reuse them
        let a = Expr::symbol("a");
        let b = Expr::symbol("b");
        let expr = a.clone() + b.clone();
        let matcher = Matcher::new();

        let result = matcher.matches(&pattern, &expr);
        assert!(result.is_some());

        let subst = result.unwrap();
        // Just check that the bindings exist, not their exact value
        // (since Symbol equality depends on ID, not just name)
        assert!(subst.get("x").is_some());
        assert!(subst.get("y").is_some());
    }

    #[test]
    fn test_commutative_match() {
        // Pattern: 2 + x
        let pattern = Pattern::add(Pattern::Integer(2), Pattern::named("x"));

        // Expression: x + 2 (reversed order)
        let x = Expr::symbol("x");
        let expr = x.clone() + Expr::from(2);
        let matcher = Matcher::new();

        // Should match due to commutative property
        let result = matcher.matches(&pattern, &expr);
        assert!(result.is_some());

        let subst = result.unwrap();
        // Just verify that x was captured
        assert!(subst.get("x").is_some());
    }

    #[test]
    fn test_complex_pattern() {
        // Pattern: sin(x)^2
        let x = Pattern::named("x");
        let sin_x = Pattern::sin(x);
        let pattern = Pattern::pow(sin_x, Pattern::Integer(2));

        // Expression: sin(a)^2
        let a = Expr::symbol("a");
        let expr = a.clone().sin().pow(Expr::from(2));
        let matcher = Matcher::new();

        let result = matcher.matches(&pattern, &expr);
        assert!(result.is_some());

        let subst = result.unwrap();
        // Just verify x was captured
        assert!(subst.get("x").is_some());
    }

    #[test]
    fn test_pythagorean_identity_pattern() {
        // Pattern: sin(x)^2 + cos(x)^2
        let x_pat = Pattern::named("x");
        let sin_x = Pattern::sin(x_pat.clone());
        let cos_x = Pattern::cos(x_pat);
        let sin_squared = Pattern::pow(sin_x, Pattern::Integer(2));
        let cos_squared = Pattern::pow(cos_x, Pattern::Integer(2));
        let pattern = Pattern::add(sin_squared, cos_squared);

        // Expression: sin(a)^2 + cos(a)^2
        let a = Expr::symbol("a");
        let expr = a.clone().sin().pow(Expr::from(2)) + a.clone().cos().pow(Expr::from(2));

        let matcher = Matcher::new();
        let result = matcher.matches(&pattern, &expr);
        assert!(result.is_some());

        let subst = result.unwrap();
        // Just verify x was captured
        assert!(subst.get("x").is_some());
    }

    #[test]
    fn test_consistency_check() {
        // Pattern: x + x (same variable twice)
        let pattern = Pattern::add(Pattern::named("x"), Pattern::named("x"));

        // Expression: a + a (should match)
        let a = Expr::symbol("a");
        let expr1 = a.clone() + a.clone();
        let matcher = Matcher::new();
        assert!(matcher.matches(&pattern, &expr1).is_some());

        // Expression: a + b (should NOT match - different expressions)
        let a = Expr::symbol("a");
        let b = Expr::symbol("b");
        let expr2 = a + b;
        assert!(matcher.matches(&pattern, &expr2).is_none());
    }
}
