//! Utilities for manifold computations
//!
//! This module provides utility functions and structures for working with
//! manifolds, including expression simplification, symbolic operations,
//! and coordinate transformations.

use rustmath_symbolic::{Expression, simplify};
use std::collections::HashMap;

/// Type alias for simplification functions
pub type SimplificationFn = fn(Expression) -> Expression;

/// Chain of simplification operations to apply to expressions
#[derive(Clone)]
pub struct SimplificationChain {
    operations: Vec<SimplificationFn>,
}

impl SimplificationChain {
    /// Create a new empty simplification chain
    pub fn new() -> Self {
        SimplificationChain {
            operations: Vec::new(),
        }
    }

    /// Add a simplification operation to the chain
    pub fn add(&mut self, op: SimplificationFn) -> &mut Self {
        self.operations.push(op);
        self
    }

    /// Apply all simplification operations in sequence
    pub fn apply(&self, expr: Expression) -> Expression {
        self.operations.iter().fold(expr, |acc, op| op(acc))
    }
}

impl Default for SimplificationChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplify expressions involving absolute values and trigonometric functions
///
/// This simplification assumes real-valued expressions and applies rules like:
/// - |sin(x)| can be simplified under certain assumptions
/// - |cos(x)| can be simplified under certain assumptions
pub fn simplify_abs_trig(expr: Expression) -> Expression {
    // For now, use the standard simplification
    // In a full implementation, this would walk the expression tree
    // and apply special rules for absolute values of trig functions
    simplify(expr)
}

/// Simplify expressions involving square roots of real numbers
///
/// This simplification assumes real-valued expressions and applies rules like:
/// - sqrt(x^2) = |x| for real x
/// - sqrt(a) * sqrt(b) = sqrt(a*b) for positive a, b
pub fn simplify_sqrt_real(expr: Expression) -> Expression {
    // For now, use the standard simplification
    // In a full implementation, this would walk the expression tree
    // and apply special rules for square roots
    simplify(expr)
}

/// Generic simplification chain for real-valued expressions
///
/// Applies a sequence of simplifications appropriate for real numbers:
/// 1. Basic algebraic simplification
/// 2. Square root simplification
/// 3. Absolute value/trigonometric simplification
pub fn simplify_chain_real(expr: Expression) -> Expression {
    let mut chain = SimplificationChain::new();
    chain.add(simplify);
    chain.add(simplify_sqrt_real);
    chain.add(simplify_abs_trig);
    chain.apply(expr)
}

/// Generic simplification chain for general expressions
///
/// Applies a sequence of simplifications appropriate for general expressions:
/// 1. Basic algebraic simplification
/// 2. Trigonometric simplification
pub fn simplify_chain_generic(expr: Expression) -> Expression {
    let mut chain = SimplificationChain::new();
    chain.add(simplify);
    chain.add(simplify_abs_trig);
    chain.apply(expr)
}

/// Compute the exterior derivative of a differential form
///
/// In differential geometry, the exterior derivative is a generalization
/// of the gradient, curl, and divergence operators.
///
/// For a 0-form (scalar field) f, df is the 1-form given by the differential.
/// For a k-form ω, dω is a (k+1)-form satisfying d(dω) = 0.
///
/// # Arguments
///
/// * `form` - The differential form to differentiate
/// * `variables` - The coordinate variables
///
/// # Returns
///
/// The exterior derivative as a new differential form
pub fn exterior_derivative(
    form: &HashMap<Vec<usize>, Expression>,
    variables: &[String],
) -> HashMap<Vec<usize>, Expression> {
    let mut result = HashMap::new();

    // For each term in the form
    for (indices, coeff) in form.iter() {
        // Compute partial derivatives with respect to each variable
        for (var_idx, var_name) in variables.iter().enumerate() {
            // Skip if this variable is already in the wedge product
            if indices.contains(&var_idx) {
                continue;
            }

            // Compute the partial derivative
            let deriv = rustmath_symbolic::differentiate(coeff.clone(), var_name);

            if !rustmath_symbolic::is_zero(&deriv) {
                // Create new index list: [var_idx] ∧ indices
                let mut new_indices = vec![var_idx];
                new_indices.extend(indices.iter().copied());

                // Sort indices and track sign from permutation
                let sign = sort_with_sign(&mut new_indices);
                let final_expr = if sign > 0 {
                    deriv
                } else {
                    rustmath_symbolic::negate(deriv)
                };

                // Add to result (or accumulate if already present)
                result.entry(new_indices)
                    .and_modify(|e| *e = rustmath_symbolic::add(e.clone(), final_expr.clone()))
                    .or_insert(final_expr);
            }
        }
    }

    result
}

/// Sort a vector and return the sign of the permutation
/// Returns 1 for even permutations, -1 for odd permutations
fn sort_with_sign(indices: &mut Vec<usize>) -> i32 {
    let n = indices.len();
    let mut sign = 1;

    // Bubble sort to count inversions
    for i in 0..n {
        for j in 0..n - 1 - i {
            if indices[j] > indices[j + 1] {
                indices.swap(j, j + 1);
                sign *= -1;
            }
        }
    }

    sign
}

/// Set labels for plot axes (placeholder for visualization)
///
/// In a full implementation, this would configure axis labels for
/// 3D plots of manifolds and vector fields.
///
/// # Arguments
///
/// * `labels` - Vector of axis labels (e.g., ["x", "y", "z"])
pub fn set_axes_labels(_labels: &[String]) {
    // Placeholder - would integrate with a plotting library
    // For now, this is a no-op since we don't have visualization
}

/// Alternative name for exterior_derivative
pub fn xder(
    form: &HashMap<Vec<usize>, Expression>,
    variables: &[String],
) -> HashMap<Vec<usize>, Expression> {
    exterior_derivative(form, variables)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_symbolic::{Expression, SymbolType};

    #[test]
    fn test_simplification_chain_creation() {
        let chain = SimplificationChain::new();
        assert_eq!(chain.operations.len(), 0);
    }

    #[test]
    fn test_simplification_chain_add() {
        let mut chain = SimplificationChain::new();
        chain.add(simplify);
        assert_eq!(chain.operations.len(), 1);
    }

    #[test]
    fn test_simplify_chain_real() {
        let x = Expression::Symbol("x".to_string(), SymbolType::Real);
        let result = simplify_chain_real(x.clone());
        assert!(matches!(result, Expression::Symbol(_, _)));
    }

    #[test]
    fn test_simplify_chain_generic() {
        let x = Expression::Symbol("x".to_string(), SymbolType::Real);
        let result = simplify_chain_generic(x.clone());
        assert!(matches!(result, Expression::Symbol(_, _)));
    }

    #[test]
    fn test_exterior_derivative_scalar_field() {
        // Test d(f) where f is a 0-form (scalar field)
        let mut form = HashMap::new();

        // f = x^2 + y (represented as 0-form with empty index)
        let x = Expression::Symbol("x".to_string(), SymbolType::Real);
        let y = Expression::Symbol("y".to_string(), SymbolType::Real);
        let x_squared = Expression::Multiply(
            Box::new(x.clone()),
            Box::new(x.clone()),
        );
        let f = Expression::Add(Box::new(x_squared), Box::new(y));

        form.insert(vec![], f);

        let variables = vec!["x".to_string(), "y".to_string()];
        let result = exterior_derivative(&form, &variables);

        // df should have terms in dx and dy
        assert!(result.len() > 0);
    }

    #[test]
    fn test_sort_with_sign() {
        let mut indices = vec![1, 0];
        let sign = sort_with_sign(&mut indices);
        assert_eq!(indices, vec![0, 1]);
        assert_eq!(sign, -1); // One swap = odd permutation

        let mut indices2 = vec![2, 1, 0];
        let sign2 = sort_with_sign(&mut indices2);
        assert_eq!(indices2, vec![0, 1, 2]);
        assert_eq!(sign2, -1); // Three swaps = odd permutation
    }

    #[test]
    fn test_xder_alias() {
        let mut form = HashMap::new();
        let x = Expression::Symbol("x".to_string(), SymbolType::Real);
        form.insert(vec![], x);

        let variables = vec!["x".to_string()];
        let result1 = exterior_derivative(&form, &variables);
        let result2 = xder(&form, &variables);

        assert_eq!(result1.len(), result2.len());
    }
}
