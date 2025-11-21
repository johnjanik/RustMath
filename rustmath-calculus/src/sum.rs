//! Symbolic summation operations
//!
//! This module provides symbolic sum computation for expressions.
//! The sum ∑(v=a to b) f(v) represents the sum of f(v) over the range [a, b].

use rustmath_core::NumericConversion;
use rustmath_symbolic::{BinaryOp, Expr};

/// Algorithm for computing symbolic sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumAlgorithm {
    /// Direct evaluation (for finite sums with numeric bounds)
    Direct,
    /// Pattern matching against known sum formulas
    Symbolic,
}

impl Default for SumAlgorithm {
    fn default() -> Self {
        SumAlgorithm::Symbolic
    }
}

/// Computes a symbolic sum ∑(v=a to b) expression.
///
/// This function evaluates sums of the form:
/// ∑(v=a to b) f(v) = f(a) + f(a+1) + ... + f(b)
///
/// # Arguments
///
/// * `expression` - The expression to sum over (function of v)
/// * `v` - The index variable
/// * `a` - Lower bound of the sum
/// * `b` - Upper bound of the sum
/// * `algorithm` - Algorithm to use for computation
/// * `hold` - If true, returns an unevaluated sum expression
///
/// # Returns
///
/// The computed sum, either evaluated or as a formal sum expression.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::Expr;
/// use rustmath_calculus::symbolic_sum;
///
/// // ∑(i=1 to 5) i = 15
/// let i = Expr::symbol("i");
/// let result = symbolic_sum(&i, "i", &Expr::from(1), &Expr::from(5), None, false);
/// // Direct evaluation returns 15
/// ```
pub fn symbolic_sum(
    expression: &Expr,
    v: &str,
    a: &Expr,
    b: &Expr,
    _algorithm: Option<SumAlgorithm>,
    hold: bool,
) -> Result<Expr, String> {
    // If hold is true, return unevaluated
    if hold {
        return Ok(formal_sum(expression, v, a, b));
    }

    // Try direct evaluation for numeric bounds
    if let (Expr::Integer(a_int), Expr::Integer(b_int)) = (a, b) {
        let a_val = a_int.to_f64().ok_or("Lower bound conversion failed")? as i64;
        let b_val = b_int.to_f64().ok_or("Upper bound conversion failed")? as i64;

        if a_val > b_val {
            // Empty sum
            return Ok(Expr::from(0));
        }

        if b_val - a_val > 10000 {
            return Err("Sum range too large for direct evaluation".to_string());
        }

        // Compute the sum directly
        let mut result = Expr::from(0);
        for i in a_val..=b_val {
            let value = substitute_value(expression, v, i);
            result = result + value;
        }

        return Ok(result);
    }

    // For symbolic bounds, try pattern matching
    // Pattern: ∑(i=1 to n) i = n(n+1)/2
    if let Expr::Symbol(s) = expression {
        if s.name() == v && a == &Expr::from(1) {
            let n = b.clone();
            return Ok(n.clone() * (n + Expr::from(1)) / Expr::from(2));
        }
        if s.name() == v && a == &Expr::from(0) {
            let n = b.clone();
            return Ok(n.clone() * (n + Expr::from(1)) / Expr::from(2));
        }
    }

    // Pattern: ∑(i=a to b) c = c*(b-a+1) (constant)
    if !contains_variable(expression, v) {
        let count = b.clone() - a.clone() + Expr::from(1);
        return Ok(expression.clone() * count);
    }

    // For other cases, return formal sum
    Ok(formal_sum(expression, v, a, b))
}

/// Creates a formal (unevaluated) sum expression.
pub fn formal_sum(expression: &Expr, v: &str, a: &Expr, b: &Expr) -> Expr {
    Expr::function(
        "sum",
        vec![
            expression.clone().into(),
            Expr::symbol(v).into(),
            a.clone().into(),
            b.clone().into(),
        ],
    )
}

/// Expands a sum into explicit addition for small finite ranges.
pub fn expand_sum(expression: &Expr, v: &str, a: i64, b: i64) -> Expr {
    if a > b {
        return Expr::from(0);
    }

    let mut result = substitute_value(expression, v, a);
    for i in (a + 1)..=b {
        let term = substitute_value(expression, v, i);
        result = result + term;
    }

    result
}

// Helper functions

/// Checks if an expression contains a variable.
fn contains_variable(expr: &Expr, var: &str) -> bool {
    match expr {
        Expr::Symbol(s) => s.name() == var,
        Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => false,
        Expr::Binary(_, left, right) => {
            contains_variable(left, var) || contains_variable(right, var)
        }
        Expr::Unary(_, inner) => contains_variable(inner, var),
        Expr::Function(_, args) => args.iter().any(|a| contains_variable(a, var)),
    }
}

/// Substitutes a numeric value for a variable in an expression.
fn substitute_value(expr: &Expr, var: &str, value: i64) -> Expr {
    substitute_expr(expr, var, &Expr::from(value))
}

/// Substitutes an expression for a variable in another expression.
fn substitute_expr(expr: &Expr, var: &str, replacement: &Expr) -> Expr {
    match expr {
        Expr::Symbol(s) if s.name() == var => replacement.clone(),
        Expr::Symbol(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => expr.clone(),
        Expr::Binary(op, left, right) => {
            let new_left = substitute_expr(left, var, replacement);
            let new_right = substitute_expr(right, var, replacement);
            match op {
                BinaryOp::Add => new_left + new_right,
                BinaryOp::Sub => new_left - new_right,
                BinaryOp::Mul => new_left * new_right,
                BinaryOp::Div => new_left / new_right,
                BinaryOp::Pow => new_left.pow(new_right),
                BinaryOp::Mod => Expr::Binary(BinaryOp::Mod, new_left.into(), new_right.into()),
            }
        }
        Expr::Unary(op, inner) => {
            let new_inner = substitute_expr(inner, var, replacement);
            Expr::Unary(*op, new_inner.into())
        }
        Expr::Function(name, args) => {
            let new_args: Vec<_> = args
                .iter()
                .map(|a| substitute_expr(a, var, replacement).into())
                .collect();
            Expr::Function(name.clone(), new_args)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_constant() {
        // ∑(i=1 to 5) 3 = 3*5 = 15
        let expr = Expr::from(3);
        let result = symbolic_sum(&expr, "i", &Expr::from(1), &Expr::from(5), None, false).unwrap();
        assert_eq!(result, Expr::from(15));
    }

    #[test]
    fn test_sum_direct_evaluation() {
        // ∑(i=1 to 5) i = 1+2+3+4+5 = 15
        let i = Expr::symbol("i");
        let result = symbolic_sum(&i, "i", &Expr::from(1), &Expr::from(5), None, false).unwrap();
        assert_eq!(result, Expr::from(15));
    }

    #[test]
    fn test_sum_empty_range() {
        // ∑(i=5 to 3) i = 0 (empty sum)
        let i = Expr::symbol("i");
        let result = symbolic_sum(&i, "i", &Expr::from(5), &Expr::from(3), None, false).unwrap();
        assert_eq!(result, Expr::from(0));
    }

    #[test]
    fn test_sum_hold() {
        // Test that hold=true returns unevaluated sum
        let i = Expr::symbol("i");
        let n = Expr::symbol("n");
        let result = symbolic_sum(&i, "i", &Expr::from(1), &n, None, true).unwrap();

        // Check it's a function call
        if let Expr::Function(name, _args) = result {
            assert_eq!(name, "sum");
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_expand_sum() {
        // ∑(i=2 to 4) i = 2+3+4 = 9
        let i = Expr::symbol("i");
        let result = expand_sum(&i, "i", 2, 4);
        assert_eq!(result, Expr::from(9));
    }
}
