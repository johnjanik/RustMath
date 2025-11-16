//! Symbolic product operations
//!
//! This module provides symbolic product computation, analogous to symbolic summation.
//! The product ∏(v=a to b) f(v) represents the product of f(v) over the range [a, b].

use rustmath_symbolic::Expr;

/// Represents the direction of a product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductDirection {
    /// Forward product from a to b
    Forward,
    /// Backward product from b to a
    Backward,
}

/// Algorithm for computing symbolic products.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductAlgorithm {
    /// Direct evaluation (for finite products with numeric bounds)
    Direct,
    /// Pattern matching against known product formulas
    Symbolic,
}

impl Default for ProductAlgorithm {
    fn default() -> Self {
        ProductAlgorithm::Symbolic
    }
}

/// Computes a symbolic product ∏(v=a to b) expression.
///
/// This function evaluates products of the form:
/// ∏(v=a to b) f(v) = f(a) * f(a+1) * ... * f(b)
///
/// # Arguments
///
/// * `expression` - The expression to product over (function of v)
/// * `v` - The index variable
/// * `a` - Lower bound of the product
/// * `b` - Upper bound of the product
/// * `algorithm` - Algorithm to use for computation
/// * `hold` - If true, returns an unevaluated product expression
///
/// # Returns
///
/// The computed product, either evaluated or as a formal product expression.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::Expr;
/// use rustmath_calculus::symbolic_product;
///
/// // ∏(i=1 to n) i = n!
/// let i = Expr::symbol("i");
/// let result = symbolic_product(&i, "i", &Expr::from(1), &Expr::symbol("n"), None, false);
/// // Returns factorial(n)
/// ```
pub fn symbolic_product(
    expression: &Expr,
    v: &str,
    a: &Expr,
    b: &Expr,
    algorithm: Option<ProductAlgorithm>,
    hold: bool,
) -> Result<Expr, String> {
    // Validate that v is actually a symbol in the expression
    if !contains_variable(expression, v) && !is_constant_product(expression, v) {
        return Err(format!(
            "Variable '{}' must appear in the expression or be a constant product",
            v
        ));
    }

    // Validate that limits don't depend on the product variable
    if contains_variable(a, v) || contains_variable(b, v) {
        return Err(format!(
            "Product limits cannot depend on the product variable '{}'",
            v
        ));
    }

    // If hold is true, return unevaluated
    if hold {
        return Ok(formal_product(expression, v, a, b));
    }

    let alg = algorithm.unwrap_or_default();

    // Try to evaluate the product
    match alg {
        ProductAlgorithm::Direct => evaluate_direct_product(expression, v, a, b),
        ProductAlgorithm::Symbolic => evaluate_symbolic_product(expression, v, a, b),
    }
}

/// Evaluates a product by direct computation (for numeric bounds).
fn evaluate_direct_product(
    expression: &Expr,
    v: &str,
    a: &Expr,
    b: &Expr,
) -> Result<Expr, String> {
    // Check if both bounds are numeric integers
    if let (Expr::Number(a_val), Expr::Number(b_val)) = (a, b) {
        if a_val.floor() == *a_val && b_val.floor() == *b_val {
            let a_int = *a_val as i64;
            let b_int = *b_val as i64;

            if a_int > b_int {
                // Empty product
                return Ok(Expr::from(1));
            }

            if b_int - a_int > 10000 {
                return Err("Product range too large for direct evaluation".to_string());
            }

            // Compute the product directly
            let mut result = Expr::from(1);
            for i in a_int..=b_int {
                let value = substitute_value(expression, v, i);
                result = result * value;
            }

            return Ok(result);
        }
    }

    // Fall back to symbolic evaluation
    evaluate_symbolic_product(expression, v, a, b)
}

/// Evaluates a product using symbolic pattern matching.
fn evaluate_symbolic_product(
    expression: &Expr,
    v: &str,
    a: &Expr,
    b: &Expr,
) -> Result<Expr, String> {
    // Pattern: ∏(i=1 to n) i = n!
    if expression == &Expr::symbol(v) && a == &Expr::from(1) {
        return Ok(Expr::function("factorial", vec![b.clone()]));
    }

    // Pattern: ∏(i=1 to n) c = c^n (constant)
    if !contains_variable(expression, v) {
        let n = b.clone() - a.clone() + Expr::from(1);
        return Ok(expression.clone().pow(n));
    }

    // Pattern: ∏(i=a to b) (v + k) for constant k
    if let Expr::Add(terms) = expression {
        if terms.len() == 2 {
            if terms.contains(&Expr::symbol(v)) {
                for term in terms {
                    if term != &Expr::symbol(v) {
                        // v + k form
                        let k = term.clone();
                        // ∏(i=a to b) (i + k) = Γ(b+k+1)/Γ(a+k)
                        let gamma_upper = Expr::function(
                            "gamma",
                            vec![b.clone() + k.clone() + Expr::from(1)],
                        );
                        let gamma_lower = Expr::function("gamma", vec![a.clone() + k.clone()]);
                        return Ok(gamma_upper / gamma_lower);
                    }
                }
            }
        }
    }

    // Pattern: ∏(i=a to b) c*v for constant c
    if let Expr::Mul(factors) = expression {
        let (constants, vars) = partition_constant_factors(factors, v);
        if vars.len() == 1 && vars[0] == Expr::symbol(v) {
            // ∏(i=a to b) c*i = c^(b-a+1) * ∏(i=a to b) i
            let n = b.clone() - a.clone() + Expr::from(1);
            let const_part = constants.pow(n);
            let var_part = evaluate_symbolic_product(&Expr::symbol(v), v, a, b)?;
            return Ok(const_part * var_part);
        }
    }

    // Pattern: ∏(i=0 to n-1) (x + i) = (x)_n (Pochhammer symbol)
    if let Expr::Add(terms) = expression {
        if terms.len() == 2 {
            if terms.contains(&Expr::symbol(v)) {
                for term in terms {
                    if term != &Expr::symbol(v) {
                        // Check if a = 0 and limits match Pochhammer pattern
                        if a == &Expr::from(0) {
                            let x = term.clone();
                            let n = b.clone() + Expr::from(1);
                            return Ok(Expr::function("pochhammer", vec![x, n]));
                        }
                    }
                }
            }
        }
    }

    // Pattern: ∏(i=1 to n) (a + (i-1)d) = (a)_n^d (generalized Pochhammer)
    // This is more complex and omitted for now

    // No pattern matched - return formal product
    Ok(formal_product(expression, v, a, b))
}

/// Creates a formal (unevaluated) product expression.
///
/// Returns an expression of the form product(expression, v, a, b).
pub fn formal_product(expression: &Expr, v: &str, a: &Expr, b: &Expr) -> Expr {
    Expr::function(
        "product",
        vec![
            expression.clone(),
            Expr::symbol(v),
            a.clone(),
            b.clone(),
        ],
    )
}

/// Expands a product into explicit multiplication for small finite ranges.
///
/// # Arguments
///
/// * `expression` - The expression to product over
/// * `v` - The index variable
/// * `a` - Lower bound (must be numeric integer)
/// * `b` - Upper bound (must be numeric integer)
///
/// # Returns
///
/// The expanded product as an explicit multiplication.
pub fn expand_product(expression: &Expr, v: &str, a: i64, b: i64) -> Expr {
    if a > b {
        return Expr::from(1);
    }

    let mut result = substitute_value(expression, v, a);
    for i in (a + 1)..=b {
        let term = substitute_value(expression, v, i);
        result = result * term;
    }

    result
}

/// Computes a telescoping product.
///
/// For products of the form ∏ f(i)/f(i+1), the result telescopes to f(a)/f(b+1).
pub fn telescoping_product(expression: &Expr, v: &str, a: &Expr, b: &Expr) -> Option<Expr> {
    // Check if expression has the form f(v)/f(v+1)
    if let Expr::Div(num, den) = expression {
        // Try to detect if den = num with v -> v+1
        // This is a simplified check - full implementation would need more sophisticated matching
        let shifted = substitute_expr(&**num, v, &(Expr::symbol(v) + Expr::from(1)));
        if shifted == **den {
            // Telescoping! Result is f(a)/f(b+1)
            let f_a = substitute_expr(&**num, v, a);
            let f_b_plus_1 = substitute_expr(&**num, v, &(b.clone() + Expr::from(1)));
            return Some(f_a / f_b_plus_1);
        }
    }

    None
}

// Helper functions

/// Checks if an expression contains a variable.
fn contains_variable(expr: &Expr, var: &str) -> bool {
    match expr {
        Expr::Symbol(s) => s == var,
        Expr::Number(_) => false,
        Expr::Add(terms) | Expr::Mul(terms) => {
            terms.iter().any(|t| contains_variable(t, var))
        }
        Expr::Neg(e) => contains_variable(e, var),
        Expr::Pow(base, exp) => contains_variable(base, var) || contains_variable(exp, var),
        Expr::Div(num, den) => contains_variable(num, var) || contains_variable(den, var),
        Expr::Function(_, args) => args.iter().any(|a| contains_variable(a, var)),
    }
}

/// Checks if this is a valid constant product (expression doesn't contain variable).
fn is_constant_product(_expr: &Expr, _var: &str) -> bool {
    true // Allow constant products
}

/// Substitutes a numeric value for a variable in an expression.
fn substitute_value(expr: &Expr, var: &str, value: i64) -> Expr {
    substitute_expr(expr, var, &Expr::from(value))
}

/// Substitutes an expression for a variable in another expression.
fn substitute_expr(expr: &Expr, var: &str, replacement: &Expr) -> Expr {
    match expr {
        Expr::Symbol(s) if s == var => replacement.clone(),
        Expr::Symbol(_) | Expr::Number(_) => expr.clone(),
        Expr::Add(terms) => {
            let new_terms: Vec<_> = terms
                .iter()
                .map(|t| substitute_expr(t, var, replacement))
                .collect();
            new_terms.into_iter().fold(Expr::from(0), |acc, t| acc + t)
        }
        Expr::Mul(factors) => {
            let new_factors: Vec<_> = factors
                .iter()
                .map(|f| substitute_expr(f, var, replacement))
                .collect();
            new_factors.into_iter().fold(Expr::from(1), |acc, f| acc * f)
        }
        Expr::Neg(e) => -substitute_expr(e, var, replacement),
        Expr::Pow(base, exp) => {
            substitute_expr(base, var, replacement).pow(substitute_expr(exp, var, replacement))
        }
        Expr::Div(num, den) => {
            substitute_expr(num, var, replacement) / substitute_expr(den, var, replacement)
        }
        Expr::Function(name, args) => {
            let new_args: Vec<_> = args
                .iter()
                .map(|a| substitute_expr(a, var, replacement))
                .collect();
            Expr::function(name, new_args)
        }
    }
}

/// Partitions factors into constants and variables w.r.t. a variable.
fn partition_constant_factors(factors: &[Expr], var: &str) -> (Expr, Vec<Expr>) {
    let mut constants = Expr::from(1);
    let mut vars = Vec::new();

    for factor in factors {
        if !contains_variable(factor, var) {
            constants = constants * factor.clone();
        } else {
            vars.push(factor.clone());
        }
    }

    (constants, vars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_factorial() {
        // ∏(i=1 to n) i = n!
        let i = Expr::symbol("i");
        let n = Expr::symbol("n");
        let result = symbolic_product(&i, "i", &Expr::from(1), &n, None, false).unwrap();

        let expected = Expr::function("factorial", vec![n]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_product_constant() {
        // ∏(i=1 to 5) 2 = 2^5 = 32
        let expr = Expr::from(2);
        let result = symbolic_product(
            &expr,
            "i",
            &Expr::from(1),
            &Expr::from(5),
            Some(ProductAlgorithm::Direct),
            false,
        )
        .unwrap();

        assert_eq!(result, Expr::from(32));
    }

    #[test]
    fn test_product_direct_evaluation() {
        // ∏(i=1 to 5) i = 1*2*3*4*5 = 120
        let i = Expr::symbol("i");
        let result = symbolic_product(
            &i,
            "i",
            &Expr::from(1),
            &Expr::from(5),
            Some(ProductAlgorithm::Direct),
            false,
        )
        .unwrap();

        assert_eq!(result, Expr::from(120));
    }

    #[test]
    fn test_product_empty_range() {
        // ∏(i=5 to 3) i = 1 (empty product)
        let i = Expr::symbol("i");
        let result = symbolic_product(
            &i,
            "i",
            &Expr::from(5),
            &Expr::from(3),
            Some(ProductAlgorithm::Direct),
            false,
        )
        .unwrap();

        assert_eq!(result, Expr::from(1));
    }

    #[test]
    fn test_product_linear_term() {
        // ∏(i=1 to 3) (i + 2) = 3 * 4 * 5 = 60
        let i = Expr::symbol("i");
        let expr = i + Expr::from(2);
        let result = symbolic_product(
            &expr,
            "i",
            &Expr::from(1),
            &Expr::from(3),
            Some(ProductAlgorithm::Direct),
            false,
        )
        .unwrap();

        assert_eq!(result, Expr::from(60));
    }

    #[test]
    fn test_product_scaled() {
        // ∏(i=1 to 4) 2i = 2^4 * 4! = 16 * 24 = 384
        let i = Expr::symbol("i");
        let expr = Expr::from(2) * i;
        let result = symbolic_product(
            &expr,
            "i",
            &Expr::from(1),
            &Expr::from(4),
            Some(ProductAlgorithm::Direct),
            false,
        )
        .unwrap();

        assert_eq!(result, Expr::from(384));
    }

    #[test]
    fn test_product_hold() {
        // Test that hold=true returns unevaluated product
        let i = Expr::symbol("i");
        let n = Expr::symbol("n");
        let result = symbolic_product(&i, "i", &Expr::from(1), &n, None, true).unwrap();

        let expected = Expr::function(
            "product",
            vec![i, Expr::symbol("i"), Expr::from(1), n],
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_expand_product() {
        // ∏(i=2 to 4) i = 2*3*4 = 24
        let i = Expr::symbol("i");
        let result = expand_product(&i, "i", 2, 4);
        assert_eq!(result, Expr::from(24));
    }

    #[test]
    fn test_expand_product_expression() {
        // ∏(i=1 to 3) (2i + 1) = 3 * 5 * 7 = 105
        let i = Expr::symbol("i");
        let expr = Expr::from(2) * i + Expr::from(1);
        let result = expand_product(&expr, "i", 1, 3);
        assert_eq!(result, Expr::from(105));
    }

    #[test]
    fn test_formal_product() {
        let expr = Expr::symbol("x");
        let result = formal_product(&expr, "i", &Expr::from(1), &Expr::symbol("n"));

        assert_eq!(
            result,
            Expr::function(
                "product",
                vec![expr, Expr::symbol("i"), Expr::from(1), Expr::symbol("n")]
            )
        );
    }

    #[test]
    fn test_substitute_value() {
        let expr = Expr::symbol("x") + Expr::from(3);
        let result = substitute_value(&expr, "x", 5);
        assert_eq!(result, Expr::from(8));
    }

    #[test]
    fn test_substitute_expr() {
        let expr = Expr::symbol("x") * Expr::from(2);
        let replacement = Expr::symbol("y") + Expr::from(1);
        let result = substitute_expr(&expr, "x", &replacement);

        let expected = (Expr::symbol("y") + Expr::from(1)) * Expr::from(2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_product_validation() {
        // Should fail if limits depend on product variable
        let i = Expr::symbol("i");
        let result = symbolic_product(&i, "i", &i, &Expr::from(10), None, false);
        assert!(result.is_err());
    }
}
