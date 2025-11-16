//! Pochhammer symbol (rising factorial)
//!
//! The Pochhammer symbol (x)_n, also known as the rising factorial, is defined as:
//! (x)_n = x(x+1)(x+2)...(x+n-1) = Γ(x+n)/Γ(x)
//!
//! This is commonly used in combinatorics, special functions, and hypergeometric series.

use rustmath_symbolic::Expr;

/// Computes the Pochhammer symbol (rising factorial).
///
/// The Pochhammer symbol (x)_n is defined as:
/// - (x)_0 = 1
/// - (x)_n = x(x+1)(x+2)...(x+n-1) for n > 0
/// - (x)_n = 1/[(x-1)(x-2)...(x-|n|)] for n < 0
///
/// Using the gamma function: (x)_n = Γ(x+n)/Γ(x)
///
/// # Arguments
///
/// * `x` - The base of the Pochhammer symbol
/// * `n` - The number of terms
///
/// # Returns
///
/// An expression representing the Pochhammer symbol, typically expressed using gamma functions.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::Expr;
/// use rustmath_calculus::pochhammer;
///
/// // (5)_3 = 5 * 6 * 7 = 210
/// let result = pochhammer(&Expr::from(5), &Expr::from(3));
/// // Result is gamma(5+3)/gamma(5) = gamma(8)/gamma(5)
/// ```
pub fn pochhammer(x: &Expr, n: &Expr) -> Expr {
    // (x)_n = Γ(x+n)/Γ(x)
    let gamma_x_plus_n = Expr::function("gamma", vec![x.clone() + n.clone()]);
    let gamma_x = Expr::function("gamma", vec![x.clone()]);
    gamma_x_plus_n / gamma_x
}

/// Evaluates the Pochhammer symbol for numeric integer arguments.
///
/// For concrete integer values, this computes the exact product rather than
/// using the gamma function representation.
///
/// # Arguments
///
/// * `x` - Numeric base value
/// * `n` - Integer number of terms
///
/// # Returns
///
/// The numeric value of (x)_n if computable, otherwise returns the symbolic form.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::pochhammer_eval;
///
/// // (5)_3 = 5 * 6 * 7 = 210
/// let result = pochhammer_eval(5.0, 3);
/// assert_eq!(result, 210.0);
///
/// // (3)_0 = 1
/// let result = pochhammer_eval(3.0, 0);
/// assert_eq!(result, 1.0);
/// ```
pub fn pochhammer_eval(x: f64, n: i64) -> f64 {
    match n {
        0 => 1.0,
        n if n > 0 => {
            let mut result = 1.0;
            for i in 0..n {
                result *= x + (i as f64);
            }
            result
        }
        n => {
            // n < 0: (x)_n = 1/[(x-1)(x-2)...(x-|n|)]
            let mut result = 1.0;
            for i in 1..=(-n) {
                result *= x - (i as f64);
            }
            1.0 / result
        }
    }
}

/// Creates a formal Pochhammer symbol expression.
///
/// This creates an unevaluated Pochhammer symbol for use in symbolic expressions.
///
/// # Arguments
///
/// * `x` - The base expression
/// * `n` - The number of terms
///
/// # Returns
///
/// A symbolic expression representing pochhammer(x, n).
pub fn formal_pochhammer(x: &Expr, n: &Expr) -> Expr {
    Expr::function("pochhammer", vec![x.clone(), n.clone()])
}

/// Dummy Pochhammer symbol for Maxima compatibility.
///
/// This function provides compatibility with SageMath's dummy_pochhammer,
/// which is used when Maxima returns an unevaluated Pochhammer symbol.
///
/// # Arguments
///
/// * `x` - The base expression
/// * `n` - The number of terms
///
/// # Returns
///
/// The Pochhammer symbol as a ratio of gamma functions.
pub fn dummy_pochhammer(x: &Expr, n: &Expr) -> Expr {
    pochhammer(x, n)
}

/// Expands the Pochhammer symbol as an explicit product for small integer n.
///
/// For small positive integer values of n, this returns the expanded product form:
/// (x)_n = x(x+1)(x+2)...(x+n-1)
///
/// # Arguments
///
/// * `x` - The base expression
/// * `n` - The number of terms (should be a small positive integer)
///
/// # Returns
///
/// The expanded product if n is a small positive integer, otherwise returns
/// the gamma function representation.
pub fn pochhammer_expand(x: &Expr, n: &Expr) -> Expr {
    // Check if n is a small positive integer
    if let Expr::Number(n_val) = n {
        if *n_val >= 0.0 && *n_val == n_val.floor() && *n_val <= 10.0 {
            let n_int = *n_val as i64;
            if n_int == 0 {
                return Expr::from(1);
            }

            // Build the product x(x+1)(x+2)...(x+n-1)
            let mut result = x.clone();
            for i in 1..n_int {
                result = result * (x.clone() + Expr::from(i));
            }
            return result;
        }
    }

    // Otherwise return gamma function form
    pochhammer(x, n)
}

/// Converts a falling factorial to Pochhammer symbol.
///
/// The falling factorial is (x)_n↓ = x(x-1)(x-2)...(x-n+1)
/// Related to Pochhammer by: (x)_n↓ = (-1)^n * (-x)_n
///
/// # Arguments
///
/// * `x` - The base expression
/// * `n` - The number of terms
///
/// # Returns
///
/// The Pochhammer symbol equivalent to the falling factorial.
pub fn falling_factorial_to_pochhammer(x: &Expr, n: &Expr) -> Expr {
    // (x)_n↓ = (-1)^n * (-x)_n
    let neg_one_pow_n = Expr::from(-1).pow(n.clone());
    let neg_x = -x.clone();
    neg_one_pow_n * pochhammer(&neg_x, n)
}

/// Simplifies products of Pochhammer symbols.
///
/// Applies identities such as:
/// - (x)_m * (x+m)_n = (x)_(m+n)
/// - (x)_n / (x)_m = (x+m)_(n-m) for n > m
///
/// # Arguments
///
/// * `expr` - Expression potentially containing Pochhammer symbols
///
/// # Returns
///
/// Simplified expression with combined Pochhammer symbols where possible.
pub fn simplify_pochhammer(expr: &Expr) -> Expr {
    // This is a placeholder for future implementation
    // Would require pattern matching and term collection
    expr.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pochhammer_gamma_form() {
        // (x)_n = gamma(x+n)/gamma(x)
        let x = Expr::symbol("x");
        let n = Expr::symbol("n");
        let result = pochhammer(&x, &n);

        let expected = Expr::function("gamma", vec![x.clone() + n.clone()]) /
                       Expr::function("gamma", vec![x.clone()]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pochhammer_eval_zero() {
        // (x)_0 = 1
        assert_eq!(pochhammer_eval(5.0, 0), 1.0);
        assert_eq!(pochhammer_eval(100.0, 0), 1.0);
    }

    #[test]
    fn test_pochhammer_eval_positive() {
        // (5)_3 = 5 * 6 * 7 = 210
        assert_eq!(pochhammer_eval(5.0, 3), 210.0);

        // (2)_4 = 2 * 3 * 4 * 5 = 120
        assert_eq!(pochhammer_eval(2.0, 4), 120.0);

        // (1)_n = n!
        assert_eq!(pochhammer_eval(1.0, 5), 120.0); // 5!
    }

    #[test]
    fn test_pochhammer_eval_negative() {
        // (5)_(-2) = 1/(4*3) = 1/12
        let result = pochhammer_eval(5.0, -2);
        assert!((result - 1.0/12.0).abs() < 1e-10);
    }

    #[test]
    fn test_pochhammer_expand_zero() {
        let x = Expr::symbol("x");
        let n = Expr::from(0);
        let result = pochhammer_expand(&x, &n);
        assert_eq!(result, Expr::from(1));
    }

    #[test]
    fn test_pochhammer_expand_small() {
        let x = Expr::symbol("x");

        // (x)_1 = x
        let result = pochhammer_expand(&x, &Expr::from(1));
        assert_eq!(result, x);

        // (x)_2 = x(x+1)
        let result = pochhammer_expand(&x, &Expr::from(2));
        let expected = x.clone() * (x.clone() + Expr::from(1));
        assert_eq!(result, expected);

        // (x)_3 = x(x+1)(x+2)
        let result = pochhammer_expand(&x, &Expr::from(3));
        let expected = x.clone() * (x.clone() + Expr::from(1)) * (x.clone() + Expr::from(2));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pochhammer_expand_large() {
        // For n > 10, should return gamma form
        let x = Expr::symbol("x");
        let n = Expr::from(15);
        let result = pochhammer_expand(&x, &n);

        let expected = Expr::function("gamma", vec![x.clone() + n.clone()]) /
                       Expr::function("gamma", vec![x.clone()]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_formal_pochhammer() {
        let x = Expr::symbol("x");
        let n = Expr::symbol("n");
        let result = formal_pochhammer(&x, &n);

        assert_eq!(
            result,
            Expr::function("pochhammer", vec![x, n])
        );
    }

    #[test]
    fn test_dummy_pochhammer() {
        // Should be same as regular pochhammer
        let x = Expr::symbol("x");
        let n = Expr::symbol("n");
        let result = dummy_pochhammer(&x, &n);
        let expected = pochhammer(&x, &n);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_falling_factorial_relation() {
        // Test the relation between falling factorial and Pochhammer
        let x = Expr::from(5);
        let n = Expr::from(3);
        let _result = falling_factorial_to_pochhammer(&x, &n);
        // Just testing it compiles and doesn't panic
    }
}
