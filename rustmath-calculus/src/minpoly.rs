//! Minimal polynomial computation
//!
//! This module provides functionality to compute the minimal polynomial of algebraic numbers
//! and expressions. The minimal polynomial is the monic polynomial of least degree that has
//! the number as a root.

use rustmath_polynomials::Polynomial;
use rustmath_rationals::Rational;
use rustmath_symbolic::Expr;

/// Algorithm for computing minimal polynomials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinpolyAlgorithm {
    /// Numerical algorithm using LLL or PSLQ
    Numerical,
    /// Algebraic algorithm using field extensions
    Algebraic,
    /// Automatic selection based on input
    Auto,
}

impl Default for MinpolyAlgorithm {
    fn default() -> Self {
        MinpolyAlgorithm::Auto
    }
}

/// Options for minimal polynomial computation.
#[derive(Debug, Clone)]
pub struct MinpolyOptions {
    /// Algorithm to use
    pub algorithm: MinpolyAlgorithm,
    /// Precision in bits for numerical computation
    pub bits: Option<usize>,
    /// Expected degree of the minimal polynomial
    pub degree: Option<usize>,
    /// Error tolerance for numerical algorithms
    pub epsilon: f64,
}

impl Default for MinpolyOptions {
    fn default() -> Self {
        MinpolyOptions {
            algorithm: MinpolyAlgorithm::Auto,
            bits: None,
            degree: None,
            epsilon: 0.0,
        }
    }
}

/// Computes the minimal polynomial of an algebraic number or expression.
///
/// The minimal polynomial is the monic polynomial of smallest degree with rational
/// coefficients that has the given number as a root.
///
/// # Arguments
///
/// * `expr` - The expression or number to find the minimal polynomial of
/// * `var` - The variable name for the polynomial (default "x")
/// * `options` - Computation options (algorithm, precision, etc.)
///
/// # Returns
///
/// A polynomial with rational coefficients, or an error if the computation fails.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::Expr;
/// use rustmath_calculus::{minpoly, MinpolyOptions};
///
/// // Minimal polynomial of sqrt(2)
/// let sqrt2 = Expr::function("sqrt", vec![Expr::from(2)]);
/// let opts = MinpolyOptions::default();
/// let result = minpoly(&sqrt2, Some("x"), Some(opts));
/// // Should return x^2 - 2
/// ```
pub fn minpoly(
    expr: &Expr,
    var: Option<&str>,
    options: Option<MinpolyOptions>,
) -> Result<Polynomial<Rational>, String> {
    let var_name = var.unwrap_or("x");
    let opts = options.unwrap_or_default();

    // Determine which algorithm to use
    let algorithm = match opts.algorithm {
        MinpolyAlgorithm::Auto => select_algorithm(expr),
        alg => alg,
    };

    match algorithm {
        MinpolyAlgorithm::Numerical => minpoly_numerical(expr, var_name, &opts),
        MinpolyAlgorithm::Algebraic => minpoly_algebraic(expr, var_name, &opts),
        MinpolyAlgorithm::Auto => unreachable!(),
    }
}

/// Computes minimal polynomial using numerical algorithms (LLL/PSLQ).
fn minpoly_numerical(
    expr: &Expr,
    _var: &str,
    opts: &MinpolyOptions,
) -> Result<Polynomial<Rational>, String> {
    // First, evaluate the expression to a numerical value
    let value = evaluate_numeric(expr)?;

    // Try multiple precisions and degrees
    let precisions = if let Some(bits) = opts.bits {
        vec![bits]
    } else {
        vec![100, 200, 500, 1000]
    };

    let degrees = if let Some(deg) = opts.degree {
        vec![deg]
    } else {
        vec![2, 4, 8, 12, 24]
    };

    for &_precision in &precisions {
        for &degree in &degrees {
            // Use the PSLQ or LLL algorithm to find integer relations
            if let Ok(poly) = find_integer_relation(value, degree, opts.epsilon) {
                // Verify the polynomial
                if verify_polynomial(&poly, value, opts.epsilon) {
                    return Ok(poly);
                }
            }
        }
    }

    Err("Could not find minimal polynomial numerically".to_string())
}

/// Computes minimal polynomial using algebraic algorithms.
fn minpoly_algebraic(
    expr: &Expr,
    var: &str,
    _opts: &MinpolyOptions,
) -> Result<Polynomial<Rational>, String> {
    // Handle common algebraic numbers symbolically
    match expr {
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            // sqrt(n) has minimal polynomial x^2 - n
            if let Expr::Number(n) = &args[0] {
                // Check if n is a perfect square
                let sqrt_n = n.sqrt();
                if (sqrt_n * sqrt_n - n).abs() < 1e-10 {
                    // It's a perfect square - minimal polynomial is x - sqrt(n)
                    return Ok(linear_polynomial(Rational::from(-sqrt_n as i64), var));
                }

                // x^2 - n
                return Ok(quadratic_from_sqrt(*n as i64, var));
            }
            Err("Cannot compute minimal polynomial for non-numeric sqrt argument".to_string())
        }
        Expr::Function(name, args) if name == "cbrt" && args.len() == 1 => {
            // cbrt(n) has minimal polynomial x^3 - n
            if let Expr::Number(n) = &args[0] {
                return Ok(cubic_from_cbrt(*n as i64, var));
            }
            Err("Cannot compute minimal polynomial for non-numeric cbrt argument".to_string())
        }
        Expr::Pow(base, exp) => {
            // Handle rational powers
            if let (Expr::Number(b), Expr::Div(num, den)) = (&**base, &**exp) {
                if let (Expr::Number(p), Expr::Number(q)) = (&**num, &**den) {
                    // b^(p/q) has minimal polynomial x^q - b^p
                    return Ok(rational_power_minpoly(*b as i64, *p as i64, *q as i64, var));
                }
            }
            Err("Cannot compute minimal polynomial for this power expression".to_string())
        }
        Expr::Number(n) => {
            // Rational number - minimal polynomial is x - n
            Ok(linear_polynomial(Rational::from(-(*n as i64)), var))
        }
        _ => Err(format!(
            "Algebraic minimal polynomial not implemented for expression: {:?}",
            expr
        )),
    }
}

/// Selects an appropriate algorithm based on the expression type.
fn select_algorithm(expr: &Expr) -> MinpolyAlgorithm {
    match expr {
        Expr::Function(name, _) if name == "sqrt" || name == "cbrt" => {
            MinpolyAlgorithm::Algebraic
        }
        Expr::Pow(_, _) => MinpolyAlgorithm::Algebraic,
        Expr::Number(_) => MinpolyAlgorithm::Algebraic,
        _ => MinpolyAlgorithm::Numerical,
    }
}

/// Evaluates an expression to a numeric value.
fn evaluate_numeric(expr: &Expr) -> Result<f64, String> {
    match expr {
        Expr::Number(n) => Ok(*n),
        Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
            let arg = evaluate_numeric(&args[0])?;
            Ok(arg.sqrt())
        }
        Expr::Function(name, args) if name == "cbrt" && args.len() == 1 => {
            let arg = evaluate_numeric(&args[0])?;
            Ok(arg.cbrt())
        }
        Expr::Pow(base, exp) => {
            let b = evaluate_numeric(base)?;
            let e = evaluate_numeric(exp)?;
            Ok(b.powf(e))
        }
        Expr::Add(terms) => {
            let mut sum = 0.0;
            for term in terms {
                sum += evaluate_numeric(term)?;
            }
            Ok(sum)
        }
        Expr::Mul(factors) => {
            let mut product = 1.0;
            for factor in factors {
                product *= evaluate_numeric(factor)?;
            }
            Ok(product)
        }
        Expr::Neg(e) => Ok(-evaluate_numeric(e)?),
        Expr::Div(num, den) => {
            let n = evaluate_numeric(num)?;
            let d = evaluate_numeric(den)?;
            if d.abs() < 1e-15 {
                return Err("Division by zero in numeric evaluation".to_string());
            }
            Ok(n / d)
        }
        _ => Err(format!("Cannot evaluate expression numerically: {:?}", expr)),
    }
}

/// Finds integer relations using PSLQ-like algorithm.
fn find_integer_relation(
    value: f64,
    degree: usize,
    epsilon: f64,
) -> Result<Polynomial<Rational>, String> {
    // Build vector [1, value, value^2, ..., value^degree]
    let mut powers = Vec::new();
    let mut power = 1.0;
    for _ in 0..=degree {
        powers.push(power);
        power *= value;
    }

    // Simple integer relation finding using rational approximation
    // This is a placeholder for a full PSLQ/LLL implementation

    // For common algebraic numbers, try small integer coefficients
    for a in -10..=10 {
        for b in -10..=10 {
            if b == 0 {
                continue;
            }
            // Try x^2 - a/b
            let test_value = value * value - (a as f64) / (b as f64);
            if test_value.abs() < epsilon.max(1e-6) {
                // Found it! x^2 - a/b = 0
                let mut coeffs = vec![
                    Rational::new(-a, b),
                    Rational::from(0),
                    Rational::from(1),
                ];
                return Ok(Polynomial::new(coeffs));
            }
        }
    }

    Err("No integer relation found".to_string())
}

/// Verifies that a polynomial has the given value as an approximate root.
fn verify_polynomial(poly: &Polynomial<Rational>, value: f64, epsilon: f64) -> bool {
    let result = evaluate_polynomial_at(poly, value);
    result.abs() < epsilon.max(1e-6)
}

/// Evaluates a polynomial at a numeric value.
fn evaluate_polynomial_at(poly: &Polynomial<Rational>, value: f64) -> f64 {
    let coeffs = poly.coefficients();
    let mut result = 0.0;
    let mut power = 1.0;

    for coeff in coeffs {
        let c = coeff.to_f64();
        result += c * power;
        power *= value;
    }

    result
}

/// Creates a linear polynomial x - c.
fn linear_polynomial(c: Rational, _var: &str) -> Polynomial<Rational> {
    Polynomial::new(vec![c, Rational::from(1)])
}

/// Creates minimal polynomial x^2 - n for sqrt(n).
fn quadratic_from_sqrt(n: i64, _var: &str) -> Polynomial<Rational> {
    Polynomial::new(vec![Rational::from(-n), Rational::from(0), Rational::from(1)])
}

/// Creates minimal polynomial x^3 - n for cbrt(n).
fn cubic_from_cbrt(n: i64, _var: &str) -> Polynomial<Rational> {
    Polynomial::new(vec![
        Rational::from(-n),
        Rational::from(0),
        Rational::from(0),
        Rational::from(1),
    ])
}

/// Creates minimal polynomial x^q - b^p for b^(p/q).
fn rational_power_minpoly(b: i64, p: i64, q: i64, _var: &str) -> Polynomial<Rational> {
    let mut coeffs = vec![Rational::from(0); (q as usize) + 1];
    coeffs[0] = Rational::from(-(b.pow(p as u32)));
    coeffs[q as usize] = Rational::from(1);
    Polynomial::new(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minpoly_rational() {
        // Minimal polynomial of 3 is x - 3
        let expr = Expr::from(3);
        let result = minpoly(&expr, Some("x"), None).unwrap();

        let expected = Polynomial::new(vec![Rational::from(-3), Rational::from(1)]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_minpoly_sqrt2() {
        // Minimal polynomial of sqrt(2) is x^2 - 2
        let expr = Expr::function("sqrt", vec![Expr::from(2)]);
        let result = minpoly(&expr, Some("x"), None).unwrap();

        let expected = Polynomial::new(vec![
            Rational::from(-2),
            Rational::from(0),
            Rational::from(1),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_minpoly_sqrt3() {
        // Minimal polynomial of sqrt(3) is x^2 - 3
        let expr = Expr::function("sqrt", vec![Expr::from(3)]);
        let result = minpoly(&expr, Some("x"), None).unwrap();

        let expected = Polynomial::new(vec![
            Rational::from(-3),
            Rational::from(0),
            Rational::from(1),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_minpoly_cbrt2() {
        // Minimal polynomial of cbrt(2) is x^3 - 2
        let expr = Expr::function("cbrt", vec![Expr::from(2)]);
        let result = minpoly(&expr, Some("x"), None).unwrap();

        let expected = Polynomial::new(vec![
            Rational::from(-2),
            Rational::from(0),
            Rational::from(0),
            Rational::from(1),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_numeric_sqrt() {
        let expr = Expr::function("sqrt", vec![Expr::from(4)]);
        let result = evaluate_numeric(&expr).unwrap();
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_numeric_cbrt() {
        let expr = Expr::function("cbrt", vec![Expr::from(8)]);
        let result = evaluate_numeric(&expr).unwrap();
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_verify_polynomial() {
        // x^2 - 2 should have sqrt(2) as a root
        let poly = Polynomial::new(vec![
            Rational::from(-2),
            Rational::from(0),
            Rational::from(1),
        ]);
        let sqrt2 = 2.0_f64.sqrt();
        assert!(verify_polynomial(&poly, sqrt2, 1e-10));
    }

    #[test]
    fn test_select_algorithm() {
        let sqrt_expr = Expr::function("sqrt", vec![Expr::from(2)]);
        assert_eq!(select_algorithm(&sqrt_expr), MinpolyAlgorithm::Algebraic);

        let num_expr = Expr::from(5);
        assert_eq!(select_algorithm(&num_expr), MinpolyAlgorithm::Algebraic);
    }
}
