//! Laplace and inverse Laplace transforms
//!
//! This module provides symbolic Laplace and inverse Laplace transforms for expressions.
//! The Laplace transform is an integral transform that converts a function f(t) to F(s):
//!
//! L{f(t)} = F(s) = ∫₀^∞ e^(-st) f(t) dt
//!
//! The inverse Laplace transform recovers the original function from its transform.

use rustmath_symbolic::Expr;
use std::collections::HashMap;

/// Algorithm for computing Laplace transforms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaplaceAlgorithm {
    /// Table lookup with symbolic manipulation
    TableLookup,
    /// Numerical approximation (future implementation)
    Numerical,
}

impl Default for LaplaceAlgorithm {
    fn default() -> Self {
        LaplaceAlgorithm::TableLookup
    }
}

/// Computes the Laplace transform of an expression.
///
/// The Laplace transform converts a function f(t) in the time domain to F(s) in the frequency domain:
/// L{f(t)} = F(s) = ∫₀^∞ e^(-st) f(t) dt
///
/// # Arguments
///
/// * `expr` - The expression to transform
/// * `t` - The time variable (must be a symbol in expr)
/// * `s` - The transform variable (frequency domain)
/// * `algorithm` - Algorithm to use for computation
///
/// # Returns
///
/// The Laplace transform F(s) of the input expression.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::Expr;
/// use rustmath_calculus::laplace;
///
/// // L{1} = 1/s
/// let expr = Expr::from(1);
/// let result = laplace(&expr, "t", "s", None);
/// assert!(result.is_ok());
///
/// // L{t} = 1/s^2
/// let t = Expr::symbol("t");
/// let result = laplace(&t, "t", "s", None);
/// assert!(result.is_ok());
/// ```
pub fn laplace(
    expr: &Expr,
    t: &str,
    s: &str,
    algorithm: Option<LaplaceAlgorithm>,
) -> Result<Expr, String> {
    let _alg = algorithm.unwrap_or_default();

    // Build transform table
    let table = build_laplace_table(t, s);

    // Try to match expression against known transforms
    if let Some(transform) = table_lookup(expr, &table, t) {
        return Ok(transform);
    }

    // Handle composite expressions using linearity and properties
    match expr {
        Expr::Add(terms) => {
            // L{f + g} = L{f} + L{g} (linearity)
            let mut result = Expr::from(0);
            for term in terms {
                let transform = laplace(term, t, s, algorithm)?;
                result = result + transform;
            }
            Ok(result)
        }
        Expr::Mul(factors) => {
            // Check if it's of the form c*f(t) where c is constant w.r.t. t
            let (constants, vars) = partition_constant_factors(factors, t);

            if vars.len() == 1 {
                // c * f(t) -> c * L{f(t)}
                let transform = laplace(&vars[0], t, s, algorithm)?;
                Ok(constants * transform)
            } else if vars.is_empty() {
                // Pure constant: L{c} = c/s
                Ok(constants / Expr::symbol(s))
            } else {
                // Complex multiplication - return formal result
                Ok(formal_laplace(expr, t, s))
            }
        }
        Expr::Pow(base, exp) => {
            // Handle t^n case
            if **base == Expr::symbol(t) {
                if let Expr::Number(n) = **exp {
                    if n >= 0.0 && n == n.floor() {
                        // L{t^n} = n!/s^(n+1)
                        let n_int = n as i64;
                        return Ok(Expr::from(factorial(n_int)) /
                                  Expr::symbol(s).pow(Expr::from(n_int + 1)));
                    }
                }
            }
            // Unsupported power expression
            Ok(formal_laplace(expr, t, s))
        }
        Expr::Symbol(sym) => {
            if sym == t {
                // L{t} = 1/s^2
                Ok(Expr::from(1) / Expr::symbol(s).pow(Expr::from(2)))
            } else {
                // Constant w.r.t. t: L{c} = c/s
                Ok(expr.clone() / Expr::symbol(s))
            }
        }
        Expr::Number(_) => {
            // L{c} = c/s
            Ok(expr.clone() / Expr::symbol(s))
        }
        Expr::Function(name, args) => {
            // Handle special functions
            match name.as_str() {
                "exp" if args.len() == 1 => {
                    // L{e^(at)} = 1/(s-a) if arg is a*t
                    if let Expr::Mul(factors) = &args[0] {
                        let (constants, vars) = partition_constant_factors(factors, t);
                        if vars.len() == 1 && vars[0] == Expr::symbol(t) {
                            return Ok(Expr::from(1) / (Expr::symbol(s) - constants));
                        }
                    } else if args[0] == Expr::symbol(t) {
                        // L{e^t} = 1/(s-1)
                        return Ok(Expr::from(1) / (Expr::symbol(s) - Expr::from(1)));
                    }
                    Ok(formal_laplace(expr, t, s))
                }
                "sin" if args.len() == 1 => {
                    // L{sin(at)} = a/(s^2 + a^2)
                    if let Some(a) = extract_coefficient(&args[0], t) {
                        let s_expr = Expr::symbol(s);
                        return Ok(a.clone() / (s_expr.pow(Expr::from(2)) + a.pow(Expr::from(2))));
                    }
                    Ok(formal_laplace(expr, t, s))
                }
                "cos" if args.len() == 1 => {
                    // L{cos(at)} = s/(s^2 + a^2)
                    if let Some(a) = extract_coefficient(&args[0], t) {
                        let s_expr = Expr::symbol(s);
                        return Ok(s_expr.clone() / (s_expr.pow(Expr::from(2)) + a.pow(Expr::from(2))));
                    }
                    Ok(formal_laplace(expr, t, s))
                }
                "sinh" if args.len() == 1 => {
                    // L{sinh(at)} = a/(s^2 - a^2)
                    if let Some(a) = extract_coefficient(&args[0], t) {
                        let s_expr = Expr::symbol(s);
                        return Ok(a.clone() / (s_expr.pow(Expr::from(2)) - a.pow(Expr::from(2))));
                    }
                    Ok(formal_laplace(expr, t, s))
                }
                "cosh" if args.len() == 1 => {
                    // L{cosh(at)} = s/(s^2 - a^2)
                    if let Some(a) = extract_coefficient(&args[0], t) {
                        let s_expr = Expr::symbol(s);
                        return Ok(s_expr.clone() / (s_expr.pow(Expr::from(2)) - a.pow(Expr::from(2))));
                    }
                    Ok(formal_laplace(expr, t, s))
                }
                _ => Ok(formal_laplace(expr, t, s)),
            }
        }
        _ => Ok(formal_laplace(expr, t, s)),
    }
}

/// Computes the inverse Laplace transform of an expression.
///
/// The inverse Laplace transform converts a function F(s) in the frequency domain
/// back to f(t) in the time domain.
///
/// # Arguments
///
/// * `expr` - The expression to transform
/// * `s` - The frequency variable (must be a symbol in expr)
/// * `t` - The time variable (result domain)
/// * `algorithm` - Algorithm to use for computation
///
/// # Returns
///
/// The inverse Laplace transform f(t) of the input expression.
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::Expr;
/// use rustmath_calculus::inverse_laplace;
///
/// // L^(-1){1/s} = 1
/// let s = Expr::symbol("s");
/// let expr = Expr::from(1) / s;
/// let result = inverse_laplace(&expr, "s", "t", None);
/// assert!(result.is_ok());
/// ```
pub fn inverse_laplace(
    expr: &Expr,
    s: &str,
    t: &str,
    algorithm: Option<LaplaceAlgorithm>,
) -> Result<Expr, String> {
    let _alg = algorithm.unwrap_or_default();

    // Build inverse transform table
    let table = build_inverse_laplace_table(s, t);

    // Try to match expression against known inverse transforms
    if let Some(transform) = table_lookup(expr, &table, s) {
        return Ok(transform);
    }

    // Handle composite expressions using linearity
    match expr {
        Expr::Add(terms) => {
            // L^(-1){F + G} = L^(-1){F} + L^(-1){G}
            let mut result = Expr::from(0);
            for term in terms {
                let transform = inverse_laplace(term, s, t, algorithm)?;
                result = result + transform;
            }
            Ok(result)
        }
        Expr::Mul(factors) => {
            // c * F(s) -> c * L^(-1){F(s)} if c is constant w.r.t. s
            let (constants, vars) = partition_constant_factors(factors, s);

            if vars.len() == 1 {
                let transform = inverse_laplace(&vars[0], s, t, algorithm)?;
                Ok(constants * transform)
            } else if vars.is_empty() {
                // Pure constant
                Ok(constants * dirac_delta(t))
            } else {
                Ok(formal_inverse_laplace(expr, s, t))
            }
        }
        Expr::Div(num, den) => {
            // Handle rational functions
            match (**num, **den) {
                (Expr::Number(_), Expr::Symbol(ref sym)) if sym == s => {
                    // c/s -> c (constant)
                    Ok((**num).clone())
                }
                (Expr::Number(a), Expr::Pow(ref base, ref exp))
                    if **base == Expr::symbol(s) => {
                    // a/s^n -> a*t^(n-1)/(n-1)! for n > 0
                    if let Expr::Number(n) = **exp {
                        if n > 0.0 && n == n.floor() {
                            let n_int = n as i64;
                            let t_expr = Expr::symbol(t);
                            return Ok(Expr::from(a) * t_expr.pow(Expr::from(n_int - 1)) /
                                      Expr::from(factorial(n_int - 1)));
                        }
                    }
                    Ok(formal_inverse_laplace(expr, s, t))
                }
                (Expr::Number(_), Expr::Add(_)) => {
                    // Could be 1/(s-a) or a/(s^2 + a^2) etc.
                    // Check for specific patterns
                    if let Expr::Add(terms) = &**den {
                        if terms.len() == 2 {
                            // Check for s - a pattern (exponential)
                            let s_sym = Expr::symbol(s);
                            if terms.contains(&s_sym) {
                                for term in terms {
                                    if term != &s_sym {
                                        if let Expr::Neg(inner) = term {
                                            // 1/(s - a) -> e^(at)
                                            let a = (**inner).clone();
                                            return Ok((**num).clone() *
                                                     Expr::function("exp", vec![a * Expr::symbol(t)]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(formal_inverse_laplace(expr, s, t))
                }
                _ => Ok(formal_inverse_laplace(expr, s, t)),
            }
        }
        _ => Ok(formal_inverse_laplace(expr, s, t)),
    }
}

/// Creates a formal (unevaluated) Laplace transform expression.
///
/// This is used when the transform cannot be computed symbolically.
/// Returns an expression of the form laplace(f, t, s).
pub fn formal_laplace(expr: &Expr, t: &str, s: &str) -> Expr {
    Expr::function(
        "laplace",
        vec![expr.clone(), Expr::symbol(t), Expr::symbol(s)],
    )
}

/// Creates a formal (unevaluated) inverse Laplace transform expression.
///
/// This is used when the inverse transform cannot be computed symbolically.
/// Returns an expression of the form ilt(F, s, t).
pub fn formal_inverse_laplace(expr: &Expr, s: &str, t: &str) -> Expr {
    Expr::function(
        "ilt",
        vec![expr.clone(), Expr::symbol(s), Expr::symbol(t)],
    )
}

/// Dummy Laplace transform for Maxima compatibility.
///
/// Creates a formal Laplace transform wrapper that can be used in symbolic expressions.
pub fn dummy_laplace(expr: &Expr, t: &str, s: &str) -> Expr {
    formal_laplace(expr, t, s)
}

/// Dummy inverse Laplace transform for Maxima compatibility.
///
/// Creates a formal inverse Laplace transform wrapper that can be used in symbolic expressions.
pub fn dummy_inverse_laplace(expr: &Expr, s: &str, t: &str) -> Expr {
    formal_inverse_laplace(expr, s, t)
}

// Helper functions

/// Builds a lookup table of known Laplace transforms.
fn build_laplace_table(t: &str, s: &str) -> HashMap<Expr, Expr> {
    let mut table = HashMap::new();
    let t_sym = Expr::symbol(t);
    let s_sym = Expr::symbol(s);

    // L{1} = 1/s
    table.insert(Expr::from(1), Expr::from(1) / s_sym.clone());

    // L{t} = 1/s^2
    table.insert(t_sym.clone(), Expr::from(1) / s_sym.clone().pow(Expr::from(2)));

    // L{t^2} = 2/s^3
    table.insert(
        t_sym.clone().pow(Expr::from(2)),
        Expr::from(2) / s_sym.clone().pow(Expr::from(3)),
    );

    table
}

/// Builds a lookup table of known inverse Laplace transforms.
fn build_inverse_laplace_table(s: &str, t: &str) -> HashMap<Expr, Expr> {
    let mut table = HashMap::new();
    let s_sym = Expr::symbol(s);
    let _t_sym = Expr::symbol(t);

    // L^(-1){1/s} = 1
    table.insert(Expr::from(1) / s_sym.clone(), Expr::from(1));

    // L^(-1){1/s^2} = t
    table.insert(
        Expr::from(1) / s_sym.clone().pow(Expr::from(2)),
        Expr::symbol(t),
    );

    table
}

/// Looks up an expression in a transform table.
fn table_lookup(expr: &Expr, table: &HashMap<Expr, Expr>, _var: &str) -> Option<Expr> {
    table.get(expr).cloned()
}

/// Partitions multiplication factors into constants and variables w.r.t. a variable.
fn partition_constant_factors(factors: &[Expr], var: &str) -> (Expr, Vec<Expr>) {
    let mut constants = Expr::from(1);
    let mut vars = Vec::new();

    for factor in factors {
        if is_constant_wrt(factor, var) {
            constants = constants * factor.clone();
        } else {
            vars.push(factor.clone());
        }
    }

    (constants, vars)
}

/// Checks if an expression is constant with respect to a variable.
fn is_constant_wrt(expr: &Expr, var: &str) -> bool {
    !contains_variable(expr, var)
}

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

/// Extracts coefficient from expressions like a*t -> returns a.
fn extract_coefficient(expr: &Expr, var: &str) -> Option<Expr> {
    match expr {
        Expr::Symbol(s) if s == var => Some(Expr::from(1)),
        Expr::Mul(factors) => {
            let (constants, vars) = partition_constant_factors(factors, var);
            if vars.len() == 1 && vars[0] == Expr::symbol(var) {
                Some(constants)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Computes factorial of a non-negative integer.
fn factorial(n: i64) -> i64 {
    if n <= 0 {
        1
    } else {
        (1..=n).product()
    }
}

/// Creates a Dirac delta function expression.
fn dirac_delta(var: &str) -> Expr {
    Expr::function("dirac_delta", vec![Expr::symbol(var)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace_constant() {
        // L{1} = 1/s
        let expr = Expr::from(1);
        let result = laplace(&expr, "t", "s", None).unwrap();
        let expected = Expr::from(1) / Expr::symbol("s");
        assert_eq!(result, expected);
    }

    #[test]
    fn test_laplace_identity() {
        // L{t} = 1/s^2
        let expr = Expr::symbol("t");
        let result = laplace(&expr, "t", "s", None).unwrap();
        let expected = Expr::from(1) / Expr::symbol("s").pow(Expr::from(2));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_laplace_power() {
        // L{t^2} = 2!/s^3 = 2/s^3
        let expr = Expr::symbol("t").pow(Expr::from(2));
        let result = laplace(&expr, "t", "s", None).unwrap();
        let expected = Expr::from(2) / Expr::symbol("s").pow(Expr::from(3));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_laplace_linear_combination() {
        // L{2 + 3t} = 2/s + 3/s^2
        let expr = Expr::from(2) + Expr::from(3) * Expr::symbol("t");
        let result = laplace(&expr, "t", "s", None).unwrap();

        let s = Expr::symbol("s");
        let expected = Expr::from(2) / s.clone() +
                       Expr::from(3) / s.pow(Expr::from(2));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_laplace_exponential() {
        // L{e^t} = 1/(s-1)
        let expr = Expr::function("exp", vec![Expr::symbol("t")]);
        let result = laplace(&expr, "t", "s", None).unwrap();
        let expected = Expr::from(1) / (Expr::symbol("s") - Expr::from(1));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse_laplace_constant() {
        // L^(-1){1/s} = 1
        let expr = Expr::from(1) / Expr::symbol("s");
        let result = inverse_laplace(&expr, "s", "t", None).unwrap();
        assert_eq!(result, Expr::from(1));
    }

    #[test]
    fn test_inverse_laplace_identity() {
        // L^(-1){1/s^2} = t
        let expr = Expr::from(1) / Expr::symbol("s").pow(Expr::from(2));
        let result = inverse_laplace(&expr, "s", "t", None).unwrap();
        assert_eq!(result, Expr::symbol("t"));
    }

    #[test]
    fn test_inverse_laplace_power() {
        // L^(-1){1/s^3} = t^2/2
        let expr = Expr::from(1) / Expr::symbol("s").pow(Expr::from(3));
        let result = inverse_laplace(&expr, "s", "t", None).unwrap();
        let expected = Expr::symbol("t").pow(Expr::from(2)) / Expr::from(2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dummy_laplace() {
        let expr = Expr::symbol("f");
        let result = dummy_laplace(&expr, "t", "s");
        assert_eq!(
            result,
            Expr::function("laplace", vec![expr, Expr::symbol("t"), Expr::symbol("s")])
        );
    }

    #[test]
    fn test_dummy_inverse_laplace() {
        let expr = Expr::symbol("F");
        let result = dummy_inverse_laplace(&expr, "s", "t");
        assert_eq!(
            result,
            Expr::function("ilt", vec![expr, Expr::symbol("s"), Expr::symbol("t")])
        );
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3628800);
    }

    #[test]
    fn test_contains_variable() {
        assert!(contains_variable(&Expr::symbol("x"), "x"));
        assert!(!contains_variable(&Expr::from(5), "x"));
        assert!(contains_variable(&(Expr::symbol("x") + Expr::from(1)), "x"));
        assert!(!contains_variable(&(Expr::symbol("y") + Expr::from(1)), "x"));
    }
}
