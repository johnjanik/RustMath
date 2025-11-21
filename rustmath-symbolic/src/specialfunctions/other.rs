//! Other Mathematical Functions
//!
//! This module implements various mathematical utility functions that don't
//! fit into other specialized categories.
//!
//! Corresponds to sage.functions.other
//!
//! # Functions
//!
//! - `abs_symbolic(x)`: Absolute value function |x|
//! - `ceil(x)`: Ceiling function ⌈x⌉
//! - `floor(x)`: Floor function ⌊x⌋
//! - `frac(x)`: Fractional part {x} = x - ⌊x⌋
//! - `factorial(n)`: Factorial n!
//! - `binomial(n, k)`: Binomial coefficient C(n,k)
//! - `real_part(z)`: Real part Re(z)
//! - `imag_part(z)`: Imaginary part Im(z)
//! - `conjugate(z)`: Complex conjugate z̄
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::other::*;
//! use rustmath_symbolic::Expr;
//!
//! assert_eq!(abs_symbolic(&Expr::from(-5)), Expr::from(5));
//! assert_eq!(ceil(&Expr::from(3)), Expr::from(4)); // assuming rational 3.x
//! assert_eq!(floor(&Expr::from(3)), Expr::from(3));
//! assert_eq!(factorial(&Expr::from(5)), Expr::from(120));
//! ```

use crate::expression::Expr;
use crate::symbol::Symbol;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::sync::Arc;

/// Helper function to try converting an Expr to f64
fn try_expr_to_f64(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Integer(i) => i.to_f64(),
        Expr::Rational(r) => r.to_f64(),
        _ => None,
    }
}

/// Absolute value function |x|
///
/// Returns the absolute value of an expression. For numeric values,
/// computes the result. For symbolic expressions, returns a symbolic abs.
///
/// # Arguments
///
/// * `x` - The expression to take absolute value of
///
/// # Returns
///
/// |x| as an expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::other::abs_symbolic;
/// use rustmath_symbolic::Expr;
///
/// assert_eq!(abs_symbolic(&Expr::from(-5)), Expr::from(5));
/// assert_eq!(abs_symbolic(&Expr::from(3)), Expr::from(3));
/// ```
pub fn abs_symbolic(x: &Expr) -> Expr {
    match x {
        Expr::Integer(i) => Expr::Integer(i.abs()),
        Expr::Rational(r) => Expr::Rational(r.abs()),
        _ => Expr::Function("abs".to_string(), vec![Arc::new(x.clone())]),
    }
}

/// Ceiling function ⌈x⌉
///
/// Returns the smallest integer greater than or equal to x.
///
/// # Arguments
///
/// * `x` - The expression to ceiling
///
/// # Returns
///
/// ⌈x⌉ as an expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::other::ceil;
/// use rustmath_symbolic::Expr;
/// use rustmath_rationals::Rational;
///
/// assert_eq!(ceil(&Expr::Rational(Rational::new(5, 2).unwrap())), Expr::from(3));
/// assert_eq!(ceil(&Expr::from(5)), Expr::from(5));
/// ```
pub fn ceil(x: &Expr) -> Expr {
    match x {
        Expr::Integer(i) => Expr::Integer(i.clone()),
        Expr::Rational(r) => Expr::Integer(r.ceil()),
        _ => Expr::Function("ceil".to_string(), vec![Arc::new(x.clone())]),
    }
}

/// Floor function ⌊x⌋
///
/// Returns the largest integer less than or equal to x.
///
/// # Arguments
///
/// * `x` - The expression to floor
///
/// # Returns
///
/// ⌊x⌋ as an expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::other::floor;
/// use rustmath_symbolic::Expr;
/// use rustmath_rationals::Rational;
///
/// assert_eq!(floor(&Expr::Rational(Rational::new(5, 2).unwrap())), Expr::from(2));
/// assert_eq!(floor(&Expr::from(5)), Expr::from(5));
/// ```
pub fn floor(x: &Expr) -> Expr {
    match x {
        Expr::Integer(i) => Expr::Integer(i.clone()),
        Expr::Rational(r) => Expr::Integer(r.floor()),
        _ => Expr::Function("floor".to_string(), vec![Arc::new(x.clone())]),
    }
}

/// Fractional part {x} = x - ⌊x⌋
///
/// Returns the fractional part of x, always in [0, 1).
///
/// # Arguments
///
/// * `x` - The expression to get fractional part of
///
/// # Returns
///
/// {x} as an expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::other::frac;
/// use rustmath_symbolic::Expr;
/// use rustmath_rationals::Rational;
///
/// assert_eq!(frac(&Expr::Rational(Rational::new(5, 2).unwrap())),
///            Expr::Rational(Rational::new(1, 2).unwrap()));
/// assert_eq!(frac(&Expr::from(5)), Expr::from(0));
/// ```
pub fn frac(x: &Expr) -> Expr {
    match x {
        Expr::Integer(_) => Expr::from(0),
        Expr::Rational(r) => {
            let floor_val = r.floor();
            Expr::Rational(r.clone() - Rational::from_integer(floor_val))
        }
        _ => Expr::Function("frac".to_string(), vec![Arc::new(x.clone())]),
    }
}

/// Factorial function n!
///
/// Returns n! = 1 × 2 × 3 × ... × n
///
/// # Arguments
///
/// * `n` - The expression to compute factorial of
///
/// # Returns
///
/// n! as an expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::other::factorial;
/// use rustmath_symbolic::Expr;
///
/// assert_eq!(factorial(&Expr::from(0)), Expr::from(1));
/// assert_eq!(factorial(&Expr::from(1)), Expr::from(1));
/// assert_eq!(factorial(&Expr::from(5)), Expr::from(120));
/// ```
pub fn factorial(n: &Expr) -> Expr {
    if let Expr::Integer(i) = n {
        // Compute factorial for non-negative integers
        let n_val = i.to_i64();
        if n_val >= 0 && n_val <= 20 {
            // Compute directly for small values
            let mut result = Integer::one();
            for k in 2..=n_val {
                result = result * Integer::from(k);
            }
            return Expr::Integer(result);
        }
    }

    // Return symbolic form for large or non-integer values
    Expr::Function("factorial".to_string(), vec![Arc::new(n.clone())])
}

/// Binomial coefficient C(n, k) = n! / (k! (n-k)!)
///
/// Returns the binomial coefficient "n choose k".
///
/// # Arguments
///
/// * `n` - The total number
/// * `k` - The number to choose
///
/// # Returns
///
/// C(n, k) as an expression
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::specialfunctions::other::binomial;
/// use rustmath_symbolic::Expr;
///
/// assert_eq!(binomial(&Expr::from(5), &Expr::from(2)), Expr::from(10));
/// assert_eq!(binomial(&Expr::from(10), &Expr::from(3)), Expr::from(120));
/// ```
pub fn binomial(n: &Expr, k: &Expr) -> Expr {
    // Try to compute for integer values
    if let (Expr::Integer(n_int), Expr::Integer(k_int)) = (n, k) {
        let (n_val, k_val) = (n_int.to_i64(), k_int.to_i64());
        {
            if n_val >= 0 && k_val >= 0 && k_val <= n_val && n_val <= 30 {
                // Compute using multiplicative formula: C(n,k) = n*(n-1)*...*(n-k+1) / k!
                let mut result = Integer::one();
                for i in 0..k_val {
                    result = result * Integer::from(n_val - i);
                    result = result / Integer::from(i + 1);
                }
                return Expr::Integer(result);
            }
        }
    }

    // Return symbolic form
    Expr::Function(
        "binomial".to_string(),
        vec![Arc::new(n.clone()), Arc::new(k.clone())],
    )
}

/// Real part of a complex expression Re(z)
///
/// For now, returns a symbolic representation.
///
/// # Arguments
///
/// * `z` - The expression to get real part of
///
/// # Returns
///
/// Re(z) as an expression
pub fn real_part(z: &Expr) -> Expr {
    match z {
        Expr::Integer(_) | Expr::Rational(_) => z.clone(),
        _ => Expr::Function("real_part".to_string(), vec![Arc::new(z.clone())]),
    }
}

/// Imaginary part of a complex expression Im(z)
///
/// For real numbers, returns 0. For symbolic expressions, returns symbolic form.
///
/// # Arguments
///
/// * `z` - The expression to get imaginary part of
///
/// # Returns
///
/// Im(z) as an expression
pub fn imag_part(z: &Expr) -> Expr {
    match z {
        Expr::Integer(_) | Expr::Rational(_) => Expr::from(0),
        _ => Expr::Function("imag_part".to_string(), vec![Arc::new(z.clone())]),
    }
}

/// Complex conjugate z̄
///
/// For real numbers, returns the number itself. For symbolic, returns symbolic form.
///
/// # Arguments
///
/// * `z` - The expression to conjugate
///
/// # Returns
///
/// z̄ as an expression
pub fn conjugate(z: &Expr) -> Expr {
    match z {
        Expr::Integer(_) | Expr::Rational(_) => z.clone(),
        _ => Expr::Function("conjugate".to_string(), vec![Arc::new(z.clone())]),
    }
}

/// Complex argument function arg(z)
///
/// Returns the argument (angle/phase) of a complex number.
///
/// Corresponds to sage.functions.other.Function_arg
pub fn arg(z: &Expr) -> Expr {
    // For real numbers, return 0 if positive, π if negative
    match z {
        Expr::Integer(i) => {
            if i > &Integer::zero() {
                Expr::from(0)
            } else if i < &Integer::zero() {
                Expr::Function("pi".to_string(), vec![])
            } else {
                Expr::Function("arg".to_string(), vec![Arc::new(z.clone())])
            }
        }
        Expr::Rational(r) => {
            if r > &Rational::zero() {
                Expr::from(0)
            } else if r < &Rational::zero() {
                Expr::Function("pi".to_string(), vec![])
            } else {
                Expr::Function("arg".to_string(), vec![Arc::new(z.clone())])
            }
        }
        _ => Expr::Function("arg".to_string(), vec![Arc::new(z.clone())]),
    }
}

/// Real nth root function
///
/// Corresponds to sage.functions.other.Function_real_nth_root
pub fn real_nth_root(x: &Expr, n: &Expr) -> Expr {
    Expr::Function(
        "real_nth_root".to_string(),
        vec![Arc::new(x.clone()), Arc::new(n.clone())],
    )
}

/// Square root function
///
/// Corresponds to sage.functions.other.Function_sqrt
pub fn sqrt_function(x: &Expr) -> Expr {
    x.clone().sqrt()
}

/// Symbolic limit function
///
/// Represents lim_{var→point} expr
///
/// Corresponds to sage.functions.other.Function_limit
pub fn limit(expr: &Expr, var: &Symbol, point: &Expr) -> Expr {
    Expr::Function(
        "limit".to_string(),
        vec![
            Arc::new(expr.clone()),
            Arc::new(Expr::Symbol(var.clone())),
            Arc::new(point.clone()),
        ],
    )
}

/// Symbolic sum function
///
/// Represents Σ_{var=lower}^{upper} expr
///
/// Corresponds to sage.functions.other.Function_sum
pub fn sum(expr: &Expr, var: &Symbol, lower: &Expr, upper: &Expr) -> Expr {
    Expr::Function(
        "sum".to_string(),
        vec![
            Arc::new(expr.clone()),
            Arc::new(Expr::Symbol(var.clone())),
            Arc::new(lower.clone()),
            Arc::new(upper.clone()),
        ],
    )
}

/// Symbolic product function
///
/// Represents Π_{var=lower}^{upper} expr
///
/// Corresponds to sage.functions.other.Function_prod
pub fn product(expr: &Expr, var: &Symbol, lower: &Expr, upper: &Expr) -> Expr {
    Expr::Function(
        "product".to_string(),
        vec![
            Arc::new(expr.clone()),
            Arc::new(Expr::Symbol(var.clone())),
            Arc::new(lower.clone()),
            Arc::new(upper.clone()),
        ],
    )
}

/// Order function O(expr) for asymptotic analysis
///
/// Corresponds to sage.functions.other.Function_Order
pub fn order(expr: &Expr, var: &Symbol, point: &Expr) -> Expr {
    Expr::Function(
        "order".to_string(),
        vec![
            Arc::new(expr.clone()),
            Arc::new(Expr::Symbol(var.clone())),
            Arc::new(point.clone()),
        ],
    )
}

/// Case-based function
///
/// Corresponds to sage.functions.other.Function_cases
pub fn cases(cases_list: &[(Expr, Expr)]) -> Expr {
    let cases_vec: Vec<Arc<Expr>> = cases_list
        .iter()
        .flat_map(|(cond, val)| vec![Arc::new(cond.clone()), Arc::new(val.clone())])
        .collect();

    Expr::Function("cases".to_string(), cases_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    #[test]
    fn test_abs_integer() {
        assert_eq!(abs_symbolic(&Expr::from(-5)), Expr::from(5));
        assert_eq!(abs_symbolic(&Expr::from(5)), Expr::from(5));
        assert_eq!(abs_symbolic(&Expr::from(0)), Expr::from(0));
    }

    #[test]
    fn test_abs_rational() {
        assert_eq!(
            abs_symbolic(&Expr::Rational(Rational::new(-3, 4).unwrap())),
            Expr::Rational(Rational::new(3, 4).unwrap())
        );
        assert_eq!(
            abs_symbolic(&Expr::Rational(Rational::new(3, 4).unwrap())),
            Expr::Rational(Rational::new(3, 4).unwrap())
        );
    }

    #[test]
    fn test_abs_symbolic() {
        let x = Symbol::new("x");
        let abs_x = abs_symbolic(&Expr::Symbol(x));
        assert!(matches!(abs_x, Expr::Function(name, _) if name == "abs"));
    }

    #[test]
    fn test_ceil_integer() {
        assert_eq!(ceil(&Expr::from(5)), Expr::from(5));
        assert_eq!(ceil(&Expr::from(-3)), Expr::from(-3));
    }

    #[test]
    fn test_ceil_rational() {
        assert_eq!(
            ceil(&Expr::Rational(Rational::new(5, 2).unwrap())),
            Expr::from(3)
        );
        assert_eq!(
            ceil(&Expr::Rational(Rational::new(-5, 2).unwrap())),
            Expr::from(-2)
        );
        assert_eq!(
            ceil(&Expr::Rational(Rational::new(10, 5).unwrap())),
            Expr::from(2)
        );
    }

    #[test]
    fn test_floor_integer() {
        assert_eq!(floor(&Expr::from(5)), Expr::from(5));
        assert_eq!(floor(&Expr::from(-3)), Expr::from(-3));
    }

    #[test]
    fn test_floor_rational() {
        assert_eq!(
            floor(&Expr::Rational(Rational::new(5, 2).unwrap())),
            Expr::from(2)
        );
        assert_eq!(
            floor(&Expr::Rational(Rational::new(-5, 2).unwrap())),
            Expr::from(-3)
        );
        assert_eq!(
            floor(&Expr::Rational(Rational::new(10, 5).unwrap())),
            Expr::from(2)
        );
    }

    #[test]
    fn test_frac_integer() {
        assert_eq!(frac(&Expr::from(5)), Expr::from(0));
        assert_eq!(frac(&Expr::from(0)), Expr::from(0));
    }

    #[test]
    fn test_frac_rational() {
        assert_eq!(
            frac(&Expr::Rational(Rational::new(5, 2).unwrap())),
            Expr::Rational(Rational::new(1, 2).unwrap())
        );
        assert_eq!(
            frac(&Expr::Rational(Rational::new(7, 3).unwrap())),
            Expr::Rational(Rational::new(1, 3).unwrap())
        );
        // 10/5 = 2, which is an integer, so frac should be 0
        let result = frac(&Expr::Rational(Rational::new(10, 5).unwrap()));
        // The result could be either Rational(0/1) or Integer(0)
        match result {
            Expr::Integer(i) => assert_eq!(i, Integer::zero()),
            Expr::Rational(r) => assert_eq!(r, Rational::new(0, 1).unwrap()),
            _ => panic!("Expected integer or rational zero"),
        }
    }

    #[test]
    fn test_factorial_small() {
        assert_eq!(factorial(&Expr::from(0)), Expr::from(1));
        assert_eq!(factorial(&Expr::from(1)), Expr::from(1));
        assert_eq!(factorial(&Expr::from(2)), Expr::from(2));
        assert_eq!(factorial(&Expr::from(3)), Expr::from(6));
        assert_eq!(factorial(&Expr::from(4)), Expr::from(24));
        assert_eq!(factorial(&Expr::from(5)), Expr::from(120));
    }

    #[test]
    fn test_factorial_larger() {
        assert_eq!(factorial(&Expr::from(10)), Expr::from(3628800));
    }

    #[test]
    fn test_factorial_symbolic() {
        let x = Symbol::new("x");
        let fact_x = factorial(&Expr::Symbol(x));
        assert!(matches!(fact_x, Expr::Function(name, _) if name == "factorial"));
    }

    #[test]
    fn test_binomial_small() {
        assert_eq!(binomial(&Expr::from(5), &Expr::from(0)), Expr::from(1));
        assert_eq!(binomial(&Expr::from(5), &Expr::from(1)), Expr::from(5));
        assert_eq!(binomial(&Expr::from(5), &Expr::from(2)), Expr::from(10));
        assert_eq!(binomial(&Expr::from(5), &Expr::from(3)), Expr::from(10));
        assert_eq!(binomial(&Expr::from(5), &Expr::from(5)), Expr::from(1));
    }

    #[test]
    fn test_binomial_larger() {
        assert_eq!(binomial(&Expr::from(10), &Expr::from(3)), Expr::from(120));
        assert_eq!(binomial(&Expr::from(10), &Expr::from(5)), Expr::from(252));
    }

    #[test]
    fn test_binomial_properties() {
        // C(n, k) = C(n, n-k)
        let n = Expr::from(8);
        let k1 = Expr::from(3);
        let k2 = Expr::from(5);
        assert_eq!(binomial(&n, &k1), binomial(&n, &k2));
    }

    #[test]
    fn test_real_part() {
        assert_eq!(real_part(&Expr::from(5)), Expr::from(5));
        assert_eq!(
            real_part(&Expr::Rational(Rational::new(3, 4).unwrap())),
            Expr::Rational(Rational::new(3, 4).unwrap())
        );
    }

    #[test]
    fn test_imag_part() {
        assert_eq!(imag_part(&Expr::from(5)), Expr::from(0));
        assert_eq!(imag_part(&Expr::Rational(Rational::new(3, 4).unwrap())), Expr::from(0));
    }

    #[test]
    fn test_conjugate() {
        assert_eq!(conjugate(&Expr::from(5)), Expr::from(5));
        assert_eq!(
            conjugate(&Expr::Rational(Rational::new(3, 4).unwrap())),
            Expr::Rational(Rational::new(3, 4).unwrap())
        );
    }

    #[test]
    fn test_arg() {
        // arg of positive real is 0
        assert_eq!(arg(&Expr::from(5)), Expr::from(0));
    }

    #[test]
    fn test_real_nth_root() {
        let x = Symbol::new("x");
        let root = real_nth_root(&Expr::Symbol(x), &Expr::from(3));
        assert!(matches!(root, Expr::Function(name, _) if name == "real_nth_root"));
    }

    #[test]
    fn test_sqrt_function() {
        let x = Symbol::new("x");
        let sqrt_x = sqrt_function(&Expr::Symbol(x));
        // sqrt should return a power with exponent 1/2
        assert!(!sqrt_x.is_constant());
    }

    #[test]
    fn test_limit_symbolic() {
        let x = Symbol::new("x");
        let lim = limit(&Expr::Symbol(x.clone()), &x, &Expr::from(0));
        assert!(matches!(lim, Expr::Function(name, _) if name == "limit"));
    }

    #[test]
    fn test_sum_symbolic() {
        let i = Symbol::new("i");
        let s = sum(&Expr::Symbol(i.clone()), &i, &Expr::from(1), &Expr::from(10));
        assert!(matches!(s, Expr::Function(name, _) if name == "sum"));
    }

    #[test]
    fn test_product_symbolic() {
        let i = Symbol::new("i");
        let p = product(&Expr::Symbol(i.clone()), &i, &Expr::from(1), &Expr::from(5));
        assert!(matches!(p, Expr::Function(name, _) if name == "product"));
    }
}
