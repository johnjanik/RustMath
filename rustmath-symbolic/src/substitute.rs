//! Expression substitution and evaluation

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::HashMap;
use std::sync::Arc;

impl Expr {
    /// Substitute a symbol with an expression
    ///
    /// Replaces all occurrences of the given symbol with the provided expression.
    pub fn substitute(&self, sym: &Symbol, replacement: &Expr) -> Expr {
        match self {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => self.clone(),
            Expr::Symbol(s) => {
                if s == sym {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            Expr::Binary(op, left, right) => {
                let new_left = left.substitute(sym, replacement);
                let new_right = right.substitute(sym, replacement);
                Expr::Binary(*op, Arc::new(new_left), Arc::new(new_right))
            }
            Expr::Unary(op, inner) => {
                let new_inner = inner.substitute(sym, replacement);
                Expr::Unary(*op, Arc::new(new_inner))
            }
            Expr::Function(name, args) => {
                let new_args: Vec<Arc<Expr>> = args
                    .iter()
                    .map(|arg| Arc::new(arg.substitute(sym, replacement)))
                    .collect();
                Expr::Function(name.clone(), new_args)
            }
        }
    }

    /// Substitute multiple symbols at once
    ///
    /// Takes a map of symbol -> expression and performs all substitutions.
    pub fn substitute_many(&self, substitutions: &HashMap<Symbol, Expr>) -> Expr {
        match self {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => self.clone(),
            Expr::Symbol(s) => substitutions.get(s).cloned().unwrap_or_else(|| self.clone()),
            Expr::Binary(op, left, right) => {
                let new_left = left.substitute_many(substitutions);
                let new_right = right.substitute_many(substitutions);
                Expr::Binary(*op, Arc::new(new_left), Arc::new(new_right))
            }
            Expr::Unary(op, inner) => {
                let new_inner = inner.substitute_many(substitutions);
                Expr::Unary(*op, Arc::new(new_inner))
            }
            Expr::Function(name, args) => {
                let new_args: Vec<Arc<Expr>> = args
                    .iter()
                    .map(|arg| Arc::new(arg.substitute_many(substitutions)))
                    .collect();
                Expr::Function(name.clone(), new_args)
            }
        }
    }

    /// Evaluate the expression to a rational number
    ///
    /// Returns Some(value) if the expression can be fully evaluated to a number,
    /// None if it contains symbols or unsupported operations.
    pub fn eval_rational(&self) -> Option<Rational> {
        match self {
            Expr::Integer(n) => Rational::new(n.clone(), Integer::one()).ok(),
            Expr::Rational(r) => Some(r.clone()),
            Expr::Real(_) => None, // Real numbers are not exact rationals
            Expr::Symbol(_) => None,
            Expr::Binary(op, left, right) => {
                let l = left.eval_rational()?;
                let r = right.eval_rational()?;

                match op {
                    BinaryOp::Add => Some(l + r),
                    BinaryOp::Sub => Some(l - r),
                    BinaryOp::Mul => Some(l * r),
                    BinaryOp::Div => {
                        if r.is_zero() {
                            None
                        } else {
                            Some(l / r)
                        }
                    }
                    BinaryOp::Pow => {
                        // Only handle integer powers
                        if let Expr::Integer(exp) = right.as_ref() {
                            if let Some(e) = exp.to_i64() {
                                if e >= 0 && e <= u32::MAX as i64 {
                                    Some(l.pow(e as u32))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    BinaryOp::Mod => {
                        // Modulo operation - not supported for rational evaluation
                        None
                    }
                }
            }
            Expr::Unary(op, inner) => {
                let val = inner.eval_rational()?;

                match op {
                    UnaryOp::Neg => Some(-val),
                    // Transcendental functions can't be exactly evaluated to rationals
                    _ => None,
                }
            }
            Expr::Function(_, _) => None,
        }
    }

    /// Evaluate the expression to a floating-point number
    ///
    /// This uses f64 approximations for transcendental functions.
    pub fn eval_float(&self) -> Option<f64> {
        use rustmath_core::NumericConversion as _;
        match self {
            Expr::Integer(n) => n.to_i64().map(|i| i as f64),
            Expr::Rational(r) => r.to_f64(),
            Expr::Real(x) => Some(*x),
            Expr::Symbol(_) => None,
            Expr::Binary(op, left, right) => {
                let l = left.eval_float()?;
                let r = right.eval_float()?;

                match op {
                    BinaryOp::Add => Some(l + r),
                    BinaryOp::Sub => Some(l - r),
                    BinaryOp::Mul => Some(l * r),
                    BinaryOp::Div => {
                        if r.abs() < f64::EPSILON {
                            None
                        } else {
                            Some(l / r)
                        }
                    }
                    BinaryOp::Pow => Some(l.powf(r)),
                    BinaryOp::Mod => Some(l % r),
                }
            }
            Expr::Unary(op, inner) => {
                let val = inner.eval_float()?;

                match op {
                    UnaryOp::Neg => Some(-val),
                    UnaryOp::Sin => Some(val.sin()),
                    UnaryOp::Cos => Some(val.cos()),
                    UnaryOp::Tan => Some(val.tan()),
                    UnaryOp::Exp => Some(val.exp()),
                    UnaryOp::Log => {
                        if val > 0.0 {
                            Some(val.ln())
                        } else {
                            None
                        }
                    }
                    UnaryOp::Sqrt => {
                        if val >= 0.0 {
                            Some(val.sqrt())
                        } else {
                            None
                        }
                    }
                    UnaryOp::Abs => Some(val.abs()),
                    UnaryOp::Sign => Some(if val > 0.0 {
                        1.0
                    } else if val < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }),
                    UnaryOp::Sinh => Some(val.sinh()),
                    UnaryOp::Cosh => Some(val.cosh()),
                    UnaryOp::Tanh => Some(val.tanh()),
                    UnaryOp::Arcsin => {
                        if val >= -1.0 && val <= 1.0 {
                            Some(val.asin())
                        } else {
                            None
                        }
                    }
                    UnaryOp::Arccos => {
                        if val >= -1.0 && val <= 1.0 {
                            Some(val.acos())
                        } else {
                            None
                        }
                    }
                    UnaryOp::Arctan => Some(val.atan()),
                    UnaryOp::Arcsinh => Some(val.asinh()),
                    UnaryOp::Arccosh => {
                        if val >= 1.0 {
                            Some(val.acosh())
                        } else {
                            None
                        }
                    }
                    UnaryOp::Arctanh => {
                        if val > -1.0 && val < 1.0 {
                            Some(val.atanh())
                        } else {
                            None
                        }
                    }
                    UnaryOp::Gamma => {
                        // Use Stirling's approximation for gamma function
                        // Γ(x) ≈ √(2π/x) * (x/e)^x for large x
                        // For small positive integers, use exact values
                        if val > 0.0 && val <= 20.0 && (val - val.floor()).abs() < 1e-10 {
                            // Integer case: Γ(n) = (n-1)!
                            let n = val.floor() as i32;
                            if n > 0 {
                                let mut result = 1.0;
                                for i in 1..(n as i32) {
                                    result *= i as f64;
                                }
                                return Some(result);
                            }
                        }
                        // General case using lgamma
                        Some(gamma_approx(val))
                    }
                    UnaryOp::Factorial => {
                        // n! for non-negative integers
                        if val >= 0.0 && (val - val.floor()).abs() < 1e-10 {
                            let n = val.floor() as i32;
                            if n <= 20 {
                                let mut result = 1.0;
                                for i in 1..=n {
                                    result *= i as f64;
                                }
                                return Some(result);
                            }
                        }
                        None
                    }
                    UnaryOp::Erf => {
                        // Error function approximation
                        Some(erf_approx(val))
                    }

                    UnaryOp::Zeta => {
                        // Riemann zeta function ζ(s) = Σ(n=1 to ∞) 1/n^s
                        Some(zeta_approx(val))
                    }
                }
            }
            Expr::Function(name, args) => {
                match name.as_str() {
                    "bessel_j" => {
                        // J_n(x) - Bessel function of the first kind
                        if args.len() == 2 {
                            let order = args[0].eval_float()?;
                            let x = args[1].eval_float()?;
                            Some(bessel_j_approx(order, x))
                        } else {
                            None
                        }
                    }
                    "bessel_y" => {
                        // Y_n(x) - Bessel function of the second kind
                        if args.len() == 2 {
                            let order = args[0].eval_float()?;
                            let x = args[1].eval_float()?;
                            Some(bessel_y_approx(order, x))
                        } else {
                            None
                        }
                    }
                    "bessel_i" => {
                        // I_n(x) - Modified Bessel function of the first kind
                        if args.len() == 2 {
                            let order = args[0].eval_float()?;
                            let x = args[1].eval_float()?;
                            Some(bessel_i_approx(order, x))
                        } else {
                            None
                        }
                    }
                    "bessel_k" => {
                        // K_n(x) - Modified Bessel function of the second kind
                        if args.len() == 2 {
                            let order = args[0].eval_float()?;
                            let x = args[1].eval_float()?;
                            Some(bessel_k_approx(order, x))
                        } else {
                            None
                        }
                    }
                    _ => None, // Unknown function
                }
            }
        }
    }

    /// Collect all symbols in the expression
    pub fn symbols(&self) -> Vec<Symbol> {
        let mut syms = Vec::new();
        self.collect_symbols(&mut syms);
        syms.sort();
        syms.dedup();
        syms
    }

    fn collect_symbols(&self, syms: &mut Vec<Symbol>) {
        match self {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => {}
            Expr::Symbol(s) => syms.push(s.clone()),
            Expr::Binary(_, left, right) => {
                left.collect_symbols(syms);
                right.collect_symbols(syms);
            }
            Expr::Unary(_, inner) => {
                inner.collect_symbols(syms);
            }
            Expr::Function(_, args) => {
                for arg in args {
                    arg.collect_symbols(syms);
                }
            }
        }
    }
}

/// Approximate gamma function using Stirling's formula
fn gamma_approx(x: f64) -> f64 {
    use std::f64::consts::{E, PI};
    if x <= 0.0 {
        return f64::NAN;
    }
    // Stirling's approximation: Γ(x) ≈ √(2π/x) * (x/e)^x
    (2.0 * PI / x).sqrt() * (x / E).powf(x)
}

/// Approximate error function using Taylor series
fn erf_approx(x: f64) -> f64 {
    use std::f64::consts::PI;

    // For |x| > 3, erf(x) ≈ ±1
    if x.abs() > 3.0 {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }

    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Approximate Riemann zeta function
fn zeta_approx(s: f64) -> f64 {
    use std::f64::consts::PI;

    // Special values
    if (s - 0.0).abs() < 1e-10 {
        return -0.5; // ζ(0) = -1/2
    }
    if (s - 1.0).abs() < 1e-10 {
        return f64::INFINITY; // ζ(1) has a pole
    }
    if (s - 2.0).abs() < 1e-10 {
        return PI * PI / 6.0; // ζ(2) = π²/6
    }
    if (s - 4.0).abs() < 1e-10 {
        return PI.powi(4) / 90.0; // ζ(4) = π⁴/90
    }

    // For Re(s) > 1, use series approximation
    // ζ(s) = Σ(n=1 to ∞) 1/n^s
    if s > 1.0 {
        let mut sum = 0.0;
        let max_terms = 1000;
        for n in 1..=max_terms {
            let term = 1.0 / (n as f64).powf(s);
            sum += term;
            if term < 1e-10 {
                break;
            }
        }
        return sum;
    }

    // For s < 1, use functional equation:
    // ζ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s) * ζ(1-s)
    // For simplicity, return NaN for negative values
    f64::NAN
}

/// Factorial for floating point (up to 20)
fn factorial_f64(n: usize) -> f64 {
    if n > 20 {
        return f64::INFINITY;
    }
    let mut result = 1.0;
    for i in 1..=n {
        result *= i as f64;
    }
    result
}

/// Approximate Bessel J function (first kind)
/// J_n(x) = (x/2)^n * Σ(k=0 to ∞) [(-1)^k / (k! * (n+k)!)] * (x/2)^(2k)
fn bessel_j_approx(order: f64, x: f64) -> f64 {
    // Only handle non-negative integer orders for now
    if order < 0.0 || (order - order.floor()).abs() > 1e-10 {
        return f64::NAN;
    }

    let n = order.floor() as usize;
    let half_x = x / 2.0;
    let mut sum = 0.0;

    for k in 0..100 {
        let numerator = (-1.0f64).powi(k as i32) * half_x.powi(2 * k as i32 + n as i32);
        let denominator = factorial_f64(k) * factorial_f64(k + n);

        if denominator.is_infinite() {
            break;
        }

        let term = numerator / denominator;
        sum += term;

        if term.abs() < 1e-15 {
            break;
        }
    }

    sum
}

/// Approximate Bessel Y function (second kind)
/// Y_n(x) is more complex and involves logarithmic terms
fn bessel_y_approx(_order: f64, _x: f64) -> f64 {
    // Y_n is defined as: Y_n(x) = [J_n(x)cos(nπ) - J_{-n}(x)] / sin(nπ)
    // For simplicity, return NaN for now as full implementation is complex
    f64::NAN
}

/// Approximate Modified Bessel I function (first kind)
/// I_n(x) = (x/2)^n * Σ(k=0 to ∞) [1 / (k! * (n+k)!)] * (x/2)^(2k)
fn bessel_i_approx(order: f64, x: f64) -> f64 {
    // Only handle non-negative integer orders for now
    if order < 0.0 || (order - order.floor()).abs() > 1e-10 {
        return f64::NAN;
    }

    let n = order.floor() as usize;
    let half_x = x / 2.0;
    let mut sum = 0.0;

    for k in 0..100 {
        let numerator = half_x.powi(2 * k as i32 + n as i32);
        let denominator = factorial_f64(k) * factorial_f64(k + n);

        if denominator.is_infinite() {
            break;
        }

        let term = numerator / denominator;
        sum += term;

        if term.abs() < 1e-15 {
            break;
        }
    }

    sum
}

/// Approximate Modified Bessel K function (second kind)
/// K_n(x) is the modified Bessel function of the second kind
fn bessel_k_approx(_order: f64, _x: f64) -> f64 {
    // K_n has a complex definition involving I_n and I_{-n}
    // For simplicity, return NaN for now as full implementation is complex
    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_substitute() {
        // x + 1, substitute x -> 2
        let x = Expr::symbol("x");
        let expr = x.clone() + Expr::from(1);

        let result = expr.substitute(&Symbol::new("x"), &Expr::from(2));

        // Should be 2 + 1
        assert_eq!(result.eval_rational(), Some(Rational::new(3, 1).unwrap()));
    }

    #[test]
    fn test_substitute_many() {
        // x + y, substitute x -> 2, y -> 3
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let expr = x.clone() + y.clone();

        let mut subs = HashMap::new();
        subs.insert(Symbol::new("x"), Expr::from(2));
        subs.insert(Symbol::new("y"), Expr::from(3));

        let result = expr.substitute_many(&subs);

        // Should be 2 + 3 = 5
        assert_eq!(result.eval_rational(), Some(Rational::new(5, 1).unwrap()));
    }

    #[test]
    fn test_eval_rational() {
        // (2 + 3) * 4
        let expr = (Expr::from(2) + Expr::from(3)) * Expr::from(4);
        assert_eq!(expr.eval_rational(), Some(Rational::new(20, 1).unwrap()));
    }

    #[test]
    fn test_eval_float() {
        // sin(0)
        let expr = Expr::from(0).sin();
        let result = expr.eval_float().unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_symbols() {
        // x + y * z
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let z = Expr::symbol("z");
        let expr = x + (y * z);

        let syms = expr.symbols();
        assert_eq!(syms.len(), 3);
        assert!(syms.contains(&Symbol::new("x")));
        assert!(syms.contains(&Symbol::new("y")));
        assert!(syms.contains(&Symbol::new("z")));
    }

    #[test]
    fn test_eval_hyperbolic_functions() {
        use std::f64::consts::E;

        // sinh(1) ≈ (e - 1/e) / 2
        let expr = Expr::from(1).sinh();
        let result = expr.eval_float().unwrap();
        let expected = (E - 1.0 / E) / 2.0;
        assert!((result - expected).abs() < 1e-10);

        // cosh(0) = 1
        let expr = Expr::from(0).cosh();
        let result = expr.eval_float().unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // tanh(0) = 0
        let expr = Expr::from(0).tanh();
        let result = expr.eval_float().unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_inverse_trig() {
        use std::f64::consts::PI;

        // arcsin(0) = 0
        let expr = Expr::from(0).arcsin();
        let result = expr.eval_float().unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // arccos(1) = 0
        let expr = Expr::from(1).arccos();
        let result = expr.eval_float().unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // arctan(0) = 0
        let expr = Expr::from(0).arctan();
        let result = expr.eval_float().unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_abs_and_sign() {
        // abs(-5) = 5
        let expr = Expr::from(-5).abs();
        let result = expr.eval_float().unwrap();
        assert!((result - 5.0).abs() < 1e-10);

        // sign(-3) = -1
        let expr = Expr::from(-3).sign();
        let result = expr.eval_float().unwrap();
        assert!((result - (-1.0)).abs() < 1e-10);

        // sign(5) = 1
        let expr = Expr::from(5).sign();
        let result = expr.eval_float().unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // sign(0) = 0
        let expr = Expr::from(0).sign();
        let result = expr.eval_float().unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_gamma() {
        // Γ(1) = 0! = 1
        let expr = Expr::from(1).gamma();
        let result = expr.eval_float().unwrap();
        assert!((result - 1.0).abs() < 1e-6);

        // Γ(5) = 4! = 24
        let expr = Expr::from(5).gamma();
        let result = expr.eval_float().unwrap();
        assert!((result - 24.0).abs() < 1e-6);
    }

    #[test]
    fn test_eval_factorial() {
        // 0! = 1
        let expr = Expr::from(0).factorial();
        let result = expr.eval_float().unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // 5! = 120
        let expr = Expr::from(5).factorial();
        let result = expr.eval_float().unwrap();
        assert!((result - 120.0).abs() < 1e-10);

        // 10! = 3628800
        let expr = Expr::from(10).factorial();
        let result = expr.eval_float().unwrap();
        assert!((result - 3628800.0).abs() < 1e-6);
    }

    #[test]
    fn test_eval_erf() {
        // erf(0) = 0
        let expr = Expr::from(0).erf();
        let result = expr.eval_float().unwrap();
        assert!((result - 0.0).abs() < 1e-6);

        // erf(large) ≈ 1
        let expr = Expr::from(3).erf();
        let result = expr.eval_float().unwrap();
        assert!((result - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_eval_zeta() {
        use std::f64::consts::PI;

        // ζ(2) = π²/6
        let expr = Expr::from(2).zeta();
        let result = expr.eval_float().unwrap();
        let expected = PI * PI / 6.0;
        assert!((result - expected).abs() < 1e-6);

        // ζ(0) = -1/2
        let expr = Expr::from(0).zeta();
        let result = expr.eval_float().unwrap();
        assert!((result - (-0.5)).abs() < 1e-10);

        // ζ(4) = π⁴/90
        let expr = Expr::from(4).zeta();
        let result = expr.eval_float().unwrap();
        let expected = PI.powi(4) / 90.0;
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_j() {
        // J_0(0) = 1
        let expr = Expr::bessel_j(Expr::from(0), Expr::from(0));
        let result = expr.eval_float().unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // J_1(0) = 0
        let expr = Expr::bessel_j(Expr::from(1), Expr::from(0));
        let result = expr.eval_float().unwrap();
        assert!(result.abs() < 1e-10);

        // J_0(1) ≈ 0.7651976866
        let expr = Expr::bessel_j(Expr::from(0), Expr::from(1));
        let result = expr.eval_float().unwrap();
        assert!((result - 0.7651976866).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_i() {
        // I_0(0) = 1
        let expr = Expr::bessel_i(Expr::from(0), Expr::from(0));
        let result = expr.eval_float().unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // I_1(0) = 0
        let expr = Expr::bessel_i(Expr::from(1), Expr::from(0));
        let result = expr.eval_float().unwrap();
        assert!(result.abs() < 1e-10);

        // I_0(1) ≈ 1.2660658777
        let expr = Expr::bessel_i(Expr::from(0), Expr::from(1));
        let result = expr.eval_float().unwrap();
        assert!((result - 1.2660658777).abs() < 1e-6);
    }

    #[test]
    fn test_custom_function() {
        // Test custom function creation
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");

        // f(x, y) = custom_func(x, y)
        let expr = Expr::function("my_func", vec![x.clone(), y.clone()]);

        // Should have 2 symbols
        let syms = expr.symbols();
        assert_eq!(syms.len(), 2);
        assert!(syms.contains(&Symbol::new("x")));
        assert!(syms.contains(&Symbol::new("y")));

        // Display should show function name and arguments
        let display = format!("{}", expr);
        assert!(display.contains("my_func"));
        assert!(display.contains("x"));
        assert!(display.contains("y"));

        // Substitution should work
        let result = expr.substitute(&Symbol::new("x"), &Expr::from(5));
        let result_syms = result.symbols();
        assert_eq!(result_syms.len(), 1);
        assert!(result_syms.contains(&Symbol::new("y")));
    }

    #[test]
    fn test_bessel_function_display() {
        let expr = Expr::bessel_j(Expr::from(0), Expr::symbol("x"));
        let display = format!("{}", expr);
        assert!(display.contains("bessel_j"));
        assert!(display.contains("x"));

        let expr = Expr::bessel_i(Expr::from(1), Expr::symbol("y"));
        let display = format!("{}", expr);
        assert!(display.contains("bessel_i"));
        assert!(display.contains("y"));
    }
}
