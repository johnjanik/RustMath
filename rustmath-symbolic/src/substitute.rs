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
            Expr::Integer(_) | Expr::Rational(_) => self.clone(),
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
        }
    }

    /// Substitute multiple symbols at once
    ///
    /// Takes a map of symbol -> expression and performs all substitutions.
    pub fn substitute_many(&self, substitutions: &HashMap<Symbol, Expr>) -> Expr {
        match self {
            Expr::Integer(_) | Expr::Rational(_) => self.clone(),
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
            Expr::Integer(_) | Expr::Rational(_) => {}
            Expr::Symbol(s) => syms.push(s.clone()),
            Expr::Binary(_, left, right) => {
                left.collect_symbols(syms);
                right.collect_symbols(syms);
            }
            Expr::Unary(_, inner) => {
                inner.collect_symbols(syms);
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
        assert_eq!(result.eval_rational(), Some(Rational::from((3, 1))));
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
        assert_eq!(result.eval_rational(), Some(Rational::from((5, 1))));
    }

    #[test]
    fn test_eval_rational() {
        // (2 + 3) * 4
        let expr = (Expr::from(2) + Expr::from(3)) * Expr::from(4);
        assert_eq!(expr.eval_rational(), Some(Rational::from((20, 1))));
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
}
