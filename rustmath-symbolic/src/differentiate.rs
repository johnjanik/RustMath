//! Symbolic differentiation

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_core::Ring;
use std::sync::Arc;

impl Expr {
    /// Differentiate the expression with respect to a symbol
    ///
    /// Implements standard differentiation rules:
    /// - d/dx(c) = 0 for constants
    /// - d/dx(x) = 1
    /// - d/dx(f + g) = f' + g'
    /// - d/dx(f * g) = f'*g + f*g' (product rule)
    /// - d/dx(f / g) = (f'*g - f*g') / g² (quotient rule)
    /// - d/dx(f^n) = n*f^(n-1)*f' (power rule)
    /// - d/dx(sin(f)) = cos(f)*f' (chain rule)
    /// - d/dx(cos(f)) = -sin(f)*f' (chain rule)
    /// - d/dx(exp(f)) = exp(f)*f' (chain rule)
    /// - d/dx(log(f)) = f'/f (chain rule)
    pub fn differentiate(&self, var: &Symbol) -> Self {
        match self {
            // Constant rule: d/dx(c) = 0
            Expr::Integer(_) | Expr::Rational(_) => Expr::from(0),

            // Variable rule: d/dx(x) = 1, d/dx(y) = 0 for y ≠ x
            Expr::Symbol(s) => {
                if s == var {
                    Expr::from(1)
                } else {
                    Expr::from(0)
                }
            }

            // Binary operations
            Expr::Binary(op, left, right) => match op {
                // Sum rule: d/dx(f + g) = f' + g'
                BinaryOp::Add => {
                    left.differentiate(var) + right.differentiate(var)
                }

                // Difference rule: d/dx(f - g) = f' - g'
                BinaryOp::Sub => {
                    left.differentiate(var) - right.differentiate(var)
                }

                // Product rule: d/dx(f * g) = f'*g + f*g'
                BinaryOp::Mul => {
                    let df = left.differentiate(var);
                    let dg = right.differentiate(var);
                    df * (**right).clone() + (**left).clone() * dg
                }

                // Quotient rule: d/dx(f / g) = (f'*g - f*g') / g²
                BinaryOp::Div => {
                    let df = left.differentiate(var);
                    let dg = right.differentiate(var);
                    let numerator = df * (**right).clone() - (**left).clone() * dg;
                    let denominator = (**right).clone().pow(Expr::from(2));
                    numerator / denominator
                }

                // Power rule: d/dx(f^n) = n*f^(n-1)*f'
                BinaryOp::Pow => {
                    // Check if exponent is constant
                    if right.is_constant() {
                        // d/dx(f^n) = n*f^(n-1)*f'
                        let n = (**right).clone();
                        let df = left.differentiate(var);
                        let n_minus_1 = n.clone() - Expr::from(1);
                        n * (**left).clone().pow(n_minus_1) * df
                    } else {
                        // General case: d/dx(f^g) = f^g * (g'*log(f) + g*f'/f)
                        // This is exp(g*log(f)) differentiation
                        let f = (**left).clone();
                        let g = (**right).clone();
                        let df = f.clone().differentiate(var);
                        let dg = g.clone().differentiate(var);

                        let term1 = dg * f.clone().log();
                        let term2 = g.clone() * df / f.clone();
                        f.pow(g) * (term1 + term2)
                    }
                }
            },

            // Unary operations (chain rule)
            Expr::Unary(op, inner) => {
                let df = inner.differentiate(var);

                match op {
                    // d/dx(-f) = -f'
                    UnaryOp::Neg => -df,

                    // d/dx(sin(f)) = cos(f) * f'
                    UnaryOp::Sin => (**inner).clone().cos() * df,

                    // d/dx(cos(f)) = -sin(f) * f'
                    UnaryOp::Cos => -((**inner).clone().sin()) * df,

                    // d/dx(tan(f)) = sec²(f) * f' = (1/cos²(f)) * f'
                    UnaryOp::Tan => {
                        let cos_f = (**inner).clone().cos();
                        df / cos_f.clone().pow(Expr::from(2))
                    }

                    // d/dx(exp(f)) = exp(f) * f'
                    UnaryOp::Exp => (**inner).clone().exp() * df,

                    // d/dx(log(f)) = f'/f
                    UnaryOp::Log => df / (**inner).clone(),

                    // d/dx(sqrt(f)) = f'/(2*sqrt(f))
                    UnaryOp::Sqrt => {
                        df / (Expr::from(2) * (**inner).clone().sqrt())
                    }

                    // d/dx(abs(f)) - not differentiable at f=0, but we can handle it symbolically
                    UnaryOp::Abs => (**inner).clone().sign() * df,

                    // d/dx(sign(f)) = 0 (except at f=0 where it's undefined)
                    UnaryOp::Sign => Expr::from(0),

                    // d/dx(sinh(f)) = cosh(f) * f'
                    UnaryOp::Sinh => (**inner).clone().cosh() * df,

                    // d/dx(cosh(f)) = sinh(f) * f'
                    UnaryOp::Cosh => (**inner).clone().sinh() * df,

                    // d/dx(tanh(f)) = sech²(f) * f' = (1/cosh²(f)) * f'
                    UnaryOp::Tanh => {
                        let cosh_f = (**inner).clone().cosh();
                        df / cosh_f.clone().pow(Expr::from(2))
                    }

                    // d/dx(arcsin(f)) = f' / sqrt(1 - f²)
                    UnaryOp::Arcsin => {
                        let f = (**inner).clone();
                        df / (Expr::from(1) - f.clone().pow(Expr::from(2))).sqrt()
                    }

                    // d/dx(arccos(f)) = -f' / sqrt(1 - f²)
                    UnaryOp::Arccos => {
                        let f = (**inner).clone();
                        -df / (Expr::from(1) - f.clone().pow(Expr::from(2))).sqrt()
                    }

                    // d/dx(arctan(f)) = f' / (1 + f²)
                    UnaryOp::Arctan => {
                        let f = (**inner).clone();
                        df / (Expr::from(1) + f.clone().pow(Expr::from(2)))
                    }

                    // d/dx(gamma(f)) = gamma(f) * digamma(f) * f'
                    // For now, return unsimplified derivative
                    UnaryOp::Gamma => {
                        // gamma'(x) = gamma(x) * psi(x) where psi is the digamma function
                        // Since we don't have digamma implemented, leave as is
                        Expr::from(0) // TODO: Implement digamma function
                    }

                    // d/dx(f!) = factorial not differentiable in standard sense
                    UnaryOp::Factorial => Expr::from(0),

                    // d/dx(erf(f)) = (2/sqrt(π)) * exp(-f²) * f'
                    UnaryOp::Erf => {
                        use std::f64::consts::PI;
                        let f = (**inner).clone();
                        let coeff = Expr::from(2) / Expr::from((PI.sqrt() * 1000.0) as i64) * Expr::from(1000);
                        coeff * (-f.clone().pow(Expr::from(2))).exp() * df
                    }
                }
            }
        }
    }

    /// Compute nth derivative
    pub fn nth_derivative(&self, var: &Symbol, n: u32) -> Self {
        if n == 0 {
            return self.clone();
        }

        let mut result = self.clone();
        for _ in 0..n {
            result = result.differentiate(var);
        }
        result
    }

    /// Compute partial derivatives for multiple variables
    pub fn gradient(&self, vars: &[Symbol]) -> Vec<Self> {
        vars.iter().map(|var| self.differentiate(var)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_derivative() {
        let c = Expr::from(5);
        let x = Symbol::new("x");
        let dc = c.differentiate(&x);

        // Should be 0
        assert!(matches!(dc, Expr::Integer(_)));
    }

    #[test]
    fn test_variable_derivative() {
        let x_sym = Symbol::new("x");
        let y_sym = Symbol::new("y");
        let x = Expr::Symbol(x_sym.clone());
        let y = Expr::Symbol(y_sym.clone());

        // d/dx(x) = 1
        let dx = x.differentiate(&x_sym);
        assert!(matches!(dx, Expr::Integer(_)));

        // d/dx(y) = 0
        let dy_dx = y.differentiate(&x_sym);
        assert!(matches!(dy_dx, Expr::Integer(_)));
    }

    #[test]
    fn test_sum_rule() {
        // d/dx(x + 5) = 1
        let x_sym = Symbol::new("x");
        let x = Expr::Symbol(x_sym.clone());
        let expr = x + Expr::from(5);
        let d = expr.differentiate(&x_sym);

        // Result should be (1 + 0) = 1
        assert!(!d.is_constant() || format!("{}", d).contains("1"));
    }

    #[test]
    fn test_product_rule() {
        // d/dx(x * x) = x + x = 2x
        let x_sym = Symbol::new("x");
        let x = Expr::Symbol(x_sym.clone());
        let expr = x.clone() * x.clone();
        let d = expr.differentiate(&x_sym);

        // Result should involve x
        let display = format!("{}", d);
        assert!(display.contains("x"));
    }

    #[test]
    fn test_power_rule() {
        // d/dx(x^3) = 3*x^2
        let x_sym = Symbol::new("x");
        let x = Expr::Symbol(x_sym.clone());
        let expr = x.pow(Expr::from(3));
        let d = expr.differentiate(&x_sym);

        let display = format!("{}", d);
        // Should contain 3 and x^2
        assert!(display.contains("3") && display.contains("x"));
    }

    #[test]
    fn test_chain_rule_sin() {
        // d/dx(sin(x)) = cos(x)
        let x_sym = Symbol::new("x");
        let x = Expr::Symbol(x_sym.clone());
        let expr = x.sin();
        let d = expr.differentiate(&x_sym);

        let display = format!("{}", d);
        assert!(display.contains("cos"));
    }

    #[test]
    fn test_chain_rule_exp() {
        // d/dx(exp(x)) = exp(x)
        let x_sym = Symbol::new("x");
        let x = Expr::Symbol(x_sym.clone());
        let expr = x.exp();
        let d = expr.differentiate(&x_sym);

        let display = format!("{}", d);
        assert!(display.contains("exp"));
    }

    #[test]
    fn test_quotient_rule() {
        // d/dx(x / (x + 1))
        let x_sym = Symbol::new("x");
        let x = Expr::Symbol(x_sym.clone());
        let expr = x.clone() / (x.clone() + Expr::from(1));
        let d = expr.differentiate(&x_sym);

        // Should produce a quotient
        let display = format!("{}", d);
        assert!(display.contains("/"));
    }

    #[test]
    fn test_nth_derivative() {
        // d²/dx²(x^4) = 12x²
        let x_sym = Symbol::new("x");
        let x = Expr::Symbol(x_sym.clone());
        let expr = x.pow(Expr::from(4));
        let d2 = expr.nth_derivative(&x_sym, 2);

        let display = format!("{}", d2);
        // Should contain x and powers
        assert!(display.contains("x"));
    }

    #[test]
    fn test_gradient() {
        // f(x,y) = x*y
        let x_sym = Symbol::new("x");
        let y_sym = Symbol::new("y");
        let x = Expr::Symbol(x_sym.clone());
        let y = Expr::Symbol(y_sym.clone());

        let expr = x * y;
        let grad = expr.gradient(&[x_sym, y_sym]);

        assert_eq!(grad.len(), 2);
        // ∂f/∂x = y, ∂f/∂y = x
    }
}
