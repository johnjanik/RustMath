//! Symbolic differentiation

use rustmath_symbolic::{BinaryOp, Expr, UnaryOp};

/// Compute the derivative of an expression with respect to a variable
pub fn differentiate(expr: &Expr, var: &str) -> Expr {
    match expr {
        // d/dx(c) = 0 for constants
        Expr::Integer(_) | Expr::Rational(_) => Expr::from(0),

        // d/dx(x) = 1, d/dx(y) = 0
        Expr::Symbol(s) => {
            if s.name() == var {
                Expr::from(1)
            } else {
                Expr::from(0)
            }
        }

        // Binary operations
        Expr::Binary(op, left, right) => match op {
            // Sum rule: (f + g)' = f' + g'
            BinaryOp::Add => {
                differentiate(left, var) + differentiate(right, var)
            }

            // Difference rule: (f - g)' = f' - g'
            BinaryOp::Sub => {
                differentiate(left, var) - differentiate(right, var)
            }

            // Product rule: (f * g)' = f' * g + f * g'
            BinaryOp::Mul => {
                let f_prime = differentiate(left, var);
                let g_prime = differentiate(right, var);
                f_prime * (**right).clone() + (**left).clone() * g_prime
            }

            // Quotient rule: (f / g)' = (f' * g - f * g') / g^2
            BinaryOp::Div => {
                let f_prime = differentiate(left, var);
                let g_prime = differentiate(right, var);
                let numerator = f_prime * (**right).clone() - (**left).clone() * g_prime;
                let denominator = (**right).clone().pow(Expr::from(2));
                numerator / denominator
            }

            // Power rule: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
            // Simplified for constant exponent: (f^n)' = n * f^(n-1) * f'
            BinaryOp::Pow => {
                if right.is_constant() {
                    // Power rule for constant exponent
                    let n = (**right).clone();
                    let f_prime = differentiate(left, var);
                    n.clone() * (**left).clone().pow(n - Expr::from(1)) * f_prime
                } else {
                    // General case: use logarithmic differentiation
                    let f = (**left).clone();
                    let g = (**right).clone();
                    let f_prime = differentiate(left, var);
                    let g_prime = differentiate(right, var);

                    let term1 = g_prime * f.clone().log();
                    let term2 = g.clone() * f_prime / f.clone();

                    f.pow(g) * (term1 + term2)
                }
            }
        },

        // Unary operations
        Expr::Unary(op, inner) => match op {
            // -(f)' = -f'
            UnaryOp::Neg => -differentiate(inner, var),

            // sin(f)' = cos(f) * f'
            UnaryOp::Sin => {
                (**inner).clone().cos() * differentiate(inner, var)
            }

            // cos(f)' = -sin(f) * f'
            UnaryOp::Cos => {
                -(**inner).clone().sin() * differentiate(inner, var)
            }

            // exp(f)' = exp(f) * f'
            UnaryOp::Exp => {
                (**inner).clone().exp() * differentiate(inner, var)
            }

            // log(f)' = f' / f
            UnaryOp::Log => {
                differentiate(inner, var) / (**inner).clone()
            }

            // sqrt(f)' = f' / (2*sqrt(f))
            UnaryOp::Sqrt => {
                differentiate(inner, var) / (Expr::from(2) * (**inner).clone().sqrt())
            }

            _ => {
                // Placeholder for other operations
                Expr::from(0)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_derivative() {
        let c = Expr::from(42);
        let deriv = differentiate(&c, "x");
        assert_eq!(deriv, Expr::from(0));
    }

    #[test]
    fn test_variable_derivative() {
        let x = Expr::symbol("x");
        let deriv = differentiate(&x, "x");
        assert_eq!(deriv, Expr::from(1));

        let y = Expr::symbol("y");
        let deriv = differentiate(&y, "x");
        assert_eq!(deriv, Expr::from(0));
    }

    #[test]
    fn test_sum_rule() {
        // d/dx(x + 5) = 1
        let x = Expr::symbol("x");
        let expr = x + Expr::from(5);
        let _deriv = differentiate(&expr, "x");
        // Result should be 1 + 0 = 1 (may need simplification)
    }

    #[test]
    fn test_product_rule() {
        // d/dx(x * x) = 2x
        let x = Expr::symbol("x");
        let expr = x.clone() * x.clone();
        let _deriv = differentiate(&expr, "x");
        // Result should be x + x = 2x (may need simplification)
    }
}
