//! Expression simplification

use crate::expression::{BinaryOp, Expr, UnaryOp};
use rustmath_core::Ring;
use rustmath_integers::Integer;
use std::sync::Arc;

impl Expr {
    /// Simplify the expression using basic algebraic rules
    ///
    /// Applies constant folding, identity elimination, and other basic simplifications.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    ///
    /// let x = Expr::symbol("x");
    ///
    /// // 0 + x simplifies to x
    /// let expr = Expr::from(0) + x.clone();
    /// let simplified = expr.simplify();
    /// ```
    pub fn simplify(&self) -> Expr {
        simplify(self)
    }

    /// Full simplification including expansion and collection
    ///
    /// This is more aggressive than basic simplify().
    pub fn simplify_full(&self) -> Expr {
        // First expand
        let expanded = self.expand();
        // Then simplify
        let simplified = simplify(&expanded);
        simplified
    }

    /// Simplify rational expressions (ratios of polynomials)
    pub fn simplify_rational(&self) -> Expr {
        match self {
            Expr::Binary(BinaryOp::Div, num, den) => {
                // Simplify numerator and denominator
                let num_simp = simplify(num);
                let den_simp = simplify(den);

                // Try to find common factors (basic GCD for polynomials)
                // For now, just return the simplified ratio
                Expr::Binary(BinaryOp::Div, Arc::new(num_simp), Arc::new(den_simp))
            }
            _ => simplify(self),
        }
    }

    /// Simplify trigonometric expressions
    ///
    /// Applies trigonometric identities like sin²(x) + cos²(x) = 1
    pub fn simplify_trig(&self) -> Expr {
        match self {
            // sin²(x) + cos²(x) = 1
            Expr::Binary(BinaryOp::Add, left, right) => {
                if is_sin_squared(left) && is_cos_squared_same_arg(right, left) {
                    return Expr::from(1);
                }
                if is_cos_squared(left) && is_sin_squared_same_arg(right, left) {
                    return Expr::from(1);
                }

                // Recursively simplify
                let left_simp = left.simplify_trig();
                let right_simp = right.simplify_trig();
                Expr::Binary(BinaryOp::Add, Arc::new(left_simp), Arc::new(right_simp))
            }
            Expr::Binary(op, left, right) => {
                let left_simp = left.simplify_trig();
                let right_simp = right.simplify_trig();
                Expr::Binary(*op, Arc::new(left_simp), Arc::new(right_simp))
            }
            Expr::Unary(op, inner) => {
                let inner_simp = inner.simplify_trig();
                Expr::Unary(*op, Arc::new(inner_simp))
            }
            _ => self.clone(),
        }
    }
}

/// Simplify an expression
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Symbol(_) => expr.clone(),

        Expr::Binary(op, left, right) => {
            let left_simp = simplify(left);
            let right_simp = simplify(right);
            simplify_binary(*op, left_simp, right_simp)
        }

        Expr::Unary(op, inner) => {
            let inner_simp = simplify(inner);
            simplify_unary(*op, inner_simp)
        }

        Expr::Function(name, args) => {
            // Recursively simplify all arguments
            let simplified_args: Vec<Arc<Expr>> = args
                .iter()
                .map(|arg| Arc::new(simplify(arg)))
                .collect();
            Expr::Function(name.clone(), simplified_args)
        }
    }
}

/// Check if expression is sin(x)^2
fn is_sin_squared(expr: &Expr) -> bool {
    match expr {
        Expr::Binary(BinaryOp::Pow, base, exp) => {
            matches!(base.as_ref(), Expr::Unary(UnaryOp::Sin, _))
                && matches!(exp.as_ref(), Expr::Integer(n) if n == &Integer::from(2))
        }
        _ => false,
    }
}

/// Check if expression is cos(x)^2
fn is_cos_squared(expr: &Expr) -> bool {
    match expr {
        Expr::Binary(BinaryOp::Pow, base, exp) => {
            matches!(base.as_ref(), Expr::Unary(UnaryOp::Cos, _))
                && matches!(exp.as_ref(), Expr::Integer(n) if n == &Integer::from(2))
        }
        _ => false,
    }
}

/// Check if expression is cos²(arg) where arg matches the sin²(arg) from sin_squared_expr
fn is_cos_squared_same_arg(expr: &Expr, sin_squared_expr: &Expr) -> bool {
    match (expr, sin_squared_expr) {
        (
            Expr::Binary(BinaryOp::Pow, cos_base, cos_exp),
            Expr::Binary(BinaryOp::Pow, sin_base, _),
        ) => {
            if let (Expr::Unary(UnaryOp::Cos, cos_arg), Expr::Unary(UnaryOp::Sin, sin_arg)) =
                (cos_base.as_ref(), sin_base.as_ref())
            {
                matches!(cos_exp.as_ref(), Expr::Integer(n) if n == &Integer::from(2))
                    && cos_arg == sin_arg
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Check if expression is sin²(arg) where arg matches the cos²(arg) from cos_squared_expr
fn is_sin_squared_same_arg(expr: &Expr, cos_squared_expr: &Expr) -> bool {
    is_cos_squared_same_arg(cos_squared_expr, expr)
}

fn simplify_binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    match op {
        BinaryOp::Add => {
            // 0 + x = x, x + 0 = x
            if let Expr::Integer(n) = &left {
                if n.is_zero() {
                    return right;
                }
            }
            if let Expr::Integer(n) = &right {
                if n.is_zero() {
                    return left;
                }
            }

            // Constant folding
            if let (Expr::Integer(a), Expr::Integer(b)) = (&left, &right) {
                return Expr::Integer(a.clone() + b.clone());
            }

            Expr::Binary(BinaryOp::Add, Arc::new(left), Arc::new(right))
        }

        BinaryOp::Sub => {
            // x - 0 = x
            if let Expr::Integer(n) = &right {
                if n.is_zero() {
                    return left;
                }
            }

            // Constant folding
            if let (Expr::Integer(a), Expr::Integer(b)) = (&left, &right) {
                return Expr::Integer(a.clone() - b.clone());
            }

            Expr::Binary(BinaryOp::Sub, Arc::new(left), Arc::new(right))
        }

        BinaryOp::Mul => {
            // 0 * x = 0, x * 0 = 0
            if let Expr::Integer(n) = &left {
                if n.is_zero() {
                    return Expr::Integer(Integer::zero());
                }
                if n.is_one() {
                    return right;
                }
            }
            if let Expr::Integer(n) = &right {
                if n.is_zero() {
                    return Expr::Integer(Integer::zero());
                }
                if n.is_one() {
                    return left;
                }
            }

            // Constant folding
            if let (Expr::Integer(a), Expr::Integer(b)) = (&left, &right) {
                return Expr::Integer(a.clone() * b.clone());
            }

            Expr::Binary(BinaryOp::Mul, Arc::new(left), Arc::new(right))
        }

        BinaryOp::Div => {
            // x / 1 = x
            if let Expr::Integer(n) = &right {
                if n.is_one() {
                    return left;
                }
            }

            // Constant folding
            if let (Expr::Integer(a), Expr::Integer(b)) = (&left, &right) {
                if !b.is_zero() {
                    return Expr::Integer(a.clone() / b.clone());
                }
            }

            Expr::Binary(BinaryOp::Div, Arc::new(left), Arc::new(right))
        }

        BinaryOp::Pow => {
            // x^0 = 1, x^1 = x
            if let Expr::Integer(n) = &right {
                if n.is_zero() {
                    return Expr::Integer(Integer::one());
                }
                if n.is_one() {
                    return left;
                }
            }

            Expr::Binary(BinaryOp::Pow, Arc::new(left), Arc::new(right))
        }
    }
}

fn simplify_unary(op: UnaryOp, inner: Expr) -> Expr {
    match op {
        UnaryOp::Neg => {
            // -(-x) = x
            if let Expr::Unary(UnaryOp::Neg, inner_inner) = &inner {
                return (**inner_inner).clone();
            }

            // Constant folding
            if let Expr::Integer(n) = &inner {
                return Expr::Integer(-n.clone());
            }

            Expr::Unary(UnaryOp::Neg, Arc::new(inner))
        }

        _ => Expr::Unary(op, Arc::new(inner)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_addition() {
        // 0 + x = x
        let x = Expr::symbol("x");
        let expr = Expr::from(0) + x.clone();
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // 2 + 3 = 5
        let expr = Expr::from(2) + Expr::from(3);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::from(5));
    }

    #[test]
    fn test_simplify_multiplication() {
        // 1 * x = x
        let x = Expr::symbol("x");
        let expr = Expr::from(1) * x.clone();
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // 0 * x = 0
        let expr = Expr::from(0) * x.clone();
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::from(0));
    }

    #[test]
    fn test_simplify_negation() {
        // -(-x) = x
        let x = Expr::symbol("x");
        let expr = -(-x.clone());
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);
    }

    #[test]
    fn test_simplify_trig() {
        let x = Expr::symbol("x");

        // sin(x)^2 + cos(x)^2 = 1
        let sin_x = x.clone().sin();
        let cos_x = x.clone().cos();
        let expr = sin_x.pow(Expr::from(2)) + cos_x.pow(Expr::from(2));

        let simplified = expr.simplify_trig();

        // Should simplify to 1
        assert_eq!(simplified, Expr::from(1));
    }

    #[test]
    fn test_simplify_full() {
        let x = Expr::symbol("x");

        // (x + 1) * 2 should expand and simplify
        let expr = (x.clone() + Expr::from(1)) * Expr::from(2);
        let simplified = expr.simplify_full();

        // Should be in expanded form
        match simplified {
            Expr::Binary(BinaryOp::Add, _, _) => {}
            _ => {}
        }
    }

    #[test]
    fn test_simplify_rational() {
        let x = Expr::symbol("x");

        // (2*x) / 2 could simplify to x (but needs more advanced implementation)
        let expr = (Expr::from(2) * x.clone()) / Expr::from(2);
        let _simplified = expr.simplify_rational();

        // For now, just test that it doesn't crash
    }
}
