//! Expression simplification

use crate::expression::{BinaryOp, Expr, UnaryOp};
use rustmath_core::Ring;
use rustmath_integers::Integer;
use std::sync::Arc;

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
    }
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
}
