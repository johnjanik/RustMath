//! RustMath Symbolic - Symbolic computation engine
//!
//! This crate provides symbolic expression manipulation, simplification,
//! and evaluation.

pub mod expression;
pub mod simplify;
pub mod symbol;

pub use expression::{BinaryOp, Expr, UnaryOp};
pub use symbol::Symbol;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_expression() {
        // x + 1
        let x = Expr::symbol("x");
        let expr = x + Expr::from(1);

        assert!(!expr.is_constant());
    }
}
