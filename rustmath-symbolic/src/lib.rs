//! RustMath Symbolic - Symbolic computation engine
//!
//! This crate provides symbolic expression manipulation, simplification,
//! and evaluation.

pub mod assumptions;
pub mod differentiate;
pub mod expand;
pub mod expression;
pub mod polynomial;
pub mod simplify;
pub mod substitute;
pub mod symbol;

pub use assumptions::{assume, forget, forget_all, get_assumptions, has_property, Property};
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

    #[test]
    fn test_assumptions_positive() {
        // Create a symbol x and assume it's positive
        let x = Expr::symbol("x");
        let x_sym = x.as_symbol().unwrap();

        assume(x_sym, Property::Positive);

        // Check properties
        assert_eq!(x.is_positive(), Some(true));
        assert_eq!(x.is_negative(), Some(false));
        assert_eq!(x.is_real(), Some(true)); // Implied by positive

        // Clean up
        forget(x_sym);
    }

    #[test]
    fn test_assumptions_integer() {
        let n = Expr::symbol("n");
        let n_sym = n.as_symbol().unwrap();

        assume(n_sym, Property::Integer);

        assert_eq!(n.is_integer(), Some(true));
        assert_eq!(n.is_real(), Some(true)); // Implied by integer

        forget(n_sym);
    }

    #[test]
    fn test_assumptions_with_constants() {
        // Constants should have their properties determined directly
        let pos = Expr::from(5);
        let neg = Expr::from(-3);
        let zero = Expr::from(0);

        assert_eq!(pos.is_positive(), Some(true));
        assert_eq!(pos.is_negative(), Some(false));
        assert_eq!(pos.is_integer(), Some(true));
        assert_eq!(pos.is_real(), Some(true));

        assert_eq!(neg.is_positive(), Some(false));
        assert_eq!(neg.is_negative(), Some(true));
        assert_eq!(neg.is_integer(), Some(true));

        assert_eq!(zero.is_positive(), Some(false));
        assert_eq!(zero.is_negative(), Some(false));
    }

    #[test]
    fn test_assumptions_unknown() {
        // Without assumptions, symbolic expressions should return None
        let y = Expr::symbol("y");

        assert_eq!(y.is_positive(), None);
        assert_eq!(y.is_negative(), None);
        assert_eq!(y.is_real(), None);
        assert_eq!(y.is_integer(), None);
    }

    #[test]
    fn test_multiple_assumptions() {
        let x = Expr::symbol("x");
        let x_sym = x.as_symbol().unwrap();

        assume(x_sym, Property::Positive);
        assume(x_sym, Property::Integer);

        assert_eq!(x.is_positive(), Some(true));
        assert_eq!(x.is_integer(), Some(true));
        assert_eq!(x.is_real(), Some(true));

        forget(x_sym);
    }

    #[test]
    fn test_implied_properties() {
        let p = Expr::symbol("p");
        let p_sym = p.as_symbol().unwrap();

        // Assuming prime implies positive, integer, real
        assume(p_sym, Property::Prime);

        assert!(has_property(p_sym, Property::Prime));
        assert!(has_property(p_sym, Property::Positive));
        assert!(has_property(p_sym, Property::Integer));
        assert!(has_property(p_sym, Property::Real));

        forget(p_sym);
    }
}
