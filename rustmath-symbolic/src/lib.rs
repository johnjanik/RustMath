//! RustMath Symbolic - Symbolic computation engine
//!
//! This crate provides symbolic expression manipulation, simplification,
//! and evaluation.

pub mod assumptions;
pub mod differentiate;
pub mod diffeq;
pub mod expand;
pub mod expression;
pub mod integrate;
pub mod limits;
pub mod numerical;
pub mod polynomial;
pub mod series;
pub mod simplify;
pub mod substitute;
pub mod symbol;

pub use assumptions::{assume, forget, forget_all, get_assumptions, has_property, Property};
pub use diffeq::{Euler, RungeKutta, ODE, ODEType};
pub use expression::{BinaryOp, Expr, UnaryOp};
pub use limits::{Direction, LimitResult};
pub use numerical::IntegrationResult;
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

    // Integration tests
    #[test]
    fn test_calculus_integration_basic() {
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        // ∫ x dx = x²/2
        let expr = Expr::Symbol(x.clone());
        let result = expr.integrate(&x).unwrap();
        let expected = Expr::Symbol(x.clone()).pow(Expr::from(2)) / Expr::from(2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_calculus_integration_trig() {
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        // ∫ cos(x) dx = sin(x)
        let expr = Expr::Symbol(x.clone()).cos();
        let result = expr.integrate(&x).unwrap();
        assert_eq!(result, Expr::Symbol(x.clone()).sin());
    }

    // Limits tests
    #[test]
    fn test_calculus_limits_basic() {
        use crate::limits::{Direction, LimitResult};
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        // lim(x→2) (x² + 1) = 5
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2)) + Expr::from(1);
        let result = expr.limit(&x, &Expr::from(2), Direction::Both);
        assert_eq!(result, LimitResult::Finite(Expr::from(5)));
    }

    #[test]
    fn test_calculus_limits_lhopital() {
        use crate::limits::{Direction, LimitResult};
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        // lim(x→0) x/x = 1 (using L'Hôpital's rule)
        let expr = Expr::Symbol(x.clone()) / Expr::Symbol(x.clone());
        let result = expr.limit(&x, &Expr::from(0), Direction::Both);
        assert_eq!(result, LimitResult::Finite(Expr::from(1)));
    }

    // Series expansion tests
    #[test]
    fn test_calculus_taylor_series() {
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        // Taylor expansion of exp(x) around 0
        let expr = Expr::Symbol(x.clone()).exp();
        let taylor = expr.maclaurin(&x, 3);
        // Should not be constant
        assert!(!taylor.is_constant());
    }

    #[test]
    fn test_calculus_series_coefficients() {
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        // For f(x) = x², coefficients should be [0, 0, 1, 0, ...]
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let coeffs = expr.series_coefficients(&x, &Expr::from(0), 3);
        assert_eq!(coeffs.len(), 4);
        assert_eq!(coeffs[0], Expr::from(0));
        assert_eq!(coeffs[1], Expr::from(0));
    }

    // Differential equations tests
    #[test]
    fn test_calculus_ode_runge_kutta() {
        use crate::diffeq::RungeKutta;

        // Solve dy/dx = x, y(0) = 0
        // Exact solution: y = x²/2
        let mut rk = RungeKutta::new(0.0, 0.0, 0.1);
        let f = |x: f64, _y: f64| x;
        let solution = rk.solve(f, 1.0);

        let (_xf, yf) = solution.last().unwrap();
        assert!((yf - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_calculus_ode_euler() {
        use crate::diffeq::Euler;

        // Solve dy/dx = y, y(0) = 1
        let mut euler = Euler::new(0.0, 1.0, 0.1);
        let f = |_x: f64, y: f64| y;

        for _ in 0..5 {
            euler.step(f);
        }

        assert!(euler.y > 1.0);
    }

    #[test]
    fn test_calculus_complete_workflow() {
        use crate::limits::{Direction, LimitResult};
        use crate::symbol::Symbol;

        let x = Symbol::new("x");

        // Define f(x) = x²
        let f = Expr::Symbol(x.clone()).pow(Expr::from(2));

        // Differentiate: f'(x) = 2x
        let f_prime = f.differentiate(&x);

        // Integrate: ∫f'(x)dx should give us back x² (plus constant)
        let integrated = f_prime.integrate(&x);
        assert!(integrated.is_some());

        // Evaluate limit: lim(x→3) f(x) = 9
        let limit = f.limit(&x, &Expr::from(3), Direction::Both);
        assert_eq!(limit, LimitResult::Finite(Expr::from(9)));

        // Taylor series of f around x=0
        let taylor = f.taylor(&x, &Expr::from(0), 5);
        assert!(!taylor.is_constant());
    }
}
