//! RustMath Calculus - Symbolic calculus operations
//!
//! This crate provides differentiation, integration, limits, series expansions,
//! interpolation, and differential equation solvers.

pub mod desolvers;
pub mod differentiation;
pub mod expr;
pub mod functional;
pub mod integration;
pub mod interpolation;
pub mod interpolators;
pub mod limits;
pub mod ode;
pub mod riemann;
pub mod taylor;
pub mod var;

// Re-export commonly used functions
pub use differentiation::differentiate;
pub use desolvers::{
    desolve_rk4, desolve_system_rk4, eulers_method, rk45_adaptive, runge_kutta_4,
    runge_kutta_4_system, ODESolution, ODESystemSolution,
};
pub use expr::{is_constant, is_polynomial, polynomial_degree, symbolic_expression, variables};
pub use functional::{expand, simplify};
pub use integration::{integrate, nintegrate, numerical_integrate_simpson};
pub use interpolation::{spline, CubicSpline};
pub use interpolators::{complex_cubic_spline, polygon_spline, CCSpline, PSpline};
pub use limits::{lim, limit, substitute, LimitDirection};
pub use ode::{ODESolver, ODESystem};
pub use riemann::{
    analytic_boundary, analytic_interior, cauchy_kernel, complex_to_rgb, complex_to_spiderweb,
    get_derivatives, RiemannMap, RGB,
};
pub use taylor::{laurent, maclaurin, series_coefficients, taylor};
pub use var::{clear_all_vars, clear_vars, function, get_vars, var, var_from_string, vars};

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_symbolic::Expr;

    #[test]
    fn basic_derivative() {
        // d/dx(x) = 1
        let x = Expr::symbol("x");
        let _deriv = differentiate(&x, "x");
        // Test passes if it compiles - full symbolic diff testing needs more work
    }
}
