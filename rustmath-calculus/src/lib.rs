//! RustMath Calculus - Symbolic calculus operations
//!
//! This crate provides differentiation, integration, limits, series expansions,
//! interpolation, differential equation solvers, and integral transforms.

pub mod desolvers;
pub mod differentiation;
pub mod expr;
pub mod functional;
pub mod integration;
pub mod interpolation;
pub mod interpolators;
// TODO: Fix for new Expr structure
// pub mod laplace;
pub mod limits;
// TODO: Fix for new Expr structure
// pub mod maxima_compat;
// TODO: Fix for new Expr structure
// pub mod minpoly;
pub mod ode;
// TODO: Fix for new Expr structure
// pub mod pochhammer;
// TODO: Fix for new Expr structure
// pub mod product;
pub mod riemann;
pub mod sum;
pub mod taylor;
pub mod transforms;
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
// TODO: Fix for new Expr structure
// pub use laplace::{
//     dummy_inverse_laplace, dummy_laplace, inverse_laplace, laplace, LaplaceAlgorithm,
// };
pub use limits::{lim, limit, substitute, LimitDirection};
// TODO: Fix for new Expr structure
// pub use maxima_compat::{mapped_opts, maxima_options, MaximaOptions, MaximaValue};
// pub use minpoly::{minpoly, MinpolyAlgorithm, MinpolyOptions};
pub use ode::{ODESolver, ODESystem};
// TODO: Fix for new Expr structure
// pub use pochhammer::{dummy_pochhammer, pochhammer, pochhammer_eval, pochhammer_expand};
// pub use product::{expand_product, symbolic_product, ProductAlgorithm};
pub use riemann::{
    analytic_boundary, analytic_interior, cauchy_kernel, complex_to_rgb, complex_to_spiderweb,
    get_derivatives, RiemannMap, RGB,
};
pub use sum::{expand_sum, symbolic_sum, SumAlgorithm};
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
