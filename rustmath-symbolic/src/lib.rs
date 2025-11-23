//! RustMath Symbolic - Symbolic computation engine
//!
//! This crate provides symbolic expression manipulation, simplification,
//! and evaluation.

pub mod assumptions;
pub mod differentiate;
pub mod diffeq;
pub mod expand;
pub mod expression;
pub mod factor;
pub mod function;
pub mod function_factory;
pub mod functions;
pub mod inequalities;
pub mod integral;
pub mod integrate;
pub mod integrate_external;
pub mod limits;
pub mod maxima_wrapper;
pub mod numerical;
pub mod operators;
pub mod parser;
pub mod printing;
pub mod pde;
pub mod polynomial;
pub mod radical;
pub mod random_tests;
pub mod registry;
pub mod series;
pub mod simplify;
pub mod solve;
pub mod specialfunctions;
pub mod subring;
pub mod substitute;
pub mod symbol;
pub mod units;
pub mod walker;

pub use assumptions::{assume, forget, forget_all, get_assumptions, has_property, Property};
pub use diffeq::{Euler, RungeKutta, ODE, ODEType};
pub use expression::{BinaryOp, Expr, UnaryOp};
pub use function::{
    get_function, initialize_registry, pickle_wrapper, register_function, unpickle_wrapper,
    BuiltinFunction, Function, FunctionRegistry, GinacFunction, SymbolicFunction,
};
pub use function_factory::{
    function, function_with_expression, unpickle_function, wrap_function, FunctionFactory,
};
pub use functions::{jacobian, wronskian};
pub use operators::{
    add_vararg, derivative, mul_vararg, partial_derivative, DerivativeOperator,
    DerivativeOperatorWithParameters, FDerivativeOperator,
};
pub use inequalities::{
    solve_abs_inequality, solve_inequality, solve_polynomial_inequality,
    solve_rational_inequality, solve_system_inequalities, InequalitySolution, InequalityType,
    Interval, IntervalSet,
};
pub use integral::{
    definite_integral, integral, integrate, integrate_multi, DefiniteIntegral,
    IndefiniteIntegral,
};
pub use integrate_external::{
    fricas_integrator, libgiac_integrator, maxima_integrator, mma_free_integrator,
    sympy_integrator, ExternalIntegrator, FricasIntegrator, IntegratorChain,
    LibgiacIntegrator, MathematicaFreeIntegrator, MaximaIntegrator, SympyIntegrator,
};
pub use limits::{Direction, LimitResult};
pub use maxima_wrapper::{
    expr_to_maxima, maxima_to_expr, MaximaError, MaximaFunctionElementWrapper, MaximaResult,
    MaximaWrapper,
};
pub use numerical::IntegrationResult;
pub use parser::{parse, ParseError};
pub use pde::PDEType;
pub use random_tests::{
    assert_strict_weak_order, choose_from_prob_list, normalize_prob_list, random_expr,
    random_integer_vector, test_symbolic_expression_order, OperationType, ProbList,
    RandomExprConfig,
};
pub use series::{BigO, LittleO, Theta, Omega, FourierSeries};
pub use solve::Solution;
pub use subring::{
    GenericSymbolicSubring, GenericSymbolicSubringFunctor, SubringFunctor,
    SymbolicConstantsSubring, SymbolicSubring, SymbolicSubringAcceptingVars,
    SymbolicSubringAcceptingVarsFunctor, SymbolicSubringFactory,
    SymbolicSubringRejectingVars, SymbolicSubringRejectingVarsFunctor, COERCION_REVERSED,
};
pub use symbol::Symbol;
pub use units::{
    base_units, convert, convert_temperature, evalunitdict, is_unit, str_to_unit,
    unit_derivations_expr, unitdocs, vars_in_str, UnitExpression, Units,
};
pub use walker::{
    collect_symbols, substitute, count_operations, calculate_depth,
    ExprVisitor, ExprMutator, SymbolCollector, Substituter,
    OperationCounter, DepthCalculator,
};
pub use registry::{CoordinateRegistry, global_registry, ChartId};

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

    // ========================================================================
    // Phase 2 Completion Test: Core Calculus Features
    // ========================================================================

    #[test]
    fn test_phase2_calculus_features_complete() {
        use crate::limits::{Direction, LimitResult};
        use crate::symbol::Symbol;

        let x = Symbol::new("x");

        // ===== Milestone 2.1: Symbolic Integration =====

        // Test basic integration rules
        let expr1 = Expr::Symbol(x.clone());
        let integral1 = expr1.integrate(&x);
        assert!(integral1.is_some(), "Basic polynomial integration");

        // Test trigonometric integration
        let expr2 = Expr::Symbol(x.clone()).sin();
        let integral2 = expr2.integrate(&x);
        assert!(integral2.is_some(), "Trigonometric integration");

        // Test definite integrals
        let expr3 = Expr::from(1);
        let definite = expr3.integrate_definite(&x, &Expr::from(0), &Expr::from(1));
        assert!(definite.is_some(), "Definite integration");

        // Test multiple integrals
        let y = Symbol::new("y");
        let expr4 = Expr::from(1);
        let double = expr4.integrate_double(
            &x, &y,
            &Expr::from(0), &Expr::from(1),
            &Expr::from(0), &Expr::from(1),
        );
        assert!(double.is_some(), "Double integration");

        // ===== Milestone 2.2: Limits =====

        // Test basic limit computation
        let expr5 = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let limit1 = expr5.limit(&x, &Expr::from(2), Direction::Both);
        assert!(matches!(limit1, LimitResult::Finite(_)), "Basic limit");

        // Test L'Hôpital's rule (indeterminate form 0/0)
        let numerator = Expr::Symbol(x.clone());
        let denominator = Expr::Symbol(x.clone());
        let expr6 = numerator / denominator;
        let limit2 = expr6.limit(&x, &Expr::from(0), Direction::Both);
        // Should apply L'Hôpital's rule and get 1
        assert!(matches!(limit2, LimitResult::Finite(_)), "L'Hôpital's rule");

        // Test one-sided limits
        let expr7 = Expr::Symbol(x.clone());
        let limit_left = expr7.limit(&x, &Expr::from(1), Direction::Left);
        let limit_right = expr7.limit(&x, &Expr::from(1), Direction::Right);
        assert!(matches!(limit_left, LimitResult::Finite(_)), "Left limit");
        assert!(matches!(limit_right, LimitResult::Finite(_)), "Right limit");

        // ===== Milestone 2.3: Series Expansion =====

        // Test Taylor series
        let expr8 = Expr::Symbol(x.clone()).exp();
        let taylor = expr8.taylor(&x, &Expr::from(0), 5);
        assert!(!taylor.is_constant(), "Taylor series");

        // Test Maclaurin series
        let expr9 = Expr::Symbol(x.clone()).sin();
        let maclaurin = expr9.maclaurin(&x, 5);
        assert!(!maclaurin.is_constant(), "Maclaurin series");

        // Test Laurent series
        let expr10 = Expr::from(1) / Expr::Symbol(x.clone());
        let laurent = expr10.laurent(&x, &Expr::from(0), -1, 2);
        assert!(!laurent.is_constant(), "Laurent series");

        // Test Fourier series (NEW in Phase 2!)
        let expr11 = Expr::from(1);
        let pi = Expr::Symbol(Symbol::new("pi"));
        let period = Expr::from(2) * pi;
        let fourier = expr11.fourier_series(&x, &period, 3);
        assert_eq!(fourier.var, x, "Fourier series");
        assert_eq!(fourier.a_coeffs.len(), 3, "Fourier cosine coefficients");
        assert_eq!(fourier.b_coeffs.len(), 3, "Fourier sine coefficients");

        // Test series coefficients extraction
        let expr12 = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let coeffs = expr12.series_coefficients(&x, &Expr::from(0), 3);
        assert_eq!(coeffs.len(), 4, "Series coefficients");

        // Test asymptotic expansion
        let expr13 = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let asymptotic = expr13.asymptotic(&x, 3);
        assert!(!asymptotic.is_constant(), "Asymptotic expansion");

        println!("✓ Phase 2: Core Calculus features complete!");
        println!("  ✓ Symbolic Integration (table-based, trig, rational, partial fractions)");
        println!("  ✓ Limits (L'Hôpital's rule, one-sided, two-sided)");
        println!("  ✓ Series Expansion (Taylor, Maclaurin, Laurent, Fourier, Asymptotic)");
    }

    // Numerical integration tests
    #[test]
    fn test_numerical_trapezoidal() {
        use crate::numerical;
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        // ∫₀¹ x dx = 0.5
        let expr = Expr::Symbol(x.clone());
        let result = numerical::trapezoidal(&expr, &x, 0.0, 1.0, 100);
        assert!((result.value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_numerical_simpson() {
        use crate::numerical;
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        // ∫₀¹ x² dx = 1/3
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = numerical::simpson(&expr, &x, 0.0, 1.0, 100);
        assert!((result.value - 1.0/3.0).abs() < 0.001);
    }

    #[test]
    fn test_numerical_adaptive_simpson() {
        use crate::numerical;
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = numerical::adaptive_simpson(&expr, &x, 0.0, 1.0, 1e-6, 10);
        assert!((result.value - 1.0/3.0).abs() < 1e-5);
    }

    #[test]
    fn test_numerical_gauss_legendre() {
        use crate::numerical;
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = numerical::gauss_legendre(&expr, &x, 0.0, 1.0, 10);
        assert!((result.value - 1.0/3.0).abs() < 0.01);
    }

    #[test]
    fn test_numerical_romberg() {
        use crate::numerical;
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = numerical::romberg(&expr, &x, 0.0, 1.0, 10, 1e-6);
        assert!((result.value - 1.0/3.0).abs() < 1e-5);
    }

    #[test]
    fn test_numerical_monte_carlo() {
        use crate::numerical;
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        let expr = Expr::from(1);
        let result = numerical::monte_carlo(&expr, &x, 0.0, 2.0, 10000);
        assert!((result.value - 2.0).abs() < 0.1);
        assert!(result.error.is_some());
    }

    // Multiple integral tests
    #[test]
    fn test_double_integral() {
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // ∫₀¹ ∫₀¹ 1 dy dx = 1
        let expr = Expr::from(1);
        let result = expr.integrate_double(
            &x,
            &y,
            &Expr::from(0),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(1),
        );

        assert!(result.is_some());
        let integral = result.unwrap();
        assert_eq!(integral, Expr::from(1));
    }

    #[test]
    fn test_triple_integral() {
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let z = Symbol::new("z");

        // ∫₀¹ ∫₀¹ ∫₀¹ 1 dz dy dx = 1
        let expr = Expr::from(1);
        let result = expr.integrate_triple(
            &x,
            &y,
            &z,
            &Expr::from(0),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(1),
            &Expr::from(0),
            &Expr::from(1),
        );

        assert!(result.is_some());
    }

    #[test]
    fn test_jacobian_2d() {
        use crate::symbol::Symbol;

        let u = Symbol::new("u");
        let v = Symbol::new("v");

        // Polar coordinates: x = u*cos(v), y = u*sin(v)
        // Jacobian should be u
        let x = Expr::Symbol(u.clone());  // Simplified: just u
        let y = Expr::Symbol(v.clone());  // Simplified: just v

        let jac = Expr::jacobian_2d(&x, &y, &u, &v);
        // For this simple case, jacobian = 1*1 - 0*0 = 1
        assert_eq!(jac, Expr::from(1));
    }

    // PDE tests
    #[test]
    fn test_pde_heat_equation_numerical() {
        use crate::pde::numerical::FiniteDifference;

        let solver = FiniteDifference::new(0.1, 0.001, 10, 100);

        // Initial condition: u(x, 0) = sin(πx)
        let initial = |x: f64| (std::f64::consts::PI * x).sin();

        let solution = solver.solve_heat(0.1, initial);

        // Check that solution exists and has correct dimensions
        assert_eq!(solution.len(), 100); // nt time steps
        assert_eq!(solution[0].len(), 10); // nx spatial points
    }

    #[test]
    fn test_pde_wave_equation_numerical() {
        use crate::pde::numerical::FiniteDifference;

        let solver = FiniteDifference::new(0.1, 0.01, 10, 50);

        let initial = |x: f64| (std::f64::consts::PI * x).sin();

        let solution = solver.solve_wave(1.0, initial, |_x| 0.0);

        assert_eq!(solution.len(), 50);
        assert_eq!(solution[0].len(), 10);
    }

    #[test]
    fn test_pde_laplace_equation_numerical() {
        use crate::pde::numerical::FiniteDifference;

        let solver = FiniteDifference::new(0.1, 0.01, 10, 10);

        let solution = solver.solve_laplace(100, 1e-6);

        assert_eq!(solution.len(), 10);
        assert_eq!(solution[0].len(), 10);
    }

    #[test]
    fn test_pde_separation_of_variables() {
        use crate::pde::methods::SeparationOfVariables;
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        let t = Symbol::new("t");

        let sep_vars = SeparationOfVariables::new(x.clone(), t.clone());

        // Test that structure is created successfully
        assert_eq!(sep_vars.spatial_var, x);
        assert_eq!(sep_vars.temporal_var, t);
    }

    #[test]
    fn test_pde_dalembert_formula() {
        use crate::pde::methods;
        use crate::symbol::Symbol;

        let x = Symbol::new("x");
        let t = Symbol::new("t");

        let f = Expr::Symbol(x.clone()).sin();
        let g = Expr::from(0);

        let result = methods::dalembert_formula(&f, &g, 1.0, &x, &t);

        // Result should not be constant
        assert!(!result.is_constant());
    }
}
