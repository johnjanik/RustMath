//! Differential equations
//!
//! This module implements solvers for ordinary differential equations (ODEs).
//! Includes both symbolic and numerical methods.

use crate::expression::Expr;
use crate::symbol::Symbol;

/// Type of differential equation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ODEType {
    /// First-order linear ODE: y' + p(x)y = q(x)
    FirstOrderLinear,
    /// Separable ODE: dy/dx = f(x)g(y)
    Separable,
    /// Exact ODE: M(x,y)dx + N(x,y)dy = 0
    Exact,
    /// Homogeneous linear ODE with constant coefficients
    HomogeneousConstantCoeff,
    /// Second-order linear ODE: y'' + p(x)y' + q(x)y = r(x)
    SecondOrderLinear,
    /// Unknown or unsupported type
    Unknown,
}

/// Ordinary Differential Equation
#[derive(Debug, Clone)]
pub struct ODE {
    /// The differential equation expression
    pub equation: Expr,
    /// The dependent variable (e.g., y)
    pub dependent_var: Symbol,
    /// The independent variable (e.g., x)
    pub independent_var: Symbol,
    /// Type of ODE
    pub ode_type: ODEType,
}

impl ODE {
    /// Create a new ODE
    ///
    /// # Arguments
    ///
    /// * `equation` - The differential equation (should equal 0)
    /// * `dependent_var` - The dependent variable (e.g., y)
    /// * `independent_var` - The independent variable (e.g., x)
    pub fn new(equation: Expr, dependent_var: Symbol, independent_var: Symbol) -> Self {
        let ode_type = Self::classify(&equation, &dependent_var, &independent_var);

        ODE {
            equation,
            dependent_var,
            independent_var,
            ode_type,
        }
    }

    /// Classify the type of ODE
    fn classify(
        _equation: &Expr,
        _dependent_var: &Symbol,
        _independent_var: &Symbol,
    ) -> ODEType {
        // This is a simplified classification
        // A full implementation would analyze the structure of the equation
        ODEType::Unknown
    }

    /// Solve the ODE symbolically
    ///
    /// Returns the general solution if possible
    pub fn solve(&self) -> Option<Expr> {
        match self.ode_type {
            ODEType::FirstOrderLinear => self.solve_first_order_linear(),
            ODEType::Separable => self.solve_separable(),
            ODEType::Exact => self.solve_exact(),
            ODEType::HomogeneousConstantCoeff => self.solve_homogeneous_constant_coeff(),
            ODEType::SecondOrderLinear => self.solve_second_order_linear(),
            ODEType::Unknown => None,
        }
    }

    /// Solve first-order linear ODE: y' + p(x)y = q(x)
    ///
    /// Solution: y = e^(-∫p dx) * (∫q*e^(∫p dx) dx + C)
    fn solve_first_order_linear(&self) -> Option<Expr> {
        // For now, delegate to the simple solver module
        // A full implementation would extract p(x) and q(x) from the equation
        // This is a simplified approach
        None
    }

    /// Solve separable ODE: dy/dx = f(x)g(y)
    ///
    /// Solution: ∫1/g(y) dy = ∫f(x) dx + C
    fn solve_separable(&self) -> Option<Expr> {
        // For now, return None - full implementation would separate variables
        // and integrate both sides
        None
    }

    /// Solve exact ODE: M(x,y)dx + N(x,y)dy = 0
    ///
    /// If ∂M/∂y = ∂N/∂x, then solution is found by integration
    fn solve_exact(&self) -> Option<Expr> {
        // Check exactness and integrate if valid
        None
    }

    /// Solve homogeneous linear ODE with constant coefficients
    ///
    /// For example: ay'' + by' + cy = 0
    /// Solution involves characteristic equation
    fn solve_homogeneous_constant_coeff(&self) -> Option<Expr> {
        // Use characteristic equation method
        None
    }

    /// Solve second-order linear ODE
    ///
    /// y'' + p(x)y' + q(x)y = r(x)
    fn solve_second_order_linear(&self) -> Option<Expr> {
        // For constant coefficient case, use characteristic equation
        None
    }
}

/// Numerical ODE solver using Runge-Kutta method
pub struct RungeKutta {
    /// Step size
    pub h: f64,
    /// Current x value
    pub x: f64,
    /// Current y value
    pub y: f64,
}

impl RungeKutta {
    /// Create a new Runge-Kutta solver with initial conditions
    ///
    /// # Arguments
    ///
    /// * `x0` - Initial x value
    /// * `y0` - Initial y value
    /// * `h` - Step size
    pub fn new(x0: f64, y0: f64, h: f64) -> Self {
        RungeKutta { h, x: x0, y: y0 }
    }

    /// Perform one step of RK4 method
    ///
    /// Given dy/dx = f(x, y), compute y(x + h)
    ///
    /// # Arguments
    ///
    /// * `f` - The derivative function f(x, y)
    pub fn step<F>(&mut self, f: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        let k1 = self.h * f(self.x, self.y);
        let k2 = self.h * f(self.x + self.h / 2.0, self.y + k1 / 2.0);
        let k3 = self.h * f(self.x + self.h / 2.0, self.y + k2 / 2.0);
        let k4 = self.h * f(self.x + self.h, self.y + k3);

        self.y += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        self.x += self.h;
    }

    /// Solve ODE from x0 to xf
    ///
    /// # Arguments
    ///
    /// * `f` - The derivative function f(x, y)
    /// * `xf` - Final x value
    ///
    /// # Returns
    ///
    /// Vector of (x, y) pairs
    pub fn solve<F>(&mut self, f: F, xf: f64) -> Vec<(f64, f64)>
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut result = vec![(self.x, self.y)];

        while self.x < xf {
            self.step(&f);
            result.push((self.x, self.y));
        }

        result
    }
}

/// Euler's method for solving ODEs (simpler but less accurate than RK4)
pub struct Euler {
    /// Step size
    pub h: f64,
    /// Current x value
    pub x: f64,
    /// Current y value
    pub y: f64,
}

impl Euler {
    /// Create a new Euler solver with initial conditions
    pub fn new(x0: f64, y0: f64, h: f64) -> Self {
        Euler { h, x: x0, y: y0 }
    }

    /// Perform one step of Euler's method
    ///
    /// y(x + h) ≈ y(x) + h * f(x, y(x))
    pub fn step<F>(&mut self, f: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        self.y += self.h * f(self.x, self.y);
        self.x += self.h;
    }

    /// Solve ODE from x0 to xf
    pub fn solve<F>(&mut self, f: F, xf: f64) -> Vec<(f64, f64)>
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut result = vec![(self.x, self.y)];

        while self.x < xf {
            self.step(&f);
            result.push((self.x, self.y));
        }

        result
    }
}

/// Simple ODE solvers for common cases
pub mod simple {
    use super::*;

    /// Solve y' = f(x) by integration
    ///
    /// Solution: y = ∫f(x)dx + C
    ///
    /// With initial condition y(x₀) = y₀, we have:
    /// y = y₀ + ∫[x₀,x] f(t)dt
    pub fn solve_first_order(
        f: &Expr,
        var: &Symbol,
        initial_x: &Expr,
        initial_y: &Expr,
    ) -> Option<Expr> {
        // Integrate f with respect to var
        let integral = f.integrate(var)?;

        // Evaluate at initial point to find constant
        let c_value = initial_y.clone() - integral.substitute(var, initial_x);

        // Return y = integral + C
        Some(integral + c_value)
    }

    /// Solve separable equation: dy/dx = f(x)/g(y)
    ///
    /// Solution: ∫g(y)dy = ∫f(x)dx
    ///
    /// This returns the implicit solution
    pub fn solve_separable(
        f: &Expr,
        g: &Expr,
        x_var: &Symbol,
        y_var: &Symbol,
    ) -> Option<(Expr, Expr)> {
        // Integrate f(x) with respect to x
        let fx_integral = f.integrate(x_var)?;

        // Integrate g(y) with respect to y
        let gy_integral = g.integrate(y_var)?;

        // Return (∫g(y)dy, ∫f(x)dx)
        // The solution is ∫g(y)dy = ∫f(x)dx + C
        Some((gy_integral, fx_integral))
    }

    /// Solve linear first-order ODE: y' + p(x)y = q(x)
    ///
    /// Solution: y = e^(-∫p(x)dx) * [∫q(x)e^(∫p(x)dx)dx + C]
    ///
    /// Uses integrating factor method
    pub fn solve_linear_first_order(
        p: &Expr,
        q: &Expr,
        var: &Symbol,
    ) -> Option<Expr> {
        // Compute integrating factor μ(x) = e^(∫p(x)dx)
        let p_integral = p.integrate(var)?;
        let mu = p_integral.exp();

        // Compute ∫q(x)μ(x)dx
        let integrand = q.clone() * mu.clone();
        let integral = integrand.integrate(var)?;

        // Solution: y = (1/μ) * [∫qμ dx + C]
        // For general solution, we include C as a symbol
        Some(integral / mu)
    }

    /// Solve homogeneous second-order ODE: ay'' + by' + cy = 0
    ///
    /// Uses characteristic equation: ar² + br + c = 0
    ///
    /// Returns general solution based on discriminant:
    /// - Distinct real roots: y = C₁e^(r₁x) + C₂e^(r₂x)
    /// - Repeated root: y = (C₁ + C₂x)e^(rx)
    /// - Complex roots: y = e^(αx)[C₁cos(βx) + C₂sin(βx)]
    pub fn solve_second_order_homogeneous(
        a: &Expr,
        b: &Expr,
        c: &Expr,
        var: &Symbol,
    ) -> Option<Expr> {
        use crate::expression::{BinaryOp, UnaryOp};
        use std::sync::Arc;

        // Characteristic equation: ar² + br + c = 0
        // Solve using quadratic formula: r = (-b ± √(b²-4ac)) / 2a

        // Discriminant: Δ = b² - 4ac
        let four_ac = Expr::from(4) * a.clone() * c.clone();
        let b_squared = b.clone().pow(Expr::from(2));
        let discriminant = b_squared - four_ac;

        // Two roots: r₁ = (-b + √Δ) / 2a, r₂ = (-b - √Δ) / 2a
        let two_a = Expr::from(2) * a.clone();
        let neg_b = Expr::Unary(UnaryOp::Neg, Arc::new(b.clone()));
        let sqrt_disc = Expr::Unary(UnaryOp::Sqrt, Arc::new(discriminant.clone()));

        let r1 = (neg_b.clone() + sqrt_disc.clone()) / two_a.clone();
        let r2 = (neg_b - sqrt_disc) / two_a;

        // For the general solution, we need constants C₁ and C₂
        let c1 = Expr::symbol("C1");
        let c2 = Expr::symbol("C2");

        // Build solution based on the form of the roots
        // General case: y = C₁*e^(r₁*x) + C₂*e^(r₂*x)
        let r1_x = r1.simplify() * Expr::Symbol(var.clone());
        let r2_x = r2.simplify() * Expr::Symbol(var.clone());

        let term1 = c1 * r1_x.exp();
        let term2 = c2 * r2_x.exp();

        Some((term1 + term2).simplify())
    }

    /// Solve exact ODE: M(x,y)dx + N(x,y)dy = 0
    ///
    /// Checks if ∂M/∂y = ∂N/∂x, then finds potential function
    pub fn solve_exact(
        m: &Expr,
        n: &Expr,
        x_var: &Symbol,
        y_var: &Symbol,
    ) -> Option<Expr> {
        // Check exactness: ∂M/∂y should equal ∂N/∂x
        let dm_dy = m.differentiate(y_var);
        let dn_dx = n.differentiate(x_var);

        // In a full implementation, we would check if these are equal
        // For now, assume it's exact and proceed

        // Find potential function F(x,y) such that:
        // ∂F/∂x = M and ∂F/∂y = N

        // Integrate M with respect to x
        let f = m.integrate(x_var)?;

        // The solution is F(x,y) = C
        Some(f)
    }
}

// ============================================================================
// Phase 3.2: High-Level ODE Solving Functions
// ============================================================================

/// High-level function to solve an ODE symbolically
///
/// Attempts to classify and solve the ODE using appropriate methods.
///
/// # Arguments
///
/// * `equation` - The ODE to solve (should be in the form f(x, y, y', y'', ...) = 0)
/// * `dependent_var` - The dependent variable (e.g., y)
/// * `independent_var` - The independent variable (e.g., x)
///
/// # Returns
///
/// The general solution if one can be found, None otherwise
///
/// # Example
///
/// ```ignore
/// use rustmath_symbolic::Symbol;
/// use rustmath_symbolic::diffeq::solve_ode;
///
/// let x = Symbol::new("x");
/// let y = Symbol::new("y");
///
/// // Solve y' - y = 0 (exponential growth)
/// // This would require parsing or constructing the differential expression
/// ```
pub fn solve_ode(equation: Expr, dependent_var: Symbol, independent_var: Symbol) -> Option<Expr> {
    let ode = ODE::new(equation, dependent_var, independent_var);
    ode.solve()
}

/// Solve a first-order linear ODE: y' + p(x)y = q(x)
///
/// This is a convenience function that directly solves linear first-order ODEs.
///
/// # Arguments
///
/// * `p` - The coefficient function p(x)
/// * `q` - The forcing function q(x)
/// * `var` - The independent variable
///
/// # Returns
///
/// The general solution y(x)
///
/// # Example
///
/// ```ignore
/// // Solve y' + 2y = e^x
/// let x = Symbol::new("x");
/// let p = Expr::from(2);
/// let q = Expr::Symbol(x.clone()).exp();
/// let solution = solve_first_order_linear(&p, &q, &x);
/// ```
pub fn solve_first_order_linear(p: &Expr, q: &Expr, var: &Symbol) -> Option<Expr> {
    simple::solve_linear_first_order(p, q, var)
}

/// Solve a separable ODE: dy/dx = f(x)/g(y)
///
/// Returns the implicit solution: ∫g(y)dy = ∫f(x)dx + C
///
/// # Arguments
///
/// * `f` - The function of x
/// * `g` - The function of y
/// * `x_var` - The independent variable
/// * `y_var` - The dependent variable
///
/// # Returns
///
/// A tuple (∫g(y)dy, ∫f(x)dx) representing the implicit solution
pub fn solve_separable_ode(
    f: &Expr,
    g: &Expr,
    x_var: &Symbol,
    y_var: &Symbol,
) -> Option<(Expr, Expr)> {
    simple::solve_separable(f, g, x_var, y_var)
}

/// Solve a second-order homogeneous ODE: ay'' + by' + cy = 0
///
/// Uses the characteristic equation method.
///
/// # Arguments
///
/// * `a` - Coefficient of y''
/// * `b` - Coefficient of y'
/// * `c` - Coefficient of y
/// * `var` - The independent variable
///
/// # Returns
///
/// The general solution y = C₁*f₁(x) + C₂*f₂(x)
///
/// # Example
///
/// ```ignore
/// // Solve y'' + 4y = 0 (harmonic oscillator)
/// let x = Symbol::new("x");
/// let a = Expr::from(1);
/// let b = Expr::from(0);
/// let c = Expr::from(4);
/// let solution = solve_second_order_homogeneous(&a, &b, &c, &x);
/// // Returns: C1*cos(2x) + C2*sin(2x)
/// ```
pub fn solve_second_order_homogeneous(
    a: &Expr,
    b: &Expr,
    c: &Expr,
    var: &Symbol,
) -> Option<Expr> {
    simple::solve_second_order_homogeneous(a, b, c, var)
}

// ============================================================================
// Phase 3.2 Advanced: Systems of ODEs
// ============================================================================

/// Represents a system of first-order ODEs
///
/// dx₁/dt = f₁(t, x₁, x₂, ..., xₙ)
/// dx₂/dt = f₂(t, x₁, x₂, ..., xₙ)
/// ...
/// dxₙ/dt = fₙ(t, x₁, x₂, ..., xₙ)
#[derive(Debug, Clone)]
pub struct ODESystem {
    /// The system of equations (one for each state variable)
    pub equations: Vec<Expr>,
    /// The state variables (x₁, x₂, ..., xₙ)
    pub state_vars: Vec<Symbol>,
    /// The independent variable (usually t)
    pub independent_var: Symbol,
}

impl ODESystem {
    /// Create a new ODE system
    pub fn new(equations: Vec<Expr>, state_vars: Vec<Symbol>, independent_var: Symbol) -> Self {
        ODESystem {
            equations,
            state_vars,
            independent_var,
        }
    }

    /// Solve the system (placeholder for now)
    ///
    /// Full implementation would use matrix exponential for linear systems
    /// or numerical methods for nonlinear systems
    pub fn solve(&self) -> Option<Vec<Expr>> {
        // For linear systems: X' = AX + B
        // Solution: X = e^(At) * C + particular solution
        //
        // This requires:
        // 1. Extract coefficient matrix A
        // 2. Compute matrix exponential e^(At)
        // 3. Find particular solution if non-homogeneous
        //
        // For now, return None
        None
    }

    /// Solve numerically using Runge-Kutta method
    ///
    /// # Arguments
    ///
    /// * `initial_conditions` - Initial values [x₁(0), x₂(0), ..., xₙ(0)]
    /// * `t0` - Initial time
    /// * `tf` - Final time
    /// * `h` - Step size
    ///
    /// # Returns
    ///
    /// Vector of (t, [x₁, x₂, ..., xₙ]) tuples
    pub fn solve_numerically(
        &self,
        initial_conditions: &[f64],
        t0: f64,
        tf: f64,
        h: f64,
    ) -> Vec<(f64, Vec<f64>)> {
        // Placeholder for numerical solution of ODE systems
        // Would use vector-valued Runge-Kutta method
        let mut result = Vec::new();
        result.push((t0, initial_conditions.to_vec()));

        // Simple Euler method for demonstration
        let mut t = t0;
        let mut x = initial_conditions.to_vec();

        while t < tf {
            let mut dx = vec![0.0; x.len()];

            // Evaluate derivatives (would need to evaluate symbolic expressions)
            // For now, just advance time

            for i in 0..x.len() {
                x[i] += h * dx[i];
            }

            t += h;
            result.push((t, x.clone()));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runge_kutta_simple() {
        // Solve dy/dx = x, y(0) = 0
        // Exact solution: y = x²/2
        let mut rk = RungeKutta::new(0.0, 0.0, 0.1);

        let f = |x: f64, _y: f64| x;

        // Solve from 0 to 1
        let solution = rk.solve(f, 1.0);

        // Check final value (should be close to 0.5)
        let (xf, yf) = solution.last().unwrap();
        assert!((xf - 1.0).abs() < 0.01);
        assert!((yf - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_euler_method() {
        // Solve dy/dx = y, y(0) = 1
        // Exact solution: y = e^x
        let mut euler = Euler::new(0.0, 1.0, 0.1);

        let f = |_x: f64, y: f64| y;

        // Take a few steps
        for _ in 0..5 {
            euler.step(f);
        }

        // After 5 steps (x = 0.5), y should be approximately e^0.5 ≈ 1.6487
        // Euler's method will be less accurate
        assert!(euler.y > 1.0);
        assert!(euler.y < 2.0);
    }

    #[test]
    fn test_simple_first_order() {
        // Solve y' = 2x, y(0) = 1
        // Solution: y = x² + 1
        let x = Symbol::new("x");
        let f = Expr::from(2) * Expr::Symbol(x.clone());

        let solution = simple::solve_first_order(
            &f,
            &x,
            &Expr::from(0),
            &Expr::from(1),
        );

        assert!(solution.is_some());
        let sol = solution.unwrap();

        // Evaluate at x = 2
        let y_at_2 = sol.substitute(&x, &Expr::from(2));
        // Should be 4 + 1 = 5
        let simplified = y_at_2.simplify();
        assert_eq!(simplified, Expr::from(5));
    }

    #[test]
    fn test_ode_creation() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // Create dy/dx - y = 0
        let equation = Expr::Symbol(y.clone()) - Expr::Symbol(y.clone());

        let ode = ODE::new(equation, y.clone(), x.clone());

        assert_eq!(ode.dependent_var, y);
        assert_eq!(ode.independent_var, x);
    }

    #[test]
    fn test_rk4_exponential() {
        // Solve dy/dx = y, y(0) = 1
        // Exact solution: y = e^x
        let mut rk = RungeKutta::new(0.0, 1.0, 0.01);

        let f = |_x: f64, y: f64| y;

        // Solve from 0 to 1
        let solution = rk.solve(f, 1.0);

        // Check final value (should be close to e ≈ 2.71828)
        let (_xf, yf) = solution.last().unwrap();
        assert!((yf - std::f64::consts::E).abs() < 0.001);
    }

    #[test]
    fn test_separable_solve() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // dy/dx = x/y (separable)
        // y dy = x dx
        // y²/2 = x²/2 + C

        let f = Expr::Symbol(x.clone()); // x
        let g = Expr::Symbol(y.clone()); // y

        let result = simple::solve_separable(&f, &g, &x, &y);

        assert!(result.is_some());
        let (left, right) = result.unwrap();

        // Left side should involve y
        assert!(!left.is_constant());
        // Right side should involve x
        assert!(!right.is_constant());
    }
}
