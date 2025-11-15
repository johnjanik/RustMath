//! Object-oriented ODE solver interface
//!
//! This module provides class-based ODE solver structures that wrap
//! the functional ODE solving methods from the desolvers module.

use crate::desolvers::{
    runge_kutta_4, runge_kutta_4_system, ODESolution as FunctionalODESolution,
    ODESystemSolution as FunctionalODESystemSolution,
};

/// Object-oriented wrapper for ODE solving
///
/// This struct provides an object-oriented interface to solve ordinary
/// differential equations.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::ode::ODESolver;
///
/// let solver = ODESolver::new(|t, y| -y, 0.0, 1.0);
/// let solution = solver.solve_rk4(2.0, 100);
/// ```
pub struct ODESolver<F>
where
    F: Fn(f64, f64) -> f64,
{
    /// The ODE function f(t, y) where dy/dt = f(t, y)
    f: F,
    /// Initial time
    t0: f64,
    /// Initial value
    y0: f64,
}

impl<F> ODESolver<F>
where
    F: Fn(f64, f64) -> f64,
{
    /// Create a new ODE solver
    ///
    /// # Arguments
    ///
    /// * `f` - The right-hand side function f(t, y) where dy/dt = f(t, y)
    /// * `t0` - Initial time
    /// * `y0` - Initial value
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_calculus::ode::ODESolver;
    ///
    /// // Solve dy/dt = -y with y(0) = 1
    /// let solver = ODESolver::new(|t, y| -y, 0.0, 1.0);
    /// ```
    pub fn new(f: F, t0: f64, y0: f64) -> Self {
        ODESolver { f, t0, y0 }
    }

    /// Solve the ODE using 4th-order Runge-Kutta method
    ///
    /// # Arguments
    ///
    /// * `t_end` - End time
    /// * `n_steps` - Number of steps
    ///
    /// # Returns
    ///
    /// Solution containing time and value arrays
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_calculus::ode::ODESolver;
    ///
    /// let solver = ODESolver::new(|t, y| y, 0.0, 1.0);
    /// let solution = solver.solve_rk4(1.0, 100);
    /// ```
    pub fn solve_rk4(&self, t_end: f64, n_steps: usize) -> FunctionalODESolution {
        runge_kutta_4(&self.f, self.t0, self.y0, t_end, n_steps)
    }

    /// Set new initial conditions
    ///
    /// # Arguments
    ///
    /// * `t0` - New initial time
    /// * `y0` - New initial value
    pub fn reset(&mut self, t0: f64, y0: f64) {
        self.t0 = t0;
        self.y0 = y0;
    }
}

/// Object-oriented wrapper for systems of ODEs
///
/// This struct provides an object-oriented interface to solve systems
/// of ordinary differential equations.
///
/// # Examples
///
/// ```
/// use rustmath_calculus::ode::ODESystem;
///
/// // Solve harmonic oscillator: y'' + y = 0
/// // As a system: y1' = y2, y2' = -y1
/// let f = |t: f64, y: &[f64]| vec![y[1], -y[0]];
/// let solver = ODESystem::new(f, 0.0, vec![1.0, 0.0]);
/// let solution = solver.solve_rk4(10.0, 1000);
/// ```
pub struct ODESystem<F>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    /// The system function f(t, y) where dy/dt = f(t, y)
    f: F,
    /// Initial time
    t0: f64,
    /// Initial state vector
    y0: Vec<f64>,
}

impl<F> ODESystem<F>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    /// Create a new ODE system solver
    ///
    /// # Arguments
    ///
    /// * `f` - The right-hand side function f(t, y) where dy/dt = f(t, y)
    /// * `t0` - Initial time
    /// * `y0` - Initial state vector
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_calculus::ode::ODESystem;
    ///
    /// // Solve harmonic oscillator
    /// let f = |t: f64, y: &[f64]| vec![y[1], -y[0]];
    /// let solver = ODESystem::new(f, 0.0, vec![1.0, 0.0]);
    /// ```
    pub fn new(f: F, t0: f64, y0: Vec<f64>) -> Self {
        ODESystem { f, t0, y0 }
    }

    /// Solve the system using 4th-order Runge-Kutta method
    ///
    /// # Arguments
    ///
    /// * `t_end` - End time
    /// * `n_steps` - Number of steps
    ///
    /// # Returns
    ///
    /// Solution containing time and state vector arrays
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_calculus::ode::ODESystem;
    ///
    /// let f = |t: f64, y: &[f64]| vec![y[1], -y[0]];
    /// let solver = ODESystem::new(f, 0.0, vec![1.0, 0.0]);
    /// let solution = solver.solve_rk4(10.0, 1000);
    /// ```
    pub fn solve_rk4(&self, t_end: f64, n_steps: usize) -> FunctionalODESystemSolution {
        runge_kutta_4_system(&self.f, self.t0, &self.y0, t_end, n_steps)
    }

    /// Set new initial conditions
    ///
    /// # Arguments
    ///
    /// * `t0` - New initial time
    /// * `y0` - New initial state vector
    pub fn reset(&mut self, t0: f64, y0: Vec<f64>) {
        self.t0 = t0;
        self.y0 = y0;
    }

    /// Get the dimension of the system
    pub fn dimension(&self) -> usize {
        self.y0.len()
    }
}

/// Python-style function wrapper (placeholder)
///
/// This is a placeholder for wrapping Python functions for ODE solving.
/// In a full implementation with Python bindings, this would allow passing
/// Python callables to the ODE solver.
pub struct PyFunctionWrapper {
    // Placeholder - would contain Python function reference in real implementation
    _phantom: std::marker::PhantomData<()>,
}

impl PyFunctionWrapper {
    /// Create a new Python function wrapper (placeholder)
    pub fn new() -> Self {
        PyFunctionWrapper {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Default for PyFunctionWrapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ode_solver_creation() {
        let solver = ODESolver::new(|_t, y| -y, 0.0, 1.0);
        let solution = solver.solve_rk4(1.0, 100);
        assert!(solution.t.len() > 0);
        assert_eq!(solution.t.len(), solution.y.len());
    }

    #[test]
    fn test_ode_solver_exponential_decay() {
        // dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
        let solver = ODESolver::new(|_t, y| -y, 0.0, 1.0);
        let solution = solver.solve_rk4(1.0, 1000);

        let y_final = solution.y[solution.y.len() - 1];
        let expected = (-1.0_f64).exp();
        assert!((y_final - expected).abs() < 0.001);
    }

    #[test]
    fn test_ode_solver_reset() {
        let mut solver = ODESolver::new(|_t, y| y, 0.0, 1.0);
        solver.reset(1.0, 2.0);
        // After reset, solve from new initial conditions
        let solution = solver.solve_rk4(2.0, 100);
        assert_eq!(solution.t[0], 1.0);
    }

    #[test]
    fn test_ode_system_creation() {
        let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let solver = ODESystem::new(f, 0.0, vec![1.0, 0.0]);
        assert_eq!(solver.dimension(), 2);
    }

    #[test]
    fn test_ode_system_harmonic_oscillator() {
        // y'' + y = 0, or as a system:
        // y1' = y2, y2' = -y1
        // Initial: y1(0) = 1, y2(0) = 0
        // Solution: y1(t) = cos(t), y2(t) = -sin(t)
        let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let solver = ODESystem::new(f, 0.0, vec![1.0, 0.0]);

        let solution = solver.solve_rk4(2.0 * std::f64::consts::PI, 1000);

        let y_final = &solution.y[solution.y.len() - 1];
        // After one period, should return to initial conditions
        assert!((y_final[0] - 1.0).abs() < 0.01);
        assert!(y_final[1].abs() < 0.01);
    }

    #[test]
    fn test_ode_system_reset() {
        let f = |_t: f64, y: &[f64]| vec![y[0], y[1]];
        let mut solver = ODESystem::new(f, 0.0, vec![1.0, 2.0]);
        solver.reset(1.0, vec![3.0, 4.0]);
        assert_eq!(solver.dimension(), 2);
    }

    #[test]
    fn test_py_function_wrapper() {
        let _wrapper = PyFunctionWrapper::new();
        let _wrapper2 = PyFunctionWrapper::default();
        // Just test that it compiles
    }
}
