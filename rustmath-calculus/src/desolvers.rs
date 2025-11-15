//! Differential equation solvers
//!
//! This module provides numerical methods for solving ordinary differential
//! equations (ODEs), including Euler's method, Runge-Kutta methods, and
//! adaptive stepsize algorithms.

use std::f64;

/// Result type for ODE solutions
///
/// Contains arrays of t values and corresponding y values
#[derive(Debug, Clone)]
pub struct ODESolution {
    /// Time/independent variable values
    pub t: Vec<f64>,
    /// Solution values
    pub y: Vec<f64>,
}

impl ODESolution {
    /// Create a new ODE solution
    pub fn new(t: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(t.len(), y.len(), "t and y must have same length");
        ODESolution { t, y }
    }

    /// Interpolate the solution at a given time
    pub fn eval(&self, t: f64) -> f64 {
        use crate::interpolation::piecewise_linear;
        piecewise_linear(&self.t, &self.y, t)
    }
}

/// Result type for systems of ODEs
#[derive(Debug, Clone)]
pub struct ODESystemSolution {
    /// Time values
    pub t: Vec<f64>,
    /// Solution vectors (each inner vector is the state at one time)
    pub y: Vec<Vec<f64>>,
}

impl ODESystemSolution {
    /// Create a new system solution
    pub fn new(t: Vec<f64>, y: Vec<Vec<f64>>) -> Self {
        assert_eq!(t.len(), y.len(), "t and y must have same length");
        ODESystemSolution { t, y }
    }
}

/// Solve ODE using Euler's method
///
/// Solves dy/dt = f(t, y) with initial condition y(t0) = y0
///
/// # Arguments
///
/// * `f` - Right-hand side function f(t, y)
/// * `t0` - Initial time
/// * `y0` - Initial value
/// * `t_end` - End time
/// * `n_steps` - Number of steps
///
/// # Returns
///
/// ODESolution containing the numerical solution
///
/// # Examples
///
/// ```
/// use rustmath_calculus::desolvers::eulers_method;
///
/// // Solve dy/dt = y with y(0) = 1
/// // Exact solution: y(t) = e^t
/// let solution = eulers_method(&|t, y| y, 0.0, 1.0, 1.0, 100);
/// ```
pub fn eulers_method<F>(f: &F, t0: f64, y0: f64, t_end: f64, n_steps: usize) -> ODESolution
where
    F: Fn(f64, f64) -> f64,
{
    let h = (t_end - t0) / n_steps as f64;
    let mut t = vec![t0];
    let mut y = vec![y0];

    for i in 0..n_steps {
        let t_curr = t[i];
        let y_curr = y[i];
        let y_next = y_curr + h * f(t_curr, y_curr);
        t.push(t_curr + h);
        y.push(y_next);
    }

    ODESolution::new(t, y)
}

/// Solve ODE using the 4th-order Runge-Kutta method (RK4)
///
/// This is more accurate than Euler's method.
///
/// # Arguments
///
/// * `f` - Right-hand side function f(t, y)
/// * `t0` - Initial time
/// * `y0` - Initial value
/// * `t_end` - End time
/// * `n_steps` - Number of steps
///
/// # Returns
///
/// ODESolution containing the numerical solution
///
/// # Examples
///
/// ```
/// use rustmath_calculus::desolvers::runge_kutta_4;
///
/// // Solve dy/dt = -y with y(0) = 1
/// // Exact solution: y(t) = e^(-t)
/// let solution = runge_kutta_4(&|t, y| -y, 0.0, 1.0, 2.0, 100);
/// ```
pub fn runge_kutta_4<F>(f: &F, t0: f64, y0: f64, t_end: f64, n_steps: usize) -> ODESolution
where
    F: Fn(f64, f64) -> f64,
{
    let h = (t_end - t0) / n_steps as f64;
    let mut t = vec![t0];
    let mut y = vec![y0];

    for i in 0..n_steps {
        let t_curr = t[i];
        let y_curr = y[i];

        let k1 = h * f(t_curr, y_curr);
        let k2 = h * f(t_curr + h / 2.0, y_curr + k1 / 2.0);
        let k3 = h * f(t_curr + h / 2.0, y_curr + k2 / 2.0);
        let k4 = h * f(t_curr + h, y_curr + k3);

        let y_next = y_curr + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        t.push(t_curr + h);
        y.push(y_next);
    }

    ODESolution::new(t, y)
}

/// Solve a system of ODEs using Euler's method
///
/// Solves dy/dt = f(t, y) where y is a vector
///
/// # Arguments
///
/// * `f` - Right-hand side function f(t, y) returning dy/dt
/// * `t0` - Initial time
/// * `y0` - Initial state vector
/// * `t_end` - End time
/// * `n_steps` - Number of steps
///
/// # Returns
///
/// ODESystemSolution containing the numerical solution
///
/// # Examples
///
/// ```
/// use rustmath_calculus::desolvers::eulers_method_system;
///
/// // Solve harmonic oscillator: y'' + y = 0
/// // As a system: y1' = y2, y2' = -y1
/// let f = |t: f64, y: &[f64]| vec![y[1], -y[0]];
/// let solution = eulers_method_system(&f, 0.0, &[1.0, 0.0], 10.0, 1000);
/// ```
pub fn eulers_method_system<F>(
    f: &F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    n_steps: usize,
) -> ODESystemSolution
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let h = (t_end - t0) / n_steps as f64;
    let mut t = vec![t0];
    let mut y = vec![y0.to_vec()];

    for i in 0..n_steps {
        let t_curr = t[i];
        let y_curr = &y[i];
        let dydt = f(t_curr, y_curr);

        let mut y_next = vec![0.0; y_curr.len()];
        for j in 0..y_curr.len() {
            y_next[j] = y_curr[j] + h * dydt[j];
        }

        t.push(t_curr + h);
        y.push(y_next);
    }

    ODESystemSolution::new(t, y)
}

/// Solve a system of ODEs using RK4
///
/// # Arguments
///
/// * `f` - Right-hand side function f(t, y) returning dy/dt
/// * `t0` - Initial time
/// * `y0` - Initial state vector
/// * `t_end` - End time
/// * `n_steps` - Number of steps
///
/// # Returns
///
/// ODESystemSolution containing the numerical solution
///
/// # Examples
///
/// ```
/// use rustmath_calculus::desolvers::runge_kutta_4_system;
///
/// // Solve harmonic oscillator
/// let f = |t: f64, y: &[f64]| vec![y[1], -y[0]];
/// let solution = runge_kutta_4_system(&f, 0.0, &[1.0, 0.0], 10.0, 1000);
/// ```
pub fn runge_kutta_4_system<F>(
    f: &F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    n_steps: usize,
) -> ODESystemSolution
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let h = (t_end - t0) / n_steps as f64;
    let dim = y0.len();
    let mut t = vec![t0];
    let mut y = vec![y0.to_vec()];

    for i in 0..n_steps {
        let t_curr = t[i];
        let y_curr = &y[i];

        // k1 = f(t, y)
        let k1 = f(t_curr, y_curr);

        // k2 = f(t + h/2, y + h*k1/2)
        let mut y_temp = vec![0.0; dim];
        for j in 0..dim {
            y_temp[j] = y_curr[j] + h * k1[j] / 2.0;
        }
        let k2 = f(t_curr + h / 2.0, &y_temp);

        // k3 = f(t + h/2, y + h*k2/2)
        for j in 0..dim {
            y_temp[j] = y_curr[j] + h * k2[j] / 2.0;
        }
        let k3 = f(t_curr + h / 2.0, &y_temp);

        // k4 = f(t + h, y + h*k3)
        for j in 0..dim {
            y_temp[j] = y_curr[j] + h * k3[j];
        }
        let k4 = f(t_curr + h, &y_temp);

        // y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        let mut y_next = vec![0.0; dim];
        for j in 0..dim {
            y_next[j] = y_curr[j] + h * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]) / 6.0;
        }

        t.push(t_curr + h);
        y.push(y_next);
    }

    ODESystemSolution::new(t, y)
}

/// Solve ODE using adaptive stepsize RK4 (RKF45)
///
/// This uses the Runge-Kutta-Fehlberg method with adaptive stepsize control.
///
/// # Arguments
///
/// * `f` - Right-hand side function
/// * `t0` - Initial time
/// * `y0` - Initial value
/// * `t_end` - End time
/// * `tol` - Error tolerance
///
/// # Returns
///
/// ODESolution with variable step sizes
///
/// # Examples
///
/// ```
/// use rustmath_calculus::desolvers::rk45_adaptive;
///
/// // Solve dy/dt = y with adaptive stepsize
/// let solution = rk45_adaptive(&|t, y| y, 0.0, 1.0, 1.0, 1e-6);
/// ```
pub fn rk45_adaptive<F>(f: &F, t0: f64, y0: f64, t_end: f64, tol: f64) -> ODESolution
where
    F: Fn(f64, f64) -> f64,
{
    let mut t = vec![t0];
    let mut y = vec![y0];
    let mut h = (t_end - t0) / 100.0; // Initial step size

    let mut t_curr = t0;
    let mut y_curr = y0;

    while t_curr < t_end {
        if t_curr + h > t_end {
            h = t_end - t_curr;
        }

        // RK4 step
        let k1 = h * f(t_curr, y_curr);
        let k2 = h * f(t_curr + h / 2.0, y_curr + k1 / 2.0);
        let k3 = h * f(t_curr + h / 2.0, y_curr + k2 / 2.0);
        let k4 = h * f(t_curr + h, y_curr + k3);
        let y_next_rk4 = y_curr + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

        // Lower order estimate (RK3)
        let y_next_rk3 = y_curr + (k1 + 4.0 * k2 + k4) / 6.0;

        // Error estimate
        let error = (y_next_rk4 - y_next_rk3).abs();

        if error < tol || h < 1e-10 {
            // Accept step
            t_curr += h;
            y_curr = y_next_rk4;
            t.push(t_curr);
            y.push(y_curr);

            // Increase step size if error is small
            if error < tol / 10.0 && h < (t_end - t0) / 10.0 {
                h *= 2.0;
            }
        } else {
            // Reject step and decrease step size
            h *= 0.5;
        }
    }

    ODESolution::new(t, y)
}

/// Convenience alias for RK4 (matches SageMath naming)
pub fn desolve_rk4<F>(f: &F, t0: f64, y0: f64, t_end: f64, n_steps: usize) -> ODESolution
where
    F: Fn(f64, f64) -> f64,
{
    runge_kutta_4(f, t0, y0, t_end, n_steps)
}

/// Convenience alias for system RK4
pub fn desolve_system_rk4<F>(
    f: &F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    n_steps: usize,
) -> ODESystemSolution
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    runge_kutta_4_system(f, t0, y0, t_end, n_steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eulers_method() {
        // dy/dt = 0, y(0) = 1 => y(t) = 1
        let solution = eulers_method(&|_t, _y| 0.0, 0.0, 1.0, 2.0, 10);
        assert_eq!(solution.t.len(), 11);
        assert!((solution.y[10] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rk4_constant() {
        // dy/dt = 2, y(0) = 0 => y(t) = 2t
        let solution = runge_kutta_4(&|_t, _y| 2.0, 0.0, 0.0, 1.0, 100);
        let y_final = solution.y[solution.y.len() - 1];
        assert!((y_final - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_rk4_exponential() {
        // dy/dt = y, y(0) = 1 => y(t) = e^t
        let solution = runge_kutta_4(&|_t, y| y, 0.0, 1.0, 1.0, 100);
        let y_final = solution.y[solution.y.len() - 1];
        let expected = 1.0_f64.exp();
        assert!((y_final - expected).abs() < 0.01);
    }

    #[test]
    fn test_eulers_method_system() {
        // dy1/dt = 0, dy2/dt = 0
        let f = |_t: f64, _y: &[f64]| vec![0.0, 0.0];
        let solution = eulers_method_system(&f, 0.0, &[1.0, 2.0], 1.0, 10);

        assert_eq!(solution.t.len(), 11);
        assert!((solution.y[10][0] - 1.0).abs() < 1e-10);
        assert!((solution.y[10][1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rk4_system_harmonic_oscillator() {
        // y'' + y = 0, or as a system:
        // y1' = y2
        // y2' = -y1
        // Initial conditions: y1(0) = 1, y2(0) = 0
        // Solution: y1(t) = cos(t), y2(t) = -sin(t)
        let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let solution = runge_kutta_4_system(&f, 0.0, &[1.0, 0.0], 2.0 * std::f64::consts::PI, 1000);

        let y_final = &solution.y[solution.y.len() - 1];
        // After one period, should return to initial conditions
        assert!((y_final[0] - 1.0).abs() < 0.01);
        assert!(y_final[1].abs() < 0.01);
    }

    #[test]
    fn test_rk45_adaptive() {
        // dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
        let solution = rk45_adaptive(&|_t, y| -y, 0.0, 1.0, 1.0, 1e-4);
        let y_final = solution.y[solution.y.len() - 1];
        let expected = (-1.0_f64).exp();
        assert!((y_final - expected).abs() < 0.001);
    }

    #[test]
    fn test_ode_solution_eval() {
        let t = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];
        let solution = ODESolution::new(t, y);

        // Test interpolation
        let y_interp = solution.eval(0.5);
        assert!((y_interp - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_desolve_aliases() {
        // Test that aliases work
        let solution = desolve_rk4(&|_t, y| y, 0.0, 1.0, 0.1, 10);
        assert!(solution.t.len() > 0);

        let f = |_t: f64, y: &[f64]| vec![y[0]];
        let system_solution = desolve_system_rk4(&f, 0.0, &[1.0], 0.1, 10);
        assert!(system_solution.t.len() > 0);
    }
}
