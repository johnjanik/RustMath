//! Continuous dynamical systems
//!
//! Systems defined by differential equations: dx/dt = f(t, x)

/// A continuous dynamical system defined by dx/dt = f(t, x)
pub struct ContinuousSystem<F>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    pub derivative: F,
    pub dimension: usize,
}

impl<F> ContinuousSystem<F>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    pub fn new(derivative: F, dimension: usize) -> Self {
        ContinuousSystem {
            derivative,
            dimension,
        }
    }

    /// Solve using Runge-Kutta 4th order method
    pub fn solve_rk4(&self, x0: &[f64], t0: f64, tf: f64, dt: f64) -> Vec<(f64, Vec<f64>)> {
        runge_kutta_4(&self.derivative, x0, t0, tf, dt)
    }

    /// Solve using Euler's method
    pub fn solve_euler(&self, x0: &[f64], t0: f64, tf: f64, dt: f64) -> Vec<(f64, Vec<f64>)> {
        euler_method(&self.derivative, x0, t0, tf, dt)
    }
}

/// Runge-Kutta 4th order method for solving ODEs
pub fn runge_kutta_4<F>(f: F, x0: &[f64], t0: f64, tf: f64, dt: f64) -> Vec<(f64, Vec<f64>)>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let n = ((tf - t0) / dt).ceil() as usize;
    let mut result = Vec::with_capacity(n + 1);

    let mut t = t0;
    let mut x = x0.to_vec();
    result.push((t, x.clone()));

    for _ in 0..n {
        // k1 = f(t, x)
        let k1 = f(t, &x);

        // k2 = f(t + dt/2, x + k1*dt/2)
        let x_temp: Vec<f64> = x
            .iter()
            .zip(k1.iter())
            .map(|(xi, k1i)| xi + k1i * dt / 2.0)
            .collect();
        let k2 = f(t + dt / 2.0, &x_temp);

        // k3 = f(t + dt/2, x + k2*dt/2)
        let x_temp: Vec<f64> = x
            .iter()
            .zip(k2.iter())
            .map(|(xi, k2i)| xi + k2i * dt / 2.0)
            .collect();
        let k3 = f(t + dt / 2.0, &x_temp);

        // k4 = f(t + dt, x + k3*dt)
        let x_temp: Vec<f64> = x.iter().zip(k3.iter()).map(|(xi, k3i)| xi + k3i * dt).collect();
        let k4 = f(t + dt, &x_temp);

        // x_{n+1} = x_n + (k1 + 2*k2 + 2*k3 + k4) * dt/6
        for i in 0..x.len() {
            x[i] += (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * dt / 6.0;
        }

        t += dt;
        result.push((t, x.clone()));
    }

    result
}

/// Euler's method for solving ODEs
pub fn euler_method<F>(f: F, x0: &[f64], t0: f64, tf: f64, dt: f64) -> Vec<(f64, Vec<f64>)>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let n = ((tf - t0) / dt).ceil() as usize;
    let mut result = Vec::with_capacity(n + 1);

    let mut t = t0;
    let mut x = x0.to_vec();
    result.push((t, x.clone()));

    for _ in 0..n {
        let dx = f(t, &x);
        for i in 0..x.len() {
            x[i] += dx[i] * dt;
        }
        t += dt;
        result.push((t, x.clone()));
    }

    result
}

/// Lorenz system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz
pub fn lorenz_system(sigma: f64, rho: f64, beta: f64) -> impl Fn(f64, &[f64]) -> Vec<f64> {
    move |_t, x| {
        vec![
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ]
    }
}

/// Rossler system: dx/dt = -y-z, dy/dt = x+ay, dz/dt = b+z(x-c)
pub fn rossler_system(a: f64, b: f64, c: f64) -> impl Fn(f64, &[f64]) -> Vec<f64> {
    move |_t, x| {
        vec![
            -x[1] - x[2],
            x[0] + a * x[1],
            b + x[2] * (x[0] - c),
        ]
    }
}

/// Van der Pol oscillator: dx/dt = y, dy/dt = μ(1-x²)y - x
pub fn van_der_pol(mu: f64) -> impl Fn(f64, &[f64]) -> Vec<f64> {
    move |_t, x| vec![x[1], mu * (1.0 - x[0] * x[0]) * x[1] - x[0]]
}

/// Simple harmonic oscillator: dx/dt = y, dy/dt = -ω²x
pub fn harmonic_oscillator(omega: f64) -> impl Fn(f64, &[f64]) -> Vec<f64> {
    move |_t, x| vec![x[1], -omega * omega * x[0]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_constant() {
        // dx/dt = 0, x(0) = 5, solution is x(t) = 5
        let f = |_t: f64, _x: &[f64]| vec![0.0];
        let solution = euler_method(f, &[5.0], 0.0, 1.0, 0.1);

        for (_t, x) in solution {
            assert!((x[0] - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_euler_linear() {
        // dx/dt = 1, x(0) = 0, solution is x(t) = t
        let f = |_t: f64, _x: &[f64]| vec![1.0];
        let solution = euler_method(f, &[0.0], 0.0, 1.0, 0.1);

        for (t, x) in solution {
            assert!((x[0] - t).abs() < 0.01);
        }
    }

    #[test]
    fn test_rk4_exponential() {
        // dx/dt = x, x(0) = 1, solution is x(t) = e^t
        let f = |_t: f64, x: &[f64]| vec![x[0]];
        let solution = runge_kutta_4(f, &[1.0], 0.0, 1.0, 0.01);

        let (_tf, xf) = solution.last().unwrap();
        let expected = 1.0_f64.exp(); // e^1
        assert!((xf[0] - expected).abs() < 0.001);
    }

    #[test]
    fn test_harmonic_oscillator() {
        // dx/dt = y, dy/dt = -x (ω = 1)
        // Solution: x(t) = cos(t), y(t) = -sin(t) with x(0)=1, y(0)=0
        let omega = 1.0;
        let system = ContinuousSystem::new(harmonic_oscillator(omega), 2);

        let solution = system.solve_rk4(&[1.0, 0.0], 0.0, 6.28, 0.01);

        // At t ≈ 2π, should return to initial conditions
        let (_t, x_final) = solution.last().unwrap();
        assert!((x_final[0] - 1.0).abs() < 0.1);
        assert!(x_final[1].abs() < 0.1);
    }

    #[test]
    fn test_continuous_system_creation() {
        let f = |_t: f64, x: &[f64]| vec![x[0], -x[1]];
        let system = ContinuousSystem::new(f, 2);
        assert_eq!(system.dimension, 2);
    }

    #[test]
    fn test_lorenz_system() {
        let f = lorenz_system(10.0, 28.0, 8.0 / 3.0);
        let deriv = f(0.0, &[1.0, 1.0, 1.0]);
        assert_eq!(deriv.len(), 3);
    }

    #[test]
    fn test_rossler_system() {
        let f = rossler_system(0.2, 0.2, 5.7);
        let deriv = f(0.0, &[0.0, 0.0, 0.0]);
        assert_eq!(deriv.len(), 3);
    }

    #[test]
    fn test_van_der_pol() {
        let f = van_der_pol(1.0);
        let deriv = f(0.0, &[1.0, 0.0]);
        assert_eq!(deriv.len(), 2);
        assert_eq!(deriv[0], 0.0);
        assert_eq!(deriv[1], -1.0);
    }

    #[test]
    fn test_rk4_multidimensional() {
        // System: dx/dt = y, dy/dt = -x
        let f = |_t: f64, x: &[f64]| vec![x[1], -x[0]];
        let solution = runge_kutta_4(f, &[1.0, 0.0], 0.0, 1.0, 0.01);

        assert!(!solution.is_empty());
        assert_eq!(solution[0].1.len(), 2);
    }
}
