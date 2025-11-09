//! RustMath Dynamics - Dynamical systems, fractals, and chaos theory
//!
//! This crate provides implementations of:
//! - Discrete dynamical systems (iteration of maps)
//! - Continuous dynamical systems (ODEs)
//! - Fractals (Mandelbrot set, Julia sets)
//! - Chaos theory (Lyapunov exponents, bifurcation diagrams)

pub mod discrete;
pub mod continuous;
pub mod fractals;
pub mod chaos;

pub use discrete::{DiscreteSystem, OrbitPoint, iterate, fixed_points};
pub use continuous::{ContinuousSystem, runge_kutta_4, euler_method};
pub use fractals::{mandelbrot, julia_set, MandelbrotResult, JuliaResult};
pub use chaos::{lyapunov_exponent, bifurcation_diagram, BifurcationPoint};
