//! RustMath Numerical - Numerical computation methods
//!
//! This crate provides implementations of:
//! - Root finding algorithms
//! - Optimization methods
//! - Linear programming
//! - Numerical integration (quadrature)
//! - Interpolation
//! - Fast Fourier Transform (FFT)
//! - Optimization backends (GLPK, CVXOPT, generic)

pub mod rootfinding;
pub mod optimization;
pub mod integration;
pub mod interpolation;
pub mod fft;
pub mod linear_programming;
pub mod backends;

pub use rootfinding::{find_root, bisection, newton_raphson, secant};
pub use optimization::{minimize, gradient_descent, nelder_mead};
pub use integration::{integrate, simpson, trapezoid, romberg};
pub use interpolation::{lagrange_interpolate, spline_interpolate};
pub use fft::{fft, ifft};
pub use linear_programming::simplex;
