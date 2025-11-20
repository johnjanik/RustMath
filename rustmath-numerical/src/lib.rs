//! RustMath Numerical - Numerical computation methods
//!
//! This crate provides implementations of:
//! - Root finding algorithms (Brent's method, bisection, Newton-Raphson, secant)
//! - Optimization methods (golden section search, gradient descent, Nelder-Mead)
//! - Curve fitting (linear least squares)
//! - Linear programming
//! - Numerical integration (quadrature)
//! - Gauss-Legendre quadrature
//! - Interpolation
//! - Fast Fourier Transform (FFT)
//! - Optimization backends (GLPK, CVXOPT, generic)

pub mod rootfinding;
pub mod optimization;
pub mod optimize;
pub mod integration;
pub mod gauss_legendre;
pub mod interpolation;
pub mod fft;
pub mod linear_programming;
pub mod backends;

pub use rootfinding::{find_root, bisection, newton_raphson, secant};
pub use optimization::{minimize, gradient_descent, nelder_mead};
pub use integration::{integrate, simpson, trapezoid, romberg};
pub use gauss_legendre::{nodes, nodes_uncached, estimate_error, integrate_vector, integrate_vector_n};
pub use interpolation::{lagrange_interpolate, spline_interpolate};
pub use fft::{fft, ifft};
pub use linear_programming::simplex;

// Export the comprehensive optimize module functions
pub use optimize::{
    find_root as optimize_find_root,
    find_local_minimum,
    find_local_maximum,
    find_fit,
    brent_root,
    golden_section_search,
    RootResult as OptimizeRootResult,
    OptimizationResult,
    FitResult,
};
