//! Numerical integration methods
//!
//! This module provides various numerical integration (quadrature) techniques:
//! - **Basic quadrature**: Simpson's rule, trapezoidal rule, Romberg integration
//! - **Gauss-Legendre quadrature**: High-accuracy integration using Legendre polynomial roots
//!
//! # Precision Guarantees
//!
//! All integration methods in this module provide approximate results with the following
//! precision characteristics:
//!
//! - **Simpson's rule**: O(h⁴) error, where h = (b-a)/n
//!   - Typically achieves 1e-6 relative error with n=100 for smooth functions
//!   - Exact for polynomials of degree ≤ 3
//!
//! - **Trapezoidal rule**: O(h²) error
//!   - Typically achieves 1e-4 relative error with n=1000 for smooth functions
//!   - Exact for linear functions
//!
//! - **Romberg integration**: O(h^(2k)) error for k iterations
//!   - Typically achieves 1e-10 relative error with 10 iterations
//!   - Uses Richardson extrapolation for accelerated convergence
//!
//! - **Gauss-Legendre quadrature**: Exact for polynomials of degree ≤ 2n-1
//!   - Adaptive algorithm typically achieves machine precision (1e-15)
//!   - Most efficient for smooth integrands
//!
//! # Examples
//!
//! ```
//! use rustmath_numerical::integration::{simpson, gauss_legendre};
//!
//! // Integrate x^2 from 0 to 1 (exact: 1/3)
//! let f = |x: f64| x * x;
//! let result = simpson(f, 0.0, 1.0, 100);
//! assert!((result - 1.0/3.0).abs() < 1e-6);
//! ```

pub mod quadrature;
pub mod gauss_legendre;

// Re-export commonly used functions
pub use quadrature::{integrate, simpson, trapezoid, romberg};
pub use gauss_legendre::{nodes, nodes_uncached, estimate_error, integrate_vector, integrate_vector_n};
