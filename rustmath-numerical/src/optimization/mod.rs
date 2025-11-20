//! Numerical optimization methods
//!
//! This module provides various optimization algorithms for finding minima and maxima
//! of real-valued functions:
//!
//! - **Gradient-based methods**: Gradient descent, Nelder-Mead simplex
//! - **Derivative-free methods**: Golden section search, Brent's method
//! - **Curve fitting**: Linear and nonlinear least squares
//!
//! # Precision Guarantees
//!
//! All optimization methods provide approximate results with the following characteristics:
//!
//! - **Gradient descent**: Linear convergence for convex functions
//!   - Learning rate affects stability and convergence speed
//!   - Typical tolerance: 1e-6
//!   - May fail to converge for non-convex functions
//!
//! - **Golden section search**: Linear convergence, O(φ^n) where φ ≈ 1.618
//!   - Guaranteed convergence for unimodal functions
//!   - Typical tolerance: 1e-8
//!   - No derivative required
//!
//! - **Brent's method**: Superlinear convergence (~1.6 order)
//!   - Combines inverse quadratic interpolation with golden section
//!   - Typical tolerance: 1e-10
//!   - Most efficient for smooth functions
//!
//! - **Nelder-Mead simplex**: No guaranteed convergence rate
//!   - Works well for non-smooth functions
//!   - Can handle noisy objective functions
//!   - Typical tolerance: 1e-8
//!
//! # Numerical Stability
//!
//! All algorithms use IEEE 754 double precision (f64) arithmetic:
//! - Machine epsilon: ~2.22e-16
//! - Effective precision: ~15-16 decimal digits
//! - Underflow threshold: ~2.23e-308
//! - Overflow threshold: ~1.80e+308
//!
//! # Examples
//!
//! ```
//! use rustmath_numerical::optimization::{minimize, golden_section_search};
//!
//! // Minimize f(x) = (x - 3)^2
//! let f = |x: f64| (x - 3.0) * (x - 3.0);
//! let result = minimize(f, 0.0);
//! assert!((result.unwrap().x - 3.0).abs() < 0.1);
//! ```

pub mod gradient;
pub mod brent;

// Re-export commonly used functions
pub use gradient::{minimize, gradient_descent, nelder_mead, OptResult};
pub use brent::{
    find_root, find_local_minimum, find_local_maximum, find_fit,
    brent_root, golden_section_search,
    RootResult, OptimizationResult, FitResult,
};
