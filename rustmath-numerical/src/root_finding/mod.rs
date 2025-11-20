//! Root finding algorithms
//!
//! This module provides numerical methods for finding zeros (roots) of real-valued functions:
//!
//! - **Bisection method**: Robust, guaranteed convergence
//! - **Newton-Raphson method**: Fast quadratic convergence (requires derivative)
//! - **Secant method**: Fast superlinear convergence (no derivative needed)
//! - **Brent's method**: Hybrid approach (see optimization module)
//!
//! # Precision Guarantees
//!
//! All root-finding methods provide approximate solutions with the following characteristics:
//!
//! - **Bisection method**: Linear convergence, halves interval each iteration
//!   - Error after n iterations: (b-a) / 2^n
//!   - Guaranteed convergence if f(a) and f(b) have opposite signs
//!   - Typical tolerance: 1e-10
//!   - Maximum iterations: 100
//!
//! - **Newton-Raphson method**: Quadratic convergence near root
//!   - Error ~ C * error^2 each iteration (when converging)
//!   - Requires derivative f'(x)
//!   - May fail if f'(x) ≈ 0 or far from root
//!   - Typical tolerance: 1e-10
//!   - Maximum iterations: 100
//!
//! - **Secant method**: Superlinear convergence (~1.618 order)
//!   - Error ~ C * error^1.618 each iteration
//!   - Approximates derivative numerically
//!   - May fail if function is not smooth
//!   - Typical tolerance: 1e-10
//!   - Maximum iterations: 100
//!
//! # Convergence Criteria
//!
//! All methods use multiple convergence criteria:
//! - **Absolute error in x**: |x_new - x_old| < tol
//! - **Function value**: |f(x)| < tol (for bisection)
//! - **Interval width**: |b - a| < tol (for bisection)
//! - **Maximum iterations**: Prevents infinite loops
//!
//! # Numerical Precision
//!
//! - All computations use IEEE 754 double precision (f64)
//! - Machine epsilon: ~2.22e-16
//! - Practical limit: ~1e-15 relative accuracy
//! - Derivatives approximated with h=1e-8 (secant method)
//! - Minimum derivative threshold: 1e-14 (Newton-Raphson)
//!
//! # When Methods Fail
//!
//! - **Bisection**: Returns `None` if f(a) and f(b) have the same sign
//! - **Newton-Raphson**: Returns `None` if |f'(x)| < 1e-14
//! - **Secant**: Returns `None` if |f(x) - f(x_prev)| < 1e-14
//!
//! # Examples
//!
//! ```
//! use rustmath_numerical::root_finding::{bisection, newton_raphson};
//!
//! // Find sqrt(2) using f(x) = x^2 - 2
//! let f = |x: f64| x * x - 2.0;
//! let result = bisection(f, 0.0, 2.0, 1e-10, 100).unwrap();
//! assert!((result.root - 2.0_f64.sqrt()).abs() < 1e-9);
//!
//! // Newton-Raphson with derivative
//! let df = |x: f64| 2.0 * x;
//! let result = newton_raphson(f, df, 1.0, 1e-10, 100).unwrap();
//! assert!((result.root - 2.0_f64.sqrt()).abs() < 1e-9);
//! ```

pub mod methods;

// Re-export commonly used functions
pub use methods::{find_root, bisection, newton_raphson, secant, RootResult};
