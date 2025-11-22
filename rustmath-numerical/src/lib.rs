//! RustMath Numerical - Numerical computation methods
//!
//! This crate provides implementations of numerical algorithms for:
//! - **Root finding**: Bisection, Newton-Raphson, secant method, Brent's method
//! - **Optimization**: Gradient descent, golden section search, Nelder-Mead, curve fitting
//! - **Numerical integration**: Simpson's rule, trapezoidal rule, Romberg, Gauss-Legendre quadrature
//! - **Interpolation**: Lagrange, spline
//! - **Linear programming**: Simplex method
//! - **Fast Fourier Transform (FFT)**
//!
//! # Precision Guarantees
//!
//! All numerical methods in this crate provide **approximate** results with quantified error bounds.
//! Understanding these guarantees is essential for scientific computing.
//!
//! ## IEEE 754 Double Precision (f64)
//!
//! All computations use IEEE 754 double precision floating-point arithmetic:
//! - **Mantissa**: 53 bits (15-17 decimal digits)
//! - **Machine epsilon (ε)**: 2.22e-16 (relative rounding error per operation)
//! - **Range**: ±2.23e-308 to ±1.80e+308
//! - **Special values**: ±∞, NaN (Not a Number)
//!
//! ## Fundamental Limitations
//!
//! 1. **Rounding errors accumulate**: Each operation introduces O(ε) relative error
//! 2. **Cancellation error**: Subtracting nearly equal numbers loses precision
//! 3. **Condition number**: Ill-conditioned problems amplify errors
//! 4. **No exact representation**: Most real numbers cannot be represented exactly
//!
//! ## Algorithm-Specific Guarantees
//!
//! ### Root Finding (see [`root_finding`] module)
//! - Bisection: Error ≤ (b-a)/2^n after n iterations
//! - Newton-Raphson: Quadratic convergence if near root and f'(x) ≠ 0
//! - Secant: Superlinear convergence (~1.618 order)
//! - Default tolerance: 1e-10
//!
//! ### Optimization (see [`optimization`] module)
//! - Gradient descent: Depends on learning rate and convexity
//! - Golden section: O(φ^n) convergence, φ ≈ 1.618
//! - Brent's method: Superlinear convergence (~1.6 order)
//! - Default tolerance: 1e-8
//!
//! ### Integration (see [`integration`] module)
//! - Simpson's rule: O(h⁴) error, h = (b-a)/n
//! - Trapezoidal: O(h²) error
//! - Romberg: O(h^(2k)) error for k iterations
//! - Gauss-Legendre: Exact for polynomials of degree ≤ 2n-1
//! - Typical achievable error: 1e-10 to 1e-15
//!
//! ## Checking Results
//!
//! All result types include convergence flags and error estimates:
//! ```
//! use rustmath_numerical::root_finding::bisection;
//!
//! let f = |x: f64| x * x - 2.0;
//! let result = bisection(f, 0.0, 2.0, 1e-10, 100).unwrap();
//!
//! if result.converged {
//!     println!("Root: {}, Error: {}", result.root, result.error);
//! }
//! ```
//!
//! ## Arbitrary Precision (Future)
//!
//! Future versions will support arbitrary precision arithmetic via:
//! - MPFR (Multiple Precision Floating-Point Reliable)
//! - GMP (GNU Multiple Precision Arithmetic Library)
//! - Enable with `cargo build --features mpfr`
//!
//! # Feature Flags
//!
//! - **`std`** (default): Standard library support
//! - **`openblas`**: Use OpenBLAS for linear algebra (mutually exclusive with netlib/intel-mkl)
//! - **`netlib`**: Use Netlib reference BLAS/LAPACK
//! - **`intel-mkl`**: Use Intel Math Kernel Library (fastest, proprietary)
//!
//! # Traits
//!
//! This module defines traits for numerical computation:

use num_traits::{Float, FromPrimitive};

/// Trait for types that support numerical root finding
///
/// Types implementing this trait can be used with root-finding algorithms.
/// The trait requires basic arithmetic, comparison, and conversion operations.
pub trait NumericalScalar: Float + FromPrimitive {
    /// Machine epsilon for this type
    fn epsilon() -> Self;

    /// Check if value is within tolerance of zero
    fn is_near_zero(&self, tol: Self) -> bool {
        self.abs() < tol
    }

    /// Check if two values are approximately equal
    fn approx_eq(&self, other: Self, tol: Self) -> bool {
        (*self - other).abs() < tol
    }
}

/// Trait for types that support numerical integration
///
/// Integration requires function evaluation and basic arithmetic.
pub trait Integrable<T: NumericalScalar> {
    /// Evaluate the function at a point
    fn eval(&self, x: T) -> T;
}

/// Trait for types that support numerical optimization
///
/// Optimization may use gradient information if available.
pub trait Optimizable<T: NumericalScalar> {
    /// Evaluate the objective function
    fn eval(&self, x: T) -> T;

    /// Evaluate the gradient (if available)
    fn gradient(&self, _x: T) -> Option<T> {
        None
    }

    /// Check if the function is convex (if known)
    fn is_convex(&self) -> bool {
        false
    }
}

// Implement NumericalScalar for f64
impl NumericalScalar for f64 {
    fn epsilon() -> Self {
        std::f64::EPSILON
    }
}

// Implement NumericalScalar for f32
impl NumericalScalar for f32 {
    fn epsilon() -> Self {
        std::f32::EPSILON
    }
}

// Implement Integrable for closures
impl<F, T> Integrable<T> for F
where
    F: Fn(T) -> T,
    T: NumericalScalar,
{
    fn eval(&self, x: T) -> T {
        self(x)
    }
}

// Implement Optimizable for closures
impl<F, T> Optimizable<T> for F
where
    F: Fn(T) -> T,
    T: NumericalScalar,
{
    fn eval(&self, x: T) -> T {
        self(x)
    }
}

// Module declarations
pub mod root_finding;
pub mod optimization;
pub mod integration;
pub mod interpolation;
pub mod fft;
pub mod linear_programming;
pub mod backends;

// Re-export commonly used items
pub use root_finding::{find_root, bisection, newton_raphson, secant};
pub use optimization::{minimize, gradient_descent, nelder_mead};
pub use integration::{integrate, simpson, trapezoid, romberg};
pub use integration::{nodes, nodes_uncached, estimate_error, integrate_vector, integrate_vector_n};
pub use interpolation::{lagrange_interpolate, spline_interpolate};
pub use fft::{fft, ifft};
pub use linear_programming::simplex;

// Re-export comprehensive optimize module functions
pub use optimization::{
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_scalar_f64() {
        let x = 1.0f64;
        let epsilon = f64::epsilon();

        assert!(epsilon < 1e-15);
        assert!(x.is_near_zero(2.0));
        assert!(!x.is_near_zero(0.5));
        assert!(x.approx_eq(1.0 + epsilon, epsilon * 2.0));
    }

    #[test]
    fn test_integrable_closure() {
        let f = |x: f64| x * x;
        assert_eq!(f.eval(2.0), 4.0);
    }

    #[test]
    fn test_optimizable_closure() {
        let f = |x: f64| x * x + 2.0 * x + 1.0;
        assert_eq!(f.eval(0.0), 1.0);
        assert_eq!(f.gradient(0.0), None); // Closures don't provide gradients by default
        assert!(!f.is_convex()); // Default is false
    }
}
