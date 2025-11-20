//! RustMath Stats - Statistics and probability distributions
//!
//! This crate provides implementations of:
//! - Generic statistical functions over algebraic structures (Ring, Field traits)
//! - Probability distributions
//! - Random variables
//! - Hypothesis testing
//! - Linear regression
//!
//! ## Architecture Overview
//!
//! The statistics crate is designed around two complementary modes of operation:
//!
//! ### 1. Exact Arithmetic Mode
//!
//! When working with types that implement `Field` trait (like `Rational`), all
//! statistical computations preserve exact values without rounding errors:
//!
//! ```ignore
//! use rustmath_stats::basic_stats::mean;
//! use rustmath_rationals::Rational;
//!
//! let data = vec![
//!     Rational::new(1, 3).unwrap(),
//!     Rational::new(1, 2).unwrap(),
//!     Rational::new(2, 3).unwrap(),
//! ];
//!
//! // Result is exactly 1/2, not 0.5 with floating-point error
//! let m = mean(&data).unwrap();
//! assert_eq!(m, Rational::new(1, 2).unwrap());
//! ```
//!
//! **Benefits of Exact Arithmetic:**
//! - No rounding errors in intermediate calculations
//! - Results can be verified symbolically
//! - Useful for theoretical analysis and algorithm verification
//! - Compatible with symbolic computation systems
//!
//! **Limitations:**
//! - Cannot compute transcendental functions (sqrt, exp, log) exactly
//! - Some operations (standard deviation) require approximation
//! - Performance may be slower for large datasets
//! - Memory usage grows with precision requirements
//!
//! ### 2. Approximate Arithmetic Mode
//!
//! When working with floating-point types (like `f64`), computations use
//! standard numerical algorithms with floating-point arithmetic:
//!
//! ```ignore
//! use rustmath_stats::statistics::mean;
//!
//! let data = vec![1.5, 2.7, 3.2, 4.1];
//! let m = mean(&data).unwrap();
//! assert!((m - 2.875).abs() < 1e-10);
//! ```
//!
//! **Benefits of Approximate Arithmetic:**
//! - Fast computation for large datasets
//! - Supports all mathematical functions (sqrt, exp, log, etc.)
//! - Compatible with standard statistical libraries
//! - Efficient memory usage
//!
//! **Limitations:**
//! - Subject to floating-point rounding errors
//! - Catastrophic cancellation in poorly-conditioned problems
//! - Numerical stability concerns for some algorithms
//!
//! ### Choosing the Right Mode
//!
//! | Use Case | Recommended Mode | Rationale |
//! |----------|-----------------|-----------|
//! | Algorithm verification | Exact (Rational) | No rounding errors |
//! | Theoretical proofs | Exact (Rational) | Symbolic manipulation |
//! | Large-scale data analysis | Approximate (f64) | Performance |
//! | Machine learning | Approximate (f64) | Speed and library compatibility |
//! | Financial calculations | Exact (Rational/Decimal) | Regulatory requirements |
//! | Scientific computing | Approximate (f64) | Standard practice |
//!
//! ## Modules
//!
//! - **`basic_stats`**: Generic statistical functions over algebraic structures (Field trait)
//!   - Works with any type implementing `Field` (Rational, f64, etc.)
//!   - Includes: mean, median, mode, variance, standard deviation, moving average
//!   - Preserves exact arithmetic when possible
//!
//! - **`statistics`**: f64-specific statistical functions
//!   - Optimized for floating-point performance
//!   - Includes specialized algorithms for numerical stability
//!   - Compatible with standard statistical software
//!
//! - **`distributions`**: Probability distributions
//!   - Normal, Binomial, Uniform, Poisson, Exponential
//!   - PDF, CDF, sampling, moments
//!   - Currently f64-based (exact distributions planned)
//!
//! - **`hypothesis`**: Hypothesis testing
//!   - t-tests, chi-squared tests
//!   - p-values and confidence intervals
//!
//! - **`regression`**: Linear regression
//!   - Ordinary least squares
//!   - Coefficient estimation and diagnostics
//!
//! ## Trait Definitions
//!
//! The crate defines several traits for statistical operations:
//!
//! ### Core Statistical Traits
//!
//! These traits enable generic statistical operations over different numeric types:
//!
//! - **`StatisticalMoments`**: Compute mean, variance, standard deviation
//! - **`Distribution`**: Probability distribution with PDF, CDF, sampling
//! - **`SquareRoot`**: Types supporting square root (needed for std dev)
//!
//! ## Examples
//!
//! ### Computing Mean with Exact Arithmetic
//!
//! ```ignore
//! use rustmath_stats::basic_stats;
//! use rustmath_rationals::Rational;
//!
//! // Dataset of exact rational numbers
//! let grades = vec![
//!     Rational::new(85, 100).unwrap(),  // 85%
//!     Rational::new(92, 100).unwrap(),  // 92%
//!     Rational::new(78, 100).unwrap(),  // 78%
//!     Rational::new(88, 100).unwrap(),  // 88%
//! ];
//!
//! // Exact mean: (85 + 92 + 78 + 88) / 4 = 343/4
//! let avg = basic_stats::mean(&grades).unwrap();
//! assert_eq!(avg, Rational::new(343, 400).unwrap());
//! ```
//!
//! ### Computing Variance with Bias Control
//!
//! ```ignore
//! use rustmath_stats::basic_stats::variance;
//! use rustmath_rationals::Rational;
//!
//! let data: Vec<Rational> = vec![/* ... */];
//!
//! // Sample variance (unbiased estimator, divides by n-1)
//! let sample_var = variance(&data, false).unwrap();
//!
//! // Population variance (biased estimator, divides by n)
//! let pop_var = variance(&data, true).unwrap();
//! ```
//!
//! ### Working with Distributions
//!
//! ```ignore
//! use rustmath_stats::distributions::{Normal, Distribution};
//!
//! let normal = Normal::new(100.0, 15.0).unwrap();
//!
//! // Probability density at x = 110
//! let density = normal.pdf(110.0);
//!
//! // Cumulative probability P(X <= 110)
//! let prob = normal.cdf(110.0);
//!
//! // Sample random value
//! let sample = normal.sample();
//! ```
//!
//! ## Performance Considerations
//!
//! ### Memory Usage
//!
//! - **Exact arithmetic (Rational)**: Memory usage grows with numerator/denominator size
//!   - Simple operations: O(log n) space per number
//!   - After many operations: Can grow to hundreds of bits
//!   - Use simplification/approximation for long computation chains
//!
//! - **Approximate arithmetic (f64)**: Fixed 64-bit per number
//!   - Constant memory regardless of operations
//!   - Suitable for large-scale datasets
//!
//! ### Computational Complexity
//!
//! | Operation | Exact (Rational) | Approximate (f64) |
//! |-----------|-----------------|-------------------|
//! | Addition | O(log n) | O(1) |
//! | Multiplication | O(n²) | O(1) |
//! | Division | O(n²) | O(1) |
//! | Mean | O(n × log m) | O(n) |
//! | Variance | O(n × log m) | O(n) |
//!
//! where n = dataset size, m = average number size in bits
//!
//! ## Future Enhancements
//!
//! - **Exact distributions**: Symbolic probability distributions over rationals
//! - **Multivariate statistics**: Covariance matrices, multivariate normal
//! - **Time series**: Autocorrelation, spectral analysis
//! - **Non-parametric tests**: Rank tests, resampling methods
//! - **Bayesian inference**: Prior/posterior calculations with exact arithmetic
//! - **Robust statistics**: M-estimators, trimmed means

pub mod basic_stats;
pub mod distributions;
pub mod statistics;
pub mod hypothesis;
pub mod regression;

// Re-export commonly used items
pub use distributions::{Distribution, Normal, Binomial, Uniform, Poisson, Exponential};
pub use statistics::{mean, variance, std_dev, median, mode, correlation, covariance};
pub use hypothesis::{t_test, chi_squared_test};
pub use regression::LinearRegression;

/// Trait for types that support statistical moment calculations
///
/// This trait provides a unified interface for computing statistical moments
/// (mean, variance, etc.) over different numeric types.
pub trait StatisticalMoments: Sized {
    /// Compute the mean (first moment) of a dataset
    fn mean(data: &[Self]) -> Option<Self>;

    /// Compute the variance (second central moment) of a dataset
    ///
    /// # Arguments
    /// * `bias` - If false, use sample variance (n-1 divisor); if true, use population variance (n divisor)
    fn variance(data: &[Self], bias: bool) -> Option<Self>;

    /// Compute the standard deviation (square root of variance)
    fn std_dev(data: &[Self], bias: bool) -> Option<Self> {
        Self::variance(data, bias).and_then(|var| var.sqrt())
    }
}

/// Trait for types that support square root operation
///
/// Required for computing standard deviation and other statistics
/// that involve square roots.
pub trait SquareRoot {
    /// Compute the square root, returning None if undefined (e.g., negative numbers)
    fn sqrt(self) -> Option<Self>;
}

impl SquareRoot for f64 {
    fn sqrt(self) -> Option<Self> {
        if self < 0.0 {
            None
        } else {
            Some(f64::sqrt(self))
        }
    }
}

/// Trait for computing statistical order statistics
///
/// Order statistics are quantities that depend on the ordering of the data,
/// such as median, quantiles, and percentiles.
pub trait OrderStatistic: Sized + PartialOrd + Clone {
    /// Compute the median (50th percentile)
    fn median(data: &[Self]) -> Option<Self>;

    /// Compute an arbitrary quantile (0.0 to 1.0)
    fn quantile(data: &[Self], q: f64) -> Option<Self>;

    /// Compute the minimum value
    fn min(data: &[Self]) -> Option<Self> {
        data.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).cloned()
    }

    /// Compute the maximum value
    fn max(data: &[Self]) -> Option<Self> {
        data.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).cloned()
    }
}

/// Trait for computing statistical measures of association
///
/// Measures how two variables relate to each other (correlation, covariance).
pub trait Association: Sized {
    /// Compute covariance between two datasets
    fn covariance(x: &[Self], y: &[Self]) -> Option<Self>;

    /// Compute Pearson correlation coefficient
    fn correlation(x: &[Self], y: &[Self]) -> Option<Self>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // This test just verifies that all modules compile and are accessible
        // Actual functionality is tested in each module
    }
}
