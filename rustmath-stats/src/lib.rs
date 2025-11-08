//! RustMath Stats - Statistics and probability distributions
//!
//! This crate provides implementations of:
//! - Probability distributions
//! - Random variables
//! - Statistical functions
//! - Hypothesis testing
//! - Linear regression

pub mod distributions;
pub mod statistics;
pub mod hypothesis;
pub mod regression;

pub use distributions::{Distribution, Normal, Binomial, Uniform, Poisson, Exponential};
pub use statistics::{mean, variance, std_dev, median, mode, correlation, covariance};
pub use hypothesis::{t_test, chi_squared_test};
pub use regression::LinearRegression;
