//! RustMath Stats - Statistics and probability distributions
//!
//! This crate provides implementations of:
//! - Probability distributions
//! - Random variables
//! - Statistical functions (both generic and f64-specific)
//! - Hypothesis testing
//! - Linear regression
//!
//! ## Modules
//!
//! - `basic_stats`: Generic statistical functions over algebraic structures (Field trait)
//! - `statistics`: f64-specific statistical functions
//! - `distributions`: Probability distributions
//! - `hypothesis`: Hypothesis testing
//! - `regression`: Linear regression

pub mod basic_stats;
pub mod distributions;
pub mod statistics;
pub mod hypothesis;
pub mod regression;

pub use distributions::{Distribution, Normal, Binomial, Uniform, Poisson, Exponential};
pub use statistics::{mean, variance, std_dev, median, mode, correlation, covariance};
pub use hypothesis::{t_test, chi_squared_test};
pub use regression::LinearRegression;
