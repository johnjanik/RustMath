//! Basic statistical functions
//!
//! This module provides fundamental statistical operations that work generically
//! over algebraic structures (Ring and Field traits), enabling both exact arithmetic
//! (e.g., Rational) and approximate arithmetic (e.g., f64).
//!
//! # Functions
//!
//! - `mean`: Arithmetic mean (average) of values
//! - `median`: Middle value when sorted
//! - `mode`: Most frequently occurring value
//! - `variance`: Measure of spread around the mean
//! - `std`: Standard deviation (square root of variance)
//! - `moving_average`: Sliding window averages
//!
//! # Bias Parameter
//!
//! The `bias` parameter in `variance` and `std` functions controls the divisor:
//! - `bias = false` (default): Sample variance, divides by (n-1) - unbiased estimator
//! - `bias = true`: Population variance, divides by n - biased but maximum likelihood estimator
//!
//! For a population (all data available), use `bias = true`.
//! For a sample (estimating population parameters), use `bias = false`.

use rustmath_core::{Field, MathError, NumericConversion, Result};
use std::collections::HashMap;
use std::hash::Hash;
use std::cmp::Ordering;

/// Compute the arithmetic mean (average) of a dataset
///
/// Returns the sum of all values divided by the count.
/// Requires at least one element.
///
/// # Type Constraints
///
/// - Generic over any `Field` type (supports division)
/// - Requires `NumericConversion` for converting the count to the field type
///
/// # Examples
///
/// ```
/// use rustmath_stats::basic_stats::mean;
/// use rustmath_rationals::Rational;
///
/// // Exact rational arithmetic
/// let data = vec![
///     Rational::new(1, 2).unwrap(),
///     Rational::new(3, 4).unwrap(),
///     Rational::new(5, 4).unwrap(),
/// ];
/// let m = mean(&data).unwrap();
/// assert_eq!(m, Rational::new(5, 6).unwrap());
/// ```
pub fn mean<T>(data: &[T]) -> Result<T>
where
    T: Field + NumericConversion,
{
    if data.is_empty() {
        return Err(MathError::InvalidArgument("Empty dataset".to_string()));
    }

    let sum = data.iter().fold(T::zero(), |acc, x| acc + x.clone());
    let count = T::from_u64(data.len() as u64);

    sum.divide(&count)
}

/// Compute the median (middle value) of a dataset
///
/// For odd-length datasets, returns the middle value.
/// For even-length datasets, returns the average of the two middle values.
///
/// # Type Constraints
///
/// - Generic over any `Field` type that is `PartialOrd` (can be sorted)
/// - Requires `NumericConversion` for division by 2
///
/// # Examples
///
/// ```
/// use rustmath_stats::basic_stats::median;
/// use rustmath_rationals::Rational;
///
/// // Odd length - returns middle value
/// let data = vec![
///     Rational::new(1, 1).unwrap(),
///     Rational::new(3, 1).unwrap(),
///     Rational::new(5, 1).unwrap(),
/// ];
/// let m = median(&data).unwrap();
/// assert_eq!(m, Rational::new(3, 1).unwrap());
///
/// // Even length - returns average of middle two
/// let data2 = vec![
///     Rational::new(1, 1).unwrap(),
///     Rational::new(2, 1).unwrap(),
///     Rational::new(3, 1).unwrap(),
///     Rational::new(4, 1).unwrap(),
/// ];
/// let m2 = median(&data2).unwrap();
/// assert_eq!(m2, Rational::new(5, 2).unwrap());
/// ```
pub fn median<T>(data: &[T]) -> Result<T>
where
    T: Field + NumericConversion + PartialOrd,
{
    if data.is_empty() {
        return Err(MathError::InvalidArgument("Empty dataset".to_string()));
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let mid = sorted.len() / 2;

    if sorted.len() % 2 == 0 {
        // Even length: average of two middle values
        let sum = sorted[mid - 1].clone() + sorted[mid].clone();
        let two = T::from_u64(2);
        sum.divide(&two)
    } else {
        // Odd length: middle value
        Ok(sorted[mid].clone())
    }
}

/// Compute the mode (most frequently occurring value) of a dataset
///
/// Returns the value that appears most often. If there are multiple modes,
/// returns one of them (unspecified which one).
///
/// # Type Constraints
///
/// - Generic over any type that is `Eq` and `Hash` (can be used in HashMap)
/// - Must also be `Clone`
///
/// # Examples
///
/// ```
/// use rustmath_stats::basic_stats::mode;
/// use rustmath_rationals::Rational;
///
/// let data = vec![
///     Rational::new(1, 1).unwrap(),
///     Rational::new(2, 1).unwrap(),
///     Rational::new(2, 1).unwrap(),
///     Rational::new(3, 1).unwrap(),
/// ];
/// let m = mode(&data).unwrap();
/// assert_eq!(m, Rational::new(2, 1).unwrap());
/// ```
pub fn mode<T>(data: &[T]) -> Result<T>
where
    T: Eq + Hash + Clone,
{
    if data.is_empty() {
        return Err(MathError::InvalidArgument("Empty dataset".to_string()));
    }

    let mut counts = HashMap::new();
    for value in data {
        *counts.entry(value).or_insert(0) += 1;
    }

    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(value, _)| value.clone())
        .ok_or(MathError::InvalidArgument("Empty dataset".to_string()))
}

/// Compute the variance of a dataset
///
/// Variance measures how far values are spread out from the mean.
/// It is the average of squared deviations from the mean.
///
/// # Arguments
///
/// - `data`: The dataset
/// - `bias`: Controls the divisor
///   - `false`: Sample variance (divides by n-1) - unbiased estimator
///   - `true`: Population variance (divides by n) - biased estimator
///
/// # Type Constraints
///
/// - Generic over any `Field` type
/// - Requires `NumericConversion` for converting counts
///
/// # Examples
///
/// ```
/// use rustmath_stats::basic_stats::variance;
/// use rustmath_rationals::Rational;
///
/// let data = vec![
///     Rational::new(1, 1).unwrap(),
///     Rational::new(2, 1).unwrap(),
///     Rational::new(3, 1).unwrap(),
///     Rational::new(4, 1).unwrap(),
///     Rational::new(5, 1).unwrap(),
/// ];
///
/// // Sample variance (unbiased)
/// let var_sample = variance(&data, false).unwrap();
/// assert_eq!(var_sample, Rational::new(5, 2).unwrap()); // 2.5
///
/// // Population variance (biased)
/// let var_pop = variance(&data, true).unwrap();
/// assert_eq!(var_pop, Rational::new(2, 1).unwrap()); // 2.0
/// ```
pub fn variance<T>(data: &[T], bias: bool) -> Result<T>
where
    T: Field + NumericConversion,
{
    let min_len = if bias { 1 } else { 2 };
    if data.len() < min_len {
        return Err(MathError::InvalidArgument(
            format!("variance requires at least {} elements", min_len)
        ));
    }

    let m = mean(data)?;

    // Sum of squared differences from mean
    let sum_squared_diff = data.iter()
        .map(|x| {
            let diff = x.clone() - m.clone();
            diff.clone() * diff
        })
        .fold(T::zero(), |acc, x| acc + x);

    // Divisor depends on bias parameter
    let divisor = if bias {
        T::from_u64(data.len() as u64)
    } else {
        T::from_u64((data.len() - 1) as u64)
    };

    sum_squared_diff.divide(&divisor)
}

/// Trait for types that support square root operation
///
/// This trait is needed for computing standard deviation, which requires
/// taking the square root of the variance.
pub trait SquareRoot: Sized {
    /// Compute the square root
    ///
    /// Returns `None` if the square root cannot be computed (e.g., negative number)
    fn sqrt(&self) -> Option<Self>;
}

impl SquareRoot for f64 {
    fn sqrt(&self) -> Option<Self> {
        if *self < 0.0 {
            None
        } else {
            Some(f64::sqrt(*self))
        }
    }
}

/// Compute the standard deviation of a dataset
///
/// Standard deviation is the square root of the variance.
/// It measures spread in the same units as the original data.
///
/// # Arguments
///
/// - `data`: The dataset
/// - `bias`: Controls the divisor (same as in `variance`)
///   - `false`: Sample standard deviation (divides variance by n-1)
///   - `true`: Population standard deviation (divides variance by n)
///
/// # Type Constraints
///
/// - Requires `Field`, `NumericConversion`, and `SquareRoot`
/// - Currently works for f64; Rational would need approximation
///
/// # Examples
///
/// ```no_run
/// use rustmath_stats::basic_stats::std;
///
/// // Note: f64 doesn't implement required traits in this project yet.
/// // This example is for illustration only.
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
///
/// // Sample standard deviation
/// let std_sample = std(&data, false).unwrap();
/// assert!((std_sample - 1.5811).abs() < 0.001);
///
/// // Population standard deviation
/// let std_pop = std(&data, true).unwrap();
/// assert!((std_pop - 1.4142).abs() < 0.001);
/// ```
pub fn std<T>(data: &[T], bias: bool) -> Result<T>
where
    T: Field + NumericConversion + SquareRoot,
{
    let var = variance(data, bias)?;
    var.sqrt().ok_or(MathError::InvalidArgument(
        "Cannot compute square root of variance".to_string()
    ))
}

/// Compute the moving average of a dataset
///
/// Returns a vector where each element is the average of a window of `n` consecutive
/// elements from the input. The output length is `data.len() - n + 1`.
///
/// # Arguments
///
/// - `data`: The dataset
/// - `n`: Window size (must be > 0 and <= data.len())
///
/// # Type Constraints
///
/// - Generic over any `Field` type
/// - Requires `NumericConversion` for division
///
/// # Examples
///
/// ```
/// use rustmath_stats::basic_stats::moving_average;
/// use rustmath_rationals::Rational;
///
/// let data = vec![
///     Rational::new(1, 1).unwrap(),
///     Rational::new(2, 1).unwrap(),
///     Rational::new(3, 1).unwrap(),
///     Rational::new(4, 1).unwrap(),
///     Rational::new(5, 1).unwrap(),
/// ];
///
/// let ma = moving_average(&data, 3).unwrap();
/// assert_eq!(ma[0], Rational::new(2, 1).unwrap()); // (1+2+3)/3 = 2
/// assert_eq!(ma[1], Rational::new(3, 1).unwrap()); // (2+3+4)/3 = 3
/// assert_eq!(ma[2], Rational::new(4, 1).unwrap()); // (3+4+5)/3 = 4
/// ```
pub fn moving_average<T>(data: &[T], n: usize) -> Result<Vec<T>>
where
    T: Field + NumericConversion,
{
    if n == 0 {
        return Err(MathError::InvalidArgument(
            "Window size must be greater than 0".to_string()
        ));
    }

    if n > data.len() {
        return Err(MathError::InvalidArgument(
            format!("Window size {} exceeds data length {}", n, data.len())
        ));
    }

    let window_count = T::from_u64(n as u64);
    let mut result = Vec::with_capacity(data.len() - n + 1);

    // Compute first window sum
    let mut window_sum = data[..n].iter()
        .fold(T::zero(), |acc, x| acc + x.clone());

    result.push(window_sum.clone().divide(&window_count)?);

    // Slide the window: subtract first element, add new element
    for i in n..data.len() {
        window_sum = window_sum - data[i - n].clone() + data[i].clone();
        result.push(window_sum.clone().divide(&window_count)?);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    // === Mean Tests ===

    #[test]
    fn test_mean_rational() {
        let data = vec![
            Rational::new(1, 2).unwrap(),
            Rational::new(3, 4).unwrap(),
            Rational::new(5, 4).unwrap(),
        ];
        let m = mean(&data).unwrap();
        // (1/2 + 3/4 + 5/4) / 3 = (2/4 + 3/4 + 5/4) / 3 = (10/4) / 3 = 10/12 = 5/6
        assert_eq!(m, Rational::new(5, 6).unwrap());
    }


    #[test]
    fn test_mean_empty() {
        let data: Vec<Rational> = vec![];
        assert!(mean(&data).is_err());
    }

    // === Median Tests ===

    #[test]
    fn test_median_rational_odd() {
        let data = vec![
            Rational::new(1, 1).unwrap(),
            Rational::new(3, 1).unwrap(),
            Rational::new(5, 1).unwrap(),
        ];
        let m = median(&data).unwrap();
        assert_eq!(m, Rational::new(3, 1).unwrap());
    }

    #[test]
    fn test_median_rational_even() {
        let data = vec![
            Rational::new(1, 1).unwrap(),
            Rational::new(2, 1).unwrap(),
            Rational::new(3, 1).unwrap(),
            Rational::new(4, 1).unwrap(),
        ];
        let m = median(&data).unwrap();
        // (2 + 3) / 2 = 5/2
        assert_eq!(m, Rational::new(5, 2).unwrap());
    }




    // === Mode Tests ===

    #[test]
    fn test_mode_rational() {
        let data = vec![
            Rational::new(1, 1).unwrap(),
            Rational::new(2, 1).unwrap(),
            Rational::new(2, 1).unwrap(),
            Rational::new(3, 1).unwrap(),
        ];
        let m = mode(&data).unwrap();
        assert_eq!(m, Rational::new(2, 1).unwrap());
    }


    #[test]
    fn test_mode_single_element() {
        let data = vec![Rational::new(5, 1).unwrap()];
        let m = mode(&data).unwrap();
        assert_eq!(m, Rational::new(5, 1).unwrap());
    }

    // === Variance Tests ===

    #[test]
    fn test_variance_rational_unbiased() {
        let data = vec![
            Rational::new(1, 1).unwrap(),
            Rational::new(2, 1).unwrap(),
            Rational::new(3, 1).unwrap(),
            Rational::new(4, 1).unwrap(),
            Rational::new(5, 1).unwrap(),
        ];

        // Mean = 3
        // Deviations: -2, -1, 0, 1, 2
        // Squared: 4, 1, 0, 1, 4
        // Sum: 10
        // Sample variance: 10/4 = 5/2
        let var = variance(&data, false).unwrap();
        assert_eq!(var, Rational::new(5, 2).unwrap());
    }

    #[test]
    fn test_variance_rational_biased() {
        let data = vec![
            Rational::new(1, 1).unwrap(),
            Rational::new(2, 1).unwrap(),
            Rational::new(3, 1).unwrap(),
            Rational::new(4, 1).unwrap(),
            Rational::new(5, 1).unwrap(),
        ];

        // Population variance: 10/5 = 2
        let var = variance(&data, true).unwrap();
        assert_eq!(var, Rational::new(2, 1).unwrap());
    }



    #[test]
    fn test_variance_insufficient_data_unbiased() {
        let data = vec![Rational::new(1, 1).unwrap()];
        assert!(variance(&data, false).is_err());
    }

    #[test]
    fn test_variance_single_element_biased() {
        let data = vec![Rational::new(5, 1).unwrap()];
        let var = variance(&data, true).unwrap();
        assert_eq!(var, Rational::new(0, 1).unwrap());
    }

    // === Standard Deviation Tests ===




    // === Moving Average Tests ===

    #[test]
    fn test_moving_average_rational() {
        let data = vec![
            Rational::new(1, 1).unwrap(),
            Rational::new(2, 1).unwrap(),
            Rational::new(3, 1).unwrap(),
            Rational::new(4, 1).unwrap(),
            Rational::new(5, 1).unwrap(),
        ];

        let ma = moving_average(&data, 3).unwrap();
        assert_eq!(ma.len(), 3);
        assert_eq!(ma[0], Rational::new(2, 1).unwrap()); // (1+2+3)/3 = 2
        assert_eq!(ma[1], Rational::new(3, 1).unwrap()); // (2+3+4)/3 = 3
        assert_eq!(ma[2], Rational::new(4, 1).unwrap()); // (3+4+5)/3 = 4
    }


    #[test]
    fn test_moving_average_window_size_equals_length() {
        let data = vec![
            Rational::new(1, 1).unwrap(),
            Rational::new(2, 1).unwrap(),
            Rational::new(3, 1).unwrap(),
        ];

        let ma = moving_average(&data, 3).unwrap();
        assert_eq!(ma.len(), 1);
        assert_eq!(ma[0], Rational::new(2, 1).unwrap()); // Mean of all
    }




    // === Additional Edge Case Tests ===

    #[test]
    fn test_mean_single_element() {
        let data = vec![Rational::new(7, 3).unwrap()];
        let m = mean(&data).unwrap();
        assert_eq!(m, Rational::new(7, 3).unwrap());
    }

    #[test]
    fn test_median_single_element() {
        let data = vec![Rational::new(42, 1).unwrap()];
        let m = median(&data).unwrap();
        assert_eq!(m, Rational::new(42, 1).unwrap());
    }


    #[test]
    fn test_moving_average_fractional_result() {
        let data = vec![
            Rational::new(1, 1).unwrap(),
            Rational::new(2, 1).unwrap(),
            Rational::new(4, 1).unwrap(),
        ];

        let ma = moving_average(&data, 2).unwrap();
        assert_eq!(ma[0], Rational::new(3, 2).unwrap()); // (1+2)/2 = 3/2
        assert_eq!(ma[1], Rational::new(3, 1).unwrap()); // (2+4)/2 = 3
    }
}
