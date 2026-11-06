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
use rustmath_rationals::Rational;
use rustmath_reals::Real;
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

/// Floating-point square root of a `Real` (double precision).
///
/// Returns `None` for negative or NaN input.
impl SquareRoot for Real {
    fn sqrt(&self) -> Option<Self> {
        let v = self.to_f64();
        if v.is_nan() || v < 0.0 {
            None
        } else {
            Some(Real::sqrt(self))
        }
    }
}

/// Exact square root in Q.
///
/// Returns `Some(r)` with `r >= 0` and `r * r == self` when `self` is the square of a
/// rational number, and `None` otherwise (negative input, or a square root that is
/// irrational such as `sqrt(2)`). No approximation is ever returned.
impl SquareRoot for Rational {
    fn sqrt(&self) -> Option<Self> {
        // Normalise to lowest terms with a positive denominator; then p/q is a square in Q
        // if and only if p and q are both perfect squares in Z.
        let r = Rational::new(self.numerator().clone(), self.denominator().clone()).ok()?;
        let (p, q) = (r.numerator(), r.denominator());
        if p.signum() < 0 {
            return None;
        }
        let a = p.sqrt().ok()?;
        let b = q.sqrt().ok()?;
        if a.clone() * a.clone() == *p && b.clone() * b.clone() == *q {
            Rational::new(a, b).ok()
        } else {
            None
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
/// - Implemented for `Real` (floating point) and `Rational` (exact). For `Rational`
///   the result is `Err` unless the variance is the square of a rational number:
///   an irrational standard deviation is never approximated.
///
/// # Examples
///
/// ```
/// use rustmath_stats::basic_stats::std;
/// use rustmath_reals::Real;
/// use rustmath_rationals::Rational;
///
/// // Floating point: data 1..5 has sample variance 5/2 and population variance 2.
/// let data: Vec<Real> = (1..=5).map(|i| Real::from(i as f64)).collect();
///
/// // Sample standard deviation sqrt(5/2) = 1.5811388300841898...
/// let std_sample = std(&data, false).unwrap();
/// assert!((std_sample.to_f64() - 1.5811388300841898).abs() < 1e-12);
///
/// // Population standard deviation sqrt(2) = 1.4142135623730951...
/// let std_pop = std(&data, true).unwrap();
/// assert!((std_pop.to_f64() - 1.4142135623730951).abs() < 1e-12);
///
/// // Exact: {2, 4, 4, 4, 5, 5, 7, 9} has mean 5 and population variance 32/8 = 4.
/// let exact: Vec<Rational> = [2, 4, 4, 4, 5, 5, 7, 9].iter().map(|&n| Rational::from(n)).collect();
/// assert_eq!(std(&exact, true).unwrap(), Rational::from(2));
///
/// // Its sample variance 32/7 is not a square in Q, so no exact std exists.
/// assert!(std(&exact, false).is_err());
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

    #[test]
    fn test_rational_sqrt_exact() {
        let r = |n: i64, d: i64| Rational::new(n, d).unwrap();
        assert_eq!(SquareRoot::sqrt(&r(9, 4)), Some(r(3, 2)));
        assert_eq!(SquareRoot::sqrt(&r(0, 1)), Some(r(0, 1)));
        assert_eq!(SquareRoot::sqrt(&r(1, 1)), Some(r(1, 1)));
        assert_eq!(SquareRoot::sqrt(&r(144, 25)), Some(r(12, 5)));
        // Irrational or negative: no rational square root
        assert_eq!(SquareRoot::sqrt(&r(2, 1)), None);
        assert_eq!(SquareRoot::sqrt(&r(4, 3)), None);
        assert_eq!(SquareRoot::sqrt(&r(-4, 1)), None);
    }

    #[test]
    fn test_real_sqrt() {
        assert_eq!(SquareRoot::sqrt(&Real::from(16.0)).map(|x| x.to_f64()), Some(4.0));
        assert!(SquareRoot::sqrt(&Real::from(-1.0)).is_none());
        assert!(SquareRoot::sqrt(&Real::from(f64::NAN)).is_none());
    }

    #[test]
    fn test_std_rational_exact() {
        // {2,4,4,4,5,5,7,9}: mean 5, sum of squared deviations 32, population var 4.
        let data: Vec<Rational> =
            [2i64, 4, 4, 4, 5, 5, 7, 9].iter().map(|&n| Rational::from(n)).collect();
        assert_eq!(std(&data, true).unwrap(), Rational::from(2));
        // Sample variance 32/7 is not a rational square.
        assert!(std(&data, false).is_err());

        // {1,2,3}: sample variance 1, so sample std is exactly 1.
        let data: Vec<Rational> = (1i64..=3).map(Rational::from).collect();
        assert_eq!(std(&data, false).unwrap(), Rational::from(1));
    }

    #[test]
    fn test_std_real() {
        // {1,...,5}: sample var 5/2, population var 2.
        let data: Vec<Real> = (1..=5).map(|i| Real::from(i as f64)).collect();
        let s = std(&data, false).unwrap().to_f64();
        assert!((s - 2.5f64.sqrt()).abs() < 1e-15);
        let p = std(&data, true).unwrap().to_f64();
        assert!((p - 2f64.sqrt()).abs() < 1e-15);
        // Too few points for a sample std
        assert!(std(&data[..1], false).is_err());
    }

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
