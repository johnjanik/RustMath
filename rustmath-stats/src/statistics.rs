//! Statistical functions
//!
//! Common statistical measures including mean, variance, standard deviation, correlation, etc.

use std::collections::HashMap;

/// Compute the arithmetic mean of a dataset
pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    Some(data.iter().sum::<f64>() / data.len() as f64)
}

/// Compute the variance of a dataset
///
/// Uses the unbiased estimator (divides by n-1)
pub fn variance(data: &[f64]) -> Option<f64> {
    if data.len() < 2 {
        return None;
    }

    let m = mean(data)?;
    let sum_squared_diff: f64 = data.iter().map(|x| (x - m).powi(2)).sum();

    Some(sum_squared_diff / (data.len() - 1) as f64)
}

/// Compute the population variance of a dataset
///
/// Uses the biased estimator (divides by n)
pub fn population_variance(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let m = mean(data)?;
    let sum_squared_diff: f64 = data.iter().map(|x| (x - m).powi(2)).sum();

    Some(sum_squared_diff / data.len() as f64)
}

/// Compute the standard deviation of a dataset
pub fn std_dev(data: &[f64]) -> Option<f64> {
    variance(data).map(|v| v.sqrt())
}

/// Compute the population standard deviation of a dataset
pub fn population_std_dev(data: &[f64]) -> Option<f64> {
    population_variance(data).map(|v| v.sqrt())
}

/// Compute the median of a dataset
pub fn median(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some((sorted[mid - 1] + sorted[mid]) / 2.0)
    } else {
        Some(sorted[mid])
    }
}

/// Compute the mode of a dataset
///
/// Returns the most frequently occurring value
pub fn mode(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let mut counts = HashMap::new();
    for &value in data {
        *counts.entry(value.to_bits()).or_insert(0) += 1;
    }

    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(bits, _)| f64::from_bits(bits))
}

/// Compute the covariance between two datasets
pub fn covariance(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;

    let sum: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    Some(sum / (x.len() - 1) as f64)
}

/// Compute the correlation coefficient between two datasets
///
/// Returns Pearson's correlation coefficient in [-1, 1]
pub fn correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let cov = covariance(x, y)?;
    let std_x = std_dev(x)?;
    let std_y = std_dev(y)?;

    if std_x == 0.0 || std_y == 0.0 {
        return None;
    }

    Some(cov / (std_x * std_y))
}

/// Compute the minimum value in a dataset
pub fn min(data: &[f64]) -> Option<f64> {
    data.iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
}

/// Compute the maximum value in a dataset
pub fn max(data: &[f64]) -> Option<f64> {
    data.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
}

/// Compute the range of a dataset (max - min)
pub fn range(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    Some(max(data)? - min(data)?)
}

/// Compute a specific quantile of a dataset
///
/// q should be in [0, 1], where 0.5 is the median
pub fn quantile(data: &[f64], q: f64) -> Option<f64> {
    if data.is_empty() || q < 0.0 || q > 1.0 {
        return None;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = q * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        Some(sorted[lower])
    } else {
        let fraction = index - lower as f64;
        Some(sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction)
    }
}

/// Compute the interquartile range (IQR = Q3 - Q1)
pub fn interquartile_range(data: &[f64]) -> Option<f64> {
    let q1 = quantile(data, 0.25)?;
    let q3 = quantile(data, 0.75)?;
    Some(q3 - q1)
}

/// Compute the skewness of a dataset
///
/// Measures asymmetry of the distribution
pub fn skewness(data: &[f64]) -> Option<f64> {
    if data.len() < 3 {
        return None;
    }

    let m = mean(data)?;
    let s = std_dev(data)?;

    if s == 0.0 {
        return None;
    }

    let n = data.len() as f64;
    let sum: f64 = data.iter().map(|x| ((x - m) / s).powi(3)).sum();

    Some((n / ((n - 1.0) * (n - 2.0))) * sum)
}

/// Compute the kurtosis of a dataset
///
/// Measures "tailedness" of the distribution
pub fn kurtosis(data: &[f64]) -> Option<f64> {
    if data.len() < 4 {
        return None;
    }

    let m = mean(data)?;
    let s = std_dev(data)?;

    if s == 0.0 {
        return None;
    }

    let n = data.len() as f64;
    let sum: f64 = data.iter().map(|x| ((x - m) / s).powi(4)).sum();

    Some((n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * sum
        - (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&data), Some(3.0));

        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn test_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = variance(&data).unwrap();
        // Sample variance ≈ 4.571
        assert!((var - 4.571).abs() < 0.01);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&data).unwrap();
        // Sample std dev ≈ 2.138
        assert!((sd - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_median_odd() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        assert_eq!(median(&data), Some(5.0));
    }

    #[test]
    fn test_median_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&data), Some(2.5));
    }

    #[test]
    fn test_mode() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        assert_eq!(mode(&data), Some(2.0));
    }

    #[test]
    fn test_covariance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let cov = covariance(&x, &y).unwrap();
        assert!((cov - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = correlation(&x, &y).unwrap();
        // Perfect positive correlation
        assert!((corr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(min(&data), Some(1.0));
        assert_eq!(max(&data), Some(5.0));
        assert_eq!(range(&data), Some(4.0));
    }

    #[test]
    fn test_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&data, 0.5), Some(3.0)); // Median
        assert_eq!(quantile(&data, 0.0), Some(1.0)); // Min
        assert_eq!(quantile(&data, 1.0), Some(5.0)); // Max
    }

    #[test]
    fn test_interquartile_range() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let iqr = interquartile_range(&data).unwrap();
        assert!((iqr - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_population_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pop_var = population_variance(&data).unwrap();
        let samp_var = variance(&data).unwrap();
        // Population variance should be smaller
        assert!(pop_var < samp_var);
    }

    #[test]
    fn test_skewness() {
        // Symmetric data should have near-zero skewness
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let skew = skewness(&data).unwrap();
        assert!(skew.abs() < 0.1);
    }

    #[test]
    fn test_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Just test that it computes without panicking
        let _kurt = kurtosis(&data);
    }
}
