//! Hypothesis testing
//!
//! Statistical hypothesis tests including t-tests and chi-squared tests

use crate::statistics::{mean, std_dev, variance};

/// Result of a hypothesis test
#[derive(Clone, Debug)]
pub struct TestResult {
    /// Test statistic value
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Degrees of freedom (if applicable)
    pub df: Option<usize>,
}

/// Perform a one-sample t-test
///
/// Tests whether the mean of the sample differs from a hypothesized value
pub fn one_sample_t_test(data: &[f64], hypothesized_mean: f64) -> Option<TestResult> {
    if data.len() < 2 {
        return None;
    }

    let sample_mean = mean(data)?;
    let sample_std = std_dev(data)?;
    let n = data.len() as f64;

    let t_statistic = (sample_mean - hypothesized_mean) / (sample_std / n.sqrt());
    let df = data.len() - 1;

    // Simplified p-value calculation (two-tailed)
    let p_value = 2.0 * (1.0 - t_cdf(t_statistic.abs(), df));

    Some(TestResult {
        statistic: t_statistic,
        p_value,
        df: Some(df),
    })
}

/// Perform a two-sample t-test (assuming equal variances)
///
/// Tests whether two samples have different means
pub fn two_sample_t_test(data1: &[f64], data2: &[f64]) -> Option<TestResult> {
    if data1.len() < 2 || data2.len() < 2 {
        return None;
    }

    let mean1 = mean(data1)?;
    let mean2 = mean(data2)?;
    let var1 = variance(data1)?;
    let var2 = variance(data2)?;

    let n1 = data1.len() as f64;
    let n2 = data2.len() as f64;

    // Pooled variance
    let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);

    let t_statistic = (mean1 - mean2) / (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();
    let df = (n1 + n2 - 2.0) as usize;

    let p_value = 2.0 * (1.0 - t_cdf(t_statistic.abs(), df));

    Some(TestResult {
        statistic: t_statistic,
        p_value,
        df: Some(df),
    })
}

/// Simplified t-test wrapper
pub fn t_test(sample1: &[f64], sample2: Option<&[f64]>, mu: Option<f64>) -> Option<TestResult> {
    match (sample2, mu) {
        (None, Some(hypothesized)) => one_sample_t_test(sample1, hypothesized),
        (Some(s2), None) => two_sample_t_test(sample1, s2),
        _ => None,
    }
}

/// Perform a chi-squared goodness-of-fit test
///
/// Tests whether observed frequencies match expected frequencies
pub fn chi_squared_test(observed: &[f64], expected: &[f64]) -> Option<TestResult> {
    if observed.len() != expected.len() || observed.is_empty() {
        return None;
    }

    let chi_squared: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(o, e)| {
            if *e == 0.0 {
                0.0
            } else {
                (o - e).powi(2) / e
            }
        })
        .sum();

    let df = observed.len() - 1;

    // Simplified p-value calculation
    let p_value = 1.0 - chi_squared_cdf(chi_squared, df);

    Some(TestResult {
        statistic: chi_squared,
        p_value,
        df: Some(df),
    })
}

// Helper functions for distribution calculations

/// Student's t-distribution CDF (approximation)
fn t_cdf(t: f64, df: usize) -> f64 {
    if df == 0 {
        return 0.5;
    }

    // For large df, approximate with normal distribution
    if df > 30 {
        return normal_cdf(t);
    }

    // Simple approximation using beta function
    let x = df as f64 / (df as f64 + t * t);
    0.5 + 0.5 * (1.0 - x).sqrt() * t.signum()
}

/// Standard normal CDF (approximation)
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function (erf)
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Chi-squared distribution CDF (approximation)
fn chi_squared_cdf(x: f64, df: usize) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // Simple approximation for chi-squared CDF
    let k = df as f64;
    if k > 30.0 {
        // Normal approximation for large df
        return normal_cdf((x - k) / (2.0 * k).sqrt());
    }

    // Very rough approximation
    1.0 - (-x / 2.0).exp()
}

/// Compute confidence interval for a mean
pub fn confidence_interval(data: &[f64], confidence_level: f64) -> Option<(f64, f64)> {
    if data.len() < 2 || confidence_level <= 0.0 || confidence_level >= 1.0 {
        return None;
    }

    let m = mean(data)?;
    let s = std_dev(data)?;
    let n = data.len() as f64;

    // Critical value for t-distribution (simplified)
    let alpha = 1.0 - confidence_level;
    let df = data.len() - 1;
    let t_critical = t_critical_value(alpha / 2.0, df);

    let margin = t_critical * s / n.sqrt();

    Some((m - margin, m + margin))
}

/// Get critical value for t-distribution (approximation)
fn t_critical_value(_alpha: f64, df: usize) -> f64 {
    // Simplified - use normal approximation for df > 30
    if df > 30 {
        // For 95% confidence (alpha=0.025), z ≈ 1.96
        return 1.96;
    }

    // Rough t-values for common cases
    match df {
        1 => 12.71,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        _ => 2.0, // Rough approximation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_sample_t_test() {
        let data = vec![10.0, 12.0, 11.0, 13.0, 12.0];
        let result = one_sample_t_test(&data, 10.0);
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(result.statistic > 0.0); // Mean is above 10
        assert_eq!(result.df, Some(4));
    }

    #[test]
    fn test_two_sample_t_test() {
        let data1 = vec![10.0, 12.0, 11.0, 13.0, 12.0];
        let data2 = vec![8.0, 9.0, 10.0, 8.0, 9.0];
        let result = two_sample_t_test(&data1, &data2);
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(result.statistic > 0.0); // data1 has higher mean
    }

    #[test]
    fn test_t_test_wrapper() {
        let data = vec![10.0, 12.0, 11.0, 13.0, 12.0];

        // One-sample test
        let result1 = t_test(&data, None, Some(10.0));
        assert!(result1.is_some());

        // Two-sample test
        let data2 = vec![8.0, 9.0, 10.0];
        let result2 = t_test(&data, Some(&data2), None);
        assert!(result2.is_some());
    }

    #[test]
    fn test_chi_squared_test() {
        let observed = vec![10.0, 15.0, 12.0, 8.0];
        let expected = vec![11.25, 11.25, 11.25, 11.25];

        let result = chi_squared_test(&observed, &expected);
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(result.statistic >= 0.0);
        assert_eq!(result.df, Some(3));
    }

    #[test]
    fn test_confidence_interval() {
        let data = vec![10.0, 12.0, 11.0, 13.0, 12.0];
        let ci = confidence_interval(&data, 0.95);
        assert!(ci.is_some());

        let (lower, upper) = ci.unwrap();
        let m = mean(&data).unwrap();
        assert!(lower < m);
        assert!(upper > m);
        assert!(lower < upper);
    }

    #[test]
    fn test_invalid_inputs() {
        // Empty data
        assert!(one_sample_t_test(&[], 0.0).is_none());

        // Single point
        assert!(one_sample_t_test(&[1.0], 0.0).is_none());

        // Mismatched lengths
        assert!(chi_squared_test(&[1.0, 2.0], &[1.0, 2.0, 3.0]).is_none());
    }
}
