//! Mathematical tables for trigonometric and logarithmic values
//!
//! This module provides precomputed tables for common mathematical functions
//! to allow fast approximation lookups.

use once_cell::sync::Lazy;
use std::f64::consts::PI;

/// Trigonometric table with precomputed sin/cos values
pub struct TrigTable {
    /// Number of entries in the table (for 0 to 2π)
    resolution: usize,
    /// Precomputed sine values
    sin_values: Vec<f64>,
    /// Precomputed cosine values
    cos_values: Vec<f64>,
}

impl TrigTable {
    /// Create a new trigonometric table with the given resolution
    pub fn new(resolution: usize) -> Self {
        let mut sin_values = Vec::with_capacity(resolution);
        let mut cos_values = Vec::with_capacity(resolution);

        for i in 0..resolution {
            let angle = 2.0 * PI * (i as f64) / (resolution as f64);
            sin_values.push(angle.sin());
            cos_values.push(angle.cos());
        }

        TrigTable {
            resolution,
            sin_values,
            cos_values,
        }
    }

    /// Get sine value for the given angle (in radians)
    pub fn sin(&self, angle: f64) -> f64 {
        let normalized = angle.rem_euclid(2.0 * PI);
        let index = ((normalized / (2.0 * PI)) * self.resolution as f64) as usize;
        let index = index.min(self.resolution - 1);
        self.sin_values[index]
    }

    /// Get cosine value for the given angle (in radians)
    pub fn cos(&self, angle: f64) -> f64 {
        let normalized = angle.rem_euclid(2.0 * PI);
        let index = ((normalized / (2.0 * PI)) * self.resolution as f64) as usize;
        let index = index.min(self.resolution - 1);
        self.cos_values[index]
    }

    /// Get tangent value for the given angle (in radians)
    pub fn tan(&self, angle: f64) -> f64 {
        let s = self.sin(angle);
        let c = self.cos(angle);
        if c.abs() < 1e-10 {
            f64::INFINITY
        } else {
            s / c
        }
    }
}

/// Logarithm table with precomputed natural log values
pub struct LogTable {
    /// Base value (start of range)
    base: f64,
    /// Maximum value (end of range)
    max: f64,
    /// Number of entries
    resolution: usize,
    /// Precomputed log values
    log_values: Vec<f64>,
}

impl LogTable {
    /// Create a new logarithm table for the range [base, max]
    pub fn new(base: f64, max: f64, resolution: usize) -> Self {
        let mut log_values = Vec::with_capacity(resolution);
        let step = (max - base) / resolution as f64;

        for i in 0..resolution {
            let x = base + step * i as f64;
            log_values.push(x.ln());
        }

        LogTable {
            base,
            max,
            resolution,
            log_values,
        }
    }

    /// Get natural logarithm approximation for the given value
    pub fn ln(&self, x: f64) -> Option<f64> {
        if x < self.base || x > self.max {
            return None;
        }

        let ratio = (x - self.base) / (self.max - self.base);
        let index = (ratio * self.resolution as f64) as usize;
        let index = index.min(self.resolution - 1);

        Some(self.log_values[index])
    }

    /// Get base-10 logarithm approximation
    pub fn log10(&self, x: f64) -> Option<f64> {
        self.ln(x).map(|ln_x| ln_x / 10f64.ln())
    }

    /// Get logarithm with arbitrary base
    pub fn log(&self, x: f64, base: f64) -> Option<f64> {
        self.ln(x).map(|ln_x| ln_x / base.ln())
    }
}

/// Global trigonometric table (1000 entries)
static TRIG_TABLE: Lazy<TrigTable> = Lazy::new(|| TrigTable::new(1000));

/// Global logarithm table for range [1, 1000] (10000 entries)
static LOG_TABLE: Lazy<LogTable> = Lazy::new(|| LogTable::new(1.0, 1000.0, 10000));

/// Get sine approximation from the global table
pub fn get_sin_approx(angle: f64) -> f64 {
    TRIG_TABLE.sin(angle)
}

/// Get cosine approximation from the global table
pub fn get_cos_approx(angle: f64) -> f64 {
    TRIG_TABLE.cos(angle)
}

/// Get tangent approximation from the global table
pub fn get_tan_approx(angle: f64) -> f64 {
    TRIG_TABLE.tan(angle)
}

/// Get natural logarithm approximation from the global table
pub fn get_log_approx(x: f64) -> Option<f64> {
    LOG_TABLE.ln(x)
}

/// Get base-10 logarithm approximation from the global table
pub fn get_log10_approx(x: f64) -> Option<f64> {
    LOG_TABLE.log10(x)
}

/// Power table for small integer powers
pub struct PowerTable {
    /// Maximum base
    max_base: usize,
    /// Maximum exponent
    max_exp: usize,
    /// Precomputed powers
    powers: Vec<Vec<rustmath_integers::Integer>>,
}

impl PowerTable {
    /// Create a new power table
    pub fn new(max_base: usize, max_exp: usize) -> Self {
        use rustmath_integers::Integer;

        let mut powers = Vec::with_capacity(max_base + 1);

        for base in 0..=max_base {
            let mut base_powers = Vec::with_capacity(max_exp + 1);
            let base_int = Integer::from(base as u64);

            for exp in 0..=max_exp {
                let power = if exp == 0 {
                    Integer::one()
                } else {
                    let mut result = Integer::one();
                    for _ in 0..exp {
                        result = &result * &base_int;
                    }
                    result
                };
                base_powers.push(power);
            }

            powers.push(base_powers);
        }

        PowerTable {
            max_base,
            max_exp,
            powers,
        }
    }

    /// Get precomputed power base^exp
    pub fn get(&self, base: usize, exp: usize) -> Option<&rustmath_integers::Integer> {
        if base > self.max_base || exp > self.max_exp {
            return None;
        }
        Some(&self.powers[base][exp])
    }
}

/// Global power table (bases 0-100, exponents 0-20)
static POWER_TABLE: Lazy<PowerTable> = Lazy::new(|| PowerTable::new(100, 20));

/// Get precomputed power from the global table
pub fn get_power(base: usize, exp: usize) -> Option<&'static rustmath_integers::Integer> {
    POWER_TABLE.get(base, exp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trig_table() {
        let table = TrigTable::new(1000);

        // Test some known values
        let sin_0 = table.sin(0.0);
        assert!((sin_0 - 0.0).abs() < 0.01);

        let sin_pi_2 = table.sin(PI / 2.0);
        assert!((sin_pi_2 - 1.0).abs() < 0.01);

        let cos_0 = table.cos(0.0);
        assert!((cos_0 - 1.0).abs() < 0.01);

        let cos_pi = table.cos(PI);
        assert!((cos_pi - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_log_table() {
        let table = LogTable::new(1.0, 1000.0, 10000);

        let ln_1 = table.ln(1.0).unwrap();
        assert!((ln_1 - 0.0).abs() < 0.01);

        let ln_e = table.ln(std::f64::consts::E).unwrap();
        assert!((ln_e - 1.0).abs() < 0.01);

        let ln_10 = table.ln(10.0).unwrap();
        assert!((ln_10 - 10f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_global_trig_approx() {
        let sin_0 = get_sin_approx(0.0);
        assert!((sin_0 - 0.0).abs() < 0.01);

        let cos_0 = get_cos_approx(0.0);
        assert!((cos_0 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_global_log_approx() {
        let ln_1 = get_log_approx(1.0).unwrap();
        assert!((ln_1 - 0.0).abs() < 0.01);

        let log10_100 = get_log10_approx(100.0).unwrap();
        assert!((log10_100 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_power_table() {
        use rustmath_integers::Integer;

        let table = PowerTable::new(10, 5);

        assert_eq!(table.get(2, 0), Some(&Integer::from(1)));
        assert_eq!(table.get(2, 1), Some(&Integer::from(2)));
        assert_eq!(table.get(2, 2), Some(&Integer::from(4)));
        assert_eq!(table.get(2, 3), Some(&Integer::from(8)));
        assert_eq!(table.get(10, 2), Some(&Integer::from(100)));
    }

    #[test]
    fn test_global_power_table() {
        use rustmath_integers::Integer;

        assert_eq!(get_power(2, 10), Some(&Integer::from(1024)));
        assert_eq!(get_power(10, 3), Some(&Integer::from(1000)));
        assert!(get_power(1000, 100).is_none()); // Out of range
    }
}
