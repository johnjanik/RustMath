//! # Big-O Notation
//!
//! This module provides the `O()` notation for various mathematical structures,
//! including power series, p-adic numbers, polynomials, and asymptotic expansions.
//!
//! ## Overview
//!
//! Big-O notation is used to indicate truncation or precision in various contexts:
//!
//! - **Power Series**: `O(x^5)` indicates terms of degree 5 and higher are omitted
//! - **p-adic Numbers**: `O(7^6)` indicates 7-adic precision 6
//! - **Polynomials**: Converted to power series automatically
//! - **Asymptotic Expansions**: Growth bounds in asymptotic analysis
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::big_oh::{BigO, BigOTerm};
//!
//! // Power series truncation
//! let truncation = BigO::PowerSeries {
//!     variable: "x".to_string(),
//!     degree: 5,
//! };
//!
//! // p-adic precision
//! let padic = BigO::PAdicPrecision {
//!     prime: 7,
//!     precision: 6,
//! };
//! ```

use std::fmt;

/// Represents a Big-O term indicating truncation or precision
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BigO {
    /// Power series truncation: O(x^n) or O(t^n)
    PowerSeries {
        /// Variable name (e.g., "x", "t")
        variable: String,
        /// Degree of truncation
        degree: i64,
    },

    /// p-adic number precision: O(p^n)
    PAdicPrecision {
        /// Prime base
        prime: i64,
        /// Precision (can be negative for p-adic fields)
        precision: i64,
    },

    /// Puiseux series with fractional exponent: O(x^(a/b))
    PuiseuxSeries {
        /// Variable name
        variable: String,
        /// Numerator of exponent
        numerator: i64,
        /// Denominator of exponent
        denominator: i64,
    },

    /// Asymptotic expansion bound
    Asymptotic {
        /// Growth description (e.g., "n^2", "log(n)")
        growth: String,
    },

    /// Numeric precision (for general use)
    Numeric {
        /// Precision level
        precision: usize,
    },
}

impl BigO {
    /// Create a power series truncation O(var^deg)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::big_oh::BigO;
    ///
    /// let o = BigO::power_series("x", 5);
    /// assert_eq!(format!("{}", o), "O(x^5)");
    /// ```
    pub fn power_series(variable: impl Into<String>, degree: i64) -> Self {
        BigO::PowerSeries {
            variable: variable.into(),
            degree,
        }
    }

    /// Create a p-adic precision indicator O(p^prec)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::big_oh::BigO;
    ///
    /// let o = BigO::padic(7, 6);
    /// assert_eq!(format!("{}", o), "O(7^6)");
    /// ```
    pub fn padic(prime: i64, precision: i64) -> Self {
        BigO::PAdicPrecision { prime, precision }
    }

    /// Create a Puiseux series truncation O(var^(num/den))
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::big_oh::BigO;
    ///
    /// let o = BigO::puiseux("y", 1, 3);
    /// assert_eq!(format!("{}", o), "O(y^(1/3))");
    /// ```
    pub fn puiseux(variable: impl Into<String>, numerator: i64, denominator: i64) -> Self {
        BigO::PuiseuxSeries {
            variable: variable.into(),
            numerator,
            denominator,
        }
    }

    /// Create an asymptotic bound O(growth)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::big_oh::BigO;
    ///
    /// let o = BigO::asymptotic("n^2");
    /// assert_eq!(format!("{}", o), "O(n^2)");
    /// ```
    pub fn asymptotic(growth: impl Into<String>) -> Self {
        BigO::Asymptotic {
            growth: growth.into(),
        }
    }

    /// Create a numeric precision indicator
    pub fn numeric(precision: usize) -> Self {
        BigO::Numeric { precision }
    }

    /// Get the variable name if this is a series-based Big-O
    pub fn variable(&self) -> Option<&str> {
        match self {
            BigO::PowerSeries { variable, .. } => Some(variable),
            BigO::PuiseuxSeries { variable, .. } => Some(variable),
            _ => None,
        }
    }

    /// Get the degree/precision value
    pub fn precision(&self) -> Option<i64> {
        match self {
            BigO::PowerSeries { degree, .. } => Some(*degree),
            BigO::PAdicPrecision { precision, .. } => Some(*precision),
            BigO::Numeric { precision } => Some(*precision as i64),
            _ => None,
        }
    }
}

impl fmt::Display for BigO {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BigO::PowerSeries { variable, degree } => {
                if *degree == 1 {
                    write!(f, "O({})", variable)
                } else {
                    write!(f, "O({}^{})", variable, degree)
                }
            }
            BigO::PAdicPrecision { prime, precision } => {
                write!(f, "O({}^{})", prime, precision)
            }
            BigO::PuiseuxSeries {
                variable,
                numerator,
                denominator,
            } => {
                if *denominator == 1 {
                    write!(f, "O({}^{})", variable, numerator)
                } else {
                    write!(f, "O({}^({}/{}))", variable, numerator, denominator)
                }
            }
            BigO::Asymptotic { growth } => {
                write!(f, "O({})", growth)
            }
            BigO::Numeric { precision } => {
                write!(f, "O(10^-{})", precision)
            }
        }
    }
}

/// Convenience function to create Big-O notation from various inputs
///
/// This function provides a flexible interface similar to SageMath's O() function.
///
/// # Examples
///
/// ```rust
/// use rustmath_rings::big_oh::O;
///
/// // Power series: O(x^5)
/// let o1 = O("x^5");
///
/// // p-adic: O(7^6)
/// let o2 = O("7^6");
///
/// // Asymptotic: O(n^2)
/// let o3 = O("n^2");
/// ```
pub fn O(input: &str) -> Option<BigO> {
    // Try to parse as var^exp or p^exp
    if let Some((base, exp_str)) = input.split_once('^') {
        // Check if exponent has fraction
        if exp_str.contains('/') {
            let parts: Vec<&str> = exp_str.split('/').collect();
            if parts.len() == 2 {
                if let (Ok(num), Ok(den)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>()) {
                    return Some(BigO::puiseux(base, num, den));
                }
            }
        }

        // Try numeric exponent
        if let Ok(exp) = exp_str.parse::<i64>() {
            // Check if base is numeric (p-adic) or symbolic (power series)
            if base.parse::<i64>().is_ok() {
                // Numeric base -> p-adic
                let prime = base.parse::<i64>().unwrap();
                return Some(BigO::padic(prime, exp));
            } else {
                // Symbolic base -> power series
                return Some(BigO::power_series(base, exp));
            }
        }
    }

    // Try just a variable (implies ^1)
    if input.chars().all(|c| c.is_alphabetic() || c == '_') {
        return Some(BigO::power_series(input, 1));
    }

    // Try numeric p-adic like O(7^6)
    if let Ok(num) = input.parse::<i64>() {
        // Treat as O(p^1) where p is the number
        return Some(BigO::padic(num, 1));
    }

    // Fall back to asymptotic
    Some(BigO::asymptotic(input))
}

/// Trait for types that support Big-O truncation
pub trait HasBigO {
    /// Get the Big-O truncation/precision of this value
    fn precision(&self) -> Option<BigO>;

    /// Set a new Big-O truncation/precision
    fn with_precision(&self, big_o: BigO) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_series_display() {
        let o = BigO::power_series("x", 5);
        assert_eq!(format!("{}", o), "O(x^5)");

        let o2 = BigO::power_series("t", 1);
        assert_eq!(format!("{}", o2), "O(t)");
    }

    #[test]
    fn test_padic_display() {
        let o = BigO::padic(7, 6);
        assert_eq!(format!("{}", o), "O(7^6)");

        let o2 = BigO::padic(11, -32);
        assert_eq!(format!("{}", o2), "O(11^-32)");
    }

    #[test]
    fn test_puiseux_display() {
        let o = BigO::puiseux("y", 1, 3);
        assert_eq!(format!("{}", o), "O(y^(1/3))");

        let o2 = BigO::puiseux("z", 5, 2);
        assert_eq!(format!("{}", o2), "O(z^(5/2))");
    }

    #[test]
    fn test_asymptotic_display() {
        let o = BigO::asymptotic("n^2");
        assert_eq!(format!("{}", o), "O(n^2)");

        let o2 = BigO::asymptotic("log(n)");
        assert_eq!(format!("{}", o2), "O(log(n))");
    }

    #[test]
    fn test_o_function_power_series() {
        let o = O("x^5").unwrap();
        assert_eq!(o.variable(), Some("x"));
        assert_eq!(o.precision(), Some(5));
        assert_eq!(format!("{}", o), "O(x^5)");
    }

    #[test]
    fn test_o_function_padic() {
        let o = O("7^6").unwrap();
        assert_eq!(o.precision(), Some(6));
        assert_eq!(format!("{}", o), "O(7^6)");
    }

    #[test]
    fn test_o_function_puiseux() {
        let o = O("y^1/3").unwrap();
        assert_eq!(o.variable(), Some("y"));
        assert_eq!(format!("{}", o), "O(y^(1/3))");
    }

    #[test]
    fn test_o_function_simple_var() {
        let o = O("x").unwrap();
        assert_eq!(o.variable(), Some("x"));
        assert_eq!(o.precision(), Some(1));
        assert_eq!(format!("{}", o), "O(x)");
    }

    #[test]
    fn test_numeric_precision() {
        let o = BigO::numeric(10);
        assert_eq!(o.precision(), Some(10));
        assert_eq!(format!("{}", o), "O(10^-10)");
    }

    #[test]
    fn test_variable_extraction() {
        let o1 = BigO::power_series("x", 5);
        assert_eq!(o1.variable(), Some("x"));

        let o2 = BigO::padic(7, 6);
        assert_eq!(o2.variable(), None);

        let o3 = BigO::puiseux("z", 1, 2);
        assert_eq!(o3.variable(), Some("z"));
    }

    #[test]
    fn test_precision_extraction() {
        let o1 = BigO::power_series("x", 5);
        assert_eq!(o1.precision(), Some(5));

        let o2 = BigO::padic(7, 6);
        assert_eq!(o2.precision(), Some(6));

        let o3 = BigO::asymptotic("n^2");
        assert_eq!(o3.precision(), None);
    }
}
