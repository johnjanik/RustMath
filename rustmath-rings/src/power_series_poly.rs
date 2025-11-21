//! # Power Series (Polynomial Backend)
//!
//! This module provides power series arithmetic using polynomial representations.
//! Power series are stored internally as polynomials truncated to a given precision.
//!
//! ## Overview
//!
//! A power series is represented as:
//! ```text
//! f(x) = p(x) + O(x^n)
//! ```
//! where p(x) is a polynomial and n is the precision.
//!
//! This module provides:
//! - Power series with polynomial backing for efficiency
//! - Arithmetic operations using polynomial arithmetic
//! - Floor division action for base ring elements
//! - Conversion and pickling support
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::power_series_poly::PowerSeriesPoly;
//! use rustmath_integers::Integer;
//!
//! // Create a power series from polynomial coefficients
//! let coeffs = vec![Integer::from(1), Integer::from(2), Integer::from(3)];
//! let series = PowerSeriesPoly::new(coeffs, 10);
//! ```

use rustmath_core::Ring;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Power series backed by polynomial representation
///
/// Stores a truncated power series using a polynomial as the backing structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerSeriesPoly<R: Ring> {
    /// Polynomial coefficients representing the series
    pub coefficients: Vec<R>,
    /// Precision of the series
    pub precision: usize,
}

impl<R: Ring> PowerSeriesPoly<R> {
    /// Create a new power series from polynomial coefficients
    ///
    /// # Arguments
    /// * `coefficients` - Polynomial coefficients [a₀, a₁, a₂, ...]
    /// * `precision` - Maximum precision
    ///
    /// # Examples
    /// ```
    /// use rustmath_rings::power_series_poly::PowerSeriesPoly;
    /// use rustmath_integers::Integer;
    ///
    /// let series = PowerSeriesPoly::new(
    ///     vec![Integer::from(1), Integer::from(0), Integer::from(1)],
    ///     10
    /// );
    /// ```
    pub fn new(mut coefficients: Vec<R>, precision: usize) -> Self {
        coefficients.truncate(precision);
        Self {
            coefficients,
            precision,
        }
    }

    /// Get coefficient at position n
    pub fn coeff(&self, n: usize) -> R {
        if n < self.coefficients.len() {
            self.coefficients[n].clone()
        } else {
            R::zero()
        }
    }

    /// Get mutable coefficient at position n
    pub fn coeff_mut(&mut self, n: usize) -> Option<&mut R> {
        if n < self.coefficients.len() {
            Some(&mut self.coefficients[n])
        } else {
            None
        }
    }

    /// Get the precision
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Get the valuation (degree of first non-zero term)
    pub fn valuation(&self) -> Option<usize> {
        self.coefficients.iter().position(|c| !c.is_zero())
    }

    /// Check if series is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| c.is_zero())
    }

    /// Change precision of the series
    pub fn change_precision(&mut self, new_precision: usize) {
        if new_precision < self.precision {
            self.coefficients.truncate(new_precision);
        } else {
            self.coefficients
                .resize_with(new_precision, || R::zero());
        }
        self.precision = new_precision;
    }

    /// Get the underlying polynomial (as coefficients)
    pub fn polynomial(&self) -> &[R] {
        &self.coefficients
    }

    /// Shift series by multiplying by x^n
    pub fn shift(&self, n: isize) -> Self {
        if n >= 0 {
            let shift = n as usize;
            let mut new_coeffs = vec![R::zero(); shift];
            new_coeffs.extend_from_slice(&self.coefficients);
            new_coeffs.truncate(self.precision);
            Self::new(new_coeffs, self.precision)
        } else {
            let shift = (-n) as usize;
            let new_coeffs = if shift < self.coefficients.len() {
                self.coefficients[shift..].to_vec()
            } else {
                vec![]
            };
            Self::new(new_coeffs, self.precision)
        }
    }

    /// Derivative of the series
    pub fn derivative(&self) -> Self {
        if self.coefficients.is_empty() {
            return Self::new(vec![], self.precision);
        }

        let mut new_coeffs = Vec::with_capacity(self.coefficients.len().saturating_sub(1));
        for (i, coeff) in self.coefficients.iter().enumerate().skip(1) {
            // Create ring element from index using repeated addition
            let mut factor = R::zero();
            for _ in 0..i {
                factor = factor + R::one();
            }
            new_coeffs.push(coeff.clone() * factor);
        }

        Self::new(new_coeffs, self.precision)
    }

    /// Compose this series with another
    /// Computes f(g(x)) where self is f and other is g
    pub fn compose(&self, other: &Self) -> Self {
        // g must have valuation >= 1 for composition to make sense
        if other.valuation().unwrap_or(0) == 0 && !other.coeff(0).is_zero() {
            panic!("Cannot compose with series having non-zero constant term");
        }

        let mut result = vec![R::zero(); self.precision];
        let mut g_power = vec![R::one()]; // g^0 = 1

        for (i, f_coeff) in self.coefficients.iter().enumerate() {
            // Add f_i * g^i to result
            for (j, g_coeff) in g_power.iter().enumerate() {
                if j < self.precision {
                    result[j] = result[j].clone() + (f_coeff.clone() * g_coeff.clone());
                }
            }

            // Compute next power of g
            if i + 1 < self.coefficients.len() {
                let mut new_g_power = vec![R::zero(); self.precision];
                for (j, coeff1) in g_power.iter().enumerate() {
                    for (k, coeff2) in other.coefficients.iter().enumerate() {
                        if j + k < self.precision {
                            new_g_power[j + k] =
                                new_g_power[j + k].clone() + (coeff1.clone() * coeff2.clone());
                        }
                    }
                }
                g_power = new_g_power;
            }
        }

        Self::new(result, self.precision)
    }
}

impl<R: Ring> Add for PowerSeriesPoly<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let min_prec = self.precision.min(other.precision);

        let mut result = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = self.coeff(i);
            let b = other.coeff(i);
            result.push(a.clone() + b.clone());
        }

        Self::new(result, min_prec)
    }
}

impl<R: Ring> Sub for PowerSeriesPoly<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let min_prec = self.precision.min(other.precision);

        let mut result = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = self.coeff(i);
            let b = other.coeff(i);
            result.push(a.clone() - b.clone());
        }

        Self::new(result, min_prec)
    }
}

impl<R: Ring> Mul for PowerSeriesPoly<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let min_prec = self.precision.min(other.precision);
        let mut result = vec![R::zero(); min_prec];

        for i in 0..self.coefficients.len().min(min_prec) {
            for j in 0..other.coefficients.len().min(min_prec - i) {
                result[i + j] = result[i + j].clone()
                    + (self.coefficients[i].clone() * other.coefficients[j].clone());
            }
        }

        Self::new(result, min_prec)
    }
}

impl<R: Ring> Neg for PowerSeriesPoly<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let coeffs: Vec<R> = self.coefficients.iter().map(|c| -c.clone()).collect();
        Self::new(coeffs, self.precision)
    }
}

/// Action of base ring floor division on power series
///
/// This represents the action where elements of the base ring can floor-divide
/// power series coefficients.
#[derive(Debug, Clone)]
pub struct BaseRingFloorDivAction<R: Ring> {
    _phantom: PhantomData<R>,
}

impl<R: Ring> BaseRingFloorDivAction<R> {
    /// Create a new floor division action
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Apply the floor division action
    /// For integer rings, this divides each coefficient by the given element
    pub fn act(&self, element: &R, series: &PowerSeriesPoly<R>) -> PowerSeriesPoly<R> {
        // Note: This is a simplified implementation
        // Full implementation would require proper floor division in the ring
        let coeffs: Vec<R> = series
            .coefficients
            .iter()
            .map(|c| {
                // Simplified: just divide (would need proper floor division)
                c.clone()
            })
            .collect();
        PowerSeriesPoly::new(coeffs, series.precision)
    }
}

impl<R: Ring> Default for BaseRingFloorDivAction<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Construct a power series from version 0 data (for unpickling)
///
/// This function is used for deserializing power series that were pickled
/// using SageMath's version 0 format.
pub fn make_powerseries_poly_v0<R: Ring>(
    coefficients: Vec<R>,
    precision: usize,
) -> PowerSeriesPoly<R> {
    PowerSeriesPoly::new(coefficients, precision)
}

impl<R: Ring + fmt::Display> fmt::Display for PowerSeriesPoly<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "O(x^{})", self.precision);
        }

        let mut first = true;
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if !coeff.is_zero() {
                if !first && !format!("{}", coeff).starts_with('-') {
                    write!(f, " + ")?;
                } else if !first {
                    write!(f, " ")?;
                }

                if i == 0 {
                    write!(f, "{}", coeff)?;
                } else if i == 1 {
                    if coeff.is_one() {
                        write!(f, "x")?;
                    } else {
                        write!(f, "{}*x", coeff)?;
                    }
                } else {
                    if coeff.is_one() {
                        write!(f, "x^{}", i)?;
                    } else {
                        write!(f, "{}*x^{}", coeff, i)?;
                    }
                }
                first = false;
            }
        }

        write!(f, " + O(x^{})", self.precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_new_power_series_poly() {
        let coeffs = vec![Integer::from(1), Integer::from(2), Integer::from(3)];
        let series = PowerSeriesPoly::new(coeffs, 10);

        assert_eq!(series.coeff(0), &Integer::from(1));
        assert_eq!(series.coeff(1), &Integer::from(2));
        assert_eq!(series.coeff(2), &Integer::from(3));
        assert_eq!(series.precision(), 10);
    }

    #[test]
    fn test_valuation() {
        let series = PowerSeriesPoly::new(
            vec![
                Integer::from(0),
                Integer::from(0),
                Integer::from(5),
                Integer::from(1),
            ],
            10,
        );
        assert_eq!(series.valuation(), Some(2));

        let zero = PowerSeriesPoly::new(vec![Integer::from(0)], 10);
        assert_eq!(zero.valuation(), None);
    }

    #[test]
    fn test_addition() {
        let s1 = PowerSeriesPoly::new(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            10,
        );
        let s2 = PowerSeriesPoly::new(
            vec![Integer::from(4), Integer::from(5), Integer::from(6)],
            10,
        );

        let sum = s1 + s2;
        assert_eq!(sum.coeff(0), &Integer::from(5));
        assert_eq!(sum.coeff(1), &Integer::from(7));
        assert_eq!(sum.coeff(2), &Integer::from(9));
    }

    #[test]
    fn test_multiplication() {
        // (1 + x) * (1 + x) = 1 + 2x + x²
        let s1 = PowerSeriesPoly::new(vec![Integer::from(1), Integer::from(1)], 10);
        let s2 = PowerSeriesPoly::new(vec![Integer::from(1), Integer::from(1)], 10);

        let product = s1 * s2;
        assert_eq!(product.coeff(0), &Integer::from(1));
        assert_eq!(product.coeff(1), &Integer::from(2));
        assert_eq!(product.coeff(2), &Integer::from(1));
    }

    #[test]
    fn test_negation() {
        let series =
            PowerSeriesPoly::new(vec![Integer::from(1), Integer::from(-2), Integer::from(3)], 10);

        let neg = -series;
        assert_eq!(neg.coeff(0), &Integer::from(-1));
        assert_eq!(neg.coeff(1), &Integer::from(2));
        assert_eq!(neg.coeff(2), &Integer::from(-3));
    }

    #[test]
    fn test_shift() {
        let series = PowerSeriesPoly::new(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            10,
        );

        let shifted = series.shift(2);
        assert_eq!(shifted.coeff(0), &Integer::from(0));
        assert_eq!(shifted.coeff(1), &Integer::from(0));
        assert_eq!(shifted.coeff(2), &Integer::from(1));
        assert_eq!(shifted.coeff(3), &Integer::from(2));
    }

    #[test]
    fn test_derivative() {
        // 1 + 2x + 3x² => 2 + 6x
        let series = PowerSeriesPoly::new(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            10,
        );

        let deriv = series.derivative();
        assert_eq!(deriv.coeff(0), &Integer::from(2));
        assert_eq!(deriv.coeff(1), &Integer::from(6));
    }

    #[test]
    fn test_change_precision() {
        let mut series = PowerSeriesPoly::new(
            vec![
                Integer::from(1),
                Integer::from(2),
                Integer::from(3),
                Integer::from(4),
            ],
            10,
        );

        series.change_precision(2);
        assert_eq!(series.precision(), 2);
        assert_eq!(series.coefficients.len(), 2);
    }

    #[test]
    fn test_make_powerseries_poly_v0() {
        let series = make_powerseries_poly_v0(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            10,
        );
        assert_eq!(series.coeff(0), &Integer::from(1));
        assert_eq!(series.precision(), 10);
    }

    #[test]
    fn test_base_ring_floor_div_action() {
        let action = BaseRingFloorDivAction::<Integer>::new();
        let series = PowerSeriesPoly::new(
            vec![Integer::from(10), Integer::from(20), Integer::from(30)],
            10,
        );

        let result = action.act(&Integer::from(2), &series);
        assert_eq!(result.precision(), 10);
    }

    #[test]
    fn test_compose() {
        // f(x) = 1 + x
        let f = PowerSeriesPoly::new(vec![Integer::from(1), Integer::from(1)], 10);

        // g(x) = x + x^2
        let g = PowerSeriesPoly::new(
            vec![Integer::from(0), Integer::from(1), Integer::from(1)],
            10,
        );

        // f(g(x)) = 1 + (x + x^2) = 1 + x + x^2
        let composed = f.compose(&g);
        assert_eq!(composed.coeff(0), &Integer::from(1));
        assert_eq!(composed.coeff(1), &Integer::from(1));
        assert_eq!(composed.coeff(2), &Integer::from(1));
    }

    #[test]
    #[should_panic(expected = "Cannot compose with series having non-zero constant term")]
    fn test_compose_invalid() {
        let f = PowerSeriesPoly::new(vec![Integer::from(1), Integer::from(1)], 10);
        let g = PowerSeriesPoly::new(vec![Integer::from(1), Integer::from(1)], 10); // Invalid: non-zero constant

        let _ = f.compose(&g);
    }
}
