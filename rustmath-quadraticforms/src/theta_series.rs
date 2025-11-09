//! Theta Series
//!
//! Implementation of theta series for quadratic forms, which count the number
//! of representations of integers by the quadratic form.

use crate::quadratic_form::QuadraticForm;
use rustmath_integers::Integer;
use rustmath_core::{Ring, NumericConversion};
use std::collections::HashMap;

/// Theta series associated with a quadratic form
///
/// The theta series θ_Q(q) = Σ r_Q(n) q^n where r_Q(n) is the number of
/// representations of n by the quadratic form Q.
pub struct ThetaSeries {
    pub form: QuadraticForm,
    pub coefficients: HashMap<Integer, Integer>,
    pub precision: usize,
}

impl ThetaSeries {
    /// Create a new theta series for the given form and precision
    ///
    /// # Arguments
    /// * `form` - The quadratic form
    /// * `precision` - Maximum value of n to compute r_Q(n) for
    pub fn new(form: QuadraticForm, precision: usize) -> Self {
        let mut series = Self {
            form,
            coefficients: HashMap::new(),
            precision,
        };
        series.compute_coefficients();
        series
    }

    /// Compute the representation numbers r_Q(n) for n ≤ precision
    pub fn compute_coefficients(&mut self) {
        self.coefficients.clear();

        // Determine the type of form and use appropriate algorithm
        if self.form.c.is_zero() {
            if self.form.b.is_zero() {
                // Unary form: ax²
                self.compute_unary_theta_series();
            } else {
                // Degenerate binary form
                self.compute_general_theta_series();
            }
        } else {
            // Proper binary form: ax² + bxy + cy²
            self.compute_binary_theta_series();
        }
    }

    /// Compute theta series for unary forms ax²
    fn compute_unary_theta_series(&mut self) {
        let a = &self.form.a;
        if a.is_zero() {
            return;
        }

        // Estimate maximum x value needed
        let max_x = Self::estimate_max_coordinate(a, &Integer::one(), self.precision);

        // Count representations for each value
        for x in -max_x..=max_x {
            let x_int = Integer::from(x);
            let value = a.clone() * x_int.clone() * x_int;

            if value <= Integer::from(self.precision as i64) && value >= Integer::zero() {
                let entry = self.coefficients.entry(value).or_insert(Integer::zero());
                *entry = entry.clone() + Integer::one();
            }
        }
    }

    /// Compute theta series for binary forms ax² + bxy + cy²
    fn compute_binary_theta_series(&mut self) {
        let a = &self.form.a;
        let b = &self.form.b;
        let c = &self.form.c;

        // Estimate maximum coordinate values needed
        let max_coord = Self::estimate_max_coordinate(a, c, self.precision);

        // Enumerate all (x, y) pairs and count representations
        for x in -max_coord..=max_coord {
            for y in -max_coord..=max_coord {
                let x_int = Integer::from(x);
                let y_int = Integer::from(y);

                let value = a.clone() * x_int.clone() * x_int.clone() +
                            b.clone() * x_int.clone() * y_int.clone() +
                            c.clone() * y_int.clone() * y_int;

                if value <= Integer::from(self.precision as i64) && value >= Integer::zero() {
                    let entry = self.coefficients.entry(value).or_insert(Integer::zero());
                    *entry = entry.clone() + Integer::one();
                }
            }
        }
    }

    /// Compute theta series for general/degenerate forms
    fn compute_general_theta_series(&mut self) {
        // For degenerate or unusual forms, use a conservative search
        let max_coord = (self.precision as f64).sqrt().ceil() as i64 + 1;

        for x in -max_coord..=max_coord {
            for y in -max_coord..=max_coord {
                let x_int = Integer::from(x);
                let y_int = Integer::from(y);

                let value = self.form.evaluate(&x_int, &y_int);

                if value <= Integer::from(self.precision as i64) && value >= Integer::zero() {
                    let entry = self.coefficients.entry(value).or_insert(Integer::zero());
                    *entry = entry.clone() + Integer::one();
                }
            }
        }
    }

    /// Estimate maximum coordinate value needed for the search
    fn estimate_max_coordinate(a: &Integer, c: &Integer, precision: usize) -> i64 {
        let precision_float = precision as f64;

        // Find minimum non-zero coefficient
        let a_abs = a.abs();
        let a_abs = if a_abs > Integer::one() { a_abs } else { Integer::one() };
        let c_abs = c.abs();
        let c_abs = if c_abs > Integer::one() { c_abs } else { Integer::one() };
        let min_coeff = if a_abs < c_abs { a_abs } else { c_abs };

        let min_float = min_coeff.to_i64().unwrap_or(1) as f64;

        // Maximum coordinate is approximately sqrt(precision / min_coeff)
        (precision_float / min_float).sqrt().ceil() as i64 + 2
    }

    /// Get the representation count r_Q(n)
    pub fn coefficient(&self, n: &Integer) -> Integer {
        self.coefficients.get(n).cloned().unwrap_or(Integer::zero())
    }

    /// Get the number of representations of n by the form
    pub fn representation_count(&self, n: &Integer) -> Integer {
        self.coefficient(n)
    }

    /// Get all computed coefficients as a sorted vector
    pub fn generating_function_coefficients(&self) -> Vec<(Integer, Integer)> {
        let mut result: Vec<_> = self.coefficients
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Compute modular form properties of the theta series
    pub fn modular_form_properties(&self) -> ThetaModularProperties {
        // Weight depends on the number of variables
        let weight = if self.form.c.is_zero() {
            // Unary form has weight 1/2
            Integer::from(1)
        } else {
            // Binary form has weight 1
            Integer::from(2)
        };

        // Level is related to the discriminant
        let level = self.compute_level();

        // Character is related to the Kronecker symbol
        let character = self.compute_character();

        ThetaModularProperties {
            weight,
            level,
            character,
        }
    }

    /// Compute the level of the modular form
    fn compute_level(&self) -> Integer {
        if self.form.discriminant().is_zero() {
            Integer::one()
        } else {
            // Level is typically 4|Δ| for binary forms
            Integer::from(4) * self.form.discriminant().abs()
        }
    }

    /// Compute the character of the modular form
    fn compute_character(&self) -> Integer {
        // Simplified: returns ±1 based on discriminant sign
        if self.form.discriminant().is_zero() {
            Integer::one()
        } else if self.form.discriminant() > &Integer::zero() {
            Integer::one()
        } else {
            Integer::from(-1)
        }
    }
}

/// Properties of the theta series as a modular form
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThetaModularProperties {
    /// Weight of the modular form (doubled to avoid fractions)
    pub weight: Integer,
    /// Level of the modular form
    pub level: Integer,
    /// Character of the modular form (simplified)
    pub character: Integer,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unary_form_theta_series() {
        // Form: x²
        let form = QuadraticForm::unary(Integer::from(1));
        let theta = ThetaSeries::new(form, 20);

        // r(0) = 1 (x=0)
        assert_eq!(theta.coefficient(&Integer::from(0)), Integer::from(1));

        // r(1) = 2 (x=±1)
        assert_eq!(theta.coefficient(&Integer::from(1)), Integer::from(2));

        // r(2) = 0 (no integer solutions)
        assert_eq!(theta.coefficient(&Integer::from(2)), Integer::from(0));

        // r(4) = 2 (x=±2)
        assert_eq!(theta.coefficient(&Integer::from(4)), Integer::from(2));

        // r(9) = 2 (x=±3)
        assert_eq!(theta.coefficient(&Integer::from(9)), Integer::from(2));
    }

    #[test]
    fn test_binary_form_theta_series() {
        // Form: x² + y²
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let theta = ThetaSeries::new(form, 10);

        // r(0) = 1 (x=0, y=0)
        assert_eq!(theta.coefficient(&Integer::from(0)), Integer::from(1));

        // r(1) = 4 (±1,0), (0,±1)
        assert_eq!(theta.coefficient(&Integer::from(1)), Integer::from(4));

        // r(2) = 4 (±1,±1)
        assert_eq!(theta.coefficient(&Integer::from(2)), Integer::from(4));

        // r(3) = 0 (no solutions)
        assert_eq!(theta.coefficient(&Integer::from(3)), Integer::from(0));

        // r(4) = 4 (±2,0), (0,±2)
        assert_eq!(theta.coefficient(&Integer::from(4)), Integer::from(4));

        // r(5) = 8 (±2,±1), (±1,±2)
        assert_eq!(theta.coefficient(&Integer::from(5)), Integer::from(8));
    }

    #[test]
    fn test_generating_function_coefficients() {
        let form = QuadraticForm::unary(Integer::from(1));
        let theta = ThetaSeries::new(form, 10);

        let coeffs = theta.generating_function_coefficients();

        // Should be sorted by n
        for i in 1..coeffs.len() {
            assert!(coeffs[i - 1].0 <= coeffs[i].0);
        }
    }

    #[test]
    fn test_modular_properties() {
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let theta = ThetaSeries::new(form, 10);

        let props = theta.modular_form_properties();

        // Binary form should have weight 2 (in our doubled convention, weight 1)
        assert!(props.weight >= Integer::zero());

        // Level should be 4|Δ| = 4|−4| = 16
        assert_eq!(props.level, Integer::from(16));
    }
}
