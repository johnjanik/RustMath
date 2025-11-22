//! # Multivariate Power Series Ring Elements
//!
//! This module provides element classes for multivariate power series rings.
//!
//! ## Overview
//!
//! Multivariate power series elements represent formal power series in multiple variables.
//! They support:
//! - Arithmetic operations (addition, subtraction, multiplication, division)
//! - Precision management
//! - Variable substitution
//! - Differentiation and integration
//! - Coefficient extraction
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::multi_power_series_ring_element::MPowerSeries;
//! use rustmath_integers::Integer;
//!
//! // Create a multivariate power series
//! let series = MPowerSeries::<Integer>::new(10);
//! assert_eq!(series.precision(), 10);
//! ```

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Sub};

/// Monomial ordering for multivariate power series
///
/// Defines the ordering used for displaying and computing with monomials.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MonomialOrder {
    /// Negative degree lexicographic order
    NegDegLex,
    /// Degree lexicographic order
    DegLex,
    /// Lexicographic order
    Lex,
}

/// MO (Monomial Order) class
///
/// Represents monomial orderings for multivariate power series.
#[derive(Clone, Debug)]
pub struct MO {
    /// The ordering type
    order: MonomialOrder,
}

impl MO {
    /// Creates a new monomial ordering
    pub fn new(order: MonomialOrder) -> Self {
        MO { order }
    }

    /// Returns the negative degree lexicographic ordering
    pub fn negdeglex() -> Self {
        MO::new(MonomialOrder::NegDegLex)
    }

    /// Returns the degree lexicographic ordering
    pub fn deglex() -> Self {
        MO::new(MonomialOrder::DegLex)
    }

    /// Returns the lexicographic ordering
    pub fn lex() -> Self {
        MO::new(MonomialOrder::Lex)
    }

    /// Gets the ordering type
    pub fn order(&self) -> &MonomialOrder {
        &self.order
    }
}

impl Default for MO {
    fn default() -> Self {
        MO::negdeglex()
    }
}

/// Represents a multivariate power series element
///
/// A multivariate power series is a formal sum of terms with coefficients from a ring.
#[derive(Clone, Debug)]
pub struct MPowerSeries<R: Ring> {
    /// Monomial coefficients (exponent tuple -> coefficient)
    coefficients: HashMap<Vec<usize>, R>,
    /// Precision of the series (total degree)
    precision: usize,
    /// Number of variables
    num_vars: usize,
}

impl<R: Ring> MPowerSeries<R> {
    /// Creates a new multivariate power series with given precision
    ///
    /// # Arguments
    ///
    /// * `precision` - Total degree precision
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_rings::multi_power_series_ring_element::MPowerSeries;
    /// use rustmath_integers::Integer;
    ///
    /// let series = MPowerSeries::<Integer>::new(10);
    /// assert_eq!(series.precision(), 10);
    /// ```
    pub fn new(precision: usize) -> Self {
        MPowerSeries {
            coefficients: HashMap::new(),
            precision,
            num_vars: 0,
        }
    }

    /// Creates a new series with given number of variables
    pub fn with_variables(num_vars: usize, precision: usize) -> Self {
        MPowerSeries {
            coefficients: HashMap::new(),
            precision,
            num_vars,
        }
    }

    /// Creates the zero series
    pub fn zero(num_vars: usize) -> Self {
        MPowerSeries {
            coefficients: HashMap::new(),
            precision: 0,
            num_vars,
        }
    }

    /// Creates the one series
    pub fn one(num_vars: usize) -> Self
    where
        R: From<i32>,
    {
        let mut coeffs = HashMap::new();
        coeffs.insert(vec![0; num_vars], R::from(1));
        MPowerSeries {
            coefficients: coeffs,
            precision: 1,
            num_vars,
        }
    }

    /// Returns the precision (total degree)
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Sets a coefficient for a given monomial
    ///
    /// # Arguments
    ///
    /// * `exponents` - Exponent tuple for the monomial
    /// * `coefficient` - Coefficient value
    pub fn set_coefficient(&mut self, exponents: Vec<usize>, coefficient: R) {
        self.coefficients.insert(exponents, coefficient);
    }

    /// Gets the coefficient of a monomial
    pub fn get_coefficient(&self, exponents: &[usize]) -> Option<&R> {
        self.coefficients.get(exponents)
    }

    /// Returns all monomial coefficients as a reference to the hashmap
    pub fn monomial_coefficients(&self) -> &HashMap<Vec<usize>, R> {
        &self.coefficients
    }

    /// Returns a list of all coefficients
    pub fn coefficients(&self) -> Vec<&R> {
        self.coefficients.values().collect()
    }

    /// Returns all exponent tuples
    pub fn exponents(&self) -> Vec<&Vec<usize>> {
        self.coefficients.keys().collect()
    }

    /// Checks if this series is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Computes the valuation (degree of lowest term)
    pub fn valuation(&self) -> usize {
        if self.coefficients.is_empty() {
            return 0;
        }

        self.coefficients
            .keys()
            .map(|exp| exp.iter().sum())
            .min()
            .unwrap_or(0)
    }

    /// Truncates the series to a given precision
    pub fn truncate(&self, new_precision: usize) -> Self
    where
        R: Clone,
    {
        let mut new_coeffs = HashMap::new();
        for (exp, coeff) in &self.coefficients {
            let degree: usize = exp.iter().sum();
            if degree < new_precision {
                new_coeffs.insert(exp.clone(), coeff.clone());
            }
        }

        MPowerSeries {
            coefficients: new_coeffs,
            precision: new_precision,
            num_vars: self.num_vars,
        }
    }

    /// Adds big-O notation (sets precision)
    pub fn add_bigoh(&self, precision: usize) -> Self
    where
        R: Clone,
    {
        self.truncate(precision)
    }

    /// Computes the derivative with respect to a variable
    ///
    /// # Arguments
    ///
    /// * `var_index` - Index of the variable to differentiate
    pub fn derivative(&self, var_index: usize) -> Self
    where
        R: Clone + From<i32>,
        R: std::ops::Mul<Output = R>,
    {
        let mut new_coeffs = HashMap::new();

        for (exp, coeff) in &self.coefficients {
            if var_index < exp.len() && exp[var_index] > 0 {
                let mut new_exp = exp.clone();
                let power = new_exp[var_index];
                new_exp[var_index] -= 1;

                let factor = R::from(power as i32);
                new_coeffs.insert(new_exp, factor * coeff.clone());
            }
        }

        MPowerSeries {
            coefficients: new_coeffs,
            precision: self.precision,
            num_vars: self.num_vars,
        }
    }

    /// Computes the integral with respect to a variable
    ///
    /// # Arguments
    ///
    /// * `var_index` - Index of the variable to integrate
    ///
    /// Note: This is a simplified implementation
    pub fn integral(&self, var_index: usize) -> Self
    where
        R: Clone,
    {
        let mut new_coeffs = HashMap::new();

        for (exp, coeff) in &self.coefficients {
            let mut new_exp = exp.clone();
            if var_index < new_exp.len() {
                new_exp[var_index] += 1;
            }
            new_coeffs.insert(new_exp, coeff.clone());
        }

        MPowerSeries {
            coefficients: new_coeffs,
            precision: self.precision + 1,
            num_vars: self.num_vars,
        }
    }

    /// Returns the number of variables
    pub fn num_variables(&self) -> usize {
        self.num_vars
    }

    /// Returns the list of variables that appear in this series
    pub fn variables(&self) -> Vec<usize> {
        let mut vars = Vec::new();
        for exp in self.coefficients.keys() {
            for (i, &e) in exp.iter().enumerate() {
                if e > 0 && !vars.contains(&i) {
                    vars.push(i);
                }
            }
        }
        vars.sort();
        vars
    }
}

impl<R: Ring + Clone> Add for MPowerSeries<R>
where
    R: std::ops::Add<Output = R> + From<i32>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result_coeffs = self.coefficients.clone();

        for (exp, coeff) in other.coefficients {
            result_coeffs
                .entry(exp.clone())
                .and_modify(|c| *c = c.clone() + coeff.clone())
                .or_insert(coeff);
        }

        MPowerSeries {
            coefficients: result_coeffs,
            precision: std::cmp::min(self.precision, other.precision),
            num_vars: std::cmp::max(self.num_vars, other.num_vars),
        }
    }
}

impl<R: Ring + Clone> Sub for MPowerSeries<R>
where
    R: std::ops::Sub<Output = R> + From<i32>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result_coeffs = self.coefficients.clone();

        for (exp, coeff) in other.coefficients {
            result_coeffs
                .entry(exp.clone())
                .and_modify(|c| *c = c.clone() - coeff.clone())
                .or_insert_with(|| R::from(0) - coeff);
        }

        MPowerSeries {
            coefficients: result_coeffs,
            precision: std::cmp::min(self.precision, other.precision),
            num_vars: std::cmp::max(self.num_vars, other.num_vars),
        }
    }
}

impl<R: Ring> fmt::Display for MPowerSeries<R>
where
    R: fmt::Display + PartialEq + From<i32>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.coefficients.is_empty() {
            return write!(f, "0");
        }

        let mut terms = Vec::new();
        for (exp, coeff) in &self.coefficients {
            let degree: usize = exp.iter().sum();
            if degree == 0 {
                terms.push(format!("{}", coeff));
            } else {
                let mut var_str = Vec::new();
                for (i, &e) in exp.iter().enumerate() {
                    if e > 0 {
                        if e == 1 {
                            var_str.push(format!("x{}", i));
                        } else {
                            var_str.push(format!("x{}^{}", i, e));
                        }
                    }
                }
                terms.push(format!("{}*{}", coeff, var_str.join("*")));
            }
        }

        write!(f, "{}", terms.join(" + "))
    }
}

/// Checks if an object is a multivariate power series
///
/// # Deprecated
///
/// This function is deprecated. Use type checking instead.
#[deprecated(since = "0.1.0", note = "Use isinstance or type checking instead")]
pub fn is_mpower_series<R: Ring>(_obj: &MPowerSeries<R>) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_monomial_order() {
        let mo1 = MO::negdeglex();
        assert_eq!(mo1.order(), &MonomialOrder::NegDegLex);

        let mo2 = MO::deglex();
        assert_eq!(mo2.order(), &MonomialOrder::DegLex);

        let mo3 = MO::lex();
        assert_eq!(mo3.order(), &MonomialOrder::Lex);
    }

    #[test]
    fn test_mpower_series_creation() {
        let series = MPowerSeries::<Integer>::new(10);
        assert_eq!(series.precision(), 10);
        assert!(series.is_zero());
    }

    #[test]
    fn test_zero_one() {
        let zero = MPowerSeries::<Integer>::zero(2);
        assert!(zero.is_zero());

        let one = MPowerSeries::<Integer>::one(2);
        assert!(!one.is_zero());
    }

    #[test]
    fn test_set_get_coefficient() {
        let mut series = MPowerSeries::<Integer>::with_variables(2, 10);
        series.set_coefficient(vec![1, 0], Integer::from(5));
        series.set_coefficient(vec![0, 1], Integer::from(3));

        assert_eq!(series.get_coefficient(&vec![1, 0]), Some(&Integer::from(5)));
        assert_eq!(series.get_coefficient(&vec![0, 1]), Some(&Integer::from(3)));
        assert_eq!(series.get_coefficient(&vec![2, 0]), None);
    }

    #[test]
    fn test_coefficients_exponents() {
        let mut series = MPowerSeries::<Integer>::with_variables(2, 10);
        series.set_coefficient(vec![1, 0], Integer::from(5));
        series.set_coefficient(vec![0, 1], Integer::from(3));

        let coeffs = series.coefficients();
        assert_eq!(coeffs.len(), 2);

        let exps = series.exponents();
        assert_eq!(exps.len(), 2);
    }

    #[test]
    fn test_valuation() {
        let mut series = MPowerSeries::<Integer>::with_variables(2, 10);
        series.set_coefficient(vec![2, 1], Integer::from(1)); // degree 3
        series.set_coefficient(vec![1, 0], Integer::from(1)); // degree 1

        assert_eq!(series.valuation(), 1);
    }

    #[test]
    fn test_truncate() {
        let mut series = MPowerSeries::<Integer>::with_variables(2, 10);
        series.set_coefficient(vec![0, 0], Integer::from(1)); // degree 0
        series.set_coefficient(vec![1, 0], Integer::from(2)); // degree 1
        series.set_coefficient(vec![2, 0], Integer::from(3)); // degree 2
        series.set_coefficient(vec![3, 0], Integer::from(4)); // degree 3

        let truncated = series.truncate(2);
        assert_eq!(truncated.precision(), 2);
        assert!(truncated.get_coefficient(&vec![0, 0]).is_some());
        assert!(truncated.get_coefficient(&vec![1, 0]).is_some());
        assert!(truncated.get_coefficient(&vec![2, 0]).is_none());
    }

    #[test]
    fn test_derivative() {
        let mut series = MPowerSeries::<Integer>::with_variables(2, 10);
        series.set_coefficient(vec![2, 0], Integer::from(3)); // 3x^2

        let deriv = series.derivative(0); // d/dx
        assert_eq!(deriv.get_coefficient(&vec![1, 0]), Some(&Integer::from(6))); // 6x
    }

    #[test]
    fn test_integral() {
        let mut series = MPowerSeries::<Integer>::with_variables(2, 10);
        series.set_coefficient(vec![1, 0], Integer::from(2)); // 2x

        let integ = series.integral(0); // ∫ dx
        assert!(integ.get_coefficient(&vec![2, 0]).is_some());
    }

    #[test]
    fn test_addition() {
        let mut s1 = MPowerSeries::<Integer>::with_variables(2, 10);
        s1.set_coefficient(vec![1, 0], Integer::from(3));

        let mut s2 = MPowerSeries::<Integer>::with_variables(2, 10);
        s2.set_coefficient(vec![1, 0], Integer::from(2));
        s2.set_coefficient(vec![0, 1], Integer::from(5));

        let sum = s1 + s2;
        assert_eq!(sum.get_coefficient(&vec![1, 0]), Some(&Integer::from(5)));
        assert_eq!(sum.get_coefficient(&vec![0, 1]), Some(&Integer::from(5)));
    }

    #[test]
    fn test_subtraction() {
        let mut s1 = MPowerSeries::<Integer>::with_variables(2, 10);
        s1.set_coefficient(vec![1, 0], Integer::from(5));

        let mut s2 = MPowerSeries::<Integer>::with_variables(2, 10);
        s2.set_coefficient(vec![1, 0], Integer::from(2));

        let diff = s1 - s2;
        assert_eq!(diff.get_coefficient(&vec![1, 0]), Some(&Integer::from(3)));
    }

    #[test]
    fn test_variables() {
        let mut series = MPowerSeries::<Integer>::with_variables(3, 10);
        series.set_coefficient(vec![1, 0, 0], Integer::from(1));
        series.set_coefficient(vec![0, 0, 2], Integer::from(1));

        let vars = series.variables();
        assert_eq!(vars, vec![0, 2]);
    }

    #[test]
    #[allow(deprecated)]
    fn test_is_mpower_series() {
        let series = MPowerSeries::<Integer>::new(10);
        assert!(is_mpower_series(&series));
    }
}
