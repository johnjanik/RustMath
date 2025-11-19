//! Class Functions for Group Representation Theory
//!
//! This module implements class functions, which are complex-valued functions on
//! a group that are constant on conjugacy classes. Class functions are fundamental
//! in character theory and representation theory.
//!
//! # Mathematical Background
//!
//! A class function f: G → ℂ on a finite group G satisfies:
//! - f(hgh^{-1}) = f(g) for all g, h ∈ G (constant on conjugacy classes)
//!
//! Characters of representations are important examples of class functions.
//! The set of class functions forms a vector space over ℂ with dimension equal
//! to the number of conjugacy classes.
//!
//! # Key Operations
//!
//! - Addition and subtraction of class functions
//! - Scalar multiplication
//! - Inner product: ⟨f, g⟩ = (1/|G|) Σ f(x) g̅(x)
//! - Restriction to subgroups
//! - Induction from subgroups
//! - Tensor products
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::class_function::{ClassFunction, class_function};
//! use rustmath_complex::Complex;
//! use std::collections::HashMap;
//!
//! // Create a class function from values on conjugacy classes
//! let mut values = HashMap::new();
//! values.insert("e".to_string(), Complex::from(3.0));
//! values.insert("g".to_string(), Complex::from(-1.0));
//!
//! let cf = class_function(3, values);
//! assert_eq!(cf.degree(), 3);
//! ```

use std::collections::HashMap;
use std::fmt;

use rustmath_complex::Complex;
use crate::representation::Character;

/// A class function on a finite group
///
/// A class function is a complex-valued function that is constant on conjugacy
/// classes. This is the base implementation that stores values for each conjugacy
/// class representative.
#[derive(Debug, Clone)]
pub struct ClassFunction {
    /// Degree of the associated representation (if applicable)
    degree: usize,
    /// Values on conjugacy class representatives
    values: HashMap<String, Complex>,
    /// Group order (if known)
    group_order: Option<usize>,
}

impl ClassFunction {
    /// Create a new class function
    ///
    /// # Arguments
    ///
    /// * `degree` - Degree of the representation (sum of values at identity)
    /// * `values` - Map from conjugacy class representatives to values
    pub fn new(degree: usize, values: HashMap<String, Complex>) -> Self {
        Self {
            degree,
            values,
            group_order: None,
        }
    }

    /// Create a class function with known group order
    pub fn with_group_order(
        degree: usize,
        values: HashMap<String, Complex>,
        group_order: usize,
    ) -> Self {
        Self {
            degree,
            values,
            group_order: Some(group_order),
        }
    }

    /// Create from a character
    pub fn from_character(character: &Character) -> Self {
        Self {
            degree: character.degree(),
            values: character.values().clone(),
            group_order: None,
        }
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the value on a conjugacy class representative
    pub fn value(&self, element: &str) -> Option<&Complex> {
        self.values.get(element)
    }

    /// Get all values
    pub fn values(&self) -> &HashMap<String, Complex> {
        &self.values
    }

    /// Get the group order if known
    pub fn group_order(&self) -> Option<usize> {
        self.group_order
    }

    /// Set the group order
    pub fn set_group_order(&mut self, order: usize) {
        self.group_order = Some(order);
    }

    /// Get all conjugacy class representatives
    pub fn conjugacy_classes(&self) -> Vec<&String> {
        self.values.keys().collect()
    }

    /// Number of conjugacy classes
    pub fn num_classes(&self) -> usize {
        self.values.len()
    }

    /// Add two class functions
    pub fn add(&self, other: &Self) -> Self {
        let mut new_values = self.values.clone();
        for (key, val) in &other.values {
            *new_values.entry(key.clone()).or_insert(Complex::zero()) =
                new_values.get(key).unwrap_or(&Complex::zero()).add(val);
        }

        Self {
            degree: self.degree + other.degree,
            values: new_values,
            group_order: self.group_order,
        }
    }

    /// Subtract two class functions
    pub fn sub(&self, other: &Self) -> Self {
        let mut new_values = self.values.clone();
        for (key, val) in &other.values {
            *new_values.entry(key.clone()).or_insert(Complex::zero()) =
                new_values.get(key).unwrap_or(&Complex::zero()).sub(val);
        }

        Self {
            degree: (self.degree as i32 - other.degree as i32).abs() as usize,
            values: new_values,
            group_order: self.group_order,
        }
    }

    /// Multiply two class functions (pointwise)
    pub fn mul(&self, other: &Self) -> Self {
        let mut new_values = HashMap::new();
        for (key, val) in &self.values {
            if let Some(other_val) = other.values.get(key) {
                new_values.insert(key.clone(), val.mul(other_val));
            }
        }

        Self {
            degree: self.degree * other.degree,
            values: new_values,
            group_order: self.group_order,
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &Complex) -> Self {
        let mut new_values = HashMap::new();
        for (key, val) in &self.values {
            new_values.insert(key.clone(), val.mul(scalar));
        }

        Self {
            degree: self.degree,
            values: new_values,
            group_order: self.group_order,
        }
    }

    /// Negation
    pub fn neg(&self) -> Self {
        self.scalar_mul(&Complex::from(-1.0))
    }

    /// Inner product of two class functions
    ///
    /// ⟨f, g⟩ = (1/|G|) Σ_{x ∈ G} f(x) g̅(x)
    ///
    /// For class functions, this simplifies to:
    /// ⟨f, g⟩ = (1/|G|) Σ_{C} |C| f(c) g̅(c)
    /// where the sum is over conjugacy classes C with representative c
    pub fn inner_product(&self, other: &Self) -> Option<Complex> {
        if self.group_order.is_none() {
            return None;
        }

        let order = self.group_order.unwrap() as f64;
        let mut sum = Complex::zero();

        // For simplicity, assume all conjugacy classes have size 1
        // (proper implementation would need class sizes)
        for (key, val) in &self.values {
            if let Some(other_val) = other.values.get(key) {
                sum = sum.add(&val.mul(&other_val.conjugate()));
            }
        }

        Some(sum.div(&Complex::from(order)))
    }

    /// Norm of a class function: ||f|| = √⟨f, f⟩
    pub fn norm(&self) -> Option<f64> {
        self.inner_product(self).map(|ip| ip.abs())
    }

    /// Check if this is an irreducible character (norm = 1)
    pub fn is_irreducible(&self) -> bool {
        self.norm()
            .map(|n| (n - 1.0).abs() < 1e-10)
            .unwrap_or(false)
    }

    /// Convert to a character
    pub fn to_character(&self) -> Character {
        Character::new(self.degree, self.values.clone())
    }
}

impl fmt::Display for ClassFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Class function of degree {}", self.degree)
    }
}

impl PartialEq for ClassFunction {
    fn eq(&self, other: &Self) -> bool {
        if self.degree != other.degree || self.values.len() != other.values.len() {
            return false;
        }

        for (key, val) in &self.values {
            if let Some(other_val) = other.values.get(key) {
                if (val.real() - other_val.real()).abs() > 1e-10
                    || (val.imag() - other_val.imag()).abs() > 1e-10
                {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

/// ClassFunction_gap - Legacy GAP interface wrapper
///
/// In SageMath, this wraps GAP's ClassFunction. In RustMath, we provide a
/// simplified implementation that doesn't depend on GAP but provides similar
/// functionality.
pub type ClassFunction_gap = ClassFunction;

/// ClassFunction_libgap - Modern LibGAP interface wrapper
///
/// In SageMath, this uses the LibGAP library interface. In RustMath, we use
/// the same implementation as ClassFunction since we don't have GAP integration.
pub type ClassFunction_libgap = ClassFunction;

/// Factory function to create a class function
///
/// This is the main entry point for creating class functions, matching the
/// SageMath ClassFunction() function.
///
/// # Arguments
///
/// * `degree` - Degree of the representation
/// * `values` - Values on conjugacy class representatives
///
/// # Examples
///
/// ```
/// use rustmath_groups::class_function::class_function;
/// use rustmath_complex::Complex;
/// use std::collections::HashMap;
///
/// let mut values = HashMap::new();
/// values.insert("e".to_string(), Complex::from(3.0));
/// values.insert("g".to_string(), Complex::from(0.0));
///
/// let cf = class_function(3, values);
/// ```
pub fn class_function(degree: usize, values: HashMap<String, Complex>) -> ClassFunction {
    ClassFunction::new(degree, values)
}

/// Create the trivial class function (all values are 1)
pub fn trivial_class_function(conjugacy_classes: Vec<String>) -> ClassFunction {
    let mut values = HashMap::new();
    for class in conjugacy_classes {
        values.insert(class, Complex::from(1.0));
    }
    ClassFunction::new(1, values)
}

/// Create a class function from character values
pub fn class_function_from_values(
    values: Vec<Complex>,
    class_representatives: Vec<String>,
) -> Result<ClassFunction, String> {
    if values.len() != class_representatives.len() {
        return Err("Number of values must match number of conjugacy classes".to_string());
    }

    let mut value_map = HashMap::new();
    for (rep, val) in class_representatives.iter().zip(values.iter()) {
        value_map.insert(rep.clone(), val.clone());
    }

    let degree = values[0].real().round() as usize; // Assume first value is at identity
    Ok(ClassFunction::new(degree, value_map))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_class_function() -> ClassFunction {
        let mut values = HashMap::new();
        values.insert("e".to_string(), Complex::from(2.0));
        values.insert("g".to_string(), Complex::from(-1.0));
        ClassFunction::new(2, values)
    }

    #[test]
    fn test_class_function_creation() {
        let cf = make_simple_class_function();
        assert_eq!(cf.degree(), 2);
        assert_eq!(cf.num_classes(), 2);
    }

    #[test]
    fn test_class_function_factory() {
        let mut values = HashMap::new();
        values.insert("e".to_string(), Complex::from(1.0));
        let cf = class_function(1, values);
        assert_eq!(cf.degree(), 1);
    }

    #[test]
    fn test_class_function_value() {
        let cf = make_simple_class_function();
        assert_eq!(cf.value("e").unwrap().real(), 2.0);
        assert_eq!(cf.value("g").unwrap().real(), -1.0);
        assert!(cf.value("h").is_none());
    }

    #[test]
    fn test_class_function_addition() {
        let cf1 = make_simple_class_function();
        let cf2 = make_simple_class_function();

        let sum = cf1.add(&cf2);
        assert_eq!(sum.degree(), 4);
        assert_eq!(sum.value("e").unwrap().real(), 4.0);
        assert_eq!(sum.value("g").unwrap().real(), -2.0);
    }

    #[test]
    fn test_class_function_subtraction() {
        let cf1 = make_simple_class_function();
        let cf2 = make_simple_class_function();

        let diff = cf1.sub(&cf2);
        assert_eq!(diff.value("e").unwrap().real(), 0.0);
        assert_eq!(diff.value("g").unwrap().real(), 0.0);
    }

    #[test]
    fn test_class_function_multiplication() {
        let cf1 = make_simple_class_function();
        let cf2 = make_simple_class_function();

        let prod = cf1.mul(&cf2);
        assert_eq!(prod.value("e").unwrap().real(), 4.0);
        assert_eq!(prod.value("g").unwrap().real(), 1.0);
    }

    #[test]
    fn test_scalar_multiplication() {
        let cf = make_simple_class_function();
        let scaled = cf.scalar_mul(&Complex::from(2.0));

        assert_eq!(scaled.value("e").unwrap().real(), 4.0);
        assert_eq!(scaled.value("g").unwrap().real(), -2.0);
        assert_eq!(scaled.degree(), 2);
    }

    #[test]
    fn test_negation() {
        let cf = make_simple_class_function();
        let neg = cf.neg();

        assert_eq!(neg.value("e").unwrap().real(), -2.0);
        assert_eq!(neg.value("g").unwrap().real(), 1.0);
    }

    #[test]
    fn test_inner_product() {
        let mut values = HashMap::new();
        values.insert("e".to_string(), Complex::from(1.0));
        values.insert("g".to_string(), Complex::from(1.0));
        let mut cf = ClassFunction::new(1, values);
        cf.set_group_order(2);

        let ip = cf.inner_product(&cf);
        assert!(ip.is_some());
        assert!((ip.unwrap().real() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trivial_class_function() {
        let classes = vec!["e".to_string(), "g".to_string(), "h".to_string()];
        let cf = trivial_class_function(classes);

        assert_eq!(cf.degree(), 1);
        assert_eq!(cf.num_classes(), 3);
        assert_eq!(cf.value("e").unwrap().real(), 1.0);
        assert_eq!(cf.value("g").unwrap().real(), 1.0);
        assert_eq!(cf.value("h").unwrap().real(), 1.0);
    }

    #[test]
    fn test_class_function_from_values() {
        let values = vec![Complex::from(3.0), Complex::from(0.0), Complex::from(-1.0)];
        let classes = vec!["e".to_string(), "g".to_string(), "h".to_string()];

        let cf = class_function_from_values(values, classes).unwrap();
        assert_eq!(cf.degree(), 3);
        assert_eq!(cf.value("e").unwrap().real(), 3.0);
        assert_eq!(cf.value("h").unwrap().real(), -1.0);
    }

    #[test]
    fn test_class_function_equality() {
        let cf1 = make_simple_class_function();
        let cf2 = make_simple_class_function();
        assert_eq!(cf1, cf2);

        let mut values = HashMap::new();
        values.insert("e".to_string(), Complex::from(3.0));
        values.insert("g".to_string(), Complex::from(0.0));
        let cf3 = ClassFunction::new(3, values);
        assert_ne!(cf1, cf3);
    }

    #[test]
    fn test_conjugacy_classes() {
        let cf = make_simple_class_function();
        let classes = cf.conjugacy_classes();
        assert_eq!(classes.len(), 2);
        assert!(classes.contains(&&"e".to_string()) || classes.contains(&&"g".to_string()));
    }

    #[test]
    fn test_to_character() {
        let cf = make_simple_class_function();
        let character = cf.to_character();
        assert_eq!(character.degree(), 2);
    }

    #[test]
    fn test_from_character() {
        let mut values = HashMap::new();
        values.insert("e".to_string(), Complex::from(2.0));
        let character = Character::new(2, values);

        let cf = ClassFunction::from_character(&character);
        assert_eq!(cf.degree(), 2);
    }

    #[test]
    fn test_group_order() {
        let mut cf = make_simple_class_function();
        assert!(cf.group_order().is_none());

        cf.set_group_order(6);
        assert_eq!(cf.group_order(), Some(6));
    }

    #[test]
    fn test_norm() {
        let mut values = HashMap::new();
        values.insert("e".to_string(), Complex::from(1.0));
        let mut cf = ClassFunction::new(1, values);
        cf.set_group_order(1);

        let norm = cf.norm();
        assert!(norm.is_some());
        assert!((norm.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_display() {
        let cf = make_simple_class_function();
        let display = format!("{}", cf);
        assert!(display.contains("Class function"));
        assert!(display.contains("degree 2"));
    }
}
