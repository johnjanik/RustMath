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
//! values.insert("e".to_string(), Complex::new(3.0, 0.0));
//! values.insert("g".to_string(), Complex::new(-1.0, 0.0));
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
    /// Class sizes `|C|` for the keys of `values`, if known (T-215).
    class_sizes: Option<HashMap<String, usize>>,
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
            class_sizes: None,
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
            class_sizes: None,
        }
    }

    /// A class function on conjugacy classes with their sizes; the group order is set to the
    /// sum of the sizes. `Err` if the keys of `class_sizes` differ from those of `values`, or a
    /// size is 0.
    ///
    /// The keys must be all the classes of the group. Nothing checks this: with only some of
    /// the classes, the group order is the partial sum of the sizes, and
    /// [`ClassFunction::inner_product`], [`ClassFunction::norm`] and
    /// [`ClassFunction::is_irreducible`] give a weighted mean over those classes only.
    pub fn with_class_sizes(
        degree: usize,
        values: HashMap<String, Complex>,
        class_sizes: HashMap<String, usize>,
    ) -> Result<Self, String> {
        if values.len() != class_sizes.len() || values.keys().any(|k| !class_sizes.contains_key(k))
        {
            return Err("class sizes and values have different keys".to_string());
        }
        if class_sizes.values().any(|&v| v == 0) {
            return Err("a class of size 0".to_string());
        }
        let order = class_sizes.values().sum();
        Ok(Self {
            degree,
            values,
            group_order: Some(order),
            class_sizes: Some(class_sizes),
        })
    }

    /// The class sizes, if known.
    pub fn class_sizes(&self) -> Option<&HashMap<String, usize>> {
        self.class_sizes.as_ref()
    }

    /// Create from a character (its class sizes are kept)
    pub fn from_character(character: &Character) -> Self {
        let class_sizes = character.class_sizes().cloned();
        Self {
            degree: character.degree(),
            values: character.values().clone(),
            group_order: class_sizes.as_ref().map(|s| s.values().sum()),
            class_sizes,
        }
    }

    /// The class sizes of a binary operation: kept only when both sides agree.
    fn common_sizes(&self, other: &Self) -> Option<HashMap<String, usize>> {
        match (&self.class_sizes, &other.class_sizes) {
            (Some(a), Some(b)) if a == b => Some(a.clone()),
            _ => None,
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
            class_sizes: self.common_sizes(other),
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
            class_sizes: self.common_sizes(other),
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
            class_sizes: self.common_sizes(other),
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
            class_sizes: self.class_sizes.clone(),
        }
    }

    /// Negation
    pub fn neg(&self) -> Self {
        self.scalar_mul(&Complex::new(-1.0, 0.0))
    }

    /// Inner product of two class functions
    ///
    /// ⟨f, g⟩ = (1/|G|) Σ_{x ∈ G} f(x) g̅(x)
    ///
    /// For class functions, this simplifies to:
    /// ⟨f, g⟩ = (1/|G|) Σ_{C} |C| f(c) g̅(c)
    /// where the sum is over conjugacy classes C with representative c.
    ///
    /// T-215: the sizes `|C|` are used. `None` unless the group order is known (and equal on
    /// both sides when both know it), the two functions have the same classes, and the sizes
    /// are known (on either side, equal when on both) and sum to the group order. Without
    /// sizes, the one case that is decided is a number of classes equal to the group order:
    /// then every class has size 1.
    pub fn inner_product(&self, other: &Self) -> Option<Complex> {
        let order = self.group_order?;
        if other.group_order.is_some_and(|o| o != order) {
            return None;
        }
        if self.values.len() != other.values.len()
            || self.values.keys().any(|k| !other.values.contains_key(k))
        {
            return None;
        }
        let sizes: Option<&HashMap<String, usize>> = match (&self.class_sizes, &other.class_sizes) {
            (Some(a), Some(b)) => {
                if a != b {
                    return None;
                }
                Some(a)
            }
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        let size_of = |key: &str| -> Option<usize> {
            match sizes {
                Some(m) => m.get(key).copied(),
                None => Some(1),
            }
        };
        match sizes {
            Some(m) => {
                if m.len() != self.values.len() || m.values().sum::<usize>() != order {
                    return None;
                }
            }
            None => {
                if self.values.len() != order {
                    return None;
                }
            }
        }

        let mut sum = Complex::zero();
        for (key, val) in &self.values {
            let other_val = &other.values[key];
            let size = size_of(key)? as f64;
            sum = sum.add(&Complex::new(size, 0.0).mul(&val.mul(&other_val.conjugate())));
        }

        sum.div(&Complex::new(order as f64, 0.0)).ok()
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

    /// Convert to a character (with the class sizes when they are known and fit the values)
    pub fn to_character(&self) -> Character {
        if let Some(sizes) = &self.class_sizes {
            if let Ok(c) =
                Character::with_class_sizes(self.degree, self.values.clone(), sizes.clone())
            {
                return c;
            }
        }
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
pub type ClassFunctionGap = ClassFunction;

/// ClassFunction_libgap - Modern LibGAP interface wrapper
///
/// In SageMath, this uses the LibGAP library interface. In RustMath, we use
/// the same implementation as ClassFunction since we don't have GAP integration.
pub type ClassFunctionLibgap = ClassFunction;

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
/// values.insert("e".to_string(), Complex::new(3.0, 0.0));
/// values.insert("g".to_string(), Complex::new(0.0, 0.0));
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
        values.insert(class, Complex::new(1.0, 0.0));
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
        values.insert("e".to_string(), Complex::new(2.0, 0.0));
        values.insert("g".to_string(), Complex::new(-1.0, 0.0));
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
        values.insert("e".to_string(), Complex::new(1.0, 0.0));
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
        let scaled = cf.scalar_mul(&Complex::new(2.0, 0.0));

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
        values.insert("e".to_string(), Complex::new(1.0, 0.0));
        values.insert("g".to_string(), Complex::new(1.0, 0.0));
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
        let values = vec![Complex::new(3.0, 0.0), Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)];
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
        values.insert("e".to_string(), Complex::new(3.0, 0.0));
        values.insert("g".to_string(), Complex::new(0.0, 0.0));
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
        values.insert("e".to_string(), Complex::new(2.0, 0.0));
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
        values.insert("e".to_string(), Complex::new(1.0, 0.0));
        let mut cf = ClassFunction::new(1, values);
        cf.set_group_order(1);

        let norm = cf.norm();
        assert!(norm.is_some());
        assert!((norm.unwrap() - 1.0).abs() < 1e-10);
    }

    fn s3(values: [f64; 3], sizes: [usize; 3]) -> ClassFunction {
        let keys = ["e", "(12)", "(123)"];
        let v = keys
            .iter()
            .zip(values)
            .map(|(k, x)| (k.to_string(), Complex::new(x, 0.0)))
            .collect();
        let s = keys
            .iter()
            .zip(sizes)
            .map(|(k, x)| (k.to_string(), x))
            .collect();
        ClassFunction::with_class_sizes(values[0] as usize, v, s).unwrap()
    }

    // T-215: the S3 standard character (2, 0, -1) on classes of sizes 1, 3, 2 has norm 1
    // (GAP 4.12.1, CharacterTable(SymmetricGroup(3))); the old code, with all sizes 1, gave 5/6.
    #[test]
    fn t215_weighted_inner_product_s3() {
        let standard = s3([2.0, 0.0, -1.0], [1, 3, 2]);
        assert_eq!(standard.group_order(), Some(6));
        let ip = standard.inner_product(&standard).unwrap();
        assert!((ip.real() - 1.0).abs() < 1e-12, "{ip:?}");
        assert!(standard.is_irreducible());
        let trivial = s3([1.0, 1.0, 1.0], [1, 3, 2]);
        let sign = s3([1.0, -1.0, 1.0], [1, 3, 2]);
        assert!(trivial.inner_product(&sign).unwrap().real().abs() < 1e-12);
        assert!(trivial.inner_product(&standard).unwrap().real().abs() < 1e-12);
        // the sizes survive a round trip through Character
        let c = standard.to_character();
        assert!(c.class_sizes().is_some());
        let back = ClassFunction::from_character(&c);
        assert!((back.inner_product(&back).unwrap().real() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn t215_inner_product_refusals() {
        let mut standard = s3([2.0, 0.0, -1.0], [1, 3, 2]);
        // sizes that do not sum to the group order
        standard.set_group_order(7);
        assert!(standard.inner_product(&standard).is_none());
        // two classes, unknown sizes, group order 6: not decided
        let mut values = HashMap::new();
        values.insert("e".to_string(), Complex::new(1.0, 0.0));
        values.insert("g".to_string(), Complex::new(1.0, 0.0));
        let mut cf = ClassFunction::new(1, values);
        cf.set_group_order(6);
        assert!(cf.inner_product(&cf).is_none());
        // review 2, n3: different class sets
        let mut other = HashMap::new();
        other.insert("e".to_string(), Complex::new(1.0, 0.0));
        other.insert("h".to_string(), Complex::new(1.0, 0.0));
        let mut cf2 = ClassFunction::new(1, other);
        cf2.set_group_order(6);
        cf.set_group_order(2);
        cf2.set_group_order(2);
        assert!(cf.inner_product(&cf).is_some());
        assert!(cf.inner_product(&cf2).is_none());
        // different size maps
        let a = s3([2.0, 0.0, -1.0], [1, 3, 2]);
        let b = s3([2.0, 0.0, -1.0], [2, 2, 2]);
        assert!(a.inner_product(&b).is_none());
        // bad size maps are refused
        let v: HashMap<String, Complex> = [("e".to_string(), Complex::new(1.0, 0.0))].into();
        let z: HashMap<String, usize> = [("e".to_string(), 0)].into();
        assert!(ClassFunction::with_class_sizes(1, v.clone(), z).is_err());
        let w: HashMap<String, usize> = [("f".to_string(), 1)].into();
        assert!(ClassFunction::with_class_sizes(1, v, w).is_err());
    }

    #[test]
    fn test_display() {
        let cf = make_simple_class_function();
        let display = format!("{}", cf);
        assert!(display.contains("Class function"));
        assert!(display.contains("degree 2"));
    }
}
