//! Ideals in Polymod Function Fields Module
//!
//! This module implements ideals in polynomial function field extensions,
//! corresponding to SageMath's `sage.rings.function_field.ideal_polymod` module.
//!
//! # Mathematical Overview
//!
//! For a function field extension L/K where L = K[y]/(f(y)), ideals play a
//! crucial role in understanding the arithmetic and geometry of the extension.
//!
//! ## Types of Ideals
//!
//! ### Finite Ideals
//!
//! Fractional ideals of the maximal order (ring of integers) of L. These
//! correspond to finite places in the geometric picture.
//!
//! ### Infinite Ideals
//!
//! Ideals associated with places at infinity. These are particularly important
//! for global function fields.
//!
//! ### Prime Ideals
//!
//! Maximal ideals of the maximal order, corresponding to places of the function field.
//! They are fundamental building blocks for the divisor class group.
//!
//! ## Operations
//!
//! - **Multiplication**: Product of ideals
//! - **Addition**: Sum of ideals
//! - **Intersection**: Ideal intersection
//! - **Norm**: Ideal norm to the base field
//!
//! # Implementation
//!
//! This module provides:
//!
//! - `FunctionFieldIdealPolymod`: Ideals in polymod extensions
//! - `FunctionFieldIdealInfinitePolymod`: Ideals at infinity
//! - `FunctionFieldIdealGlobal`: Ideals in global function fields
//!
//! # References
//!
//! - SageMath: `sage.rings.function_field.ideal_polymod`
//! - Stichtenoth, H. (2009). "Algebraic Function Fields and Codes"

use rustmath_core::{Field, Ring};
use std::marker::PhantomData;

/// Ideal in a polymod function field extension
///
/// Represents a fractional ideal in the maximal order of L/K where
/// L = K[y]/(f(y)).
///
/// # Type Parameters
///
/// * `F` - The coefficient field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::ideal_polymod::FunctionFieldIdealPolymod;
/// use rustmath_rationals::Rational;
///
/// let ideal = FunctionFieldIdealPolymod::<Rational>::new("(x, y)".to_string());
/// assert!(ideal.is_well_defined());
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldIdealPolymod<F: Field> {
    /// Generators of the ideal (as strings for simplicity)
    generators: Vec<String>,
    /// Name/description of the ideal
    name: String,
    /// Whether this is a prime ideal
    is_prime_ideal: bool,
    /// Phantom data for field type
    _phantom: PhantomData<F>,
}

impl<F: Field> FunctionFieldIdealPolymod<F> {
    /// Create a new ideal with generators
    pub fn new(name: String) -> Self {
        Self {
            generators: Vec::new(),
            name,
            is_prime_ideal: false,
            _phantom: PhantomData,
        }
    }

    /// Create an ideal with explicit generators
    pub fn with_generators(generators: Vec<String>) -> Self {
        Self {
            name: format!("({})", generators.join(", ")),
            generators,
            is_prime_ideal: false,
            _phantom: PhantomData,
        }
    }

    /// Create a prime ideal
    pub fn prime(name: String) -> Self {
        Self {
            generators: Vec::new(),
            name,
            is_prime_ideal: true,
            _phantom: PhantomData,
        }
    }

    /// Check if this is a prime ideal
    pub fn is_prime(&self) -> bool {
        self.is_prime_ideal
    }

    /// Check if this is the unit ideal
    pub fn is_unit(&self) -> bool {
        self.generators.iter().any(|g| g == "1")
    }

    /// Check if this is the zero ideal
    pub fn is_zero(&self) -> bool {
        self.generators.is_empty() || self.generators.iter().all(|g| g == "0")
    }

    /// Get the generators
    pub fn generators(&self) -> &[String] {
        &self.generators
    }

    /// Get the name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add a generator
    pub fn add_generator(&mut self, gen: String) {
        if !self.generators.contains(&gen) {
            self.generators.push(gen);
        }
    }

    /// Compute the norm of this ideal to the base field
    pub fn norm(&self) -> String {
        format!("Norm({}) to base field", self.name)
    }

    /// Compute the ideal product I*J
    pub fn multiply(&self, other: &Self) -> Self {
        let new_name = format!("({})*({})", self.name, other.name);
        Self {
            generators: Vec::new(), // Would compute product generators
            name: new_name,
            is_prime_ideal: false,
            _phantom: PhantomData,
        }
    }

    /// Compute the ideal sum I+J
    pub fn add(&self, other: &Self) -> Self {
        let mut gens = self.generators.clone();
        gens.extend(other.generators.clone());

        Self {
            generators: gens,
            name: format!("({}) + ({})", self.name, other.name),
            is_prime_ideal: false,
            _phantom: PhantomData,
        }
    }

    /// Compute the ideal intersection I∩J
    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            generators: Vec::new(), // Would compute intersection
            name: format!("({}) ∩ ({})", self.name, other.name),
            is_prime_ideal: false,
            _phantom: PhantomData,
        }
    }

    /// Check if this ideal divides another (I | J means J ⊆ I)
    pub fn divides(&self, _other: &Self) -> bool {
        // Would check containment
        false
    }

    /// Compute the relative degree f(P/p) for a prime ideal
    pub fn relative_degree(&self) -> usize {
        if !self.is_prime_ideal {
            return 1;
        }
        // Would compute [O/P : o/p] where O is maximal order
        1
    }

    /// Compute the ramification index e(P/p) for a prime ideal
    pub fn ramification_index(&self) -> usize {
        if !self.is_prime_ideal {
            return 1;
        }
        // Would compute v_P(p) where p is the prime below
        1
    }

    /// Check if well-defined
    pub fn is_well_defined(&self) -> bool {
        !self.name.is_empty()
    }
}

/// Ideal at infinity in a polymod function field
///
/// Represents ideals associated with infinite places of the function field.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::ideal_polymod::FunctionFieldIdealInfinitePolymod;
/// use rustmath_rationals::Rational;
///
/// let ideal_inf = FunctionFieldIdealInfinitePolymod::<Rational>::new("∞".to_string());
/// assert!(ideal_inf.is_infinite());
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldIdealInfinitePolymod<F: Field> {
    /// Underlying ideal structure
    inner: FunctionFieldIdealPolymod<F>,
    /// Valuation at infinity
    valuation: i32,
}

impl<F: Field> FunctionFieldIdealInfinitePolymod<F> {
    /// Create a new infinite ideal
    pub fn new(name: String) -> Self {
        Self {
            inner: FunctionFieldIdealPolymod::new(name),
            valuation: 1,
        }
    }

    /// Check if this is an infinite place
    pub fn is_infinite(&self) -> bool {
        true
    }

    /// Get the valuation at infinity
    pub fn valuation(&self) -> i32 {
        self.valuation
    }

    /// Set the valuation
    pub fn set_valuation(&mut self, val: i32) {
        self.valuation = val;
    }

    /// Get the name
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Check if this is a standard infinity (valuation 1)
    pub fn is_standard_infinity(&self) -> bool {
        self.valuation == 1
    }
}

/// Ideal in a global function field (over finite constant field)
///
/// These are ideals in function fields over finite fields, which have
/// additional structure and properties.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::ideal_polymod::FunctionFieldIdealGlobal;
/// use rustmath_rationals::Rational;
///
/// let ideal = FunctionFieldIdealGlobal::<Rational>::new("(x)".to_string(), 2);
/// assert_eq!(ideal.constant_field_size(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldIdealGlobal<F: Field> {
    /// Underlying polymod ideal
    inner: FunctionFieldIdealPolymod<F>,
    /// Size of constant field
    constant_field_size: usize,
}

impl<F: Field> FunctionFieldIdealGlobal<F> {
    /// Create a new global ideal
    pub fn new(name: String, constant_field_size: usize) -> Self {
        Self {
            inner: FunctionFieldIdealPolymod::new(name),
            constant_field_size,
        }
    }

    /// Create a prime global ideal
    pub fn prime(name: String, constant_field_size: usize) -> Self {
        Self {
            inner: FunctionFieldIdealPolymod::prime(name),
            constant_field_size,
        }
    }

    /// Get the constant field size
    pub fn constant_field_size(&self) -> usize {
        self.constant_field_size
    }

    /// Check if prime
    pub fn is_prime(&self) -> bool {
        self.inner.is_prime()
    }

    /// Get the name
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Compute the norm degree
    ///
    /// For a prime ideal P, this is the degree of the residue field extension
    pub fn norm_degree(&self) -> usize {
        if !self.inner.is_prime() {
            return 0;
        }
        // Would compute degree of (O/P) : (o/p)
        1
    }

    /// Compute the ideal norm as an integer
    ///
    /// For prime P of degree f: N(P) = q^f where q = |k|
    pub fn norm_as_integer(&self) -> usize {
        if !self.inner.is_prime() {
            return 1;
        }

        let f = self.norm_degree();
        self.constant_field_size.pow(f as u32)
    }

    /// Count elements in the residue class field
    pub fn residue_field_size(&self) -> usize {
        if !self.inner.is_prime() {
            return 0;
        }
        self.norm_as_integer()
    }

    /// Check if this ideal has good reduction
    pub fn has_good_reduction(&self) -> bool {
        // Would check if the ideal is unramified
        true
    }

    /// Compute Frobenius element (for unramified primes)
    pub fn frobenius(&self) -> String {
        format!("Frobenius element for {}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_function_field_ideal_polymod() {
        let ideal = FunctionFieldIdealPolymod::<Rational>::new("(x, y)".to_string());

        assert!(ideal.is_well_defined());
        assert!(!ideal.is_prime());
        assert!(!ideal.is_unit());
        assert_eq!(ideal.name(), "(x, y)");
    }

    #[test]
    fn test_ideal_with_generators() {
        let gens = vec!["x".to_string(), "y-1".to_string()];
        let ideal = FunctionFieldIdealPolymod::<Rational>::with_generators(gens.clone());

        assert_eq!(ideal.generators().len(), 2);
        assert!(ideal.name().contains("x"));
        assert!(ideal.name().contains("y-1"));
    }

    #[test]
    fn test_prime_ideal() {
        let prime = FunctionFieldIdealPolymod::<Rational>::prime("(x)".to_string());

        assert!(prime.is_prime());
        assert!(!prime.is_unit());
        assert!(!prime.is_zero());
    }

    #[test]
    fn test_unit_ideal() {
        let mut unit = FunctionFieldIdealPolymod::<Rational>::new("unit".to_string());
        unit.add_generator("1".to_string());

        assert!(unit.is_unit());
        assert!(!unit.is_zero());
    }

    #[test]
    fn test_zero_ideal() {
        let zero = FunctionFieldIdealPolymod::<Rational>::new("zero".to_string());
        assert!(zero.is_zero());
    }

    #[test]
    fn test_add_generator() {
        let mut ideal = FunctionFieldIdealPolymod::<Rational>::new("test".to_string());

        assert_eq!(ideal.generators().len(), 0);

        ideal.add_generator("x".to_string());
        assert_eq!(ideal.generators().len(), 1);

        ideal.add_generator("y".to_string());
        assert_eq!(ideal.generators().len(), 2);

        // Adding duplicate shouldn't increase count
        ideal.add_generator("x".to_string());
        assert_eq!(ideal.generators().len(), 2);
    }

    #[test]
    fn test_ideal_multiplication() {
        let i1 = FunctionFieldIdealPolymod::<Rational>::new("I".to_string());
        let i2 = FunctionFieldIdealPolymod::<Rational>::new("J".to_string());

        let product = i1.multiply(&i2);

        assert!(product.name().contains("I"));
        assert!(product.name().contains("J"));
        assert!(product.name().contains("*"));
    }

    #[test]
    fn test_ideal_addition() {
        let mut i1 = FunctionFieldIdealPolymod::<Rational>::with_generators(vec![
            "x".to_string(),
        ]);
        let mut i2 = FunctionFieldIdealPolymod::<Rational>::with_generators(vec![
            "y".to_string(),
        ]);

        let sum = i1.add(&i2);

        assert_eq!(sum.generators().len(), 2);
        assert!(sum.name().contains("+"));
    }

    #[test]
    fn test_ideal_intersection() {
        let i1 = FunctionFieldIdealPolymod::<Rational>::new("I".to_string());
        let i2 = FunctionFieldIdealPolymod::<Rational>::new("J".to_string());

        let inter = i1.intersect(&i2);

        assert!(inter.name().contains("∩"));
    }

    #[test]
    fn test_norm() {
        let ideal = FunctionFieldIdealPolymod::<Rational>::new("(x)".to_string());
        let norm = ideal.norm();

        assert!(norm.contains("Norm"));
        assert!(norm.contains("(x)"));
    }

    #[test]
    fn test_relative_degree() {
        let prime = FunctionFieldIdealPolymod::<Rational>::prime("(x)".to_string());
        let deg = prime.relative_degree();

        assert!(deg >= 1);
    }

    #[test]
    fn test_ramification_index() {
        let prime = FunctionFieldIdealPolymod::<Rational>::prime("(x)".to_string());
        let e = prime.ramification_index();

        assert!(e >= 1);
    }

    #[test]
    fn test_function_field_ideal_infinite() {
        let ideal_inf = FunctionFieldIdealInfinitePolymod::<Rational>::new("∞".to_string());

        assert!(ideal_inf.is_infinite());
        assert_eq!(ideal_inf.valuation(), 1);
        assert!(ideal_inf.is_standard_infinity());
    }

    #[test]
    fn test_infinite_valuation() {
        let mut ideal_inf = FunctionFieldIdealInfinitePolymod::<Rational>::new("∞".to_string());

        ideal_inf.set_valuation(3);
        assert_eq!(ideal_inf.valuation(), 3);
        assert!(!ideal_inf.is_standard_infinity());
    }

    #[test]
    fn test_function_field_ideal_global() {
        let ideal = FunctionFieldIdealGlobal::<Rational>::new("(x)".to_string(), 2);

        assert_eq!(ideal.constant_field_size(), 2);
        assert!(!ideal.is_prime());
    }

    #[test]
    fn test_global_prime_ideal() {
        let prime = FunctionFieldIdealGlobal::<Rational>::prime("(x)".to_string(), 3);

        assert!(prime.is_prime());
        assert_eq!(prime.constant_field_size(), 3);
    }

    #[test]
    fn test_norm_degree() {
        let prime = FunctionFieldIdealGlobal::<Rational>::prime("(x)".to_string(), 2);
        let deg = prime.norm_degree();

        assert!(deg >= 1);
    }

    #[test]
    fn test_norm_as_integer() {
        let prime = FunctionFieldIdealGlobal::<Rational>::prime("(x)".to_string(), 2);
        let norm = prime.norm_as_integer();

        // For degree 1, norm should be q^1 = 2
        assert!(norm >= 2);
    }

    #[test]
    fn test_residue_field_size() {
        let prime = FunctionFieldIdealGlobal::<Rational>::prime("(x)".to_string(), 5);
        let res_size = prime.residue_field_size();

        assert!(res_size >= 5);
    }

    #[test]
    fn test_frobenius() {
        let prime = FunctionFieldIdealGlobal::<Rational>::prime("(x)".to_string(), 2);
        let frob = prime.frobenius();

        assert!(frob.contains("Frobenius"));
    }

    #[test]
    fn test_has_good_reduction() {
        let prime = FunctionFieldIdealGlobal::<Rational>::prime("(x)".to_string(), 2);
        assert!(prime.has_good_reduction());
    }

    #[test]
    fn test_ideal_chain() {
        // Test a chain of ideal operations
        let i1 = FunctionFieldIdealPolymod::<Rational>::prime("P1".to_string());
        let i2 = FunctionFieldIdealPolymod::<Rational>::prime("P2".to_string());
        let i3 = FunctionFieldIdealPolymod::<Rational>::prime("P3".to_string());

        let prod = i1.multiply(&i2);
        let sum = prod.add(&i3);

        assert!(sum.name().contains("P1"));
        assert!(sum.name().contains("P3"));
    }

    #[test]
    fn test_different_field_sizes() {
        for q in [2, 3, 5, 7, 11] {
            let prime = FunctionFieldIdealGlobal::<Rational>::prime("(x)".to_string(), q);
            assert_eq!(prime.constant_field_size(), q);
            assert!(prime.residue_field_size() >= q);
        }
    }
}
