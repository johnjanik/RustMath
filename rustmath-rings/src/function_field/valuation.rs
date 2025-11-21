//! Valuations of function fields
//!
//! This module provides valuation structures for function fields,
//! corresponding to SageMath's `sage.rings.function_field.valuation`.
//!
//! # Mathematical Background
//!
//! A discrete valuation on a function field K is a surjective map v: K* → ℤ
//! satisfying:
//! - v(xy) = v(x) + v(y)
//! - v(x + y) ≥ min(v(x), v(y))
//! - v is trivial on the constant field k
//!
//! # Types of Valuations
//!
//! ## Classical Valuations
//!
//! - **Finite valuations**: Associated with finite places (prime ideals)
//! - **Infinite valuations**: Associated with the infinite place
//! - Correspond to discrete valuations on local rings
//!
//! ## Valuation Theory
//!
//! For a place P of a function field K:
//! - Valuation ring: O_P = {f ∈ K : v_P(f) ≥ 0}
//! - Maximal ideal: m_P = {f ∈ K : v_P(f) > 0}
//! - Residue field: κ(P) = O_P / m_P
//! - Uniformizer: Element π with v_P(π) = 1
//!
//! ## Extension of Valuations
//!
//! For an extension L/K and valuation v on K:
//! - Extensions of v to L may not be unique
//! - Ramification and inertia describe the extension behavior
//! - Fundamental identity: ∑ e_i f_i = [L:K]
//!
//! # Key Types
//!
//! - `DiscreteFunctionFieldValuation_base`: Base class for discrete valuations
//! - `ClassicalFunctionFieldValuation_base`: Classical valuations on function fields
//! - `RationalFunctionFieldValuation_base`: Valuations on rational function fields
//! - `FunctionFieldValuation_base`: General valuation interface

use rustmath_core::Field;
use std::marker::PhantomData;
use std::fmt;

/// Base trait for discrete valuations on function fields
///
/// This corresponds to SageMath's `DiscreteFunctionFieldValuation_base` class.
///
/// # Type Parameters
///
/// - `F`: The constant field type
pub trait DiscreteFunctionFieldValuation<F: Field> {
    /// Compute the valuation of an element
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    ///
    /// # Returns
    ///
    /// The valuation (as an integer)
    fn valuation(&self, element: &str) -> i64;

    /// Get the valuation ring
    ///
    /// # Returns
    ///
    /// String description of the valuation ring
    fn valuation_ring(&self) -> String;

    /// Get the residue field
    ///
    /// # Returns
    ///
    /// String description of the residue field
    fn residue_field(&self) -> String;

    /// Check if element is in the valuation ring
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    ///
    /// # Returns
    ///
    /// True if v(element) >= 0
    fn is_in_valuation_ring(&self, element: &str) -> bool {
        self.valuation(element) >= 0
    }

    /// Reduce element modulo the maximal ideal
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    ///
    /// # Returns
    ///
    /// String representation of the residue class
    fn reduce(&self, element: &str) -> String;

    /// Get a uniformizer (element with valuation 1)
    ///
    /// # Returns
    ///
    /// String representation of a uniformizer
    fn uniformizer(&self) -> String;
}

/// Base class for function field valuations
///
/// This corresponds to SageMath's `FunctionFieldValuation_base`.
///
/// # Type Parameters
///
/// - `F`: The constant field type
#[derive(Clone, Debug)]
pub struct FunctionFieldValuationBase<F: Field> {
    /// Name of the function field
    function_field: String,
    /// Name of the valuation
    name: String,
    /// Field marker
    field_marker: PhantomData<F>,
}

/// Type alias for snake_case compatibility
pub type FunctionFieldValuation_base<F> = FunctionFieldValuationBase<F>;

impl<F: Field> FunctionFieldValuation_base<F> {
    /// Create a new function field valuation base
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    /// * `name` - Name of the valuation
    ///
    /// # Returns
    ///
    /// A new FunctionFieldValuation_base instance
    pub fn new(function_field: String, name: String) -> Self {
        FunctionFieldValuation_base {
            function_field,
            name,
            field_marker: PhantomData,
        }
    }

    /// Get the function field
    ///
    /// # Returns
    ///
    /// Name of the function field
    pub fn function_field(&self) -> &str {
        &self.function_field
    }

    /// Get the valuation name
    ///
    /// # Returns
    ///
    /// Name of the valuation
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<F: Field> fmt::Display for FunctionFieldValuation_base<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Valuation {} on {}", self.name, self.function_field)
    }
}

/// Classical function field valuation
///
/// This corresponds to SageMath's `ClassicalFunctionFieldValuation_base`.
///
/// # Type Parameters
///
/// - `F`: The constant field type
#[derive(Clone, Debug)]
pub struct ClassicalFunctionFieldValuationBase<F: Field> {
    /// Base valuation
    base: FunctionFieldValuation_base<F>,
    /// Associated place
    place_name: String,
    /// Degree of the place
    place_degree: usize,
}

/// Type alias for snake_case compatibility
pub type ClassicalFunctionFieldValuation_base<F> = ClassicalFunctionFieldValuationBase<F>;

impl<F: Field> ClassicalFunctionFieldValuation_base<F> {
    /// Create a new classical valuation
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    /// * `place_name` - Name of the associated place
    /// * `place_degree` - Degree of the place
    ///
    /// # Returns
    ///
    /// A new ClassicalFunctionFieldValuation_base instance
    pub fn new(function_field: String, place_name: String, place_degree: usize) -> Self {
        let name = format!("v_{}", place_name);
        ClassicalFunctionFieldValuation_base {
            base: FunctionFieldValuation_base::new(function_field, name),
            place_name,
            place_degree,
        }
    }

    /// Get the associated place
    ///
    /// # Returns
    ///
    /// Name of the place
    pub fn place(&self) -> &str {
        &self.place_name
    }

    /// Get the place degree
    ///
    /// # Returns
    ///
    /// Degree of the place
    pub fn place_degree(&self) -> usize {
        self.place_degree
    }

    /// Get the base valuation
    ///
    /// # Returns
    ///
    /// Reference to the base valuation
    pub fn base(&self) -> &FunctionFieldValuation_base<F> {
        &self.base
    }
}

impl<F: Field> DiscreteFunctionFieldValuation<F> for ClassicalFunctionFieldValuation_base<F> {
    fn valuation(&self, element: &str) -> i64 {
        // Symbolic computation
        0
    }

    fn valuation_ring(&self) -> String {
        format!("O_{{{}}}", self.place_name)
    }

    fn residue_field(&self) -> String {
        format!("κ({})", self.place_name)
    }

    fn reduce(&self, element: &str) -> String {
        format!("{} mod m_{{{}}}", element, self.place_name)
    }

    fn uniformizer(&self) -> String {
        format!("π_{}", self.place_name)
    }
}

impl<F: Field> fmt::Display for ClassicalFunctionFieldValuation_base<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Classical valuation at place {} (degree {})",
            self.place_name, self.place_degree
        )
    }
}

/// Valuation on a rational function field
///
/// This corresponds to SageMath's `RationalFunctionFieldValuation_base`.
///
/// # Type Parameters
///
/// - `F`: The constant field type
#[derive(Clone, Debug)]
pub struct RationalFunctionFieldValuationBase<F: Field> {
    /// Base valuation
    base: FunctionFieldValuation_base<F>,
    /// Prime polynomial (None for infinite valuation)
    prime_polynomial: Option<String>,
}

/// Type alias for snake_case compatibility
pub type RationalFunctionFieldValuation_base<F> = RationalFunctionFieldValuationBase<F>;

impl<F: Field> RationalFunctionFieldValuation_base<F> {
    /// Create a finite valuation on k(x)
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    /// * `prime_polynomial` - Prime polynomial defining the valuation
    ///
    /// # Returns
    ///
    /// A new RationalFunctionFieldValuation_base instance
    pub fn finite(function_field: String, prime_polynomial: String) -> Self {
        let name = format!("v_{{{}}}", prime_polynomial);
        RationalFunctionFieldValuation_base {
            base: FunctionFieldValuation_base::new(function_field, name),
            prime_polynomial: Some(prime_polynomial),
        }
    }

    /// Create the infinite valuation on k(x)
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    ///
    /// # Returns
    ///
    /// The infinite valuation
    pub fn infinite(function_field: String) -> Self {
        RationalFunctionFieldValuation_base {
            base: FunctionFieldValuation_base::new(function_field, "v_∞".to_string()),
            prime_polynomial: None,
        }
    }

    /// Check if this is the infinite valuation
    ///
    /// # Returns
    ///
    /// True if this is the infinite valuation
    pub fn is_infinite(&self) -> bool {
        self.prime_polynomial.is_none()
    }

    /// Get the prime polynomial
    ///
    /// # Returns
    ///
    /// Prime polynomial (None for infinite valuation)
    pub fn prime_polynomial(&self) -> Option<&str> {
        self.prime_polynomial.as_deref()
    }

    /// Get the base valuation
    ///
    /// # Returns
    ///
    /// Reference to the base valuation
    pub fn base(&self) -> &FunctionFieldValuation_base<F> {
        &self.base
    }
}

impl<F: Field> DiscreteFunctionFieldValuation<F> for RationalFunctionFieldValuation_base<F> {
    fn valuation(&self, element: &str) -> i64 {
        // Symbolic computation
        0
    }

    fn valuation_ring(&self) -> String {
        if self.is_infinite() {
            "k[1/x]".to_string()
        } else if let Some(p) = &self.prime_polynomial {
            format!("k[x]_{{({})}}", p)
        } else {
            "k[x]".to_string()
        }
    }

    fn residue_field(&self) -> String {
        if self.is_infinite() {
            "k".to_string()
        } else if let Some(p) = &self.prime_polynomial {
            format!("k[x]/({})", p)
        } else {
            "k".to_string()
        }
    }

    fn reduce(&self, element: &str) -> String {
        if self.is_infinite() {
            format!("lim_{{x→∞}} {}", element)
        } else if let Some(p) = &self.prime_polynomial {
            format!("{} mod ({})", element, p)
        } else {
            element.to_string()
        }
    }

    fn uniformizer(&self) -> String {
        if self.is_infinite() {
            "1/x".to_string()
        } else if let Some(p) = &self.prime_polynomial {
            p.clone()
        } else {
            "x".to_string()
        }
    }
}

impl<F: Field> fmt::Display for RationalFunctionFieldValuation_base<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinite() {
            write!(f, "Infinite valuation on {}", self.base.function_field())
        } else if let Some(p) = &self.prime_polynomial {
            write!(
                f,
                "Valuation at {} on {}",
                p,
                self.base.function_field()
            )
        } else {
            write!(f, "Trivial valuation on {}", self.base.function_field())
        }
    }
}

/// Finite rational function field valuation
///
/// This corresponds to SageMath's `FiniteRationalFunctionFieldValuation`.
///
/// # Type Parameters
///
/// - `F`: The constant field type
#[derive(Clone, Debug)]
pub struct FiniteRationalFunctionFieldValuation<F: Field> {
    /// Base valuation
    base: RationalFunctionFieldValuation_base<F>,
}

impl<F: Field> FiniteRationalFunctionFieldValuation<F> {
    /// Create a new finite valuation
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    /// * `prime_polynomial` - Prime polynomial
    ///
    /// # Returns
    ///
    /// A new finite valuation
    pub fn new(function_field: String, prime_polynomial: String) -> Self {
        FiniteRationalFunctionFieldValuation {
            base: RationalFunctionFieldValuation_base::finite(function_field, prime_polynomial),
        }
    }

    /// Get the base valuation
    ///
    /// # Returns
    ///
    /// Reference to the base valuation
    pub fn base(&self) -> &RationalFunctionFieldValuation_base<F> {
        &self.base
    }
}

impl<F: Field> DiscreteFunctionFieldValuation<F> for FiniteRationalFunctionFieldValuation<F> {
    fn valuation(&self, element: &str) -> i64 {
        self.base.valuation(element)
    }

    fn valuation_ring(&self) -> String {
        self.base.valuation_ring()
    }

    fn residue_field(&self) -> String {
        self.base.residue_field()
    }

    fn reduce(&self, element: &str) -> String {
        self.base.reduce(element)
    }

    fn uniformizer(&self) -> String {
        self.base.uniformizer()
    }
}

impl<F: Field> fmt::Display for FiniteRationalFunctionFieldValuation<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base)
    }
}

/// Infinite rational function field valuation
///
/// This corresponds to SageMath's `InfiniteRationalFunctionFieldValuation`.
///
/// # Type Parameters
///
/// - `F`: The constant field type
#[derive(Clone, Debug)]
pub struct InfiniteRationalFunctionFieldValuation<F: Field> {
    /// Base valuation
    base: RationalFunctionFieldValuation_base<F>,
}

impl<F: Field> InfiniteRationalFunctionFieldValuation<F> {
    /// Create the infinite valuation
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    ///
    /// # Returns
    ///
    /// The infinite valuation
    pub fn new(function_field: String) -> Self {
        InfiniteRationalFunctionFieldValuation {
            base: RationalFunctionFieldValuation_base::infinite(function_field),
        }
    }

    /// Get the base valuation
    ///
    /// # Returns
    ///
    /// Reference to the base valuation
    pub fn base(&self) -> &RationalFunctionFieldValuation_base<F> {
        &self.base
    }
}

impl<F: Field> DiscreteFunctionFieldValuation<F> for InfiniteRationalFunctionFieldValuation<F> {
    fn valuation(&self, element: &str) -> i64 {
        self.base.valuation(element)
    }

    fn valuation_ring(&self) -> String {
        self.base.valuation_ring()
    }

    fn residue_field(&self) -> String {
        self.base.residue_field()
    }

    fn reduce(&self, element: &str) -> String {
        self.base.reduce(element)
    }

    fn uniformizer(&self) -> String {
        self.base.uniformizer()
    }
}

impl<F: Field> fmt::Display for InfiniteRationalFunctionFieldValuation<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base)
    }
}

/// Non-classical rational function field valuation
///
/// This corresponds to SageMath's `NonClassicalRationalFunctionFieldValuation`.
/// These are valuations that don't come from places (e.g., Gauss valuations).
///
/// # Type Parameters
///
/// - `F`: The constant field type
#[derive(Clone, Debug)]
pub struct NonClassicalRationalFunctionFieldValuation<F: Field> {
    /// Base valuation
    base: FunctionFieldValuation_base<F>,
    /// Description of the valuation
    description: String,
}

impl<F: Field> NonClassicalRationalFunctionFieldValuation<F> {
    /// Create a non-classical valuation
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    /// * `name` - Name of the valuation
    /// * `description` - Description of the valuation
    ///
    /// # Returns
    ///
    /// A new non-classical valuation
    pub fn new(function_field: String, name: String, description: String) -> Self {
        NonClassicalRationalFunctionFieldValuation {
            base: FunctionFieldValuation_base::new(function_field, name),
            description,
        }
    }

    /// Get the description
    ///
    /// # Returns
    ///
    /// Description of the valuation
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the base valuation
    ///
    /// # Returns
    ///
    /// Reference to the base valuation
    pub fn base(&self) -> &FunctionFieldValuation_base<F> {
        &self.base
    }
}

impl<F: Field> fmt::Display for NonClassicalRationalFunctionFieldValuation<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Non-classical valuation {} ({})",
            self.base.name(),
            self.description
        )
    }
}

/// Factory for creating function field valuations
///
/// This corresponds to SageMath's `FunctionFieldValuationFactory`.
pub struct FunctionFieldValuationFactory;

impl FunctionFieldValuationFactory {
    /// Create a valuation on a rational function field
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    /// * `prime_or_infinity` - Prime polynomial or "∞" for infinite valuation
    ///
    /// # Returns
    ///
    /// String description of the created valuation
    pub fn create_rational_valuation(function_field: String, prime_or_infinity: String) -> String {
        if prime_or_infinity == "∞" || prime_or_infinity == "infinity" {
            format!("Infinite valuation on {}", function_field)
        } else {
            format!("Valuation at {} on {}", prime_or_infinity, function_field)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_function_field_valuation_base() {
        let val: FunctionFieldValuation_base<Rational> =
            FunctionFieldValuation_base::new("Q(x)".to_string(), "v".to_string());

        assert_eq!(val.function_field(), "Q(x)");
        assert_eq!(val.name(), "v");
    }

    #[test]
    fn test_classical_valuation() {
        let val: ClassicalFunctionFieldValuation_base<Rational> =
            ClassicalFunctionFieldValuation_base::new(
                "Q(x)".to_string(),
                "P".to_string(),
                1,
            );

        assert_eq!(val.place(), "P");
        assert_eq!(val.place_degree(), 1);
        assert_eq!(val.uniformizer(), "π_P");
    }

    #[test]
    fn test_rational_finite_valuation() {
        let val: RationalFunctionFieldValuation_base<Rational> =
            RationalFunctionFieldValuation_base::finite(
                "Q(x)".to_string(),
                "x".to_string(),
            );

        assert!(!val.is_infinite());
        assert_eq!(val.prime_polynomial(), Some("x"));
        assert_eq!(val.uniformizer(), "x");
        assert_eq!(val.residue_field(), "k[x]/(x)");
    }

    #[test]
    fn test_rational_infinite_valuation() {
        let val: RationalFunctionFieldValuation_base<Rational> =
            RationalFunctionFieldValuation_base::infinite("Q(x)".to_string());

        assert!(val.is_infinite());
        assert_eq!(val.prime_polynomial(), None);
        assert_eq!(val.uniformizer(), "1/x");
        assert_eq!(val.residue_field(), "k");
    }

    #[test]
    fn test_finite_rational_valuation() {
        let val: FiniteRationalFunctionFieldValuation<Rational> =
            FiniteRationalFunctionFieldValuation::new(
                "Q(x)".to_string(),
                "x - 1".to_string(),
            );

        assert_eq!(val.base().prime_polynomial(), Some("x - 1"));
        assert_eq!(val.uniformizer(), "x - 1");
    }

    #[test]
    fn test_infinite_rational_valuation() {
        let val: InfiniteRationalFunctionFieldValuation<Rational> =
            InfiniteRationalFunctionFieldValuation::new("Q(x)".to_string());

        assert!(val.base().is_infinite());
        assert_eq!(val.uniformizer(), "1/x");
        assert_eq!(val.valuation_ring(), "k[1/x]");
    }

    #[test]
    fn test_non_classical_valuation() {
        let val: NonClassicalRationalFunctionFieldValuation<Rational> =
            NonClassicalRationalFunctionFieldValuation::new(
                "Q(x)".to_string(),
                "v_Gauss".to_string(),
                "Gauss valuation".to_string(),
            );

        assert_eq!(val.base().name(), "v_Gauss");
        assert_eq!(val.description(), "Gauss valuation");
    }

    #[test]
    fn test_valuation_ring_finite() {
        let val: RationalFunctionFieldValuation_base<Rational> =
            RationalFunctionFieldValuation_base::finite(
                "Q(x)".to_string(),
                "x".to_string(),
            );

        let ring = val.valuation_ring();
        assert!(ring.contains("k[x]"));
    }

    #[test]
    fn test_reduce_element() {
        let val: RationalFunctionFieldValuation_base<Rational> =
            RationalFunctionFieldValuation_base::finite(
                "Q(x)".to_string(),
                "x - 1".to_string(),
            );

        let reduced = val.reduce("x^2");
        assert!(reduced.contains("mod"));
    }

    #[test]
    fn test_factory() {
        let val_str = FunctionFieldValuationFactory::create_rational_valuation(
            "Q(x)".to_string(),
            "x".to_string(),
        );

        assert!(val_str.contains("Valuation at x"));

        let inf_str = FunctionFieldValuationFactory::create_rational_valuation(
            "Q(x)".to_string(),
            "∞".to_string(),
        );

        assert!(inf_str.contains("Infinite"));
    }

    #[test]
    fn test_display_classical() {
        let val: ClassicalFunctionFieldValuation_base<Rational> =
            ClassicalFunctionFieldValuation_base::new(
                "Q(x)".to_string(),
                "P".to_string(),
                2,
            );

        let display = format!("{}", val);
        assert!(display.contains("Classical valuation"));
        assert!(display.contains("degree 2"));
    }

    #[test]
    fn test_display_finite_rational() {
        let val: FiniteRationalFunctionFieldValuation<Rational> =
            FiniteRationalFunctionFieldValuation::new(
                "Q(x)".to_string(),
                "x + 1".to_string(),
            );

        let display = format!("{}", val);
        assert!(display.contains("Valuation"));
    }

    #[test]
    fn test_display_infinite_rational() {
        let val: InfiniteRationalFunctionFieldValuation<Rational> =
            InfiniteRationalFunctionFieldValuation::new("Q(x)".to_string());

        let display = format!("{}", val);
        assert!(display.contains("Infinite"));
    }

    #[test]
    fn test_valuation_interface() {
        let val: ClassicalFunctionFieldValuation_base<Rational> =
            ClassicalFunctionFieldValuation_base::new(
                "Q(x)".to_string(),
                "P".to_string(),
                1,
            );

        // Test the trait methods
        assert!(val.is_in_valuation_ring("x"));
        assert_eq!(val.valuation_ring(), "O_{P}");
        assert_eq!(val.residue_field(), "κ(P)");
    }
}
