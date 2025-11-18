//! Differentials on function fields
//!
//! This module provides differential forms on function fields, corresponding to
//! SageMath's `sage.rings.function_field.differential`.
//!
//! # Mathematical Background
//!
//! For a function field K, the module of differentials Ω_K forms a one-dimensional
//! vector space over K. A differential can be written as f·dx where f ∈ K.
//!
//! Key properties:
//! - **Addition**: (f·dx) + (g·dx) = (f+g)·dx
//! - **Scalar multiplication**: f·(g·dx) = (fg)·dx
//! - **Residue**: ∑ Res_P(ω) = 0 (residue theorem)
//! - **Divisor**: Each differential has an associated divisor
//!
//! For characteristic p > 0, the Cartier operator C acts on differentials with:
//! - C(exact differentials) = 0
//! - C(logarithmic differentials) stable
//!
//! # Key Types
//!
//! - `FunctionFieldDifferential<F>`: A differential form on a function field
//! - `DifferentialsSpace<F>`: The module of all differentials
//! - `DifferentialsSpaceInclusion<F>`: Morphisms between differential spaces
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::differential::*;
//! use rustmath_rationals::Rational;
//!
//! // Create the space of differentials on Q(x)
//! let space = DifferentialsSpace::new("Q(x)".to_string());
//!
//! // Create a differential f·dx
//! let omega = FunctionFieldDifferential::new("x^2".to_string(), "x".to_string());
//! ```

use rustmath_core::{Ring, Field};
use std::marker::PhantomData;
use std::fmt;
use std::ops::{Add, Mul, Neg, Div};

/// Differential form on a function field
///
/// Represents a differential ω = f·dx where f is a function field element
/// and dx is the basic differential.
///
/// Corresponds to SageMath's `FunctionFieldDifferential` class.
///
/// # Type Parameters
///
/// - `F`: The base field type
#[derive(Clone, Debug)]
pub struct FunctionFieldDifferential<F: Field> {
    /// Coefficient (the f in f·dx)
    coefficient: String,
    /// Differential variable (the x in dx)
    variable: String,
    _field: PhantomData<F>,
}

impl<F: Field> FunctionFieldDifferential<F> {
    /// Create a new differential
    ///
    /// # Arguments
    ///
    /// * `coeff` - The coefficient (function field element)
    /// * `var` - The differential variable
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Creates the differential (x^2)·dx
    /// let omega = FunctionFieldDifferential::new("x^2".to_string(), "x".to_string());
    /// ```
    pub fn new(coeff: String, var: String) -> Self {
        FunctionFieldDifferential {
            coefficient: coeff,
            variable: var,
            _field: PhantomData,
        }
    }

    /// Create the zero differential
    pub fn zero(var: String) -> Self {
        FunctionFieldDifferential::new("0".to_string(), var)
    }

    /// Check if this is the zero differential
    pub fn is_zero(&self) -> bool {
        self.coefficient == "0"
    }

    /// Get the coefficient
    pub fn coefficient(&self) -> &str {
        &self.coefficient
    }

    /// Get the differential variable
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Compute the divisor of this differential
    ///
    /// Returns the divisor (ω) associated with this differential.
    /// For ω = f·dx on k(x), we have (ω) = (f) + (deg(f)-1)·∞
    pub fn divisor(&self) -> String {
        // Placeholder: would compute actual divisor
        format!("div({}·d{})", self.coefficient, self.variable)
    }

    /// Compute the valuation at a place
    ///
    /// Returns v_P(ω), the order of vanishing/pole at place P.
    ///
    /// # Arguments
    ///
    /// * `place` - The place (as string identifier)
    pub fn valuation(&self, place: &str) -> i64 {
        // Placeholder: would compute actual valuation
        0
    }

    /// Compute the residue at a place
    ///
    /// Returns Res_P(ω), the residue of this differential at place P.
    /// Satisfies ∑_P Res_P(ω) = 0 (residue theorem).
    ///
    /// # Arguments
    ///
    /// * `place` - The place at which to compute the residue
    pub fn residue(&self, place: &str) -> String {
        // Placeholder: would compute actual residue
        format!("Res_{}({}·d{})", place, self.coefficient, self.variable)
    }

    /// Check if this is an exact differential
    ///
    /// A differential ω is exact if ω = df for some f.
    /// Exact differentials have residue 0 at all places.
    pub fn is_exact(&self) -> bool {
        // Placeholder: would check exactness
        false
    }

    /// Check if this is logarithmic
    ///
    /// A differential is logarithmic if it has the form df/f.
    /// Logarithmic differentials are stable under the Cartier operator.
    pub fn is_logarithmic(&self) -> bool {
        // Placeholder: would check if ω = df/f
        false
    }

    /// Apply the Cartier operator (for characteristic p > 0)
    ///
    /// Returns C(ω) where C is the Cartier operator.
    /// Properties:
    /// - C(exact differentials) = 0
    /// - C(logarithmic differentials) are stable
    /// - C is p-linear
    pub fn cartier(&self) -> Self {
        // Placeholder: Cartier operator computation
        FunctionFieldDifferential::zero(self.variable.clone())
    }

    /// Multiply by a function field element (scalar multiplication)
    ///
    /// Returns f·ω where f is a function field element.
    pub fn scale(&self, scalar: String) -> Self {
        if scalar == "0" {
            return Self::zero(self.variable.clone());
        }
        if scalar == "1" {
            return self.clone();
        }

        // Multiply coefficients: (scalar)·(coeff·dx)
        FunctionFieldDifferential::new(
            format!("({})*({})", scalar, self.coefficient),
            self.variable.clone(),
        )
    }
}

impl<F: Field> Add for FunctionFieldDifferential<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.variable, other.variable, "Differentials must have same variable");

        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }

        // (f·dx) + (g·dx) = (f+g)·dx
        FunctionFieldDifferential::new(
            format!("({}) + ({})", self.coefficient, other.coefficient),
            self.variable,
        )
    }
}

impl<F: Field> Neg for FunctionFieldDifferential<F> {
    type Output = Self;

    fn neg(self) -> Self {
        FunctionFieldDifferential::new(
            format!("-({})", self.coefficient),
            self.variable,
        )
    }
}

impl<F: Field> Div for FunctionFieldDifferential<F> {
    type Output = String; // Returns a function field element

    /// Division of differentials
    ///
    /// (f·dx) / (g·dx) = f/g (returns a function field element)
    fn div(self, other: Self) -> String {
        assert_eq!(self.variable, other.variable, "Differentials must have same variable");
        assert!(!other.is_zero(), "Division by zero differential");

        format!("({})/({})", self.coefficient, other.coefficient)
    }
}

impl<F: Field> PartialEq for FunctionFieldDifferential<F> {
    fn eq(&self, other: &Self) -> bool {
        // Placeholder: should normalize and compare
        self.coefficient == other.coefficient && self.variable == other.variable
    }
}

impl<F: Field> Eq for FunctionFieldDifferential<F> {}

impl<F: Field> fmt::Display for FunctionFieldDifferential<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else if self.coefficient == "1" {
            write!(f, "d{}", self.variable)
        } else {
            write!(f, "({})*d{}", self.coefficient, self.variable)
        }
    }
}

/// Space of differentials on a function field
///
/// Represents Ω_K, the one-dimensional K-module of differentials.
/// Corresponds to SageMath's `DifferentialsSpace`.
///
/// # Type Parameters
///
/// - `F`: The base field type
#[derive(Clone, Debug)]
pub struct DifferentialsSpace<F: Field> {
    /// The underlying function field
    function_field: String,
    /// The differential variable
    variable: String,
    _field: PhantomData<F>,
}

impl<F: Field> DifferentialsSpace<F> {
    /// Create a new differentials space
    ///
    /// # Arguments
    ///
    /// * `field` - Identifier of the function field
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let space = DifferentialsSpace::new("Q(x)".to_string());
    /// ```
    pub fn new(field: String) -> Self {
        // Extract variable from field identifier (e.g., "Q(x)" -> "x")
        let var = extract_variable(&field);
        DifferentialsSpace {
            function_field: field,
            variable: var,
            _field: PhantomData,
        }
    }

    /// Get the underlying function field
    pub fn function_field(&self) -> &str {
        &self.function_field
    }

    /// Get the differential variable
    pub fn variable(&self) -> &str {
        &self.variable
    }

    /// Get the dimension of this space as a vector space
    ///
    /// Always returns 1 (differentials form a 1-dimensional space)
    pub fn dimension(&self) -> usize {
        1
    }

    /// Get a basis for the space
    ///
    /// Returns {dx} as the basis
    pub fn basis(&self) -> Vec<FunctionFieldDifferential<F>> {
        vec![FunctionFieldDifferential::new(
            "1".to_string(),
            self.variable.clone(),
        )]
    }

    /// Get the canonical basis element
    ///
    /// Returns dx (or dt, etc. depending on the variable)
    pub fn gen(&self) -> FunctionFieldDifferential<F> {
        FunctionFieldDifferential::new("1".to_string(), self.variable.clone())
    }

    /// Create a differential from a coefficient
    ///
    /// Given f, returns f·dx
    ///
    /// # Arguments
    ///
    /// * `coeff` - The coefficient
    pub fn element(&self, coeff: String) -> FunctionFieldDifferential<F> {
        FunctionFieldDifferential::new(coeff, self.variable.clone())
    }

    /// The zero differential
    pub fn zero(&self) -> FunctionFieldDifferential<F> {
        FunctionFieldDifferential::zero(self.variable.clone())
    }
}

impl<F: Field> fmt::Display for DifferentialsSpace<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ω_{}", self.function_field)
    }
}

/// Morphism between differential spaces
///
/// Represents an inclusion or coercion map between differential spaces,
/// typically arising from function field extensions.
///
/// Corresponds to SageMath's `DifferentialsSpaceInclusion`.
#[derive(Clone, Debug)]
pub struct DifferentialsSpaceInclusion<F: Field> {
    /// Source space
    source: DifferentialsSpace<F>,
    /// Target space
    target: DifferentialsSpace<F>,
}

impl<F: Field> DifferentialsSpaceInclusion<F> {
    /// Create a new inclusion morphism
    ///
    /// # Arguments
    ///
    /// * `source` - The source differential space
    /// * `target` - The target differential space
    pub fn new(source: DifferentialsSpace<F>, target: DifferentialsSpace<F>) -> Self {
        DifferentialsSpaceInclusion { source, target }
    }

    /// Get the source space
    pub fn source(&self) -> &DifferentialsSpace<F> {
        &self.source
    }

    /// Get the target space
    pub fn target(&self) -> &DifferentialsSpace<F> {
        &self.target
    }

    /// Apply the morphism to a differential
    ///
    /// Maps a differential from the source space to the target space.
    pub fn apply(&self, omega: &FunctionFieldDifferential<F>) -> FunctionFieldDifferential<F> {
        // Placeholder: would perform coercion
        omega.clone()
    }
}

impl<F: Field> fmt::Display for DifferentialsSpaceInclusion<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} → {}", self.source, self.target)
    }
}

/// Global differential (for global function fields over finite fields)
///
/// Corresponds to SageMath's `FunctionFieldDifferential_global`.
/// Adds Cartier operator support and genus computations.
pub type FunctionFieldDifferentialGlobal<F> = FunctionFieldDifferential<F>;

/// Global differentials space
///
/// Corresponds to SageMath's `DifferentialsSpace_global`.
pub type DifferentialsSpaceGlobal<F> = DifferentialsSpace<F>;

/// Helper function to extract variable from field identifier
fn extract_variable(field: &str) -> String {
    // Simple extraction: "Q(x)" -> "x", "F_p(t)" -> "t"
    if let Some(start) = field.find('(') {
        if let Some(end) = field.find(')') {
            return field[start + 1..end].to_string();
        }
    }
    "x".to_string() // default
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_differential_creation() {
        let omega = FunctionFieldDifferential::<Rational>::new("x^2".to_string(), "x".to_string());

        assert_eq!(omega.coefficient(), "x^2");
        assert_eq!(omega.variable(), "x");
        assert!(!omega.is_zero());
    }

    #[test]
    fn test_zero_differential() {
        let zero = FunctionFieldDifferential::<Rational>::zero("x".to_string());

        assert!(zero.is_zero());
        assert_eq!(zero.coefficient(), "0");
    }

    #[test]
    fn test_differential_display() {
        let omega1 = FunctionFieldDifferential::<Rational>::new("x^2".to_string(), "x".to_string());
        assert_eq!(format!("{}", omega1), "(x^2)*dx");

        let omega2 = FunctionFieldDifferential::<Rational>::new("1".to_string(), "x".to_string());
        assert_eq!(format!("{}", omega2), "dx");

        let zero = FunctionFieldDifferential::<Rational>::zero("x".to_string());
        assert_eq!(format!("{}", zero), "0");
    }

    #[test]
    fn test_differential_addition() {
        let omega1 = FunctionFieldDifferential::<Rational>::new("x".to_string(), "x".to_string());
        let omega2 = FunctionFieldDifferential::<Rational>::new("1".to_string(), "x".to_string());

        let sum = omega1 + omega2;
        assert!(sum.coefficient().contains("x"));
        assert!(sum.coefficient().contains("1"));
    }

    #[test]
    fn test_differential_negation() {
        let omega = FunctionFieldDifferential::<Rational>::new("x".to_string(), "x".to_string());
        let neg = -omega;

        assert!(neg.coefficient().contains("-"));
    }

    #[test]
    fn test_differential_division() {
        let omega1 = FunctionFieldDifferential::<Rational>::new("x^2".to_string(), "x".to_string());
        let omega2 = FunctionFieldDifferential::<Rational>::new("x".to_string(), "x".to_string());

        let ratio = omega1 / omega2;
        assert!(ratio.contains("x^2"));
        assert!(ratio.contains("x"));
    }

    #[test]
    #[should_panic(expected = "Division by zero differential")]
    fn test_division_by_zero() {
        let omega = FunctionFieldDifferential::<Rational>::new("x".to_string(), "x".to_string());
        let zero = FunctionFieldDifferential::<Rational>::zero("x".to_string());

        let _ = omega / zero;
    }

    #[test]
    fn test_scalar_multiplication() {
        let omega = FunctionFieldDifferential::<Rational>::new("x".to_string(), "x".to_string());
        let scaled = omega.scale("2".to_string());

        assert!(scaled.coefficient().contains("2"));
        assert!(scaled.coefficient().contains("x"));
    }

    #[test]
    fn test_scale_by_zero() {
        let omega = FunctionFieldDifferential::<Rational>::new("x".to_string(), "x".to_string());
        let zero = omega.scale("0".to_string());

        assert!(zero.is_zero());
    }

    #[test]
    fn test_scale_by_one() {
        let omega = FunctionFieldDifferential::<Rational>::new("x".to_string(), "x".to_string());
        let same = omega.clone().scale("1".to_string());

        assert_eq!(omega.coefficient(), same.coefficient());
    }

    #[test]
    fn test_differentials_space() {
        let space = DifferentialsSpace::<Rational>::new("Q(x)".to_string());

        assert_eq!(space.function_field(), "Q(x)");
        assert_eq!(space.variable(), "x");
        assert_eq!(space.dimension(), 1);
    }

    #[test]
    fn test_space_basis() {
        let space = DifferentialsSpace::<Rational>::new("Q(x)".to_string());
        let basis = space.basis();

        assert_eq!(basis.len(), 1);
        assert_eq!(basis[0].coefficient(), "1");
        assert_eq!(basis[0].variable(), "x");
    }

    #[test]
    fn test_space_generator() {
        let space = DifferentialsSpace::<Rational>::new("Q(t)".to_string());
        let gen = space.gen();

        assert_eq!(gen.coefficient(), "1");
        assert_eq!(gen.variable(), "t");
    }

    #[test]
    fn test_space_element() {
        let space = DifferentialsSpace::<Rational>::new("Q(x)".to_string());
        let omega = space.element("x^2 + 1".to_string());

        assert_eq!(omega.coefficient(), "x^2 + 1");
        assert_eq!(omega.variable(), "x");
    }

    #[test]
    fn test_space_zero() {
        let space = DifferentialsSpace::<Rational>::new("Q(x)".to_string());
        let zero = space.zero();

        assert!(zero.is_zero());
    }

    #[test]
    fn test_space_display() {
        let space = DifferentialsSpace::<Rational>::new("Q(x)".to_string());
        assert_eq!(format!("{}", space), "Ω_Q(x)");
    }

    #[test]
    fn test_inclusion_morphism() {
        let source = DifferentialsSpace::<Rational>::new("Q(x)".to_string());
        let target = DifferentialsSpace::<Rational>::new("Q(x)[y]".to_string());
        let morphism = DifferentialsSpaceInclusion::new(source, target);

        assert_eq!(morphism.source().function_field(), "Q(x)");
        assert_eq!(morphism.target().function_field(), "Q(x)[y]");
    }

    #[test]
    fn test_inclusion_apply() {
        let source = DifferentialsSpace::<Rational>::new("Q(x)".to_string());
        let target = DifferentialsSpace::<Rational>::new("Q(x)[y]".to_string());
        let morphism = DifferentialsSpaceInclusion::new(source.clone(), target);

        let omega = source.element("x".to_string());
        let mapped = morphism.apply(&omega);

        assert_eq!(mapped.coefficient(), omega.coefficient());
    }

    #[test]
    fn test_divisor() {
        let omega = FunctionFieldDifferential::<Rational>::new("x^2".to_string(), "x".to_string());
        let div = omega.divisor();

        assert!(div.contains("div"));
        assert!(div.contains("x^2"));
    }

    #[test]
    fn test_valuation() {
        let omega = FunctionFieldDifferential::<Rational>::new("x^2".to_string(), "x".to_string());
        let val = omega.valuation("P");

        // Placeholder returns 0
        assert_eq!(val, 0);
    }

    #[test]
    fn test_residue() {
        let omega = FunctionFieldDifferential::<Rational>::new("1/x".to_string(), "x".to_string());
        let res = omega.residue("0");

        assert!(res.contains("Res"));
        assert!(res.contains("1/x"));
    }

    #[test]
    fn test_cartier_operator() {
        let omega = FunctionFieldDifferential::<Rational>::new("x".to_string(), "x".to_string());
        let cartier = omega.cartier();

        // Placeholder returns zero
        assert!(cartier.is_zero());
    }

    #[test]
    fn test_extract_variable() {
        assert_eq!(extract_variable("Q(x)"), "x");
        assert_eq!(extract_variable("F_p(t)"), "t");
        assert_eq!(extract_variable("K(y)"), "y");
        assert_eq!(extract_variable("NoVar"), "x"); // default
    }

    #[test]
    fn test_equality() {
        let omega1 = FunctionFieldDifferential::<Rational>::new("x".to_string(), "x".to_string());
        let omega2 = FunctionFieldDifferential::<Rational>::new("x".to_string(), "x".to_string());
        let omega3 = FunctionFieldDifferential::<Rational>::new("y".to_string(), "x".to_string());

        assert_eq!(omega1, omega2);
        assert_ne!(omega1, omega3);
    }

    #[test]
    fn test_add_with_zero() {
        let omega = FunctionFieldDifferential::<Rational>::new("x".to_string(), "x".to_string());
        let zero = FunctionFieldDifferential::<Rational>::zero("x".to_string());

        let sum1 = omega.clone() + zero.clone();
        let sum2 = zero + omega.clone();

        assert_eq!(sum1.coefficient(), omega.coefficient());
        assert_eq!(sum2.coefficient(), omega.coefficient());
    }
}
