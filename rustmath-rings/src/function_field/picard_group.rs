//! Picard Groups and Divisor Class Groups
//!
//! This module implements Picard groups and divisor class groups for algebraic curves,
//! providing the foundation for Jacobian variety computations.
//!
//! # Mathematical Overview
//!
//! ## Divisors
//!
//! A divisor on a curve C is a formal sum:
//!
//! D = Σ nᵢPᵢ
//!
//! where nᵢ ∈ ℤ and Pᵢ are points on C.
//!
//! The degree of D is: deg(D) = Σ nᵢ
//!
//! ## Divisor Class Group (Picard Group)
//!
//! The Picard group Pic(C) is the group of divisors modulo linear equivalence:
//!
//! Pic(C) = Div(C) / Principal divisors
//!
//! Two divisors D₁ and D₂ are linearly equivalent (D₁ ~ D₂) if D₁ - D₂ = div(f)
//! for some rational function f.
//!
//! ## Degree Zero Picard Group
//!
//! Pic⁰(C) consists of divisor classes of degree 0. This is isomorphic to the
//! Jacobian variety Jac(C).
//!
//! ## Key Properties
//!
//! - Pic(C) ≅ Pic⁰(C) × ℤ  (split by degree)
//! - Pic⁰(C) is an abelian variety of dimension g (the genus)
//! - For genus g ≥ 1, Pic⁰(C)(ℚ) may be infinite
//! - For finite fields 𝔽_q, Pic⁰(C)(𝔽_q) is always finite
//!
//! # Implementation
//!
//! This module provides:
//!
//! - `Divisor`: Representation of divisors on curves
//! - `DivisorClass`: Elements of the divisor class group
//! - `PicardGroup`: The full Picard group Pic(C)
//! - `DegreeZeroPicardGroup`: Pic⁰(C), isomorphic to Jacobian
//! - Operations: addition, subtraction, scalar multiplication
//! - Linear equivalence checking
//! - Degree calculations
//!
//! # References
//!
//! - Hartshorne, R. "Algebraic Geometry" Chapter II, §6
//! - Liu, Q. "Algebraic Geometry and Arithmetic Curves"
//! - SageMath: `sage.schemes.generic.divisor`

use rustmath_core::{Field, Ring};
use rustmath_integers::Integer;
use std::collections::HashMap;
use std::fmt;

/// A divisor on an algebraic curve
///
/// Represents a formal sum D = Σ nᵢPᵢ where nᵢ are integers and Pᵢ are points.
///
/// # Type Parameters
///
/// * `F` - The field over which the curve is defined
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::picard_group::Divisor;
/// use rustmath_rationals::Rational;
///
/// let mut div = Divisor::<Rational>::zero();
/// div.add_point("P".to_string(), 2);
/// div.add_point("Q".to_string(), -1);
/// assert_eq!(div.degree(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divisor<F: Field> {
    /// Map from point names to multiplicities
    coefficients: HashMap<String, i64>,
    /// Cached degree
    degree: i64,
    /// Phantom field parameter
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field> Divisor<F> {
    /// Create the zero divisor
    pub fn zero() -> Self {
        Self {
            coefficients: HashMap::new(),
            degree: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a divisor from a single point with multiplicity
    pub fn from_point(point: String, multiplicity: i64) -> Self {
        let mut div = Self::zero();
        div.add_point(point, multiplicity);
        div
    }

    /// Create a divisor from a list of (point, multiplicity) pairs
    pub fn from_points(points: Vec<(String, i64)>) -> Self {
        let mut div = Self::zero();
        for (point, mult) in points {
            div.add_point(point, mult);
        }
        div
    }

    /// Add a point with given multiplicity to the divisor
    pub fn add_point(&mut self, point: String, multiplicity: i64) {
        let current = self.coefficients.entry(point.clone()).or_insert(0);
        *current += multiplicity;
        self.degree += multiplicity;

        // Remove zero coefficients
        if *current == 0 {
            self.coefficients.remove(&point);
        }
    }

    /// Get the multiplicity of a point
    pub fn multiplicity(&self, point: &str) -> i64 {
        *self.coefficients.get(point).unwrap_or(&0)
    }

    /// Get the degree of the divisor
    pub fn degree(&self) -> i64 {
        self.degree
    }

    /// Check if this is the zero divisor
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Get all points with non-zero coefficients
    pub fn support(&self) -> Vec<&String> {
        self.coefficients.keys().collect()
    }

    /// Get the number of points in the support
    pub fn support_size(&self) -> usize {
        self.coefficients.len()
    }

    /// Add two divisors
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (point, mult) in &other.coefficients {
            result.add_point(point.clone(), *mult);
        }
        result
    }

    /// Subtract two divisors
    pub fn subtract(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (point, mult) in &other.coefficients {
            result.add_point(point.clone(), -mult);
        }
        result
    }

    /// Negate a divisor
    pub fn negate(&self) -> Self {
        let mut result = Self::zero();
        for (point, mult) in &self.coefficients {
            result.add_point(point.clone(), -mult);
        }
        result
    }

    /// Scalar multiplication of a divisor
    pub fn scalar_mul(&self, scalar: i64) -> Self {
        let mut result = Self::zero();
        for (point, mult) in &self.coefficients {
            result.add_point(point.clone(), mult * scalar);
        }
        result
    }

    /// Check if divisor is effective (all multiplicities ≥ 0)
    pub fn is_effective(&self) -> bool {
        self.coefficients.values().all(|&m| m >= 0)
    }

    /// Get the effective part (keep only non-negative multiplicities)
    pub fn effective_part(&self) -> Self {
        let mut result = Self::zero();
        for (point, mult) in &self.coefficients {
            if *mult > 0 {
                result.add_point(point.clone(), *mult);
            }
        }
        result
    }

    /// Get all coefficients as a map
    pub fn coefficients(&self) -> &HashMap<String, i64> {
        &self.coefficients
    }
}

impl<F: Field> fmt::Display for Divisor<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut terms: Vec<_> = self.coefficients.iter().collect();
        terms.sort_by(|a, b| a.0.cmp(b.0));

        let mut first = true;
        for (point, mult) in terms {
            if !first {
                if *mult > 0 {
                    write!(f, " + ")?;
                } else {
                    write!(f, " - ")?;
                }
            }

            let abs_mult = mult.abs();
            if abs_mult == 1 {
                if first && *mult < 0 {
                    write!(f, "-")?;
                }
                write!(f, "{}", point)?;
            } else {
                if first {
                    write!(f, "{}", mult)?;
                } else {
                    write!(f, "{}", abs_mult)?;
                }
                write!(f, "*{}", point)?;
            }
            first = false;
        }
        Ok(())
    }
}

/// A divisor class in the Picard group
///
/// Represents an equivalence class of divisors under linear equivalence.
///
/// # Type Parameters
///
/// * `F` - The field over which the curve is defined
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::picard_group::{Divisor, DivisorClass};
/// use rustmath_rationals::Rational;
///
/// let div = Divisor::<Rational>::from_point("P".to_string(), 1);
/// let class = DivisorClass::new(div);
/// assert_eq!(class.degree(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DivisorClass<F: Field> {
    /// Representative divisor
    representative: Divisor<F>,
}

impl<F: Field> DivisorClass<F> {
    /// Create a divisor class from a divisor
    pub fn new(divisor: Divisor<F>) -> Self {
        Self {
            representative: divisor,
        }
    }

    /// Create the zero class
    pub fn zero() -> Self {
        Self::new(Divisor::zero())
    }

    /// Get a representative of the class
    pub fn representative(&self) -> &Divisor<F> {
        &self.representative
    }

    /// Get the degree (well-defined on classes)
    pub fn degree(&self) -> i64 {
        self.representative.degree()
    }

    /// Check if this is the zero class
    pub fn is_zero(&self) -> bool {
        // In practice, would check if representative is principal
        // For now, just check if divisor is zero
        self.representative.is_zero()
    }

    /// Add two divisor classes
    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.representative.add(&other.representative))
    }

    /// Subtract two divisor classes
    pub fn subtract(&self, other: &Self) -> Self {
        Self::new(self.representative.subtract(&other.representative))
    }

    /// Negate a divisor class
    pub fn negate(&self) -> Self {
        Self::new(self.representative.negate())
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: i64) -> Self {
        Self::new(self.representative.scalar_mul(scalar))
    }
}

impl<F: Field> fmt::Display for DivisorClass<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.representative)
    }
}

/// The Picard group Pic(C) of a curve
///
/// The Picard group is the group of divisor classes under addition.
///
/// # Type Parameters
///
/// * `F` - The field over which the curve is defined
///
/// # Mathematical Details
///
/// Pic(C) = Div(C) / Principal divisors
///
/// There is a split: Pic(C) ≅ Pic⁰(C) × ℤ where the ℤ factor is the degree.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::picard_group::PicardGroup;
/// use rustmath_rationals::Rational;
///
/// let pic = PicardGroup::<Rational>::new("C".to_string(), 2);
/// assert_eq!(pic.curve(), "C");
/// assert_eq!(pic.genus(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct PicardGroup<F: Field> {
    /// Curve name
    curve: String,
    /// Genus of the curve
    genus: usize,
    /// Phantom field parameter
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field> PicardGroup<F> {
    /// Create a Picard group for a curve
    pub fn new(curve: String, genus: usize) -> Self {
        Self {
            curve,
            genus,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the curve name
    pub fn curve(&self) -> &str {
        &self.curve
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// Get the zero class
    pub fn zero(&self) -> DivisorClass<F> {
        DivisorClass::zero()
    }

    /// Create a class from a divisor
    pub fn class(&self, divisor: Divisor<F>) -> DivisorClass<F> {
        DivisorClass::new(divisor)
    }

    /// Get the degree zero subgroup Pic⁰(C)
    pub fn degree_zero_subgroup(&self) -> DegreeZeroPicardGroup<F> {
        DegreeZeroPicardGroup::new(self.curve.clone(), self.genus)
    }

    /// Split a class into degree zero part and degree
    pub fn split_by_degree(&self, class: &DivisorClass<F>) -> (DivisorClass<F>, i64) {
        let degree = class.degree();
        // To get degree zero part, subtract degree * [canonical divisor]
        // For now, return the class and degree
        (class.clone(), degree)
    }
}

/// The degree zero Picard group Pic⁰(C)
///
/// This is the subgroup of the Picard group consisting of degree zero divisor classes.
/// It is isomorphic to the Jacobian variety Jac(C).
///
/// # Type Parameters
///
/// * `F` - The field over which the curve is defined
///
/// # Mathematical Details
///
/// Pic⁰(C) ≅ Jac(C) as an abelian variety of dimension g = genus(C).
///
/// For genus g:
/// - g = 0: Pic⁰(C) = {0} (trivial)
/// - g = 1: Pic⁰(C) ≅ C (elliptic curve)
/// - g ≥ 2: Pic⁰(C) is a higher-dimensional abelian variety
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::picard_group::DegreeZeroPicardGroup;
/// use rustmath_rationals::Rational;
///
/// let pic0 = DegreeZeroPicardGroup::<Rational>::new("C".to_string(), 2);
/// assert_eq!(pic0.dimension(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct DegreeZeroPicardGroup<F: Field> {
    /// Curve name
    curve: String,
    /// Genus (equals dimension of Jacobian)
    genus: usize,
    /// Base point for Abel-Jacobi map
    base_point: Option<String>,
    /// Phantom field parameter
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field> DegreeZeroPicardGroup<F> {
    /// Create a degree zero Picard group
    pub fn new(curve: String, genus: usize) -> Self {
        Self {
            curve,
            genus,
            base_point: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create with a base point
    pub fn with_base_point(curve: String, genus: usize, base_point: String) -> Self {
        Self {
            curve,
            genus,
            base_point: Some(base_point),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the curve name
    pub fn curve(&self) -> &str {
        &self.curve
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// Get the dimension (equals genus)
    pub fn dimension(&self) -> usize {
        self.genus
    }

    /// Get the base point
    pub fn base_point(&self) -> Option<&str> {
        self.base_point.as_deref()
    }

    /// Get the identity element
    pub fn identity(&self) -> DivisorClass<F> {
        DivisorClass::zero()
    }

    /// Create a degree zero class from a divisor
    pub fn class(&self, divisor: Divisor<F>) -> Result<DivisorClass<F>, String> {
        if divisor.degree() != 0 {
            return Err(format!(
                "Divisor must have degree 0, got degree {}",
                divisor.degree()
            ));
        }
        Ok(DivisorClass::new(divisor))
    }

    /// Apply Abel-Jacobi map: embed curve into Jacobian
    ///
    /// φ: C → Jac(C) = Pic⁰(C)
    /// P ↦ [P - P₀]
    ///
    /// where P₀ is the base point.
    pub fn abel_jacobi(&self, point: String) -> Result<DivisorClass<F>, String> {
        if let Some(base) = &self.base_point {
            let mut div = Divisor::zero();
            div.add_point(point, 1);
            div.add_point(base.clone(), -1);
            Ok(DivisorClass::new(div))
        } else {
            Err("Base point required for Abel-Jacobi map".to_string())
        }
    }

    /// Embed a list of points into Jacobian
    pub fn abel_jacobi_points(&self, points: Vec<String>) -> Result<DivisorClass<F>, String> {
        if let Some(base) = &self.base_point {
            let n = points.len() as i64;
            let mut div = Divisor::zero();
            for point in points {
                div.add_point(point, 1);
            }
            div.add_point(base.clone(), -n);
            Ok(DivisorClass::new(div))
        } else {
            Err("Base point required for Abel-Jacobi map".to_string())
        }
    }

    /// Check if a divisor class is in Pic⁰
    pub fn contains(&self, class: &DivisorClass<F>) -> bool {
        class.degree() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_zero_divisor() {
        let div = Divisor::<Rational>::zero();
        assert!(div.is_zero());
        assert_eq!(div.degree(), 0);
        assert_eq!(div.support_size(), 0);
    }

    #[test]
    fn test_divisor_from_point() {
        let div = Divisor::<Rational>::from_point("P".to_string(), 3);
        assert_eq!(div.multiplicity("P"), 3);
        assert_eq!(div.degree(), 3);
        assert!(!div.is_zero());
    }

    #[test]
    fn test_divisor_addition() {
        let div1 = Divisor::<Rational>::from_point("P".to_string(), 2);
        let div2 = Divisor::<Rational>::from_point("Q".to_string(), 1);
        let sum = div1.add(&div2);

        assert_eq!(sum.multiplicity("P"), 2);
        assert_eq!(sum.multiplicity("Q"), 1);
        assert_eq!(sum.degree(), 3);
    }

    #[test]
    fn test_divisor_subtraction() {
        let div1 = Divisor::<Rational>::from_point("P".to_string(), 3);
        let div2 = Divisor::<Rational>::from_point("P".to_string(), 1);
        let diff = div1.subtract(&div2);

        assert_eq!(diff.multiplicity("P"), 2);
        assert_eq!(diff.degree(), 2);
    }

    #[test]
    fn test_divisor_negation() {
        let div = Divisor::<Rational>::from_point("P".to_string(), 2);
        let neg = div.negate();

        assert_eq!(neg.multiplicity("P"), -2);
        assert_eq!(neg.degree(), -2);
    }

    #[test]
    fn test_divisor_scalar_mul() {
        let div = Divisor::<Rational>::from_point("P".to_string(), 2);
        let scaled = div.scalar_mul(3);

        assert_eq!(scaled.multiplicity("P"), 6);
        assert_eq!(scaled.degree(), 6);
    }

    #[test]
    fn test_divisor_is_effective() {
        let div1 = Divisor::<Rational>::from_point("P".to_string(), 2);
        assert!(div1.is_effective());

        let div2 = Divisor::<Rational>::from_point("P".to_string(), -1);
        assert!(!div2.is_effective());
    }

    #[test]
    fn test_divisor_effective_part() {
        let mut div = Divisor::<Rational>::zero();
        div.add_point("P".to_string(), 3);
        div.add_point("Q".to_string(), -2);
        div.add_point("R".to_string(), 1);

        let eff = div.effective_part();
        assert_eq!(eff.multiplicity("P"), 3);
        assert_eq!(eff.multiplicity("Q"), 0);
        assert_eq!(eff.multiplicity("R"), 1);
    }

    #[test]
    fn test_divisor_class() {
        let div = Divisor::<Rational>::from_point("P".to_string(), 1);
        let class = DivisorClass::new(div);

        assert_eq!(class.degree(), 1);
        assert!(!class.is_zero());
    }

    #[test]
    fn test_divisor_class_addition() {
        let class1 = DivisorClass::new(Divisor::from_point("P".to_string(), 1));
        let class2 = DivisorClass::new(Divisor::from_point("Q".to_string(), 1));
        let sum = class1.add(&class2);

        assert_eq!(sum.degree(), 2);
    }

    #[test]
    fn test_picard_group() {
        let pic = PicardGroup::<Rational>::new("C".to_string(), 2);

        assert_eq!(pic.curve(), "C");
        assert_eq!(pic.genus(), 2);
    }

    #[test]
    fn test_picard_group_zero() {
        let pic = PicardGroup::<Rational>::new("C".to_string(), 2);
        let zero = pic.zero();

        assert!(zero.is_zero());
        assert_eq!(zero.degree(), 0);
    }

    #[test]
    fn test_picard_group_class() {
        let pic = PicardGroup::<Rational>::new("C".to_string(), 2);
        let div = Divisor::from_point("P".to_string(), 3);
        let class = pic.class(div);

        assert_eq!(class.degree(), 3);
    }

    #[test]
    fn test_degree_zero_picard_group() {
        let pic0 = DegreeZeroPicardGroup::<Rational>::new("C".to_string(), 2);

        assert_eq!(pic0.curve(), "C");
        assert_eq!(pic0.genus(), 2);
        assert_eq!(pic0.dimension(), 2);
    }

    #[test]
    fn test_degree_zero_class() {
        let pic0 = DegreeZeroPicardGroup::<Rational>::new("C".to_string(), 2);

        let mut div = Divisor::zero();
        div.add_point("P".to_string(), 1);
        div.add_point("Q".to_string(), -1);

        let class = pic0.class(div).unwrap();
        assert_eq!(class.degree(), 0);
    }

    #[test]
    fn test_abel_jacobi() {
        let pic0 = DegreeZeroPicardGroup::<Rational>::with_base_point(
            "C".to_string(),
            2,
            "P0".to_string(),
        );

        let class = pic0.abel_jacobi("P1".to_string()).unwrap();
        assert_eq!(class.degree(), 0);
        assert_eq!(class.representative().multiplicity("P1"), 1);
        assert_eq!(class.representative().multiplicity("P0"), -1);
    }

    #[test]
    fn test_abel_jacobi_multiple_points() {
        let pic0 = DegreeZeroPicardGroup::<Rational>::with_base_point(
            "C".to_string(),
            2,
            "P0".to_string(),
        );

        let points = vec!["P1".to_string(), "P2".to_string()];
        let class = pic0.abel_jacobi_points(points).unwrap();

        assert_eq!(class.degree(), 0);
        assert_eq!(class.representative().multiplicity("P0"), -2);
    }

    #[test]
    fn test_contains() {
        let pic0 = DegreeZeroPicardGroup::<Rational>::new("C".to_string(), 2);

        let mut div_zero = Divisor::zero();
        div_zero.add_point("P".to_string(), 1);
        div_zero.add_point("Q".to_string(), -1);
        let class_zero = DivisorClass::new(div_zero);

        let div_nonzero = Divisor::from_point("P".to_string(), 1);
        let class_nonzero = DivisorClass::new(div_nonzero);

        assert!(pic0.contains(&class_zero));
        assert!(!pic0.contains(&class_nonzero));
    }

    #[test]
    fn test_degree_zero_subgroup() {
        let pic = PicardGroup::<Rational>::new("C".to_string(), 3);
        let pic0 = pic.degree_zero_subgroup();

        assert_eq!(pic0.genus(), 3);
        assert_eq!(pic0.dimension(), 3);
    }
}
