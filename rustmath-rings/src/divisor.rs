//! Divisors on function fields
//!
//! This module provides divisor structures for function fields, corresponding to
//! SageMath's `sage.rings.function_field.divisor`.
//!
//! # Mathematical Background
//!
//! A divisor on a function field is a formal sum D = ∑ nₚ·P where:
//! - P ranges over places (points) of the function field
//! - nₚ ∈ ℤ are multiplicities
//! - Only finitely many nₚ are non-zero
//!
//! Key properties:
//! - **Degree**: deg(D) = ∑ nₚ·deg(P)
//! - **Addition**: Divisors form an abelian group
//! - **Principal divisors**: (f) = ∑ vₚ(f)·P for f ∈ K*
//! - **Riemann-Roch theorem**: ℓ(D) - ℓ(K_C - D) = deg(D) - g + 1
//!
//! where:
//! - ℓ(D) = dim L(D) is the dimension of the Riemann-Roch space
//! - K_C is the canonical divisor
//! - g is the genus of the function field
//!
//! # Key Types
//!
//! - `FunctionFieldDivisor<F>`: A divisor (formal sum of places)
//! - `DivisorGroup<F>`: The group of all divisors
//! - Helper functions for divisor construction
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::divisor::*;
//! use rustmath_rationals::Rational;
//!
//! // Create a prime divisor at a place
//! let D = prime_divisor("P".to_string(), 1);
//!
//! // Create a divisor from multiple places
//! let div = divisor(vec![("P", 2), ("Q", -1)]);
//! ```

use rustmath_core::{Ring, Field};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::fmt;
use std::ops::{Add, Sub, Neg, Mul};

/// Place on a function field
///
/// A place is a discrete valuation (equivalently, a prime ideal or a point
/// on the associated curve).
///
/// In SageMath, places are implemented separately, but here we use a
/// simplified string representation.
pub type Place = String;

/// Divisor on a function field
///
/// Represents a formal sum D = ∑ nₚ·P where P are places and nₚ ∈ ℤ.
///
/// Corresponds to SageMath's `FunctionFieldDivisor` class.
///
/// # Type Parameters
///
/// - `F`: The base field type
#[derive(Clone, Debug)]
pub struct FunctionFieldDivisor<F: Field> {
    /// Map from places to multiplicities
    /// Only non-zero multiplicities are stored
    places: HashMap<Place, i64>,
    _field: PhantomData<F>,
}

impl<F: Field> FunctionFieldDivisor<F> {
    /// Create a new divisor
    ///
    /// # Arguments
    ///
    /// * `places` - HashMap of places to multiplicities
    pub fn new(places: HashMap<Place, i64>) -> Self {
        // Filter out zero multiplicities
        let filtered: HashMap<Place, i64> = places
            .into_iter()
            .filter(|(_, mult)| *mult != 0)
            .collect();

        FunctionFieldDivisor {
            places: filtered,
            _field: PhantomData,
        }
    }

    /// Create the zero divisor
    pub fn zero() -> Self {
        FunctionFieldDivisor {
            places: HashMap::new(),
            _field: PhantomData,
        }
    }

    /// Check if this is the zero divisor
    pub fn is_zero(&self) -> bool {
        self.places.is_empty()
    }

    /// Get the multiplicity at a place
    ///
    /// Returns 0 if the place doesn't appear in the divisor.
    pub fn multiplicity(&self, place: &Place) -> i64 {
        self.places.get(place).copied().unwrap_or(0)
    }

    /// Get all places with non-zero multiplicity
    pub fn support(&self) -> Vec<Place> {
        self.places.keys().cloned().collect()
    }

    /// Compute the degree of the divisor
    ///
    /// deg(D) = ∑ nₚ·deg(P)
    ///
    /// For now, assumes all places have degree 1 (rational places).
    /// A full implementation would track place degrees.
    pub fn degree(&self) -> i64 {
        self.places.values().sum()
    }

    /// Check if this is an effective divisor
    ///
    /// A divisor is effective if all multiplicities are non-negative.
    pub fn is_effective(&self) -> bool {
        self.places.values().all(|&mult| mult >= 0)
    }

    /// Split into positive and negative parts
    ///
    /// Returns (D⁺, D⁻) where D = D⁺ - D⁻ with both D⁺, D⁻ effective.
    pub fn split(&self) -> (Self, Self) {
        let mut positive = HashMap::new();
        let mut negative = HashMap::new();

        for (place, &mult) in &self.places {
            if mult > 0 {
                positive.insert(place.clone(), mult);
            } else if mult < 0 {
                negative.insert(place.clone(), -mult);
            }
        }

        (
            FunctionFieldDivisor::new(positive),
            FunctionFieldDivisor::new(negative),
        )
    }

    /// Get the numerator (positive part)
    pub fn numerator(&self) -> Self {
        self.split().0
    }

    /// Get the denominator (negative part, made positive)
    pub fn denominator(&self) -> Self {
        self.split().1
    }

    /// Compute the dimension of the Riemann-Roch space L(D)
    ///
    /// L(D) = {f ∈ K* : (f) + D ≥ 0} ∪ {0}
    ///
    /// Placeholder: would use Riemann-Roch theorem or Hess' algorithm
    pub fn dimension(&self) -> usize {
        // Placeholder: simple bound from Riemann-Roch
        if self.degree() < 0 {
            0
        } else {
            (self.degree() + 1) as usize
        }
    }

    /// Compute a basis for the Riemann-Roch space L(D)
    ///
    /// Uses Hess' algorithm in SageMath.
    /// Placeholder implementation.
    pub fn basis(&self) -> Vec<String> {
        // Placeholder: would compute actual basis
        vec!["1".to_string()]
    }

    /// Compute the dimension of the differential space Ω(D)
    ///
    /// Ω(D) = {ω differential : (ω) ≥ D}
    pub fn differential_space_dimension(&self) -> usize {
        // Placeholder: uses duality in Riemann-Roch
        0
    }

    /// Compute a basis for the differential space Ω(D)
    pub fn differential_basis(&self) -> Vec<String> {
        // Placeholder
        vec![]
    }

    /// Check if a function field element is in L(D)
    ///
    /// Returns true if f ∈ L(D), i.e., (f) + D ≥ 0
    pub fn contains(&self, _element: &str) -> bool {
        // Placeholder: would check valuations
        false
    }

    /// Compute the principal divisor of a function
    ///
    /// (f) = ∑ vₚ(f)·P
    ///
    /// # Arguments
    ///
    /// * `function` - A function field element
    pub fn principal_divisor(function: String) -> Self {
        // Placeholder: would compute valuations at all places
        FunctionFieldDivisor::zero()
    }

    /// Get the list of (place, multiplicity) pairs
    pub fn list(&self) -> Vec<(Place, i64)> {
        self.places.iter().map(|(p, &m)| (p.clone(), m)).collect()
    }
}

impl<F: Field> Add for FunctionFieldDivisor<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self.places.clone();

        for (place, mult) in other.places {
            *result.entry(place).or_insert(0) += mult;
        }

        FunctionFieldDivisor::new(result)
    }
}

impl<F: Field> Sub for FunctionFieldDivisor<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = self.places.clone();

        for (place, mult) in other.places {
            *result.entry(place).or_insert(0) -= mult;
        }

        FunctionFieldDivisor::new(result)
    }
}

impl<F: Field> Neg for FunctionFieldDivisor<F> {
    type Output = Self;

    fn neg(self) -> Self {
        let negated: HashMap<Place, i64> = self
            .places
            .into_iter()
            .map(|(p, m)| (p, -m))
            .collect();

        FunctionFieldDivisor::new(negated)
    }
}

impl<F: Field> Mul<i64> for FunctionFieldDivisor<F> {
    type Output = Self;

    /// Scalar multiplication: n·D
    fn mul(self, scalar: i64) -> Self {
        if scalar == 0 {
            return Self::zero();
        }

        let scaled: HashMap<Place, i64> = self
            .places
            .into_iter()
            .map(|(p, m)| (p, m * scalar))
            .collect();

        FunctionFieldDivisor::new(scaled)
    }
}

impl<F: Field> PartialEq for FunctionFieldDivisor<F> {
    fn eq(&self, other: &Self) -> bool {
        if self.places.len() != other.places.len() {
            return false;
        }

        for (place, &mult) in &self.places {
            if other.multiplicity(place) != mult {
                return false;
            }
        }

        true
    }
}

impl<F: Field> Eq for FunctionFieldDivisor<F> {}

impl<F: Field> fmt::Display for FunctionFieldDivisor<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut parts: Vec<String> = self
            .places
            .iter()
            .map(|(place, &mult)| {
                if mult == 1 {
                    place.clone()
                } else {
                    format!("{}*{}", mult, place)
                }
            })
            .collect();

        parts.sort(); // For consistent display
        write!(f, "{}", parts.join(" + "))
    }
}

/// Group of divisors on a function field
///
/// Represents the abelian group of all divisors on a function field.
///
/// Corresponds to SageMath's `DivisorGroup` class.
///
/// # Type Parameters
///
/// - `F`: The base field type
#[derive(Clone, Debug)]
pub struct DivisorGroup<F: Field> {
    /// The underlying function field
    function_field: String,
    _field: PhantomData<F>,
}

impl<F: Field> DivisorGroup<F> {
    /// Create a new divisor group
    ///
    /// # Arguments
    ///
    /// * `field` - Identifier of the function field
    pub fn new(field: String) -> Self {
        DivisorGroup {
            function_field: field,
            _field: PhantomData,
        }
    }

    /// Get the underlying function field
    pub fn function_field(&self) -> &str {
        &self.function_field
    }

    /// Create a divisor element
    ///
    /// # Arguments
    ///
    /// * `places` - HashMap of places to multiplicities
    pub fn element(&self, places: HashMap<Place, i64>) -> FunctionFieldDivisor<F> {
        FunctionFieldDivisor::new(places)
    }

    /// The zero divisor
    pub fn zero(&self) -> FunctionFieldDivisor<F> {
        FunctionFieldDivisor::zero()
    }

    /// Create a prime divisor (single place with multiplicity 1)
    ///
    /// # Arguments
    ///
    /// * `place` - The place
    pub fn prime(&self, place: Place) -> FunctionFieldDivisor<F> {
        let mut places = HashMap::new();
        places.insert(place, 1);
        FunctionFieldDivisor::new(places)
    }

    /// Compute the canonical divisor K_C
    ///
    /// The canonical divisor is the divisor of any non-zero differential.
    /// All differentials have the same divisor (up to equivalence).
    pub fn canonical_divisor(&self) -> FunctionFieldDivisor<F> {
        // Placeholder: would compute actual canonical divisor
        FunctionFieldDivisor::zero()
    }

    /// Get the genus of the function field
    ///
    /// The genus g satisfies deg(K_C) = 2g - 2.
    pub fn genus(&self) -> usize {
        // Placeholder: would compute actual genus
        0
    }
}

impl<F: Field> fmt::Display for DivisorGroup<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Div({})", self.function_field)
    }
}

/// Create a divisor from a list of (place, multiplicity) pairs
///
/// Corresponds to SageMath's `divisor()` function.
///
/// # Arguments
///
/// * `data` - List of (place, multiplicity) pairs
///
/// # Examples
///
/// ```ignore
/// let D = divisor(vec![("P".to_string(), 2), ("Q".to_string(), -1)]);
/// // Creates 2·P - Q
/// ```
pub fn divisor<F: Field>(data: Vec<(Place, i64)>) -> FunctionFieldDivisor<F> {
    let mut places = HashMap::new();
    for (place, mult) in data {
        *places.entry(place).or_insert(0) += mult;
    }
    FunctionFieldDivisor::new(places)
}

/// Create a prime divisor (single place with multiplicity n)
///
/// Corresponds to SageMath's `prime_divisor()` function.
///
/// # Arguments
///
/// * `place` - The place
/// * `multiplicity` - The multiplicity (default 1)
///
/// # Examples
///
/// ```ignore
/// let P = prime_divisor("P".to_string(), 1); // Just P
/// let twoP = prime_divisor("P".to_string(), 2); // 2·P
/// ```
pub fn prime_divisor<F: Field>(place: Place, multiplicity: i64) -> FunctionFieldDivisor<F> {
    let mut places = HashMap::new();
    places.insert(place, multiplicity);
    FunctionFieldDivisor::new(places)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_divisor_creation() {
        let mut places = HashMap::new();
        places.insert("P".to_string(), 2);
        places.insert("Q".to_string(), -1);

        let div = FunctionFieldDivisor::<Rational>::new(places);

        assert_eq!(div.multiplicity(&"P".to_string()), 2);
        assert_eq!(div.multiplicity(&"Q".to_string()), -1);
        assert_eq!(div.multiplicity(&"R".to_string()), 0);
    }

    #[test]
    fn test_zero_divisor() {
        let zero = FunctionFieldDivisor::<Rational>::zero();

        assert!(zero.is_zero());
        assert_eq!(zero.degree(), 0);
        assert!(zero.support().is_empty());
    }

    #[test]
    fn test_divisor_degree() {
        let mut places = HashMap::new();
        places.insert("P".to_string(), 2);
        places.insert("Q".to_string(), 3);

        let div = FunctionFieldDivisor::<Rational>::new(places);

        assert_eq!(div.degree(), 5);
    }

    #[test]
    fn test_divisor_addition() {
        let div1 = prime_divisor::<Rational>("P".to_string(), 2);
        let div2 = prime_divisor::<Rational>("Q".to_string(), 1);

        let sum = div1 + div2;

        assert_eq!(sum.multiplicity(&"P".to_string()), 2);
        assert_eq!(sum.multiplicity(&"Q".to_string()), 1);
    }

    #[test]
    fn test_divisor_subtraction() {
        let div1 = prime_divisor::<Rational>("P".to_string(), 2);
        let div2 = prime_divisor::<Rational>("P".to_string(), 1);

        let diff = div1 - div2;

        assert_eq!(diff.multiplicity(&"P".to_string()), 1);
    }

    #[test]
    fn test_divisor_negation() {
        let div = prime_divisor::<Rational>("P".to_string(), 2);
        let neg = -div;

        assert_eq!(neg.multiplicity(&"P".to_string()), -2);
    }

    #[test]
    fn test_divisor_scalar_multiplication() {
        let div = prime_divisor::<Rational>("P".to_string(), 2);
        let scaled = div * 3;

        assert_eq!(scaled.multiplicity(&"P".to_string()), 6);
    }

    #[test]
    fn test_scalar_mult_by_zero() {
        let div = prime_divisor::<Rational>("P".to_string(), 2);
        let zero = div * 0;

        assert!(zero.is_zero());
    }

    #[test]
    fn test_is_effective() {
        let eff = prime_divisor::<Rational>("P".to_string(), 2);
        assert!(eff.is_effective());

        let non_eff = prime_divisor::<Rational>("P".to_string(), -1);
        assert!(!non_eff.is_effective());
    }

    #[test]
    fn test_split() {
        let mut places = HashMap::new();
        places.insert("P".to_string(), 2);
        places.insert("Q".to_string(), -1);
        places.insert("R".to_string(), 3);

        let div = FunctionFieldDivisor::<Rational>::new(places);
        let (pos, neg) = div.split();

        assert_eq!(pos.multiplicity(&"P".to_string()), 2);
        assert_eq!(pos.multiplicity(&"R".to_string()), 3);
        assert_eq!(neg.multiplicity(&"Q".to_string()), 1);
    }

    #[test]
    fn test_numerator_denominator() {
        let mut places = HashMap::new();
        places.insert("P".to_string(), 2);
        places.insert("Q".to_string(), -1);

        let div = FunctionFieldDivisor::<Rational>::new(places);

        let num = div.numerator();
        let den = div.denominator();

        assert_eq!(num.multiplicity(&"P".to_string()), 2);
        assert_eq!(den.multiplicity(&"Q".to_string()), 1);
    }

    #[test]
    fn test_support() {
        let mut places = HashMap::new();
        places.insert("P".to_string(), 2);
        places.insert("Q".to_string(), -1);

        let div = FunctionFieldDivisor::<Rational>::new(places);
        let support = div.support();

        assert_eq!(support.len(), 2);
        assert!(support.contains(&"P".to_string()));
        assert!(support.contains(&"Q".to_string()));
    }

    #[test]
    fn test_divisor_display() {
        let div = prime_divisor::<Rational>("P".to_string(), 2);
        let display = format!("{}", div);

        assert!(display.contains("P"));

        let zero = FunctionFieldDivisor::<Rational>::zero();
        assert_eq!(format!("{}", zero), "0");
    }

    #[test]
    fn test_divisor_equality() {
        let div1 = prime_divisor::<Rational>("P".to_string(), 2);
        let div2 = prime_divisor::<Rational>("P".to_string(), 2);
        let div3 = prime_divisor::<Rational>("Q".to_string(), 2);

        assert_eq!(div1, div2);
        assert_ne!(div1, div3);
    }

    #[test]
    fn test_divisor_group() {
        let group = DivisorGroup::<Rational>::new("Q(x)".to_string());

        assert_eq!(group.function_field(), "Q(x)");

        let zero = group.zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_group_prime_divisor() {
        let group = DivisorGroup::<Rational>::new("Q(x)".to_string());
        let P = group.prime("P".to_string());

        assert_eq!(P.multiplicity(&"P".to_string()), 1);
    }

    #[test]
    fn test_divisor_function() {
        let div = divisor::<Rational>(vec![
            ("P".to_string(), 2),
            ("Q".to_string(), -1),
        ]);

        assert_eq!(div.multiplicity(&"P".to_string()), 2);
        assert_eq!(div.multiplicity(&"Q".to_string()), -1);
    }

    #[test]
    fn test_prime_divisor_function() {
        let P = prime_divisor::<Rational>("P".to_string(), 1);
        let twoP = prime_divisor::<Rational>("P".to_string(), 2);

        assert_eq!(P.multiplicity(&"P".to_string()), 1);
        assert_eq!(twoP.multiplicity(&"P".to_string()), 2);
    }

    #[test]
    fn test_dimension() {
        let div = prime_divisor::<Rational>("P".to_string(), 2);
        let dim = div.dimension();

        // Placeholder returns degree + 1
        assert_eq!(dim, 3);

        let neg = prime_divisor::<Rational>("P".to_string(), -1);
        assert_eq!(neg.dimension(), 0);
    }

    #[test]
    fn test_list() {
        let div = divisor::<Rational>(vec![
            ("P".to_string(), 2),
            ("Q".to_string(), -1),
        ]);

        let list = div.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_zero_multiplicity_filtered() {
        let mut places = HashMap::new();
        places.insert("P".to_string(), 2);
        places.insert("Q".to_string(), 0); // Should be filtered out

        let div = FunctionFieldDivisor::<Rational>::new(places);

        assert_eq!(div.support().len(), 1);
        assert!(div.support().contains(&"P".to_string()));
    }

    #[test]
    fn test_add_to_zero() {
        let div = prime_divisor::<Rational>("P".to_string(), 2);
        let zero = FunctionFieldDivisor::<Rational>::zero();

        let sum = div.clone() + zero;
        assert_eq!(sum, div);
    }

    #[test]
    fn test_subtract_itself() {
        let div = prime_divisor::<Rational>("P".to_string(), 2);
        let diff = div.clone() - div;

        assert!(diff.is_zero());
    }
}
