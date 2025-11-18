//! Places of function fields
//!
//! This module provides place structures for function fields, corresponding to
//! SageMath's `sage.rings.function_field.place`.
//!
//! # Mathematical Background
//!
//! A place of a function field K is a discrete valuation on K that is trivial
//! on the constant field. Places generalize the notion of primes in number theory.
//!
//! Key properties:
//! - Each place P has a valuation v_P: K* → ℤ
//! - The degree deg(P) is the degree of the residue field extension
//! - Places correspond to prime ideals in the ring of integers
//! - The divisor group Div(K) is the free abelian group on places
//!
//! # Riemann-Roch Theorem
//!
//! For a divisor D and function field K of genus g:
//! dim L(D) - dim L(K - D) = deg(D) + 1 - g
//!
//! where L(D) = {f ∈ K : div(f) + D ≥ 0}.
//!
//! # Key Types
//!
//! - `FunctionFieldPlace`: A place of a function field
//! - `PlaceSet`: The set of all places of a function field
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::function_field::place::*;
//!
//! // Create a place
//! let place = FunctionFieldPlace::new("P".to_string(), 1);
//!
//! // Check degree
//! assert_eq!(place.degree(), 1);
//!
//! // Create the place set
//! let places = PlaceSet::new("Q(x)".to_string());
//! ```

use rustmath_core::Field;
use std::marker::PhantomData;
use std::fmt;
use std::collections::HashSet;

/// A place of a function field
///
/// Represents a discrete valuation/place of a function field.
/// This corresponds to SageMath's `FunctionFieldPlace` class.
///
/// # Type Parameters
///
/// - `F`: The constant field type
///
/// # Mathematical Details
///
/// A place P determines:
/// - A discrete valuation v_P: K* → ℤ
/// - A residue field κ(P) = O_P/m_P
/// - A degree deg(P) = [κ(P) : k]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionFieldPlace<F: Field> {
    /// Name of the place
    name: String,
    /// Degree of the place
    degree: usize,
    /// Whether this is a finite place
    is_finite: bool,
    /// Field marker
    field_marker: PhantomData<F>,
}

impl<F: Field> FunctionFieldPlace<F> {
    /// Create a new function field place
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the place
    /// * `degree` - Degree of the place
    ///
    /// # Returns
    ///
    /// A new FunctionFieldPlace instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let place = FunctionFieldPlace::new("P".to_string(), 1);
    /// assert_eq!(place.degree(), 1);
    /// ```
    pub fn new(name: String, degree: usize) -> Self {
        assert!(degree > 0, "Degree must be positive");
        FunctionFieldPlace {
            name,
            degree,
            is_finite: true,
            field_marker: PhantomData,
        }
    }

    /// Create an infinite place
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the place
    /// * `degree` - Degree of the place
    ///
    /// # Returns
    ///
    /// An infinite place
    pub fn infinite(name: String, degree: usize) -> Self {
        FunctionFieldPlace {
            name,
            degree,
            is_finite: false,
            field_marker: PhantomData,
        }
    }

    /// Get the name of the place
    ///
    /// # Returns
    ///
    /// Name of the place
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the degree
    ///
    /// # Returns
    ///
    /// Degree of the place
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Check if this is a finite place
    ///
    /// # Returns
    ///
    /// True if the place is finite
    pub fn is_finite(&self) -> bool {
        self.is_finite
    }

    /// Check if this is an infinite place
    ///
    /// # Returns
    ///
    /// True if the place is infinite
    pub fn is_infinite(&self) -> bool {
        !self.is_finite
    }

    /// Check if this is a degree 1 place
    ///
    /// # Returns
    ///
    /// True if degree equals 1
    pub fn is_degree_one(&self) -> bool {
        self.degree == 1
    }

    /// Compute local uniformizer (symbolic)
    ///
    /// # Returns
    ///
    /// String representation of a local uniformizer
    pub fn local_uniformizer(&self) -> String {
        format!("t_{}", self.name)
    }

    /// Valuation of an element (symbolic)
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    ///
    /// # Returns
    ///
    /// String representation of the valuation
    pub fn valuation(&self, element: &str) -> String {
        format!("v_{}({})", self.name, element)
    }
}

impl<F: Field> fmt::Display for FunctionFieldPlace<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_finite {
            write!(f, "Place {} (degree {})", self.name, self.degree)
        } else {
            write!(f, "Infinite place {} (degree {})", self.name, self.degree)
        }
    }
}

/// Set of all places of a function field
///
/// Represents the collection of all places of a function field.
/// This corresponds to SageMath's `PlaceSet` class.
///
/// # Type Parameters
///
/// - `F`: The constant field type
///
/// # Mathematical Details
///
/// The place set forms:
/// - A set (no natural ordering in general)
/// - The support for the divisor group
/// - The prime spectrum of the function field
#[derive(Clone, Debug)]
pub struct PlaceSet<F: Field> {
    /// Function field name
    function_field: String,
    /// Known places
    places: HashSet<String>,
    /// Genus of the function field
    genus: Option<usize>,
    /// Field marker
    field_marker: PhantomData<F>,
}

impl<F: Field> PlaceSet<F> {
    /// Create a new place set
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    ///
    /// # Returns
    ///
    /// A new PlaceSet instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let places = PlaceSet::new("Q(x)".to_string());
    /// ```
    pub fn new(function_field: String) -> Self {
        PlaceSet {
            function_field,
            places: HashSet::new(),
            genus: None,
            field_marker: PhantomData,
        }
    }

    /// Create with known genus
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    /// * `genus` - Genus of the function field
    ///
    /// # Returns
    ///
    /// A new PlaceSet with genus set
    pub fn with_genus(function_field: String, genus: usize) -> Self {
        PlaceSet {
            function_field,
            places: HashSet::new(),
            genus: Some(genus),
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

    /// Get the genus
    ///
    /// # Returns
    ///
    /// The genus if known
    pub fn genus(&self) -> Option<usize> {
        self.genus
    }

    /// Set the genus
    ///
    /// # Arguments
    ///
    /// * `g` - Genus value
    pub fn set_genus(&mut self, g: usize) {
        self.genus = Some(g);
    }

    /// Add a place
    ///
    /// # Arguments
    ///
    /// * `place` - String representation of the place
    pub fn add_place(&mut self, place: String) {
        self.places.insert(place);
    }

    /// Get the number of known places
    ///
    /// # Returns
    ///
    /// Number of places in the set
    pub fn num_places(&self) -> usize {
        self.places.len()
    }

    /// Check if a place is in the set
    ///
    /// # Arguments
    ///
    /// * `place` - String representation of the place
    ///
    /// # Returns
    ///
    /// True if the place is in the set
    pub fn contains(&self, place: &str) -> bool {
        self.places.contains(place)
    }

    /// Create a degree 1 place
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the place
    ///
    /// # Returns
    ///
    /// A degree 1 place
    pub fn degree_one_place(&self, name: String) -> FunctionFieldPlace<F> {
        FunctionFieldPlace::new(name, 1)
    }

    /// Count places of a given degree (theoretical bound)
    ///
    /// # Arguments
    ///
    /// * `degree` - Degree to count
    /// * `field_size` - Size of constant field
    ///
    /// # Returns
    ///
    /// Approximate number of degree d places
    pub fn count_places_of_degree(&self, degree: usize, field_size: usize) -> usize {
        // Approximation: roughly q^d / d places of degree d
        if degree == 0 {
            return 0;
        }
        field_size.pow(degree as u32) / degree
    }
}

impl<F: Field> fmt::Display for PlaceSet<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PlaceSet({}, {} places",
            self.function_field,
            self.places.len()
        )?;
        if let Some(g) = self.genus {
            write!(f, ", genus {}", g)?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_place_creation() {
        let place: FunctionFieldPlace<Rational> = FunctionFieldPlace::new("P".to_string(), 1);

        assert_eq!(place.name(), "P");
        assert_eq!(place.degree(), 1);
        assert!(place.is_finite());
        assert!(!place.is_infinite());
        assert!(place.is_degree_one());
    }

    #[test]
    fn test_infinite_place() {
        let place: FunctionFieldPlace<Rational> =
            FunctionFieldPlace::infinite("∞".to_string(), 1);

        assert!(place.is_infinite());
        assert!(!place.is_finite());
    }

    #[test]
    fn test_place_degree() {
        let place: FunctionFieldPlace<Rational> = FunctionFieldPlace::new("P".to_string(), 3);

        assert_eq!(place.degree(), 3);
        assert!(!place.is_degree_one());
    }

    #[test]
    #[should_panic(expected = "Degree must be positive")]
    fn test_invalid_degree() {
        let _place: FunctionFieldPlace<Rational> = FunctionFieldPlace::new("P".to_string(), 0);
    }

    #[test]
    fn test_local_uniformizer() {
        let place: FunctionFieldPlace<Rational> = FunctionFieldPlace::new("P".to_string(), 1);

        let uniformizer = place.local_uniformizer();
        assert!(uniformizer.contains("t_P"));
    }

    #[test]
    fn test_valuation() {
        let place: FunctionFieldPlace<Rational> = FunctionFieldPlace::new("P".to_string(), 1);

        let val = place.valuation("x");
        assert!(val.contains("v_P"));
        assert!(val.contains("x"));
    }

    #[test]
    fn test_place_set_creation() {
        let places: PlaceSet<Rational> = PlaceSet::new("Q(x)".to_string());

        assert_eq!(places.function_field(), "Q(x)");
        assert_eq!(places.num_places(), 0);
        assert_eq!(places.genus(), None);
    }

    #[test]
    fn test_place_set_with_genus() {
        let places: PlaceSet<Rational> = PlaceSet::with_genus("Q(x)".to_string(), 0);

        assert_eq!(places.genus(), Some(0));
    }

    #[test]
    fn test_add_place() {
        let mut places: PlaceSet<Rational> = PlaceSet::new("Q(x)".to_string());

        places.add_place("P1".to_string());
        places.add_place("P2".to_string());

        assert_eq!(places.num_places(), 2);
        assert!(places.contains("P1"));
        assert!(places.contains("P2"));
    }

    #[test]
    fn test_set_genus() {
        let mut places: PlaceSet<Rational> = PlaceSet::new("K".to_string());

        assert_eq!(places.genus(), None);
        places.set_genus(3);
        assert_eq!(places.genus(), Some(3));
    }

    #[test]
    fn test_degree_one_place() {
        let places: PlaceSet<Rational> = PlaceSet::new("Q(x)".to_string());
        let place = places.degree_one_place("P".to_string());

        assert_eq!(place.degree(), 1);
        assert!(place.is_degree_one());
    }

    #[test]
    fn test_count_places() {
        let places: PlaceSet<Rational> = PlaceSet::new("K".to_string());

        // For field size q, approximately q^d / d places of degree d
        assert_eq!(places.count_places_of_degree(1, 2), 2); // 2^1 / 1 = 2
        assert_eq!(places.count_places_of_degree(2, 2), 2); // 2^2 / 2 = 2
        assert_eq!(places.count_places_of_degree(1, 3), 3); // 3^1 / 1 = 3
    }

    #[test]
    fn test_place_display() {
        let place: FunctionFieldPlace<Rational> = FunctionFieldPlace::new("P".to_string(), 2);

        let display = format!("{}", place);
        assert!(display.contains("Place P"));
        assert!(display.contains("degree 2"));
    }

    #[test]
    fn test_infinite_place_display() {
        let place: FunctionFieldPlace<Rational> =
            FunctionFieldPlace::infinite("∞".to_string(), 1);

        let display = format!("{}", place);
        assert!(display.contains("Infinite place"));
    }

    #[test]
    fn test_place_set_display() {
        let mut places: PlaceSet<Rational> = PlaceSet::with_genus("K".to_string(), 2);
        places.add_place("P".to_string());

        let display = format!("{}", places);
        assert!(display.contains("PlaceSet"));
        assert!(display.contains("genus 2"));
    }

    #[test]
    fn test_place_clone() {
        let place1: FunctionFieldPlace<Rational> = FunctionFieldPlace::new("P".to_string(), 1);
        let place2 = place1.clone();

        assert_eq!(place1.name(), place2.name());
        assert_eq!(place1.degree(), place2.degree());
    }

    #[test]
    fn test_place_equality() {
        let place1: FunctionFieldPlace<Rational> = FunctionFieldPlace::new("P".to_string(), 1);
        let place2: FunctionFieldPlace<Rational> = FunctionFieldPlace::new("P".to_string(), 1);
        let place3: FunctionFieldPlace<Rational> = FunctionFieldPlace::new("Q".to_string(), 1);

        assert_eq!(place1, place2);
        assert_ne!(place1, place3);
    }
}
