//! Valuation rings of function fields
//!
//! This module provides valuation ring structures for function fields,
//! corresponding to SageMath's `sage.rings.function_field.valuation_ring`.
//!
//! # Mathematical Background
//!
//! The valuation ring O_v associated with a discrete valuation v on a function field K is:
//!
//! O_v = {f ∈ K : v(f) ≥ 0}
//!
//! ## Properties
//!
//! - O_v is a local ring (has unique maximal ideal)
//! - Maximal ideal: m_v = {f ∈ K : v(f) > 0}
//! - Residue field: κ_v = O_v / m_v
//! - Unit group: O_v* = {f ∈ K : v(f) = 0}
//!
//! ## Examples
//!
//! ### Rational Function Field Q(x)
//!
//! - **Valuation at x**: O_x = Q[x]_(x) = {f/g : v_x(g) = 0}
//! - **Valuation at ∞**: O_∞ = Q[1/x] = {f/g : deg(f) ≤ deg(g)}
//!
//! ### Function Field Extensions
//!
//! For L/K and place P|p:
//! - O_P ∩ K = O_p
//! - O_P is integral over O_p
//! - [O_P : O_p] = e(P|p)f(P|p) divides [L:K]
//!
//! # Key Types
//!
//! - `FunctionFieldValuationRing`: The valuation ring structure
//! - Associated types: Elements, ideals, morphisms

use rustmath_core::Field;
use std::marker::PhantomData;
use std::fmt;
use std::collections::HashSet;

/// Valuation ring of a function field
///
/// Represents the valuation ring O_v = {f ∈ K : v(f) ≥ 0} for a discrete
/// valuation v on a function field K.
///
/// This corresponds to SageMath's `FunctionFieldValuationRing` class.
///
/// # Type Parameters
///
/// - `F`: The constant field type
///
/// # Mathematical Details
///
/// The valuation ring is a Dedekind domain when K is a global function field.
/// It is always a discrete valuation ring (DVR) as it's local.
///
/// Structure:
/// - Principal ideal domain
/// - Unique factorization domain
/// - Integrally closed in K
#[derive(Clone, Debug)]
pub struct FunctionFieldValuationRing<F: Field> {
    /// Name of the function field
    function_field: String,
    /// Name of the valuation
    valuation_name: String,
    /// String representation of the uniformizer
    uniformizer: String,
    /// Whether this is a finite or infinite valuation
    is_finite: bool,
    /// Known elements in the ring
    elements: HashSet<String>,
    /// Field marker
    field_marker: PhantomData<F>,
}

impl<F: Field> FunctionFieldValuationRing<F> {
    /// Create a new valuation ring
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    /// * `valuation_name` - Name of the valuation
    /// * `uniformizer` - Uniformizing element
    /// * `is_finite` - Whether this is a finite valuation
    ///
    /// # Returns
    ///
    /// A new FunctionFieldValuationRing instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Valuation ring at x in Q(x)
    /// let ring = FunctionFieldValuationRing::new(
    ///     "Q(x)".to_string(),
    ///     "v_x".to_string(),
    ///     "x".to_string(),
    ///     true
    /// );
    /// ```
    pub fn new(
        function_field: String,
        valuation_name: String,
        uniformizer: String,
        is_finite: bool,
    ) -> Self {
        FunctionFieldValuationRing {
            function_field,
            valuation_name,
            uniformizer,
            is_finite,
            elements: HashSet::new(),
            field_marker: PhantomData,
        }
    }

    /// Create valuation ring for a finite place
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    /// * `place_name` - Name of the place
    /// * `prime_polynomial` - Prime polynomial defining the place
    ///
    /// # Returns
    ///
    /// Valuation ring at the finite place
    pub fn at_finite_place(
        function_field: String,
        place_name: String,
        prime_polynomial: String,
    ) -> Self {
        let valuation_name = format!("v_{}", place_name);
        FunctionFieldValuationRing::new(
            function_field,
            valuation_name,
            prime_polynomial,
            true,
        )
    }

    /// Create valuation ring for the infinite place
    ///
    /// # Arguments
    ///
    /// * `function_field` - Name of the function field
    ///
    /// # Returns
    ///
    /// Valuation ring at infinity
    pub fn at_infinite_place(function_field: String) -> Self {
        FunctionFieldValuationRing::new(
            function_field,
            "v_∞".to_string(),
            "1/x".to_string(),
            false,
        )
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
    pub fn valuation_name(&self) -> &str {
        &self.valuation_name
    }

    /// Get the uniformizer
    ///
    /// # Returns
    ///
    /// String representation of the uniformizer
    pub fn uniformizer(&self) -> &str {
        &self.uniformizer
    }

    /// Check if this is a finite valuation ring
    ///
    /// # Returns
    ///
    /// True if finite, false if infinite
    pub fn is_finite(&self) -> bool {
        self.is_finite
    }

    /// Check if this is the infinite valuation ring
    ///
    /// # Returns
    ///
    /// True if infinite
    pub fn is_infinite(&self) -> bool {
        !self.is_finite
    }

    /// Add an element to the ring (for symbolic tracking)
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    pub fn add_element(&mut self, element: String) {
        self.elements.insert(element);
    }

    /// Check if element is known to be in the ring
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    ///
    /// # Returns
    ///
    /// True if element is in the tracked set
    pub fn contains_element(&self, element: &str) -> bool {
        self.elements.contains(element)
    }

    /// Get the maximal ideal
    ///
    /// # Returns
    ///
    /// String description of the maximal ideal m_v
    pub fn maximal_ideal(&self) -> String {
        format!("m_{{{}}}", self.valuation_name)
    }

    /// Get the residue field
    ///
    /// # Returns
    ///
    /// String description of the residue field κ_v = O_v / m_v
    pub fn residue_field(&self) -> String {
        if self.is_finite {
            format!("κ_{{{}}}", self.valuation_name)
        } else {
            "k".to_string()
        }
    }

    /// Get the unit group (symbolically)
    ///
    /// # Returns
    ///
    /// String description of the unit group O_v*
    pub fn unit_group(&self) -> String {
        format!("O_{{{}}}*", self.valuation_name)
    }

    /// Check if ring is a DVR
    ///
    /// # Returns
    ///
    /// Always true (valuation rings are DVRs)
    pub fn is_dvr(&self) -> bool {
        true
    }

    /// Check if ring is a PID
    ///
    /// # Returns
    ///
    /// Always true (DVRs are PIDs)
    pub fn is_pid(&self) -> bool {
        true
    }

    /// Check if ring is a UFD
    ///
    /// # Returns
    ///
    /// Always true (PIDs are UFDs)
    pub fn is_ufd(&self) -> bool {
        true
    }

    /// Check if ring is integrally closed
    ///
    /// # Returns
    ///
    /// Always true (DVRs are integrally closed)
    pub fn is_integrally_closed(&self) -> bool {
        true
    }

    /// Check if ring is Noetherian
    ///
    /// # Returns
    ///
    /// Always true (DVRs are Noetherian)
    pub fn is_noetherian(&self) -> bool {
        true
    }

    /// Get the Krull dimension
    ///
    /// # Returns
    ///
    /// Always 1 (DVRs have dimension 1)
    pub fn krull_dimension(&self) -> usize {
        1
    }

    /// Localization description
    ///
    /// # Returns
    ///
    /// String describing the ring as a localization
    pub fn as_localization(&self) -> String {
        if self.is_finite {
            format!("k[x]_{{({})}}", self.uniformizer)
        } else {
            "k[1/x]".to_string()
        }
    }

    /// Represent element in terms of uniformizer (symbolic)
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    ///
    /// # Returns
    ///
    /// Representation as power series in uniformizer
    pub fn series_expansion(&self, element: &str) -> String {
        format!("{} = a_0 + a_1·{} + a_2·{}^2 + ...", element, self.uniformizer, self.uniformizer)
    }

    /// Compute valuation of an element (symbolic)
    ///
    /// # Arguments
    ///
    /// * `element` - String representation of the element
    ///
    /// # Returns
    ///
    /// String representation of the valuation
    pub fn valuation(&self, element: &str) -> String {
        format!("{}({})", self.valuation_name, element)
    }
}

impl<F: Field> fmt::Display for FunctionFieldValuationRing<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_finite {
            write!(
                f,
                "Valuation ring O_{{{}}} of {} with uniformizer {}",
                self.valuation_name, self.function_field, self.uniformizer
            )
        } else {
            write!(
                f,
                "Valuation ring at infinity of {} (k[1/x])",
                self.function_field
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_valuation_ring_creation() {
        let ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::new(
                "Q(x)".to_string(),
                "v_x".to_string(),
                "x".to_string(),
                true,
            );

        assert_eq!(ring.function_field(), "Q(x)");
        assert_eq!(ring.valuation_name(), "v_x");
        assert_eq!(ring.uniformizer(), "x");
        assert!(ring.is_finite());
    }

    #[test]
    fn test_finite_place_ring() {
        let ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::at_finite_place(
                "Q(x)".to_string(),
                "P_0".to_string(),
                "x".to_string(),
            );

        assert!(ring.is_finite());
        assert!(!ring.is_infinite());
        assert_eq!(ring.uniformizer(), "x");
    }

    #[test]
    fn test_infinite_place_ring() {
        let ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::at_infinite_place("Q(x)".to_string());

        assert!(!ring.is_finite());
        assert!(ring.is_infinite());
        assert_eq!(ring.uniformizer(), "1/x");
        assert_eq!(ring.valuation_name(), "v_∞");
    }

    #[test]
    fn test_maximal_ideal() {
        let ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::new(
                "Q(x)".to_string(),
                "v_P".to_string(),
                "x-1".to_string(),
                true,
            );

        let ideal = ring.maximal_ideal();
        assert!(ideal.contains("m_"));
    }

    #[test]
    fn test_residue_field() {
        let finite_ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::at_finite_place(
                "Q(x)".to_string(),
                "P".to_string(),
                "x".to_string(),
            );

        let infinite_ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::at_infinite_place("Q(x)".to_string());

        assert!(finite_ring.residue_field().contains("κ"));
        assert_eq!(infinite_ring.residue_field(), "k");
    }

    #[test]
    fn test_ring_properties() {
        let ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::new(
                "Q(x)".to_string(),
                "v_x".to_string(),
                "x".to_string(),
                true,
            );

        assert!(ring.is_dvr());
        assert!(ring.is_pid());
        assert!(ring.is_ufd());
        assert!(ring.is_integrally_closed());
        assert!(ring.is_noetherian());
        assert_eq!(ring.krull_dimension(), 1);
    }

    #[test]
    fn test_unit_group() {
        let ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::new(
                "Q(x)".to_string(),
                "v_x".to_string(),
                "x".to_string(),
                true,
            );

        let units = ring.unit_group();
        assert!(units.contains("O_"));
        assert!(units.contains("*"));
    }

    #[test]
    fn test_localization() {
        let finite_ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::at_finite_place(
                "Q(x)".to_string(),
                "P".to_string(),
                "x-1".to_string(),
            );

        let infinite_ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::at_infinite_place("Q(x)".to_string());

        let loc_f = finite_ring.as_localization();
        let loc_i = infinite_ring.as_localization();

        assert!(loc_f.contains("k[x]"));
        assert_eq!(loc_i, "k[1/x]");
    }

    #[test]
    fn test_add_element() {
        let mut ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::new(
                "Q(x)".to_string(),
                "v_x".to_string(),
                "x".to_string(),
                true,
            );

        assert!(!ring.contains_element("x^2"));
        ring.add_element("x^2".to_string());
        assert!(ring.contains_element("x^2"));
    }

    #[test]
    fn test_series_expansion() {
        let ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::new(
                "Q(x)".to_string(),
                "v_x".to_string(),
                "x".to_string(),
                true,
            );

        let expansion = ring.series_expansion("f");
        assert!(expansion.contains("a_0"));
        assert!(expansion.contains("x"));
    }

    #[test]
    fn test_valuation() {
        let ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::new(
                "Q(x)".to_string(),
                "v_x".to_string(),
                "x".to_string(),
                true,
            );

        let val = ring.valuation("x^2/(x-1)");
        assert!(val.contains("v_x"));
    }

    #[test]
    fn test_display_finite() {
        let ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::at_finite_place(
                "Q(x)".to_string(),
                "P".to_string(),
                "x".to_string(),
            );

        let display = format!("{}", ring);
        assert!(display.contains("Valuation ring"));
        assert!(display.contains("uniformizer"));
    }

    #[test]
    fn test_display_infinite() {
        let ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::at_infinite_place("Q(x)".to_string());

        let display = format!("{}", ring);
        assert!(display.contains("infinity"));
        assert!(display.contains("k[1/x]"));
    }

    #[test]
    fn test_clone() {
        let ring1: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::new(
                "Q(x)".to_string(),
                "v_x".to_string(),
                "x".to_string(),
                true,
            );
        let ring2 = ring1.clone();

        assert_eq!(ring1.function_field(), ring2.function_field());
        assert_eq!(ring1.uniformizer(), ring2.uniformizer());
    }

    #[test]
    fn test_multiple_valuations() {
        // Create several valuation rings
        let rings: Vec<FunctionFieldValuationRing<Rational>> = vec![
            FunctionFieldValuationRing::at_finite_place(
                "Q(x)".to_string(),
                "P_0".to_string(),
                "x".to_string(),
            ),
            FunctionFieldValuationRing::at_finite_place(
                "Q(x)".to_string(),
                "P_1".to_string(),
                "x - 1".to_string(),
            ),
            FunctionFieldValuationRing::at_infinite_place("Q(x)".to_string()),
        ];

        assert_eq!(rings.len(), 3);
        assert_eq!(rings.iter().filter(|r| r.is_finite()).count(), 2);
        assert_eq!(rings.iter().filter(|r| r.is_infinite()).count(), 1);
    }

    #[test]
    fn test_dvr_properties() {
        let ring: FunctionFieldValuationRing<Rational> =
            FunctionFieldValuationRing::new(
                "K".to_string(),
                "v_P".to_string(),
                "t".to_string(),
                true,
            );

        // A DVR has exactly one non-zero prime ideal (the maximal ideal)
        assert!(ring.is_dvr());
        assert_eq!(ring.krull_dimension(), 1);

        // Every non-zero element can be written as π^n · u where u is a unit
        let expansion = ring.series_expansion("f");
        assert!(!expansion.is_empty());
    }
}
