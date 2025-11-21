//! Complex interval field factory and field class
//!
//! Provides factory functions and field classes for creating
//! complex intervals with consistent precision settings.

use crate::complex_interval::ComplexIntervalFieldElement;
use rustmath_reals::{Interval, Real};
use std::fmt;

/// Default precision for complex interval fields (53 bits = f64 equivalent)
pub const DEFAULT_PRECISION: u32 = 53;

/// Complex interval field class
///
/// Represents a field of complex intervals with specified precision.
/// All elements created from this field share the same precision settings.
///
/// # Examples
///
/// ```
/// use rustmath_complex::ComplexIntervalFieldClass;
///
/// let field = ComplexIntervalFieldClass::new(256);
/// let z = field.make_interval(1.0, 2.0, 3.0, 4.0);
/// assert_eq!(z.precision(), 256);
/// ```
#[derive(Clone, Debug)]
pub struct ComplexIntervalFieldClass {
    /// Precision in bits for interval endpoints
    precision: u32,
}

impl ComplexIntervalFieldClass {
    /// Create a new complex interval field with specified precision
    ///
    /// # Arguments
    ///
    /// * `precision` - Number of bits of precision for interval endpoints
    pub fn new(precision: u32) -> Self {
        ComplexIntervalFieldClass { precision }
    }

    /// Create a default complex interval field (53-bit precision)
    pub fn default() -> Self {
        ComplexIntervalFieldClass {
            precision: DEFAULT_PRECISION,
        }
    }

    /// Get the precision of this field
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Create a complex interval in this field
    ///
    /// # Arguments
    ///
    /// * `real_lower` - Lower bound of real component
    /// * `real_upper` - Upper bound of real component
    /// * `imag_lower` - Lower bound of imaginary component
    /// * `imag_upper` - Upper bound of imaginary component
    pub fn make_interval(
        &self,
        real_lower: f64,
        real_upper: f64,
        imag_lower: f64,
        imag_upper: f64,
    ) -> ComplexIntervalFieldElement {
        ComplexIntervalFieldElement::with_precision(
            self.precision,
            real_lower,
            real_upper,
            imag_lower,
            imag_upper,
        )
    }

    /// Create a point interval (exact complex number) in this field
    pub fn make_point(&self, real: f64, imag: f64) -> ComplexIntervalFieldElement {
        let mut elem = ComplexIntervalFieldElement::point(real, imag);
        // Update precision (note: this is a simplified approach)
        elem = ComplexIntervalFieldElement::with_precision(
            self.precision,
            real,
            real,
            imag,
            imag,
        );
        elem
    }

    /// Create complex interval from real and imaginary intervals
    pub fn from_intervals(
        &self,
        real: Interval,
        imag: Interval,
    ) -> ComplexIntervalFieldElement {
        let mut elem = ComplexIntervalFieldElement::from_intervals(real, imag);
        // Note: precision is embedded in the element creation
        elem
    }

    /// Create zero in this field
    pub fn zero(&self) -> ComplexIntervalFieldElement {
        ComplexIntervalFieldElement::zero_with_prec(self.precision)
    }

    /// Create one in this field
    pub fn one(&self) -> ComplexIntervalFieldElement {
        ComplexIntervalFieldElement::one_with_prec(self.precision)
    }

    /// Create imaginary unit i in this field
    pub fn i(&self) -> ComplexIntervalFieldElement {
        ComplexIntervalFieldElement::i_with_prec(self.precision)
    }

    /// Create a real interval (imaginary part is zero)
    pub fn from_real(&self, lower: f64, upper: f64) -> ComplexIntervalFieldElement {
        self.make_interval(lower, upper, 0.0, 0.0)
    }

    /// Create an imaginary interval (real part is zero)
    pub fn from_imag(&self, lower: f64, upper: f64) -> ComplexIntervalFieldElement {
        self.make_interval(0.0, 0.0, lower, upper)
    }

    /// Get the characteristic of this field (always 0 for complex numbers)
    pub fn characteristic(&self) -> u32 {
        0
    }

    /// Check if this field is exact (always false for interval fields)
    pub fn is_exact(&self) -> bool {
        false
    }

    /// Get a string representation of the field
    pub fn name(&self) -> String {
        format!("Complex Interval Field with {} bits precision", self.precision)
    }
}

impl fmt::Display for ComplexIntervalFieldClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ComplexIntervalField(precision={})",
            self.precision
        )
    }
}

impl PartialEq for ComplexIntervalFieldClass {
    fn eq(&self, other: &Self) -> bool {
        self.precision == other.precision
    }
}

/// Factory function to create a complex interval field
///
/// This is the primary interface for creating complex interval fields,
/// mirroring SageMath's `ComplexIntervalField(prec)` constructor.
///
/// # Arguments
///
/// * `precision` - Number of bits of precision (default: 53)
///
/// # Examples
///
/// ```
/// use rustmath_complex::ComplexIntervalField;
///
/// // Create field with default precision (53 bits)
/// let cif = ComplexIntervalField(None);
///
/// // Create field with custom precision (256 bits)
/// let cif_256 = ComplexIntervalField(Some(256));
/// ```
pub fn complex_interval_field(precision: Option<u32>) -> ComplexIntervalFieldClass {
    ComplexIntervalFieldClass::new(precision.unwrap_or(DEFAULT_PRECISION))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_creation() {
        let field = ComplexIntervalFieldClass::new(128);
        assert_eq!(field.precision(), 128);
    }

    #[test]
    fn test_default_field() {
        let field = ComplexIntervalFieldClass::default();
        assert_eq!(field.precision(), DEFAULT_PRECISION);
    }

    #[test]
    fn test_make_interval() {
        let field = ComplexIntervalFieldClass::new(256);
        let z = field.make_interval(1.0, 2.0, 3.0, 4.0);
        assert_eq!(z.precision(), 256);
        assert!(z.contains(1.5, 3.5));
    }

    #[test]
    fn test_make_point() {
        let field = ComplexIntervalFieldClass::new(256);
        let z = field.make_point(3.0, 4.0);
        assert_eq!(z.precision(), 256);
        assert!(z.is_point());
    }

    #[test]
    fn test_zero_one_i() {
        let field = ComplexIntervalFieldClass::new(128);

        let zero = field.zero();
        assert!(zero.is_point());
        assert!(zero.contains(0.0, 0.0));

        let one = field.one();
        assert!(one.is_point());
        assert!(one.contains(1.0, 0.0));

        let i = field.i();
        assert!(i.is_point());
        assert!(i.contains(0.0, 1.0));
    }

    #[test]
    fn test_from_real() {
        let field = ComplexIntervalFieldClass::new(128);
        let z = field.from_real(2.0, 3.0);

        assert!(z.contains(2.5, 0.0));
        assert!(!z.contains(2.5, 1.0));
    }

    #[test]
    fn test_from_imag() {
        let field = ComplexIntervalFieldClass::new(128);
        let z = field.from_imag(2.0, 3.0);

        assert!(z.contains(0.0, 2.5));
        assert!(!z.contains(1.0, 2.5));
    }

    #[test]
    fn test_factory_function() {
        let field1 = ComplexIntervalField(None);
        assert_eq!(field1.precision(), DEFAULT_PRECISION);

        let field2 = ComplexIntervalField(Some(512));
        assert_eq!(field2.precision(), 512);
    }

    #[test]
    fn test_field_properties() {
        let field = ComplexIntervalFieldClass::new(256);

        assert_eq!(field.characteristic(), 0);
        assert!(!field.is_exact());
        assert!(field.name().contains("256"));
    }

    #[test]
    fn test_field_equality() {
        let field1 = ComplexIntervalFieldClass::new(128);
        let field2 = ComplexIntervalFieldClass::new(128);
        let field3 = ComplexIntervalFieldClass::new(256);

        assert_eq!(field1, field2);
        assert_ne!(field1, field3);
    }

    #[test]
    fn test_display() {
        let field = ComplexIntervalFieldClass::new(128);
        let display = format!("{}", field);
        assert!(display.contains("128"));
    }

    #[test]
    fn test_interval_arithmetic() {
        let field = ComplexIntervalFieldClass::new(256);
        let z1 = field.make_interval(1.0, 2.0, 3.0, 4.0);
        let z2 = field.make_interval(0.5, 1.0, 0.5, 1.0);

        let sum = z1 + z2;
        assert!(sum.contains(2.0, 4.0));
    }
}
