//! Complex interval arithmetic
//!
//! Complex intervals represent rectangular regions in the complex plane,
//! with separate intervals for real and imaginary components.
//!
//! This provides guaranteed containment of complex values through
//! interval arithmetic on both components.

use rustmath_core::{CommutativeRing, Field, MathError, Result, Ring};
use rustmath_reals::{Interval, Real, RealMPFR};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Default precision for complex intervals (53 bits = f64 equivalent)
pub const DEFAULT_PRECISION: u32 = 53;

/// Complex interval field element
///
/// Represents a rectangular region in the complex plane:
/// [a, b] + [c, d]i where all points (x + yi) with x ∈ [a,b] and y ∈ [c,d]
/// are possible values.
///
/// # Examples
///
/// ```
/// use rustmath_complex::ComplexIntervalFieldElement;
///
/// // Create interval [1,2] + [3,4]i
/// let z = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
/// assert!(z.contains(1.5, 3.5));
/// ```
#[derive(Clone, Debug)]
pub struct ComplexIntervalFieldElement {
    /// Interval for real component
    real: Interval,
    /// Interval for imaginary component
    imag: Interval,
    /// Precision in bits
    precision: u32,
}

impl ComplexIntervalFieldElement {
    /// Create a new complex interval from bounds
    ///
    /// # Arguments
    ///
    /// * `real_lower` - Lower bound of real component
    /// * `real_upper` - Upper bound of real component
    /// * `imag_lower` - Lower bound of imaginary component
    /// * `imag_upper` - Upper bound of imaginary component
    pub fn new(real_lower: f64, real_upper: f64, imag_lower: f64, imag_upper: f64) -> Self {
        ComplexIntervalFieldElement {
            real: Interval::from_f64(real_lower, real_upper),
            imag: Interval::from_f64(imag_lower, imag_upper),
            precision: DEFAULT_PRECISION,
        }
    }

    /// Create a new complex interval with specified precision
    pub fn with_precision(
        prec: u32,
        real_lower: f64,
        real_upper: f64,
        imag_lower: f64,
        imag_upper: f64,
    ) -> Self {
        ComplexIntervalFieldElement {
            real: Interval::from_f64(real_lower, real_upper),
            imag: Interval::from_f64(imag_lower, imag_upper),
            precision: prec,
        }
    }

    /// Create a point interval (exact complex number)
    pub fn point(real: f64, imag: f64) -> Self {
        ComplexIntervalFieldElement {
            real: Interval::point(Real::from(real)),
            imag: Interval::point(Real::from(imag)),
            precision: DEFAULT_PRECISION,
        }
    }

    /// Create from interval components
    pub fn from_intervals(real: Interval, imag: Interval) -> Self {
        ComplexIntervalFieldElement {
            real,
            imag,
            precision: DEFAULT_PRECISION,
        }
    }

    /// Get the real interval
    pub fn real_interval(&self) -> &Interval {
        &self.real
    }

    /// Get the imaginary interval
    pub fn imag_interval(&self) -> &Interval {
        &self.imag
    }

    /// Get the precision
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Get the center (midpoint) of the interval
    pub fn midpoint(&self) -> (f64, f64) {
        (
            self.real.midpoint().to_f64(),
            self.imag.midpoint().to_f64(),
        )
    }

    /// Get the width of the real component
    pub fn real_width(&self) -> f64 {
        self.real.width().to_f64()
    }

    /// Get the width of the imaginary component
    pub fn imag_width(&self) -> f64 {
        self.imag.width().to_f64()
    }

    /// Get the maximum width (diameter) of the interval
    pub fn max_width(&self) -> f64 {
        self.real_width().max(self.imag_width())
    }

    /// Check if this interval contains a complex point
    pub fn contains(&self, real: f64, imag: f64) -> bool {
        self.real.contains(&Real::from(real)) && self.imag.contains(&Real::from(imag))
    }

    /// Check if this interval contains another interval
    pub fn contains_interval(&self, other: &ComplexIntervalFieldElement) -> bool {
        self.real.contains_interval(&other.real) && self.imag.contains_interval(&other.imag)
    }

    /// Check if this interval overlaps with another
    pub fn overlaps(&self, other: &ComplexIntervalFieldElement) -> bool {
        self.real.overlaps(&other.real) && self.imag.overlaps(&other.imag)
    }

    /// Compute the intersection with another interval
    pub fn intersection(&self, other: &ComplexIntervalFieldElement) -> Option<Self> {
        let real_int = self.real.intersection(&other.real)?;
        let imag_int = self.imag.intersection(&other.imag)?;

        Some(ComplexIntervalFieldElement {
            real: real_int,
            imag: imag_int,
            precision: self.precision.max(other.precision),
        })
    }

    /// Compute the hull (smallest interval containing both)
    pub fn hull(&self, other: &ComplexIntervalFieldElement) -> Self {
        ComplexIntervalFieldElement {
            real: self.real.hull(&other.real),
            imag: self.imag.hull(&other.imag),
            precision: self.precision.max(other.precision),
        }
    }

    /// Check if this is a point interval (zero width)
    pub fn is_point(&self) -> bool {
        self.real.is_point() && self.imag.is_point()
    }

    /// Compute the conjugate interval
    pub fn conjugate(&self) -> Self {
        ComplexIntervalFieldElement {
            real: self.real.clone(),
            imag: -self.imag.clone(),
            precision: self.precision,
        }
    }

    /// Compute bounds on absolute value
    ///
    /// Returns (lower_bound, upper_bound) for |z|
    pub fn abs_bounds(&self) -> (f64, f64) {
        let r_low = self.real.lower().to_f64();
        let r_high = self.real.upper().to_f64();
        let i_low = self.imag.lower().to_f64();
        let i_high = self.imag.upper().to_f64();

        // Minimum |z| occurs at the point in the interval closest to origin
        let min_abs = if self.contains(0.0, 0.0) {
            0.0
        } else {
            // Check all corners and edges
            let mut min = f64::INFINITY;
            for &r in &[r_low, r_high] {
                for &i in &[i_low, i_high] {
                    min = min.min((r * r + i * i).sqrt());
                }
            }
            // Also check projections to axes
            if r_low <= 0.0 && 0.0 <= r_high {
                min = min.min(i_low.abs()).min(i_high.abs());
            }
            if i_low <= 0.0 && 0.0 <= i_high {
                min = min.min(r_low.abs()).min(r_high.abs());
            }
            min
        };

        // Maximum |z| occurs at a corner
        let max_abs = [
            (r_low * r_low + i_low * i_low).sqrt(),
            (r_low * r_low + i_high * i_high).sqrt(),
            (r_high * r_high + i_low * i_low).sqrt(),
            (r_high * r_high + i_high * i_high).sqrt(),
        ]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

        (min_abs, max_abs)
    }

    /// Create zero interval with specified precision
    pub fn zero_with_prec(prec: u32) -> Self {
        ComplexIntervalFieldElement {
            real: Interval::point(Real::zero()),
            imag: Interval::point(Real::zero()),
            precision: prec,
        }
    }

    /// Create one interval with specified precision
    pub fn one_with_prec(prec: u32) -> Self {
        ComplexIntervalFieldElement {
            real: Interval::point(Real::one()),
            imag: Interval::point(Real::zero()),
            precision: prec,
        }
    }

    /// Create imaginary unit interval with specified precision
    pub fn i_with_prec(prec: u32) -> Self {
        ComplexIntervalFieldElement {
            real: Interval::point(Real::zero()),
            imag: Interval::point(Real::one()),
            precision: prec,
        }
    }
}

impl fmt::Display for ComplexIntervalFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}i", self.real, self.imag)
    }
}

impl Add for ComplexIntervalFieldElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        ComplexIntervalFieldElement {
            real: self.real + other.real,
            imag: self.imag + other.imag,
            precision: self.precision.max(other.precision),
        }
    }
}

impl Sub for ComplexIntervalFieldElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        ComplexIntervalFieldElement {
            real: self.real - other.real,
            imag: self.imag - other.imag,
            precision: self.precision.max(other.precision),
        }
    }
}

impl Mul for ComplexIntervalFieldElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        let ac = self.real.clone() * other.real.clone();
        let bd = self.imag.clone() * other.imag.clone();
        let ad = self.real.clone() * other.imag.clone();
        let bc = self.imag.clone() * other.real.clone();

        ComplexIntervalFieldElement {
            real: ac - bd,
            imag: ad + bc,
            precision: self.precision.max(other.precision),
        }
    }
}

impl Div for ComplexIntervalFieldElement {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        // z1 / z2 = z1 * conj(z2) / |z2|^2
        let conj = other.conjugate();
        let norm_sq = other.real.clone().square() + other.imag.clone().square();

        let numerator = self * conj;

        ComplexIntervalFieldElement {
            real: numerator.real / norm_sq.clone(),
            imag: numerator.imag / norm_sq,
            precision: numerator.precision,
        }
    }
}

impl Neg for ComplexIntervalFieldElement {
    type Output = Self;

    fn neg(self) -> Self {
        ComplexIntervalFieldElement {
            real: -self.real,
            imag: -self.imag,
            precision: self.precision,
        }
    }
}

/// Create a complex interval field element (factory function)
///
/// # Arguments
///
/// * `real_lower` - Lower bound of real component
/// * `real_upper` - Upper bound of real component
/// * `imag_lower` - Lower bound of imaginary component
/// * `imag_upper` - Upper bound of imaginary component
pub fn create_complex_interval_field_element(
    real_lower: f64,
    real_upper: f64,
    imag_lower: f64,
    imag_upper: f64,
) -> ComplexIntervalFieldElement {
    ComplexIntervalFieldElement::new(real_lower, real_upper, imag_lower, imag_upper)
}

/// Check if a value is a ComplexIntervalFieldElement
///
/// In Python/SageMath this would check instance type.
/// In Rust, we use this for API compatibility.
pub fn is_complex_interval_field_element<T: std::any::Any>(_value: &T) -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<ComplexIntervalFieldElement>()
}

/// Create a complex interval field element for serialization (version 0)
///
/// This is for compatibility with SageMath's pickle format.
pub fn make_complex_interval_field_element0(
    real_lower: f64,
    real_upper: f64,
    imag_lower: f64,
    imag_upper: f64,
) -> ComplexIntervalFieldElement {
    ComplexIntervalFieldElement::new(real_lower, real_upper, imag_lower, imag_upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let z = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(z.real_interval().lower().to_f64(), 1.0);
        assert_eq!(z.real_interval().upper().to_f64(), 2.0);
        assert_eq!(z.imag_interval().lower().to_f64(), 3.0);
        assert_eq!(z.imag_interval().upper().to_f64(), 4.0);
    }

    #[test]
    fn test_point_interval() {
        let z = ComplexIntervalFieldElement::point(3.0, 4.0);
        assert!(z.is_point());
        assert!(z.contains(3.0, 4.0));
    }

    #[test]
    fn test_contains() {
        let z = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        assert!(z.contains(1.5, 3.5));
        assert!(!z.contains(0.5, 3.5));
        assert!(!z.contains(1.5, 5.0));
    }

    #[test]
    fn test_addition() {
        let z1 = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        let z2 = ComplexIntervalFieldElement::new(0.5, 1.0, 0.5, 1.0);
        let sum = z1 + z2;

        assert_eq!(sum.real_interval().lower().to_f64(), 1.5);
        assert_eq!(sum.real_interval().upper().to_f64(), 3.0);
        assert_eq!(sum.imag_interval().lower().to_f64(), 3.5);
        assert_eq!(sum.imag_interval().upper().to_f64(), 5.0);
    }

    #[test]
    fn test_subtraction() {
        let z1 = ComplexIntervalFieldElement::new(2.0, 3.0, 4.0, 5.0);
        let z2 = ComplexIntervalFieldElement::new(1.0, 1.5, 1.0, 1.5);
        let diff = z1 - z2;

        assert_eq!(diff.real_interval().lower().to_f64(), 0.5);
        assert_eq!(diff.real_interval().upper().to_f64(), 2.0);
    }

    #[test]
    fn test_multiplication() {
        let z1 = ComplexIntervalFieldElement::point(2.0, 0.0);
        let z2 = ComplexIntervalFieldElement::point(3.0, 0.0);
        let prod = z1 * z2;

        assert!((prod.real_interval().lower().to_f64() - 6.0).abs() < 1e-10);
        assert!((prod.imag_interval().lower().to_f64() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_conjugate() {
        let z = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        let conj = z.conjugate();

        assert_eq!(conj.real_interval().lower().to_f64(), 1.0);
        assert_eq!(conj.real_interval().upper().to_f64(), 2.0);
        assert_eq!(conj.imag_interval().lower().to_f64(), -4.0);
        assert_eq!(conj.imag_interval().upper().to_f64(), -3.0);
    }

    #[test]
    fn test_overlaps() {
        let z1 = ComplexIntervalFieldElement::new(1.0, 3.0, 1.0, 3.0);
        let z2 = ComplexIntervalFieldElement::new(2.0, 4.0, 2.0, 4.0);
        let z3 = ComplexIntervalFieldElement::new(5.0, 6.0, 5.0, 6.0);

        assert!(z1.overlaps(&z2));
        assert!(!z1.overlaps(&z3));
    }

    #[test]
    fn test_intersection() {
        let z1 = ComplexIntervalFieldElement::new(1.0, 4.0, 1.0, 4.0);
        let z2 = ComplexIntervalFieldElement::new(2.0, 5.0, 2.0, 5.0);

        let intersection = z1.intersection(&z2).unwrap();
        assert_eq!(intersection.real_interval().lower().to_f64(), 2.0);
        assert_eq!(intersection.real_interval().upper().to_f64(), 4.0);
    }

    #[test]
    fn test_hull() {
        let z1 = ComplexIntervalFieldElement::new(1.0, 2.0, 1.0, 2.0);
        let z2 = ComplexIntervalFieldElement::new(3.0, 4.0, 3.0, 4.0);

        let hull = z1.hull(&z2);
        assert_eq!(hull.real_interval().lower().to_f64(), 1.0);
        assert_eq!(hull.real_interval().upper().to_f64(), 4.0);
    }

    #[test]
    fn test_abs_bounds() {
        let z = ComplexIntervalFieldElement::point(3.0, 4.0);
        let (min_abs, max_abs) = z.abs_bounds();

        // |3+4i| = 5
        assert!((min_abs - 5.0).abs() < 1e-10);
        assert!((max_abs - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_abs_bounds_with_origin() {
        let z = ComplexIntervalFieldElement::new(-1.0, 1.0, -1.0, 1.0);
        let (min_abs, max_abs) = z.abs_bounds();

        assert_eq!(min_abs, 0.0); // Contains origin
        assert!((max_abs - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_midpoint() {
        let z = ComplexIntervalFieldElement::new(1.0, 3.0, 2.0, 4.0);
        let (r, i) = z.midpoint();

        assert_eq!(r, 2.0);
        assert_eq!(i, 3.0);
    }

    #[test]
    fn test_width() {
        let z = ComplexIntervalFieldElement::new(1.0, 4.0, 2.0, 5.0);
        assert_eq!(z.real_width(), 3.0);
        assert_eq!(z.imag_width(), 3.0);
        assert_eq!(z.max_width(), 3.0);
    }

    #[test]
    fn test_factory_functions() {
        let z1 = create_complex_interval_field_element(1.0, 2.0, 3.0, 4.0);
        assert!(z1.contains(1.5, 3.5));

        let z2 = make_complex_interval_field_element0(1.0, 2.0, 3.0, 4.0);
        assert!(z2.contains(1.5, 3.5));
    }

    #[test]
    fn test_negation() {
        let z = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        let neg = -z;

        assert_eq!(neg.real_interval().lower().to_f64(), -2.0);
        assert_eq!(neg.real_interval().upper().to_f64(), -1.0);
        assert_eq!(neg.imag_interval().lower().to_f64(), -4.0);
        assert_eq!(neg.imag_interval().upper().to_f64(), -3.0);
    }
}
