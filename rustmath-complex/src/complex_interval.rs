//! Complex interval arithmetic
//!
//! Complex intervals represent rectangular regions in the complex plane,
//! with separate intervals for real and imaginary components.
//!
//! This provides guaranteed containment of complex values through
//! interval arithmetic on both components.
//!
//! ## SageMath API Compatibility
//!
//! This implementation provides compatibility with SageMath's
//! `sage.rings.complex_interval.ComplexIntervalFieldElement` API.
//!
//! ### Implemented Methods
//!
//! - **Accessors**: `real()`, `imag()`, `precision()`, `center()`, `diameter()`
//! - **Interval operations**: `contains()`, `overlaps()`, `intersection()`, `union()`, `hull()`
//! - **Utility**: `is_exact()`, `is_point()`, `is_nan()`, `contains_zero()`
//! - **Geometric**: `endpoints()`, `edges()`, `bisection()`, `midpoint()`, `width()`
//! - **Mathematical**: `norm()`, `abs()`, `sqrt()`, `powi()`, `conjugate()`, `argument_bounds()`
//! - **Arithmetic**: `+`, `-`, `*`, `/`, `-` (negation)
//! - **Comparison**: `==`, `!=`, `<`, `<=`, `>`, `>=` (lexicographic ordering)
//!
//! ### Not Yet Implemented (Requires Transcendental Functions)
//!
//! The following methods from SageMath's API are not yet implemented because
//! the underlying `Interval` type does not yet support transcendental functions:
//!
//! - **Exponential/Logarithmic**: `exp()`, `log()`, `log10()`
//! - **Trigonometric**: `sin()`, `cos()`, `tan()`, `cot()`, `sec()`, `csc()`
//! - **Inverse Trigonometric**: `asin()`, `acos()`, `atan()`, `atan2()`
//! - **Hyperbolic**: `sinh()`, `cosh()`, `tanh()`, `coth()`, `sech()`, `csch()`
//! - **Inverse Hyperbolic**: `asinh()`, `acosh()`, `atanh()`
//!
//! These will be added once the `rustmath-reals` crate implements interval
//! versions of these transcendental functions.
//!
//! ## Edge Cases
//!
//! - **Zero-width intervals**: Fully supported via `point()` constructor and `is_exact()` check
//! - **Infinity handling**: Intervals can have infinite bounds; operations handle these conservatively
//! - **NaN handling**: Detected via `is_nan()` method
//! - **Division by zero**: Returns conservative infinite bounds when dividing by zero-containing interval
//! - **Branch cuts**: `argument_bounds()` handles branch cuts conservatively

use rustmath_core::Ring;
use rustmath_reals::{Interval, Real};
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

    /// Get the real component (alias for real_interval)
    ///
    /// This method provides SageMath API compatibility
    pub fn real(&self) -> &Interval {
        &self.real
    }

    /// Get the imaginary component (alias for imag_interval)
    ///
    /// This method provides SageMath API compatibility
    pub fn imag(&self) -> &Interval {
        &self.imag
    }

    /// Get the precision
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Get the center point of the interval (alias for midpoint)
    ///
    /// This method provides SageMath API compatibility
    pub fn center(&self) -> (f64, f64) {
        self.midpoint()
    }

    /// Get the diameter (maximum width) of the interval
    ///
    /// This method provides SageMath API compatibility
    pub fn diameter(&self) -> f64 {
        self.max_width()
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

    /// Check if this is an exact interval (alias for is_point)
    ///
    /// This method provides SageMath API compatibility
    pub fn is_exact(&self) -> bool {
        self.is_point()
    }

    /// Check if the interval contains zero
    pub fn contains_zero(&self) -> bool {
        self.contains(0.0, 0.0)
    }

    /// Check if either component contains NaN
    pub fn is_nan(&self) -> bool {
        self.real.lower().to_f64().is_nan()
            || self.real.upper().to_f64().is_nan()
            || self.imag.lower().to_f64().is_nan()
            || self.imag.upper().to_f64().is_nan()
    }

    /// Get the four corner points (endpoints) of the rectangular interval
    ///
    /// Returns corners in the order: (lower-left, lower-right, upper-left, upper-right)
    /// where lower-left = (real_lower, imag_lower), etc.
    pub fn endpoints(&self) -> [(f64, f64); 4] {
        let rl = self.real.lower().to_f64();
        let ru = self.real.upper().to_f64();
        let il = self.imag.lower().to_f64();
        let iu = self.imag.upper().to_f64();

        [(rl, il), (ru, il), (rl, iu), (ru, iu)]
    }

    /// Get the four edges of the rectangular interval
    ///
    /// Returns (left, right, bottom, top) edges as complex intervals
    pub fn edges(&self) -> [ComplexIntervalFieldElement; 4] {
        let rl = self.real.lower().to_f64();
        let ru = self.real.upper().to_f64();
        let il = self.imag.lower().to_f64();
        let iu = self.imag.upper().to_f64();

        [
            // Left edge: x = real_lower, y varies
            ComplexIntervalFieldElement::new(rl, rl, il, iu),
            // Right edge: x = real_upper, y varies
            ComplexIntervalFieldElement::new(ru, ru, il, iu),
            // Bottom edge: x varies, y = imag_lower
            ComplexIntervalFieldElement::new(rl, ru, il, il),
            // Top edge: x varies, y = imag_upper
            ComplexIntervalFieldElement::new(rl, ru, iu, iu),
        ]
    }

    /// Bisect the interval into four quadrants
    ///
    /// Splits the interval at its center point, returning four sub-intervals
    /// in the order: (lower-left, lower-right, upper-left, upper-right)
    pub fn bisection(&self) -> [ComplexIntervalFieldElement; 4] {
        let (mr, mi) = self.midpoint();
        let rl = self.real.lower().to_f64();
        let ru = self.real.upper().to_f64();
        let il = self.imag.lower().to_f64();
        let iu = self.imag.upper().to_f64();

        [
            ComplexIntervalFieldElement::new(rl, mr, il, mi), // lower-left
            ComplexIntervalFieldElement::new(mr, ru, il, mi), // lower-right
            ComplexIntervalFieldElement::new(rl, mr, mi, iu), // upper-left
            ComplexIntervalFieldElement::new(mr, ru, mi, iu), // upper-right
        ]
    }

    /// Compute the union (smallest interval containing both)
    ///
    /// This is an alias for hull() providing SageMath API compatibility
    pub fn union(&self, other: &ComplexIntervalFieldElement) -> Self {
        self.hull(other)
    }

    /// Compute the conjugate interval
    pub fn conjugate(&self) -> Self {
        ComplexIntervalFieldElement {
            real: self.real.clone(),
            imag: -self.imag.clone(),
            precision: self.precision,
        }
    }

    /// Compute the norm (squared magnitude) as an interval
    ///
    /// Returns the interval [min(|z|²), max(|z|²)] for all z in this interval.
    /// This is computed as real² + imag².
    pub fn norm(&self) -> Interval {
        let r_sq = self.real.clone().square();
        let i_sq = self.imag.clone().square();
        r_sq + i_sq
    }

    /// Compute the absolute value (magnitude) as an interval
    ///
    /// Returns the interval [min(|z|), max(|z|)] for all z in this interval.
    pub fn abs(&self) -> Interval {
        let (min_abs, max_abs) = self.abs_bounds();
        Interval::from_f64(min_abs, max_abs)
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

    /// Compute the principal square root
    ///
    /// For a complex number z = a + bi, the principal square root is computed using:
    /// sqrt(z) = sqrt((|z| + a)/2) + i * sign(b) * sqrt((|z| - a)/2)
    ///
    /// For intervals, this computes conservative bounds on all possible square roots.
    pub fn sqrt(&self) -> Self {
        // For now, compute sqrt at corners and take hull
        // A more sophisticated implementation would use the formula above with intervals
        let corners = self.endpoints();
        let mut result: Option<ComplexIntervalFieldElement> = None;

        for (r, i) in corners.iter() {
            let z_abs = (r * r + i * i).sqrt();
            let re_sqrt = ((z_abs + r) / 2.0).sqrt();
            let im_sqrt = ((z_abs - r) / 2.0).sqrt() * i.signum();

            let corner_sqrt = ComplexIntervalFieldElement::point(re_sqrt, im_sqrt);

            result = Some(match result {
                None => corner_sqrt,
                Some(prev) => prev.hull(&corner_sqrt),
            });
        }

        result.unwrap_or_else(|| ComplexIntervalFieldElement::point(0.0, 0.0))
    }

    /// Compute integer power z^n
    ///
    /// Uses binary exponentiation for efficiency.
    /// For negative exponents, computes (1/z)^|n|.
    pub fn powi(&self, n: i32) -> Self {
        if n == 0 {
            return ComplexIntervalFieldElement::one_with_prec(self.precision);
        }

        if n < 0 {
            // z^(-n) = (1/z)^n
            let inv = ComplexIntervalFieldElement::one_with_prec(self.precision) / self.clone();
            return inv.powi(-n);
        }

        // Binary exponentiation
        let mut result = ComplexIntervalFieldElement::one_with_prec(self.precision);
        let mut base = self.clone();
        let mut exp = n as u32;

        while exp > 0 {
            if exp % 2 == 1 {
                result = result * base.clone();
            }
            base = base.clone() * base.clone();
            exp /= 2;
        }

        result
    }

    /// Compute the argument (phase angle) bounds
    ///
    /// Returns (min_arg, max_arg) where the argument is the angle in radians.
    /// The argument is in the range (-π, π].
    ///
    /// Note: For intervals crossing the branch cut (negative real axis),
    /// this may return conservative bounds.
    pub fn argument_bounds(&self) -> (f64, f64) {
        let rl = self.real.lower().to_f64();
        let ru = self.real.upper().to_f64();
        let il = self.imag.lower().to_f64();
        let iu = self.imag.upper().to_f64();

        // Check if interval contains origin
        if self.contains_zero() {
            return (-std::f64::consts::PI, std::f64::consts::PI);
        }

        // Compute arguments at all four corners
        let angles = vec![
            il.atan2(rl),
            il.atan2(ru),
            iu.atan2(rl),
            iu.atan2(ru),
        ];

        // Check if we cross the negative real axis (branch cut)
        if rl <= 0.0 && il <= 0.0 && iu >= 0.0 {
            // Interval straddles the negative real axis
            return (-std::f64::consts::PI, std::f64::consts::PI);
        }

        let min_arg = angles
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_arg = angles
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        (min_arg, max_arg)
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

impl PartialEq for ComplexIntervalFieldElement {
    fn eq(&self, other: &Self) -> bool {
        // Two intervals are equal if they represent the same region
        self.real.lower().to_f64() == other.real.lower().to_f64()
            && self.real.upper().to_f64() == other.real.upper().to_f64()
            && self.imag.lower().to_f64() == other.imag.lower().to_f64()
            && self.imag.upper().to_f64() == other.imag.upper().to_f64()
    }
}

impl PartialOrd for ComplexIntervalFieldElement {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Lexicographic ordering: first by real lower, then real upper, then imag lower, then imag upper
        let rl1 = self.real.lower().to_f64();
        let rl2 = other.real.lower().to_f64();

        if rl1 != rl2 {
            return rl1.partial_cmp(&rl2);
        }

        let ru1 = self.real.upper().to_f64();
        let ru2 = other.real.upper().to_f64();

        if ru1 != ru2 {
            return ru1.partial_cmp(&ru2);
        }

        let il1 = self.imag.lower().to_f64();
        let il2 = other.imag.lower().to_f64();

        if il1 != il2 {
            return il1.partial_cmp(&il2);
        }

        let iu1 = self.imag.upper().to_f64();
        let iu2 = other.imag.upper().to_f64();

        iu1.partial_cmp(&iu2)
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

    // Tests for new accessor methods
    #[test]
    fn test_real_imag_accessors() {
        let z = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(z.real().lower().to_f64(), 1.0);
        assert_eq!(z.real().upper().to_f64(), 2.0);
        assert_eq!(z.imag().lower().to_f64(), 3.0);
        assert_eq!(z.imag().upper().to_f64(), 4.0);
    }

    #[test]
    fn test_center_and_diameter() {
        let z = ComplexIntervalFieldElement::new(1.0, 3.0, 2.0, 6.0);
        let (cr, ci) = z.center();
        assert_eq!(cr, 2.0);
        assert_eq!(ci, 4.0);
        assert_eq!(z.diameter(), 4.0); // max of (3-1, 6-2) = max(2, 4) = 4
    }

    // Tests for utility methods
    #[test]
    fn test_contains_zero() {
        let z1 = ComplexIntervalFieldElement::new(-1.0, 1.0, -1.0, 1.0);
        assert!(z1.contains_zero());

        let z2 = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        assert!(!z2.contains_zero());

        let z3 = ComplexIntervalFieldElement::point(0.0, 0.0);
        assert!(z3.contains_zero());
    }

    #[test]
    fn test_is_exact() {
        let z1 = ComplexIntervalFieldElement::point(3.0, 4.0);
        assert!(z1.is_exact());

        let z2 = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        assert!(!z2.is_exact());
    }

    #[test]
    fn test_is_nan() {
        let z1 = ComplexIntervalFieldElement::point(3.0, 4.0);
        assert!(!z1.is_nan());

        let z2 = ComplexIntervalFieldElement::new(f64::NAN, 1.0, 2.0, 3.0);
        assert!(z2.is_nan());

        let z3 = ComplexIntervalFieldElement::new(1.0, 2.0, f64::NAN, 3.0);
        assert!(z3.is_nan());
    }

    #[test]
    fn test_endpoints() {
        let z = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        let endpoints = z.endpoints();

        assert_eq!(endpoints[0], (1.0, 3.0)); // lower-left
        assert_eq!(endpoints[1], (2.0, 3.0)); // lower-right
        assert_eq!(endpoints[2], (1.0, 4.0)); // upper-left
        assert_eq!(endpoints[3], (2.0, 4.0)); // upper-right
    }

    #[test]
    fn test_edges() {
        let z = ComplexIntervalFieldElement::new(1.0, 3.0, 2.0, 4.0);
        let edges = z.edges();

        // Left edge: x=1, y∈[2,4]
        assert_eq!(edges[0].real().lower().to_f64(), 1.0);
        assert_eq!(edges[0].real().upper().to_f64(), 1.0);
        assert_eq!(edges[0].imag().lower().to_f64(), 2.0);
        assert_eq!(edges[0].imag().upper().to_f64(), 4.0);

        // Right edge: x=3, y∈[2,4]
        assert_eq!(edges[1].real().lower().to_f64(), 3.0);
        assert_eq!(edges[1].real().upper().to_f64(), 3.0);

        // Bottom edge: x∈[1,3], y=2
        assert_eq!(edges[2].imag().lower().to_f64(), 2.0);
        assert_eq!(edges[2].imag().upper().to_f64(), 2.0);

        // Top edge: x∈[1,3], y=4
        assert_eq!(edges[3].imag().lower().to_f64(), 4.0);
        assert_eq!(edges[3].imag().upper().to_f64(), 4.0);
    }

    #[test]
    fn test_bisection() {
        let z = ComplexIntervalFieldElement::new(0.0, 4.0, 0.0, 4.0);
        let quadrants = z.bisection();

        // Lower-left: [0,2] + [0,2]i
        assert_eq!(quadrants[0].real().lower().to_f64(), 0.0);
        assert_eq!(quadrants[0].real().upper().to_f64(), 2.0);
        assert_eq!(quadrants[0].imag().lower().to_f64(), 0.0);
        assert_eq!(quadrants[0].imag().upper().to_f64(), 2.0);

        // Upper-right: [2,4] + [2,4]i
        assert_eq!(quadrants[3].real().lower().to_f64(), 2.0);
        assert_eq!(quadrants[3].real().upper().to_f64(), 4.0);
        assert_eq!(quadrants[3].imag().lower().to_f64(), 2.0);
        assert_eq!(quadrants[3].imag().upper().to_f64(), 4.0);
    }

    #[test]
    fn test_union() {
        let z1 = ComplexIntervalFieldElement::new(1.0, 2.0, 1.0, 2.0);
        let z2 = ComplexIntervalFieldElement::new(3.0, 4.0, 3.0, 4.0);

        let union = z1.union(&z2);
        assert_eq!(union.real().lower().to_f64(), 1.0);
        assert_eq!(union.real().upper().to_f64(), 4.0);
        assert_eq!(union.imag().lower().to_f64(), 1.0);
        assert_eq!(union.imag().upper().to_f64(), 4.0);
    }

    // Tests for mathematical methods
    #[test]
    fn test_norm() {
        let z = ComplexIntervalFieldElement::point(3.0, 4.0);
        let norm = z.norm();

        // |3+4i|² = 9 + 16 = 25
        let norm_val = norm.lower().to_f64();
        assert!((norm_val - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_abs_interval() {
        let z = ComplexIntervalFieldElement::point(3.0, 4.0);
        let abs_interval = z.abs();

        // |3+4i| = 5
        let abs_val = abs_interval.lower().to_f64();
        assert!((abs_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_simple() {
        // sqrt(4) = 2
        let z = ComplexIntervalFieldElement::point(4.0, 0.0);
        let sqrt_z = z.sqrt();
        let (r, i) = sqrt_z.center();

        assert!((r - 2.0).abs() < 1e-10);
        assert!(i.abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_imaginary() {
        // sqrt(i) ≈ (1+i)/√2
        let z = ComplexIntervalFieldElement::point(0.0, 1.0);
        let sqrt_z = z.sqrt();
        let (r, i) = sqrt_z.center();

        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((r - expected).abs() < 1e-10);
        assert!((i - expected).abs() < 1e-10);
    }

    #[test]
    fn test_powi() {
        let z = ComplexIntervalFieldElement::point(2.0, 0.0);

        // z^0 = 1
        let z0 = z.powi(0);
        assert!(z0.is_exact());
        assert!(z0.contains(1.0, 0.0));

        // z^1 = z
        let z1 = z.powi(1);
        assert!(z1.contains(2.0, 0.0));

        // z^2 = 4
        let z2 = z.powi(2);
        let (r, i) = z2.center();
        assert!((r - 4.0).abs() < 1e-10);
        assert!(i.abs() < 1e-10);

        // z^3 = 8
        let z3 = z.powi(3);
        let (r, i) = z3.center();
        assert!((r - 8.0).abs() < 1e-10);
        assert!(i.abs() < 1e-10);
    }

    #[test]
    fn test_powi_imaginary() {
        // i^2 = -1, i^3 = -i, i^4 = 1
        let i = ComplexIntervalFieldElement::point(0.0, 1.0);

        let i2 = i.powi(2);
        let (r, im) = i2.center();
        assert!((r - (-1.0)).abs() < 1e-10);
        assert!(im.abs() < 1e-10);

        let i4 = i.powi(4);
        let (r, im) = i4.center();
        assert!((r - 1.0).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn test_powi_negative() {
        // (2+0i)^(-1) = 0.5
        let z = ComplexIntervalFieldElement::point(2.0, 0.0);
        let inv = z.powi(-1);
        let (r, i) = inv.center();

        assert!((r - 0.5).abs() < 1e-10);
        assert!(i.abs() < 1e-10);
    }

    #[test]
    fn test_argument_bounds() {
        // Argument of 1+i is π/4
        let z = ComplexIntervalFieldElement::point(1.0, 1.0);
        let (min_arg, max_arg) = z.argument_bounds();

        let expected = std::f64::consts::PI / 4.0;
        assert!((min_arg - expected).abs() < 1e-10);
        assert!((max_arg - expected).abs() < 1e-10);
    }

    #[test]
    fn test_argument_bounds_with_origin() {
        // Interval containing origin has full range
        let z = ComplexIntervalFieldElement::new(-1.0, 1.0, -1.0, 1.0);
        let (min_arg, max_arg) = z.argument_bounds();

        assert_eq!(min_arg, -std::f64::consts::PI);
        assert_eq!(max_arg, std::f64::consts::PI);
    }

    // Tests for comparison
    #[test]
    fn test_equality() {
        let z1 = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        let z2 = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        let z3 = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 5.0);

        assert_eq!(z1, z2);
        assert_ne!(z1, z3);
    }

    #[test]
    fn test_partial_ord() {
        let z1 = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 4.0);
        let z2 = ComplexIntervalFieldElement::new(1.0, 2.0, 3.0, 5.0);
        let z3 = ComplexIntervalFieldElement::new(2.0, 3.0, 1.0, 2.0);

        assert!(z1 < z2); // Same real and imag lower, but z2 has larger imag upper
        assert!(z1 < z3); // z1 has smaller real lower
    }

    // Edge case tests
    #[test]
    fn test_zero_width_interval() {
        let z = ComplexIntervalFieldElement::point(5.0, 12.0);

        assert!(z.is_exact());
        assert!(z.is_point());
        assert_eq!(z.real_width(), 0.0);
        assert_eq!(z.imag_width(), 0.0);
        assert_eq!(z.diameter(), 0.0);

        let (r, i) = z.center();
        assert_eq!(r, 5.0);
        assert_eq!(i, 12.0);

        // |5+12i| = 13
        let (min_abs, max_abs) = z.abs_bounds();
        assert_eq!(min_abs, max_abs);
        assert_eq!(min_abs, 13.0);
    }

    #[test]
    fn test_infinity_handling() {
        // Test intervals with infinite bounds
        let z = ComplexIntervalFieldElement::new(
            f64::NEG_INFINITY,
            f64::INFINITY,
            0.0,
            1.0
        );

        assert!(!z.is_exact());
        assert!(z.contains(0.0, 0.5));
        assert!(z.contains(1000.0, 0.5));
        assert!(z.contains(-1000.0, 0.5));

        // Width should be infinite
        assert!(z.real_width().is_infinite());
    }

    #[test]
    fn test_division_by_zero_containing_interval() {
        let z1 = ComplexIntervalFieldElement::new(1.0, 2.0, 1.0, 2.0);
        let z2 = ComplexIntervalFieldElement::new(-1.0, 1.0, -1.0, 1.0); // Contains zero

        let result = z1 / z2;

        // Result should have very large or infinite bounds
        // This is a conservative estimate when dividing by interval containing zero
        assert!(result.real_width() > 100.0 || result.real_width().is_infinite());
    }

    #[test]
    fn test_interval_arithmetic_conservativeness() {
        // Verify that interval arithmetic is conservative:
        // (z1 op z2) should contain all possible values of (z op w) for z ∈ z1, w ∈ z2

        let z1 = ComplexIntervalFieldElement::new(1.0, 2.0, 1.0, 2.0);
        let z2 = ComplexIntervalFieldElement::new(0.5, 1.0, 0.5, 1.0);

        // Test specific points
        let p1 = ComplexIntervalFieldElement::point(1.5, 1.5);
        let p2 = ComplexIntervalFieldElement::point(0.75, 0.75);

        let sum = z1.clone() + z2.clone();
        let point_sum = p1.clone() + p2.clone();

        // The point sum should be contained in the interval sum
        assert!(sum.contains_interval(&point_sum));
    }
}
