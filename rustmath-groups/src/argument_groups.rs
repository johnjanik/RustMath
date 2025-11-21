//! Argument Groups for Complex Numbers
//!
//! This module implements group-theoretic representations of complex arguments,
//! capturing the "angular" part of complex numbers as multiplicative group elements.
//!
//! # Mathematical Background
//!
//! A complex number z can be written in polar form as:
//! z = r · e^(iθ)
//!
//! where r = |z| is the modulus and θ = arg(z) is the argument (angle).
//! The argument forms a multiplicative group structure:
//! - e^(iθ₁) · e^(iθ₂) = e^(i(θ₁+θ₂))
//! - (e^(iθ))⁻¹ = e^(-iθ)
//!
//! This module implements several specialized argument groups:
//! - Unit circle: All complex numbers of modulus 1
//! - Roots of unity: e^(2πi·k/n) for integers k, n
//! - Sign group: {+1, -1}
//! - Argument by element: Formal arguments from complex elements
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::argument_groups::{UnitCircleGroup, RootsOfUnityGroup, SignGroup};
//!
//! // Unit circle point at angle π/2
//! let uc = UnitCircleGroup::new();
//! let point = uc.from_angle(std::f64::consts::PI / 2.0);
//!
//! // 4th roots of unity
//! let roots = RootsOfUnityGroup::new(4);
//! let zeta4 = roots.gen(); // Primitive 4th root
//!
//! // Sign group
//! let signs = SignGroup::new();
//! let pos = signs.positive();
//! let neg = signs.negative();
//! ```

use std::f64::consts::PI;
use std::fmt;

use rustmath_complex::Complex;
use rustmath_rationals::Rational;

// ============================================================================
// Abstract Argument Group Traits
// ============================================================================

/// An abstract argument element
///
/// This trait represents an element of an argument group, capturing the
/// angular/phase component of complex numbers.
pub trait AbstractArgument: Clone + fmt::Debug {
    /// Convert to a complex number on the unit circle
    fn to_complex(&self) -> Complex;

    /// Multiply two argument elements
    fn multiply(&self, other: &Self) -> Self;

    /// Compute the inverse
    fn inverse(&self) -> Self;

    /// Raise to a power
    fn pow(&self, n: i32) -> Self;

    /// Check if this is the identity
    fn is_identity(&self) -> bool;
}

/// An abstract argument group
///
/// This trait represents a group of complex arguments.
pub trait AbstractArgumentGroup {
    type Element: AbstractArgument;

    /// Get the identity element
    fn identity(&self) -> Self::Element;
}

// ============================================================================
// Unit Circle Group
// ============================================================================

/// A point on the unit circle in the complex plane
///
/// Represented as e^(2πi·exponent) where exponent ∈ [0, 1)
#[derive(Debug, Clone)]
pub struct UnitCirclePoint {
    /// The exponent in e^(2πi·exponent), normalized to [0, 1)
    exponent: f64,
}

impl UnitCirclePoint {
    /// Create a new unit circle point from an exponent
    ///
    /// The exponent is normalized to [0, 1)
    pub fn new(exponent: f64) -> Self {
        let normalized = (exponent % 1.0 + 1.0) % 1.0; // Ensure [0, 1)
        Self {
            exponent: normalized,
        }
    }

    /// Create from an angle in radians
    pub fn from_angle(angle: f64) -> Self {
        Self::new(angle / (2.0 * PI))
    }

    /// Get the exponent
    pub fn exponent(&self) -> f64 {
        self.exponent
    }

    /// Get the angle in radians
    pub fn angle(&self) -> f64 {
        self.exponent * 2.0 * PI
    }
}

impl AbstractArgument for UnitCirclePoint {
    fn to_complex(&self) -> Complex {
        let angle = self.angle();
        Complex::new(angle.cos(), angle.sin())
    }

    fn multiply(&self, other: &Self) -> Self {
        Self::new(self.exponent + other.exponent)
    }

    fn inverse(&self) -> Self {
        Self::new(-self.exponent)
    }

    fn pow(&self, n: i32) -> Self {
        Self::new(self.exponent * n as f64)
    }

    fn is_identity(&self) -> bool {
        self.exponent.abs() < 1e-10 || (self.exponent - 1.0).abs() < 1e-10
    }
}

impl fmt::Display for UnitCirclePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            write!(f, "1")
        } else if (self.exponent - 0.5).abs() < 1e-10 {
            write!(f, "-1")
        } else if (self.exponent - 0.25).abs() < 1e-10 {
            write!(f, "i")
        } else if (self.exponent - 0.75).abs() < 1e-10 {
            write!(f, "-i")
        } else {
            write!(f, "e^(2πi·{:.4})", self.exponent)
        }
    }
}

impl PartialEq for UnitCirclePoint {
    fn eq(&self, other: &Self) -> bool {
        (self.exponent - other.exponent).abs() < 1e-10
    }
}

impl Eq for UnitCirclePoint {}

/// The multiplicative group of points on the unit circle
///
/// This is isomorphic to ℝ/ℤ under addition, or S¹ as a topological group.
#[derive(Debug, Clone)]
pub struct UnitCircleGroup;

impl UnitCircleGroup {
    /// Create a new unit circle group
    pub fn new() -> Self {
        Self
    }

    /// Create a point from an angle in radians
    pub fn from_angle(&self, angle: f64) -> UnitCirclePoint {
        UnitCirclePoint::from_angle(angle)
    }

    /// Create a point from an exponent in [0, 1)
    pub fn from_exponent(&self, exponent: f64) -> UnitCirclePoint {
        UnitCirclePoint::new(exponent)
    }
}

impl AbstractArgumentGroup for UnitCircleGroup {
    type Element = UnitCirclePoint;

    fn identity(&self) -> Self::Element {
        UnitCirclePoint::new(0.0)
    }
}

impl fmt::Display for UnitCircleGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unit circle group")
    }
}

impl Default for UnitCircleGroup {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Roots of Unity Group
// ============================================================================

/// A root of unity
///
/// Represented as e^(2πi·k/n) for integers k, n with gcd(k, n) possibly > 1
#[derive(Debug, Clone)]
pub struct RootOfUnity {
    /// Numerator of the exponent
    k: i64,
    /// Denominator of the exponent (order)
    n: u64,
}

impl RootOfUnity {
    /// Create a new root of unity e^(2πi·k/n)
    pub fn new(k: i64, n: u64) -> Self {
        assert!(n > 0, "Order must be positive");
        // Normalize k to [0, n)
        let k_normalized = ((k % n as i64) + n as i64) % n as i64;
        Self {
            k: k_normalized,
            n,
        }
    }

    /// Get the numerator k
    pub fn numerator(&self) -> i64 {
        self.k
    }

    /// Get the denominator (order) n
    pub fn order(&self) -> u64 {
        self.n
    }

    /// Get the exponent as a rational
    pub fn exponent_rational(&self) -> Rational {
        Rational::new(self.k.into(), self.n.into()).unwrap()
    }

    /// Get the exponent as a float
    pub fn exponent(&self) -> f64 {
        self.k as f64 / self.n as f64
    }
}

impl AbstractArgument for RootOfUnity {
    fn to_complex(&self) -> Complex {
        let angle = 2.0 * PI * self.exponent();
        Complex::new(angle.cos(), angle.sin())
    }

    fn multiply(&self, other: &Self) -> Self {
        // Need common denominator
        let new_n = self.n * other.n / gcd(self.n, other.n);
        let k1 = self.k * (new_n / self.n) as i64;
        let k2 = other.k * (new_n / other.n) as i64;
        Self::new(k1 + k2, new_n)
    }

    fn inverse(&self) -> Self {
        Self::new(-self.k, self.n)
    }

    fn pow(&self, n: i32) -> Self {
        Self::new(self.k * n as i64, self.n)
    }

    fn is_identity(&self) -> bool {
        self.k % self.n as i64 == 0
    }
}

impl fmt::Display for RootOfUnity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            write!(f, "1")
        } else if self.k == self.n as i64 / 2 && self.n % 2 == 0 {
            write!(f, "-1")
        } else if self.n == 1 {
            write!(f, "1")
        } else {
            write!(f, "ζ_{}^{}", self.n, self.k)
        }
    }
}

impl PartialEq for RootOfUnity {
    fn eq(&self, other: &Self) -> bool {
        let exp1 = self.exponent();
        let exp2 = other.exponent();
        (exp1 - exp2).abs() < 1e-10
    }
}

impl Eq for RootOfUnity {}

/// The multiplicative group of nth roots of unity
///
/// This is a cyclic group of order n, isomorphic to ℤ/nℤ.
#[derive(Debug, Clone)]
pub struct RootsOfUnityGroup {
    /// The order n
    n: u64,
}

impl RootsOfUnityGroup {
    /// Create the group of nth roots of unity
    pub fn new(n: u64) -> Self {
        assert!(n > 0, "Order must be positive");
        Self { n }
    }

    /// Get the order
    pub fn order(&self) -> u64 {
        self.n
    }

    /// Get the primitive nth root of unity: e^(2πi/n)
    pub fn gen(&self) -> RootOfUnity {
        RootOfUnity::new(1, self.n)
    }

    /// Get the kth power of the generator
    pub fn element(&self, k: i64) -> RootOfUnity {
        RootOfUnity::new(k, self.n)
    }

    /// Get all nth roots of unity
    pub fn all_roots(&self) -> Vec<RootOfUnity> {
        (0..self.n as i64)
            .map(|k| RootOfUnity::new(k, self.n))
            .collect()
    }
}

impl AbstractArgumentGroup for RootsOfUnityGroup {
    type Element = RootOfUnity;

    fn identity(&self) -> Self::Element {
        RootOfUnity::new(0, self.n)
    }
}

impl fmt::Display for RootsOfUnityGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Group of {}-th roots of unity", self.n)
    }
}

// ============================================================================
// Sign Group
// ============================================================================

/// An element of the sign group: +1 or -1
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    /// Positive sign (+1)
    Positive,
    /// Negative sign (-1)
    Negative,
}

impl Sign {
    /// Create a sign from an integer (positive → +1, negative/zero → -1)
    pub fn from_sign(value: i32) -> Self {
        if value >= 0 {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }

    /// Convert to an integer
    pub fn to_int(&self) -> i32 {
        match self {
            Sign::Positive => 1,
            Sign::Negative => -1,
        }
    }

    /// Convert to a float
    pub fn to_f64(&self) -> f64 {
        self.to_int() as f64
    }
}

impl AbstractArgument for Sign {
    fn to_complex(&self) -> Complex {
        Complex::from_real(self.to_f64())
    }

    fn multiply(&self, other: &Self) -> Self {
        if self == other {
            Sign::Positive
        } else {
            Sign::Negative
        }
    }

    fn inverse(&self) -> Self {
        *self // Both +1 and -1 are their own inverses
    }

    fn pow(&self, n: i32) -> Self {
        if n % 2 == 0 {
            Sign::Positive
        } else {
            *self
        }
    }

    fn is_identity(&self) -> bool {
        matches!(self, Sign::Positive)
    }
}

impl fmt::Display for Sign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sign::Positive => write!(f, "+1"),
            Sign::Negative => write!(f, "-1"),
        }
    }
}

/// The sign group: {+1, -1}
///
/// This is a cyclic group of order 2, isomorphic to ℤ/2ℤ.
#[derive(Debug, Clone)]
pub struct SignGroup;

impl SignGroup {
    /// Create a new sign group
    pub fn new() -> Self {
        Self
    }

    /// Get the positive sign
    pub fn positive(&self) -> Sign {
        Sign::Positive
    }

    /// Get the negative sign
    pub fn negative(&self) -> Sign {
        Sign::Negative
    }

    /// Get all elements
    pub fn elements(&self) -> Vec<Sign> {
        vec![Sign::Positive, Sign::Negative]
    }
}

impl AbstractArgumentGroup for SignGroup {
    type Element = Sign;

    fn identity(&self) -> Self::Element {
        Sign::Positive
    }
}

impl fmt::Display for SignGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sign group {{+1, -1}}")
    }
}

impl Default for SignGroup {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Argument By Element
// ============================================================================

/// An argument defined by a complex element
///
/// Represents e^(i·arg(z)) for a complex number z, stored symbolically
#[derive(Debug, Clone)]
pub struct ArgumentByElement {
    /// The complex number whose argument we're taking
    element: Complex,
}

impl ArgumentByElement {
    /// Create a new argument from a complex element
    pub fn new(element: Complex) -> Self {
        Self { element }
    }

    /// Get the underlying complex element
    pub fn element(&self) -> &Complex {
        &self.element
    }

    /// Get the argument (angle) in radians
    pub fn argument(&self) -> f64 {
        self.element.arg()
    }
}

impl AbstractArgument for ArgumentByElement {
    fn to_complex(&self) -> Complex {
        let r = self.element.abs();
        if r == 0.0 {
            Complex::zero()
        } else {
            self.element.div(&Complex::from_real(r)).unwrap()
        }
    }

    fn multiply(&self, other: &Self) -> Self {
        Self::new(self.element.mul(&other.element))
    }

    fn inverse(&self) -> Self {
        Self::new(self.element.conjugate())
    }

    fn pow(&self, n: i32) -> Self {
        let mut result = Complex::one();
        for _ in 0..n.abs() {
            result = result.mul(&self.element);
        }
        if n < 0 {
            result = result.conjugate();
        }
        Self::new(result)
    }

    fn is_identity(&self) -> bool {
        (self.element.real() - 1.0).abs() < 1e-10 && self.element.imag().abs() < 1e-10
    }
}

impl fmt::Display for ArgumentByElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "arg({})", self.element)
    }
}

/// Group of arguments defined by complex elements
#[derive(Debug, Clone)]
pub struct ArgumentByElementGroup;

impl ArgumentByElementGroup {
    /// Create a new argument by element group
    pub fn new() -> Self {
        Self
    }

    /// Create an argument from a complex number
    pub fn from_complex(&self, z: Complex) -> ArgumentByElement {
        ArgumentByElement::new(z)
    }
}

impl AbstractArgumentGroup for ArgumentByElementGroup {
    type Element = ArgumentByElement;

    fn identity(&self) -> Self::Element {
        ArgumentByElement::new(Complex::one())
    }
}

impl fmt::Display for ArgumentByElementGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Argument by element group")
    }
}

impl Default for ArgumentByElementGroup {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Argument Group Factory
// ============================================================================

/// Factory for creating appropriate argument groups
///
/// This provides a unified interface for creating argument groups based on
/// different mathematical contexts.
#[derive(Debug, Clone)]
pub struct ArgumentGroupFactory;

impl ArgumentGroupFactory {
    /// Create a unit circle group
    pub fn unit_circle() -> UnitCircleGroup {
        UnitCircleGroup::new()
    }

    /// Create a roots of unity group of order n
    pub fn roots_of_unity(n: u64) -> RootsOfUnityGroup {
        RootsOfUnityGroup::new(n)
    }

    /// Create the sign group
    pub fn signs() -> SignGroup {
        SignGroup::new()
    }

    /// Create an argument by element group
    pub fn by_element() -> ArgumentByElementGroup {
        ArgumentByElementGroup::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute the greatest common divisor
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Compute the exponent for a complex argument group element
///
/// This is a helper function matching SageMath's exponent attribute.
pub fn exponent<A: AbstractArgument>(arg: &A) -> Complex {
    let z = arg.to_complex();
    // Return the exponent representation
    // For e^(iθ), this would be iθ
    let angle = z.arg();
    Complex::new(0.0, angle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_circle_point() {
        let p1 = UnitCirclePoint::new(0.25); // i
        let p2 = UnitCirclePoint::new(0.25); // i
        let prod = p1.multiply(&p2);

        assert!((prod.exponent() - 0.5).abs() < 1e-10); // i * i = -1
    }

    #[test]
    fn test_unit_circle_identity() {
        let id = UnitCirclePoint::new(0.0);
        assert!(id.is_identity());

        let id2 = UnitCirclePoint::new(1.0);
        assert!(id2.is_identity());
    }

    #[test]
    fn test_unit_circle_inverse() {
        let p = UnitCirclePoint::new(0.3);
        let inv = p.inverse();
        let prod = p.multiply(&inv);
        assert!(prod.is_identity());
    }

    #[test]
    fn test_unit_circle_from_angle() {
        let p = UnitCirclePoint::from_angle(PI / 2.0);
        assert!((p.exponent() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_unit_circle_group() {
        let uc = UnitCircleGroup::new();
        let id = uc.identity();
        assert!(id.is_identity());

        let p = uc.from_angle(PI);
        assert!((p.exponent() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_root_of_unity() {
        let zeta4 = RootOfUnity::new(1, 4); // i
        let zeta4_squared = zeta4.multiply(&zeta4);
        assert_eq!(zeta4_squared.numerator(), 2);
        assert_eq!(zeta4_squared.order(), 4);
    }

    #[test]
    fn test_root_of_unity_identity() {
        let one = RootOfUnity::new(0, 4);
        assert!(one.is_identity());

        let also_one = RootOfUnity::new(4, 4);
        assert!(also_one.is_identity());
    }

    #[test]
    fn test_root_of_unity_inverse() {
        let zeta = RootOfUnity::new(1, 5);
        let inv = zeta.inverse();
        let prod = zeta.multiply(&inv);
        assert!(prod.is_identity());
    }

    #[test]
    fn test_roots_of_unity_group() {
        let g = RootsOfUnityGroup::new(6);
        assert_eq!(g.order(), 6);

        let gen = g.gen();
        assert_eq!(gen.numerator(), 1);
        assert_eq!(gen.order(), 6);

        let roots = g.all_roots();
        assert_eq!(roots.len(), 6);
    }

    #[test]
    fn test_sign() {
        let pos = Sign::Positive;
        let neg = Sign::Negative;

        assert_eq!(pos.to_int(), 1);
        assert_eq!(neg.to_int(), -1);

        let prod = pos.multiply(&neg);
        assert_eq!(prod, Sign::Negative);

        let prod2 = neg.multiply(&neg);
        assert_eq!(prod2, Sign::Positive);
    }

    #[test]
    fn test_sign_identity() {
        let pos = Sign::Positive;
        assert!(pos.is_identity());

        let neg = Sign::Negative;
        assert!(!neg.is_identity());
    }

    #[test]
    fn test_sign_inverse() {
        let pos = Sign::Positive;
        assert_eq!(pos.inverse(), pos);

        let neg = Sign::Negative;
        assert_eq!(neg.inverse(), neg);
    }

    #[test]
    fn test_sign_pow() {
        let neg = Sign::Negative;
        assert_eq!(neg.pow(2), Sign::Positive);
        assert_eq!(neg.pow(3), Sign::Negative);
        assert_eq!(neg.pow(4), Sign::Positive);
    }

    #[test]
    fn test_sign_group() {
        let g = SignGroup::new();
        let pos = g.positive();
        let neg = g.negative();

        assert!(pos.is_identity());
        assert!(!neg.is_identity());

        let elems = g.elements();
        assert_eq!(elems.len(), 2);
    }

    #[test]
    fn test_argument_by_element() {
        let z = Complex::new(1.0, 1.0);
        let arg = ArgumentByElement::new(z);

        let angle = arg.argument();
        assert!((angle - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_argument_by_element_multiply() {
        let z1 = Complex::new(1.0, 0.0);
        let z2 = Complex::new(0.0, 1.0);
        let arg1 = ArgumentByElement::new(z1);
        let arg2 = ArgumentByElement::new(z2);

        let prod = arg1.multiply(&arg2);
        let expected = Complex::new(0.0, 1.0);
        assert!((prod.element().real() - expected.real()).abs() < 1e-10);
        assert!((prod.element().imag() - expected.imag()).abs() < 1e-10);
    }

    #[test]
    fn test_argument_group_factory() {
        let uc = ArgumentGroupFactory::unit_circle();
        let _id = uc.identity();

        let roots = ArgumentGroupFactory::roots_of_unity(8);
        assert_eq!(roots.order(), 8);

        let signs = ArgumentGroupFactory::signs();
        let _pos = signs.positive();
    }

    #[test]
    fn test_display() {
        let p = UnitCirclePoint::new(0.25);
        assert_eq!(format!("{}", p), "i");

        let p2 = UnitCirclePoint::new(0.5);
        assert_eq!(format!("{}", p2), "-1");

        let zeta = RootOfUnity::new(1, 3);
        assert!(format!("{}", zeta).contains("ζ_3"));

        let pos = Sign::Positive;
        assert_eq!(format!("{}", pos), "+1");
    }

    #[test]
    fn test_to_complex() {
        let p = UnitCirclePoint::new(0.0);
        let c = p.to_complex();
        assert!((c.real() - 1.0).abs() < 1e-10);
        assert!(c.imag().abs() < 1e-10);

        let p2 = UnitCirclePoint::new(0.25);
        let c2 = p2.to_complex();
        assert!(c2.real().abs() < 1e-10);
        assert!((c2.imag() - 1.0).abs() < 1e-10);
    }
}
