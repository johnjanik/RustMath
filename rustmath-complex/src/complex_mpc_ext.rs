//! Extended MPC functionality: morphisms and helper functions
//!
//! This module provides additional functionality for multi-precision complex numbers:
//! - Type morphisms (conversions between different number types)
//! - String parsing utilities
//! - Late imports and initialization
//!
//! These complement the core ComplexMPFR implementation in mpc.rs.

use crate::mpc::ComplexMPFR;
use crate::complex::Complex;
use rustmath_integers::Integer;
use rustmath_reals::RealMPFR;
use std::fmt;

/// Default precision for MPC numbers (53 bits = f64 equivalent)
pub const DEFAULT_PRECISION: u32 = 53;

/// Morphism from Complex (f64-based) to MPC (arbitrary precision)
///
/// Converts standard precision complex numbers to arbitrary precision.
/// This is useful for upgrading precision in computations.
///
/// # Examples
///
/// ```
/// use rustmath_complex::{Complex, ComplexMPFR, CCtoMPC};
///
/// let morph = CCtoMPC::new(256); // Target precision: 256 bits
/// let z = Complex::new(3.0, 4.0);
/// let z_mpc = morph.apply(&z);
/// assert_eq!(z_mpc.precision(), 256);
/// ```
#[derive(Clone, Debug)]
pub struct CCtoMPC {
    /// Target precision in bits
    precision: u32,
}

impl CCtoMPC {
    /// Create a new CC to MPC morphism with specified target precision
    pub fn new(precision: u32) -> Self {
        CCtoMPC { precision }
    }

    /// Apply the morphism: convert Complex to ComplexMPFR
    pub fn apply(&self, z: &Complex) -> ComplexMPFR {
        ComplexMPFR::with_val(self.precision, (z.real(), z.imag()))
    }

    /// Get the target precision
    pub fn precision(&self) -> u32 {
        self.precision
    }
}

impl fmt::Display for CCtoMPC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Morphism: Complex (f64) -> ComplexMPFR ({} bits)",
            self.precision
        )
    }
}

/// Morphism from Integer to MPC
///
/// Converts integers to multi-precision complex numbers (with zero imaginary part).
///
/// # Examples
///
/// ```
/// use rustmath_complex::{ComplexMPFR, INTEGERtoMPC};
/// use rustmath_integers::Integer;
///
/// let morph = INTEGERtoMPC::new(128);
/// let n = Integer::from(42);
/// let z = morph.apply(&n);
/// assert_eq!(z.real(), 42.0);
/// assert_eq!(z.imag(), 0.0);
/// ```
#[derive(Clone, Debug)]
pub struct INTEGERtoMPC {
    /// Target precision in bits
    precision: u32,
}

impl INTEGERtoMPC {
    /// Create a new Integer to MPC morphism
    pub fn new(precision: u32) -> Self {
        INTEGERtoMPC { precision }
    }

    /// Apply the morphism: convert Integer to ComplexMPFR
    pub fn apply(&self, n: &Integer) -> ComplexMPFR {
        let i = n.to_i64();
        let f64_val = if i >= i64::MIN && i <= i64::MAX {
            i as f64
        } else {
            // For very large integers, parse through string
            n.to_string().parse::<f64>().unwrap_or(f64::INFINITY)
        };
        ComplexMPFR::with_val(self.precision, (f64_val, 0.0))
    }

    /// Get the target precision
    pub fn precision(&self) -> u32 {
        self.precision
    }
}

impl fmt::Display for INTEGERtoMPC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Morphism: Integer -> ComplexMPFR ({} bits)",
            self.precision
        )
    }
}

/// Morphism from MPC to MPC (precision conversion)
///
/// Converts between ComplexMPFR values with different precisions.
/// Useful for precision management in computations.
///
/// # Examples
///
/// ```
/// use rustmath_complex::{ComplexMPFR, MPCtoMPC};
///
/// let morph = MPCtoMPC::new(512); // Convert to 512-bit precision
/// let z = ComplexMPFR::with_val(128, (3.0, 4.0));
/// let z_high = morph.apply(&z);
/// assert_eq!(z_high.precision(), 512);
/// ```
#[derive(Clone, Debug)]
pub struct MPCtoMPC {
    /// Target precision in bits
    precision: u32,
}

impl MPCtoMPC {
    /// Create a new MPC to MPC morphism with target precision
    pub fn new(precision: u32) -> Self {
        MPCtoMPC { precision }
    }

    /// Apply the morphism: convert to different precision
    pub fn apply(&self, z: &ComplexMPFR) -> ComplexMPFR {
        ComplexMPFR::with_val(self.precision, (z.real(), z.imag()))
    }

    /// Get the target precision
    pub fn precision(&self) -> u32 {
        self.precision
    }
}

impl fmt::Display for MPCtoMPC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Morphism: ComplexMPFR -> ComplexMPFR ({} bits)",
            self.precision
        )
    }
}

/// Morphism from MPFR (real) to MPC
///
/// Converts arbitrary precision real numbers to complex numbers
/// with zero imaginary part.
///
/// # Examples
///
/// ```
/// use rustmath_complex::{ComplexMPFR, MPFRtoMPC};
/// use rustmath_reals::RealMPFR;
///
/// let morph = MPFRtoMPC::new(256);
/// let r = RealMPFR::with_val(128, 3.14159);
/// let z = morph.apply(&r);
/// assert_eq!(z.precision(), 256);
/// ```
#[derive(Clone, Debug)]
pub struct MPFRtoMPC {
    /// Target precision in bits
    precision: u32,
}

impl MPFRtoMPC {
    /// Create a new MPFR to MPC morphism
    pub fn new(precision: u32) -> Self {
        MPFRtoMPC { precision }
    }

    /// Apply the morphism: convert RealMPFR to ComplexMPFR
    pub fn apply(&self, r: &RealMPFR) -> ComplexMPFR {
        ComplexMPFR::with_val(self.precision, (r.to_f64(), 0.0))
    }

    /// Get the target precision
    pub fn precision(&self) -> u32 {
        self.precision
    }
}

impl fmt::Display for MPFRtoMPC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Morphism: RealMPFR -> ComplexMPFR ({} bits)",
            self.precision
        )
    }
}

/// Factory function to create an MPC field with specified precision
///
/// This mirrors SageMath's `MPComplexField(prec)` constructor.
///
/// # Arguments
///
/// * `precision` - Number of bits of precision (default: 53)
///
/// # Examples
///
/// ```
/// use rustmath_complex::mp_complex_field;
///
/// let field = mp_complex_field(Some(256));
/// // Use field for creating high-precision complex numbers
/// ```
pub fn mpcomplex_field(precision: Option<u32>) -> MpcomplexFieldClass {
    MpcomplexFieldClass::new(precision.unwrap_or(DEFAULT_PRECISION))
}

/// Multi-precision complex field class
///
/// Represents a field of multi-precision complex numbers with specified precision.
/// All operations within this field maintain the specified precision.
///
/// # Examples
///
/// ```
/// use rustmath_complex::MpcomplexFieldClass;
///
/// let field = MpcomplexFieldClass::new(256);
/// let z = field.make_complex(3.0, 4.0);
/// assert_eq!(z.precision(), 256);
/// ```
#[derive(Clone, Debug)]
pub struct MpcomplexFieldClass {
    /// Precision in bits
    precision: u32,
}

impl MpcomplexFieldClass {
    /// Create a new multi-precision complex field
    pub fn new(precision: u32) -> Self {
        MpcomplexFieldClass { precision }
    }

    /// Create default field (53-bit precision)
    pub fn default() -> Self {
        MpcomplexFieldClass {
            precision: DEFAULT_PRECISION,
        }
    }

    /// Get the precision of this field
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Create a complex number in this field
    pub fn make_complex(&self, real: f64, imag: f64) -> ComplexMPFR {
        ComplexMPFR::with_val(self.precision, (real, imag))
    }

    /// Create from Integer components
    pub fn from_integers(&self, real: &Integer, imag: &Integer) -> ComplexMPFR {
        ComplexMPFR::with_val_integers(self.precision, real, imag)
    }

    /// Create from RealMPFR components
    pub fn from_reals(&self, real: RealMPFR, imag: RealMPFR) -> ComplexMPFR {
        ComplexMPFR::with_val_reals(real, imag)
    }

    /// Create zero
    pub fn zero(&self) -> ComplexMPFR {
        ComplexMPFR::zero_with_prec(self.precision)
    }

    /// Create one
    pub fn one(&self) -> ComplexMPFR {
        ComplexMPFR::one_with_prec(self.precision)
    }

    /// Create imaginary unit i
    pub fn i(&self) -> ComplexMPFR {
        ComplexMPFR::i_with_prec(self.precision)
    }

    /// Parse a complex number from string
    pub fn parse(&self, s: &str) -> Result<ComplexMPFR, String> {
        let (real_str, imag_str) = split_complex_string(s)?;

        let real: f64 = real_str.parse().map_err(|e| format!("Invalid real part: {}", e))?;
        let imag: f64 = imag_str.parse().map_err(|e| format!("Invalid imaginary part: {}", e))?;

        Ok(self.make_complex(real, imag))
    }

    /// Get the characteristic of this field (always 0 for complex numbers)
    pub fn characteristic(&self) -> u32 {
        0
    }

    /// Get a string representation of the field
    pub fn name(&self) -> String {
        format!("Complex Field with {} bits precision", self.precision)
    }
}

impl fmt::Display for MpcomplexFieldClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MPComplexField(precision={})", self.precision)
    }
}

impl PartialEq for MpcomplexFieldClass {
    fn eq(&self, other: &Self) -> bool {
        self.precision == other.precision
    }
}

/// Alias for ComplexMPFR to match SageMath naming
pub type MPComplexNumber = ComplexMPFR;

/// Late import helper (for lazy initialization)
///
/// This function is called to perform late imports of modules
/// that may not be needed at startup. In Rust, this is primarily
/// used for API compatibility with SageMath.
pub fn late_import() {
    // In Python/SageMath, this would import heavy dependencies lazily.
    // In Rust, all dependencies are statically linked, so this is a no-op.
    // We keep it for API compatibility.
}

/// Split a complex number string into real and imaginary parts
///
/// Parses strings in various formats:
/// - "3.0 + 4.0i"
/// - "3.0 + 4.0*I"
/// - "3.0+4.0j"
/// - "3 - 2i"
/// - "5i" (real part = 0)
/// - "7" (imaginary part = 0)
///
/// # Examples
///
/// ```
/// use rustmath_complex::split_complex_string;
///
/// let (re, im) = split_complex_string("3.0 + 4.0i").unwrap();
/// assert_eq!(re, "3.0");
/// assert_eq!(im, "4.0");
///
/// let (re, im) = split_complex_string("5i").unwrap();
/// assert_eq!(re, "0");
/// assert_eq!(im, "5");
/// ```
pub fn split_complex_string(s: &str) -> Result<(String, String), String> {
    let s = s.trim();

    // Remove whitespace
    let s = s.replace(" ", "");

    // Check for imaginary unit markers
    let has_i = s.contains('i') || s.contains('I') || s.contains('j');

    if !has_i {
        // Pure real number
        return Ok((s.to_string(), "0".to_string()));
    }

    // Find the imaginary unit and extract parts
    let s = s.replace("*I", "i").replace("*i", "i").replace("j", "i");

    // Handle pure imaginary: "5i" or "-3i"
    if !s.contains('+') && !s.contains('-') || (s.starts_with('-') && s.matches('-').count() == 1) {
        // Check if it's pure imaginary
        if s.ends_with('i') {
            let imag_part = s.trim_end_matches('i');
            if imag_part.is_empty() || imag_part == "+" {
                return Ok(("0".to_string(), "1".to_string()));
            } else if imag_part == "-" {
                return Ok(("0".to_string(), "-1".to_string()));
            } else {
                return Ok(("0".to_string(), imag_part.to_string()));
            }
        } else {
            return Ok((s.to_string(), "0".to_string()));
        }
    }

    // Find the operator between real and imaginary parts
    // Look for + or - that's not at the beginning
    let mut split_pos = None;
    let mut last_op = '+';

    for (i, c) in s.char_indices() {
        if i > 0 && (c == '+' || c == '-') {
            // Check if this is not part of an exponent (e.g., 1e-5)
            if let Some(prev_char) = s.chars().nth(i - 1) {
                if prev_char != 'e' && prev_char != 'E' {
                    split_pos = Some(i);
                    last_op = c;
                }
            }
        }
    }

    if let Some(pos) = split_pos {
        let real_part = &s[..pos];
        let imag_part = &s[pos..].trim_start_matches('+').trim_end_matches('i');

        let imag_with_sign = if last_op == '-' {
            format!("-{}", imag_part.trim_start_matches('-'))
        } else {
            imag_part.to_string()
        };

        // Handle empty imaginary coefficient (means 1 or -1)
        let imag_final = if imag_with_sign.is_empty() || imag_with_sign == "+" {
            "1".to_string()
        } else if imag_with_sign == "-" {
            "-1".to_string()
        } else {
            imag_with_sign
        };

        Ok((real_part.to_string(), imag_final))
    } else {
        Err(format!("Could not parse complex string: {}", s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cc_to_mpc() {
        let morph = CCtoMPC::new(256);
        let z = Complex::new(3.0, 4.0);
        let z_mpc = morph.apply(&z);

        assert_eq!(z_mpc.precision(), 256);
        assert!((z_mpc.real() - 3.0).abs() < 1e-10);
        assert!((z_mpc.imag() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_integer_to_mpc() {
        let morph = INTEGERtoMPC::new(128);
        let n = Integer::from(42);
        let z = morph.apply(&n);

        assert_eq!(z.precision(), 128);
        assert_eq!(z.real(), 42.0);
        assert_eq!(z.imag(), 0.0);
    }

    #[test]
    fn test_mpc_to_mpc() {
        let morph = MPCtoMPC::new(512);
        let z = ComplexMPFR::with_val(128, (3.0, 4.0));
        let z_high = morph.apply(&z);

        assert_eq!(z_high.precision(), 512);
        assert!((z_high.real() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mpfr_to_mpc() {
        let morph = MPFRtoMPC::new(256);
        let r = RealMPFR::with_val(128, 3.14159);
        let z = morph.apply(&r);

        assert_eq!(z.precision(), 256);
        assert!((z.real() - 3.14159).abs() < 1e-5);
        assert_eq!(z.imag(), 0.0);
    }

    #[test]
    fn test_mp_complex_field() {
        let field = mp_complex_field(Some(256));
        assert_eq!(field.precision(), 256);

        let z = field.make_complex(3.0, 4.0);
        assert_eq!(z.precision(), 256);
    }

    #[test]
    fn test_mp_complex_field_class() {
        let field = MpcomplexFieldClass::new(256);

        let zero = field.zero();
        assert!(zero.is_zero());

        let one = field.one();
        assert!(one.is_one());

        let i = field.i();
        assert_eq!(i.real(), 0.0);
        assert_eq!(i.imag(), 1.0);
    }

    #[test]
    fn test_split_complex_string_standard() {
        let (re, im) = split_complex_string("3.0 + 4.0i").unwrap();
        assert_eq!(re, "3.0");
        assert_eq!(im, "4.0");

        let (re, im) = split_complex_string("3.0 - 4.0i").unwrap();
        assert_eq!(re, "3.0");
        assert_eq!(im, "-4.0");
    }

    #[test]
    fn test_split_complex_string_pure_real() {
        let (re, im) = split_complex_string("42.0").unwrap();
        assert_eq!(re, "42.0");
        assert_eq!(im, "0");
    }

    #[test]
    fn test_split_complex_string_pure_imag() {
        let (re, im) = split_complex_string("5i").unwrap();
        assert_eq!(re, "0");
        assert_eq!(im, "5");

        let (re, im) = split_complex_string("-3i").unwrap();
        assert_eq!(re, "0");
        assert_eq!(im, "-3");

        let (re, im) = split_complex_string("i").unwrap();
        assert_eq!(re, "0");
        assert_eq!(im, "1");
    }

    #[test]
    fn test_split_complex_string_no_spaces() {
        let (re, im) = split_complex_string("3+4i").unwrap();
        assert_eq!(re, "3");
        assert_eq!(im, "4");
    }

    #[test]
    fn test_split_complex_string_with_j() {
        let (re, im) = split_complex_string("3.0 + 4.0j").unwrap();
        assert_eq!(re, "3.0");
        assert_eq!(im, "4.0");
    }

    #[test]
    fn test_field_parse() {
        let field = MpcomplexFieldClass::new(128);

        let z = field.parse("3.0 + 4.0i").unwrap();
        assert!((z.real() - 3.0).abs() < 1e-10);
        assert!((z.imag() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_field_from_integers() {
        let field = MpcomplexFieldClass::new(128);
        let real = Integer::from(3);
        let imag = Integer::from(4);

        let z = field.from_integers(&real, &imag);
        assert_eq!(z.real(), 3.0);
        assert_eq!(z.imag(), 4.0);
    }

    #[test]
    fn test_field_properties() {
        let field = MpcomplexFieldClass::new(256);
        assert_eq!(field.characteristic(), 0);
        assert!(field.name().contains("256"));
    }

    #[test]
    fn test_late_import() {
        // Should not panic
        late_import();
    }
}
