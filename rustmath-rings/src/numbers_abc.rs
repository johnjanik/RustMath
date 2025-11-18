//! Abstract base classes for number types
//!
//! This module provides abstract base class definitions for various number types,
//! corresponding to SageMath's `sage.rings.numbers_abc`.
//!
//! # Overview
//!
//! This module defines marker traits for different categories of numbers following
//! the algebraic hierarchy:
//!
//! Number
//!  ├─ Rational
//!  ├─ Real
//!  ├─ Complex
//!  └─ Integral
//!       └─ Integer
//!
//! These traits are used for:
//! - Type checking and classification
//! - Generic programming with number constraints
//! - Runtime type identification
//! - Interface documentation
//!
//! # Python's numbers ABC
//!
//! This corresponds to Python's `numbers` module abstract base classes,
//! which SageMath uses for interoperability.

use std::fmt;

/// Marker trait for number types
///
/// Base trait for all numeric types. Corresponds to `numbers.Number`.
pub trait Number: fmt::Display + Clone {}

/// Marker trait for complex number types
///
/// Types that behave like complex numbers. Corresponds to `numbers.Complex`.
///
/// # Properties
///
/// - Contains real and imaginary parts
/// - Closed under addition, subtraction, multiplication, division
/// - Forms an algebraically closed field
pub trait Complex: Number {
    /// Get the real part
    ///
    /// # Returns
    ///
    /// String representation of real part
    fn real_part(&self) -> String;

    /// Get the imaginary part
    ///
    /// # Returns
    ///
    /// String representation of imaginary part
    fn imag_part(&self) -> String;

    /// Compute complex conjugate
    ///
    /// # Returns
    ///
    /// String representation of conjugate
    fn conjugate(&self) -> String;

    /// Compute absolute value (modulus)
    ///
    /// # Returns
    ///
    /// String representation of |z|
    fn abs(&self) -> String;
}

/// Marker trait for real number types
///
/// Types that behave like real numbers. Corresponds to `numbers.Real`.
///
/// # Properties
///
/// - Totally ordered
/// - Closed under addition, subtraction, multiplication, division
/// - Contains rationals as dense subset
pub trait Real: Complex {
    /// Compare with another real number
    ///
    /// # Returns
    ///
    /// Ordering relationship
    fn compare(&self, other: &Self) -> String;

    /// Check if positive
    ///
    /// # Returns
    ///
    /// True if self > 0
    fn is_positive(&self) -> bool;

    /// Check if negative
    ///
    /// # Returns
    ///
    /// True if self < 0
    fn is_negative(&self) -> bool;
}

/// Marker trait for rational number types
///
/// Types that behave like rational numbers. Corresponds to `numbers.Rational`.
///
/// # Properties
///
/// - Can be expressed as p/q where p, q are integers
/// - Closed under addition, subtraction, multiplication, division (by non-zero)
/// - Dense in the reals
pub trait Rational: Real {
    /// Get the numerator
    ///
    /// # Returns
    ///
    /// String representation of numerator
    fn numerator(&self) -> String;

    /// Get the denominator
    ///
    /// # Returns
    ///
    /// String representation of denominator
    fn denominator(&self) -> String;
}

/// Marker trait for integral number types
///
/// Types that behave like integers. Corresponds to `numbers.Integral`.
///
/// # Properties
///
/// - Closed under addition, subtraction, multiplication
/// - Supports division with remainder
/// - Forms a Euclidean domain
pub trait Integral: Rational {
    /// Compute quotient in division
    ///
    /// # Returns
    ///
    /// String representation of quotient
    fn div(&self, other: &Self) -> String;

    /// Compute remainder in division
    ///
    /// # Returns
    ///
    /// String representation of remainder
    fn mod_op(&self, other: &Self) -> String;

    /// Compute power
    ///
    /// # Returns
    ///
    /// String representation of self^n
    fn pow(&self, n: u32) -> String;
}

/// Check if a value is a number
///
/// # Returns
///
/// Always true for types implementing Number
pub fn is_number<T: Number>(_value: &T) -> bool {
    true
}

/// Check if a value is complex
///
/// # Returns
///
/// True for complex number types
pub fn is_complex<T: Complex>(_value: &T) -> bool {
    true
}

/// Check if a value is real
///
/// # Returns
///
/// True for real number types
pub fn is_real<T: Real>(_value: &T) -> bool {
    true
}

/// Check if a value is rational
///
/// # Returns
///
/// True for rational number types
pub fn is_rational<T: Rational>(_value: &T) -> bool {
    true
}

/// Check if a value is integral
///
/// # Returns
///
/// True for integral types
pub fn is_integral<T: Integral>(_value: &T) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example implementation for testing
    #[derive(Clone)]
    struct TestInteger(i64);

    impl fmt::Display for TestInteger {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl Number for TestInteger {}

    impl Complex for TestInteger {
        fn real_part(&self) -> String {
            format!("{}", self.0)
        }

        fn imag_part(&self) -> String {
            "0".to_string()
        }

        fn conjugate(&self) -> String {
            format!("{}", self.0)
        }

        fn abs(&self) -> String {
            format!("{}", self.0.abs())
        }
    }

    impl Real for TestInteger {
        fn compare(&self, other: &Self) -> String {
            if self.0 < other.0 {
                "less".to_string()
            } else if self.0 > other.0 {
                "greater".to_string()
            } else {
                "equal".to_string()
            }
        }

        fn is_positive(&self) -> bool {
            self.0 > 0
        }

        fn is_negative(&self) -> bool {
            self.0 < 0
        }
    }

    impl Rational for TestInteger {
        fn numerator(&self) -> String {
            format!("{}", self.0)
        }

        fn denominator(&self) -> String {
            "1".to_string()
        }
    }

    impl Integral for TestInteger {
        fn div(&self, other: &Self) -> String {
            format!("{}", self.0 / other.0)
        }

        fn mod_op(&self, other: &Self) -> String {
            format!("{}", self.0 % other.0)
        }

        fn pow(&self, n: u32) -> String {
            format!("{}", self.0.pow(n))
        }
    }

    #[test]
    fn test_is_number() {
        let n = TestInteger(42);
        assert!(is_number(&n));
    }

    #[test]
    fn test_is_complex() {
        let n = TestInteger(42);
        assert!(is_complex(&n));
    }

    #[test]
    fn test_is_real() {
        let n = TestInteger(42);
        assert!(is_real(&n));
    }

    #[test]
    fn test_is_rational() {
        let n = TestInteger(42);
        assert!(is_rational(&n));
    }

    #[test]
    fn test_is_integral() {
        let n = TestInteger(42);
        assert!(is_integral(&n));
    }

    #[test]
    fn test_complex_operations() {
        let n = TestInteger(5);
        assert_eq!(n.real_part(), "5");
        assert_eq!(n.imag_part(), "0");
        assert_eq!(n.conjugate(), "5");
        assert_eq!(n.abs(), "5");
    }

    #[test]
    fn test_real_operations() {
        let n1 = TestInteger(5);
        let n2 = TestInteger(3);

        assert_eq!(n1.compare(&n2), "greater");
        assert!(n1.is_positive());
        assert!(!n1.is_negative());
    }

    #[test]
    fn test_rational_operations() {
        let n = TestInteger(7);
        assert_eq!(n.numerator(), "7");
        assert_eq!(n.denominator(), "1");
    }

    #[test]
    fn test_integral_operations() {
        let n1 = TestInteger(17);
        let n2 = TestInteger(5);

        assert_eq!(n1.div(&n2), "3");
        assert_eq!(n1.mod_op(&n2), "2");
        assert_eq!(n1.pow(2), "289");
    }

    #[test]
    fn test_negative_number() {
        let n = TestInteger(-5);
        assert!(!n.is_positive());
        assert!(n.is_negative());
        assert_eq!(n.abs(), "5");
    }
}
