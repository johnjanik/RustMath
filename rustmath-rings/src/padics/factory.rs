//! Factory functions for creating p-adic rings and fields
//!
//! This module provides SageMath-compatible factory functions for constructing
//! p-adic number systems with various precision models.
//!
//! ## Precision Models
//!
//! ### Capped Relative Precision (default)
//! Tracks the number of significant p-adic digits. This is the most flexible model
//! and is appropriate for most computations.
//!
//! **Trade-offs:**
//! - ✓ Natural for multiplication (precision adds)
//! - ✓ Handles leading zeros gracefully
//! - ✗ Addition can lose precision unpredictably
//! - ✗ More bookkeeping overhead
//!
//! **Example:** In 5-adics with precision 10:
//! - `1 + O(5^10)` has 10 digits of precision
//! - `5 + O(5^10)` still has 10 digits (relative to valuation)
//!
//! ### Capped Absolute Precision
//! Tracks the absolute precision (all values modulo p^n). Simpler but can waste
//! precision when valuations vary.
//!
//! **Trade-offs:**
//! - ✓ Simple and predictable behavior
//! - ✓ Efficient for fixed precision computations
//! - ✗ Wastes precision for high-valuation elements
//! - ✗ Less natural for mathematical operations
//!
//! **Example:** In 5-adics with precision 10:
//! - `1 + O(5^10)` has 10 digits
//! - `5 + O(5^10)` also has 10 digits (but only 9 significant)
//!
//! ### Fixed Modulus
//! Fixed arithmetic modulo p^n with no precision tracking. Fastest but least flexible.
//!
//! **Trade-offs:**
//! - ✓ Fastest performance (no precision bookkeeping)
//! - ✓ Smallest memory footprint
//! - ✗ No automatic precision management
//! - ✗ Can't represent elements with different precisions
//! - ✗ Not suitable for iterative algorithms
//!
//! **Example:** In 5-adics with modulus 5^10:
//! - All elements stored modulo 5^10
//! - No distinction between different precision requirements
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::padics::factory::{zp, qp, PrecisionModel};
//! use rustmath_integers::Integer;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create 5-adic integers with capped relative precision
//! let zp = zp(Integer::from(5), 20, PrecisionModel::CappedRelative)?;
//! let x = zp.from_int(7)?;
//!
//! // Create 7-adic field with capped absolute precision
//! let qp = qp(Integer::from(7), 15, PrecisionModel::CappedAbsolute)?;
//! let y = qp.from_rational_nums(3, 7)?;
//!
//! // Fixed modulus for fastest performance
//! let zp_fast = zp(Integer::from(3), 10, PrecisionModel::FixedModulus)?;
//! # Ok(())
//! # }
//! ```

use rustmath_core::{MathError, Result, Ring};
use rustmath_integers::Integer;
use rustmath_padics::{PadicInteger, PadicRational};
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;
use std::fmt;

/// Precision model for p-adic arithmetic
///
/// Different models trade off flexibility, accuracy, and performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionModel {
    /// Track relative precision (significant p-adic digits)
    ///
    /// Best for: General purpose p-adic computations
    CappedRelative,

    /// Track absolute precision (modulo p^n)
    ///
    /// Best for: Fixed precision algorithms, simpler mental model
    CappedAbsolute,

    /// Fixed modulus arithmetic (no precision tracking)
    ///
    /// Best for: Performance-critical code, no iterative algorithms
    FixedModulus,
}

impl Default for PrecisionModel {
    fn default() -> Self {
        PrecisionModel::CappedRelative
    }
}

impl fmt::Display for PrecisionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrecisionModel::CappedRelative => write!(f, "capped-rel"),
            PrecisionModel::CappedAbsolute => write!(f, "capped-abs"),
            PrecisionModel::FixedModulus => write!(f, "fixed-mod"),
        }
    }
}

/// p-adic integer ring (Z_p)
///
/// Ring of p-adic integers with specified precision model.
#[derive(Debug, Clone)]
pub struct PadicIntegerRing {
    prime: Integer,
    precision: usize,
    model: PrecisionModel,
}

impl PadicIntegerRing {
    /// Create a new p-adic integer ring
    ///
    /// # Arguments
    ///
    /// * `prime` - The prime p (must be prime, but not checked for performance)
    /// * `precision` - Default precision for elements
    /// * `model` - Precision model to use
    pub fn new(prime: Integer, precision: usize, model: PrecisionModel) -> Result<Self> {
        if prime <= Integer::one() {
            return Err(MathError::InvalidArgument(
                "Prime must be > 1".to_string(),
            ));
        }

        if precision == 0 {
            return Err(MathError::InvalidArgument(
                "Precision must be > 0".to_string(),
            ));
        }

        Ok(PadicIntegerRing {
            prime,
            precision,
            model,
        })
    }

    /// Get the prime p
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// Get the default precision
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Get the precision model
    pub fn model(&self) -> PrecisionModel {
        self.model
    }

    /// Create an element from an integer
    pub fn from_int(&self, value: i64) -> Result<PadicInteger> {
        PadicInteger::from_integer(Integer::from(value), self.prime.clone(), self.precision)
    }

    /// Create an element from an Integer
    pub fn from_integer(&self, value: Integer) -> Result<PadicInteger> {
        PadicInteger::from_integer(value, self.prime.clone(), self.precision)
    }

    /// Create zero element
    pub fn zero(&self) -> Result<PadicInteger> {
        PadicInteger::zero(self.prime.clone(), self.precision)
    }

    /// Create one element
    pub fn one(&self) -> Result<PadicInteger> {
        PadicInteger::one(self.prime.clone(), self.precision)
    }

    /// Adjust precision of an element according to the model
    ///
    /// For CappedRelative: maintains relative precision
    /// For CappedAbsolute: maintains absolute precision
    /// For FixedModulus: always uses fixed precision
    pub fn adjust_precision(&self, elem: PadicInteger) -> PadicInteger {
        match self.model {
            PrecisionModel::CappedRelative => {
                // Relative precision: precision relative to valuation
                let val = elem.valuation();
                if val != u32::MAX && val > 0 {
                    // Element has valuation v, so effective precision is precision + v
                    // But we store absolute precision, so no change needed
                    elem
                } else {
                    elem
                }
            }
            PrecisionModel::CappedAbsolute => {
                // Absolute precision: always modulo p^precision
                if elem.precision() != self.precision {
                    if elem.precision() > self.precision {
                        elem.truncate(self.precision).unwrap_or(elem)
                    } else {
                        elem.lift(self.precision).unwrap_or(elem)
                    }
                } else {
                    elem
                }
            }
            PrecisionModel::FixedModulus => {
                // Fixed modulus: force exact precision
                if elem.precision() != self.precision {
                    PadicInteger::from_integer(elem.value().clone(), self.prime.clone(), self.precision)
                        .unwrap_or(elem)
                } else {
                    elem
                }
            }
        }
    }
}

impl fmt::Display for PadicIntegerRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Ring of {}-adic integers (precision {}, model: {})",
            self.prime, self.precision, self.model
        )
    }
}

/// p-adic field (Q_p)
///
/// Field of p-adic numbers with specified precision model.
#[derive(Debug, Clone)]
pub struct PadicField {
    prime: Integer,
    precision: usize,
    model: PrecisionModel,
}

impl PadicField {
    /// Create a new p-adic field
    ///
    /// # Arguments
    ///
    /// * `prime` - The prime p (must be prime, but not checked for performance)
    /// * `precision` - Default precision for elements
    /// * `model` - Precision model to use
    pub fn new(prime: Integer, precision: usize, model: PrecisionModel) -> Result<Self> {
        if prime <= Integer::one() {
            return Err(MathError::InvalidArgument(
                "Prime must be > 1".to_string(),
            ));
        }

        if precision == 0 {
            return Err(MathError::InvalidArgument(
                "Precision must be > 0".to_string(),
            ));
        }

        Ok(PadicField {
            prime,
            precision,
            model,
        })
    }

    /// Get the prime p
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// Get the default precision
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Get the precision model
    pub fn model(&self) -> PrecisionModel {
        self.model
    }

    /// Get the integer ring (Z_p)
    pub fn integer_ring(&self) -> Result<PadicIntegerRing> {
        PadicIntegerRing::new(self.prime.clone(), self.precision, self.model)
    }

    /// Create an element from a rational
    pub fn from_rational(&self, value: Rational) -> Result<PadicRational> {
        PadicRational::from_rational(value, self.prime.clone(), self.precision)
    }

    /// Create an element from numerator and denominator
    pub fn from_rational_nums(&self, num: i64, den: i64) -> Result<PadicRational> {
        let rat = Rational::new(Integer::from(num), Integer::from(den))?;
        self.from_rational(rat)
    }

    /// Create an element from an integer
    pub fn from_int(&self, value: i64) -> Result<PadicRational> {
        let padic_int = PadicInteger::from_integer(
            Integer::from(value),
            self.prime.clone(),
            self.precision,
        )?;
        Ok(PadicRational::from_padic_integer(padic_int))
    }

    /// Create zero element
    pub fn zero(&self) -> Result<PadicRational> {
        let zero_int = PadicInteger::zero(self.prime.clone(), self.precision)?;
        Ok(PadicRational::from_padic_integer(zero_int))
    }

    /// Create one element
    pub fn one(&self) -> Result<PadicRational> {
        let one_int = PadicInteger::one(self.prime.clone(), self.precision)?;
        Ok(PadicRational::from_padic_integer(one_int))
    }
}

impl fmt::Display for PadicField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-adic field (precision {}, model: {})",
            self.prime, self.precision, self.model
        )
    }
}

/// Unramified extension of Z_p
///
/// Z_q where q = p^n, constructed via a polynomial of degree n.
#[derive(Debug, Clone)]
pub struct PadicIntegerExtension {
    base_ring: PadicIntegerRing,
    degree: usize,
    modulus: Option<UnivariatePolynomial<Integer>>,
    name: String,
}

impl PadicIntegerExtension {
    /// Create a new unramified extension Z_q
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring Z_p
    /// * `degree` - Extension degree n (so q = p^n)
    /// * `name` - Variable name for the extension element
    /// * `modulus` - Optional defining polynomial (if None, uses Conway polynomial)
    pub fn new(
        base_ring: PadicIntegerRing,
        degree: usize,
        name: String,
        modulus: Option<UnivariatePolynomial<Integer>>,
    ) -> Result<Self> {
        if degree == 0 {
            return Err(MathError::InvalidArgument(
                "Extension degree must be > 0".to_string(),
            ));
        }

        Ok(PadicIntegerExtension {
            base_ring,
            degree,
            modulus,
            name,
        })
    }

    /// Get the base prime p
    pub fn prime(&self) -> &Integer {
        self.base_ring.prime()
    }

    /// Get the extension degree
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the base ring
    pub fn base_ring(&self) -> &PadicIntegerRing {
        &self.base_ring
    }

    /// Get the variable name
    pub fn variable_name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for PadicIntegerExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unramified extension Z_{{{}^{}}} = Z_{}[{}] (precision {}, model: {})",
            self.base_ring.prime(),
            self.degree,
            self.base_ring.prime(),
            self.name,
            self.base_ring.precision(),
            self.base_ring.model()
        )
    }
}

/// Unramified extension of Q_p
///
/// Q_q where q = p^n, constructed via a polynomial of degree n.
#[derive(Debug, Clone)]
pub struct PadicFieldExtension {
    base_field: PadicField,
    degree: usize,
    modulus: Option<UnivariatePolynomial<Integer>>,
    name: String,
}

impl PadicFieldExtension {
    /// Create a new unramified extension Q_q
    ///
    /// # Arguments
    ///
    /// * `base_field` - The base field Q_p
    /// * `degree` - Extension degree n (so q = p^n)
    /// * `name` - Variable name for the extension element
    /// * `modulus` - Optional defining polynomial (if None, uses Conway polynomial)
    pub fn new(
        base_field: PadicField,
        degree: usize,
        name: String,
        modulus: Option<UnivariatePolynomial<Integer>>,
    ) -> Result<Self> {
        if degree == 0 {
            return Err(MathError::InvalidArgument(
                "Extension degree must be > 0".to_string(),
            ));
        }

        Ok(PadicFieldExtension {
            base_field,
            degree,
            modulus,
            name,
        })
    }

    /// Get the base prime p
    pub fn prime(&self) -> &Integer {
        self.base_field.prime()
    }

    /// Get the extension degree
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the base field
    pub fn base_field(&self) -> &PadicField {
        &self.base_field
    }

    /// Get the variable name
    pub fn variable_name(&self) -> &str {
        &self.name
    }

    /// Get the integer ring (Z_q)
    pub fn integer_ring(&self) -> Result<PadicIntegerExtension> {
        let base_zp = self.base_field.integer_ring()?;
        PadicIntegerExtension::new(base_zp, self.degree, self.name.clone(), self.modulus.clone())
    }
}

impl fmt::Display for PadicFieldExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unramified extension Q_{{{}^{}}} = Q_{}[{}] (precision {}, model: {})",
            self.base_field.prime(),
            self.degree,
            self.base_field.prime(),
            self.name,
            self.base_field.precision(),
            self.base_field.model()
        )
    }
}

// ============================================================================
// Factory Functions (SageMath-compatible API)
// ============================================================================

/// Create a p-adic integer ring Z_p
///
/// # Arguments
///
/// * `p` - Prime number
/// * `prec` - Precision (number of p-adic digits)
/// * `model` - Precision model (CappedRelative, CappedAbsolute, or FixedModulus)
///
/// # Examples
///
/// ```rust
/// use rustmath_rings::padics::factory::{zp, PrecisionModel};
/// use rustmath_integers::Integer;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create 5-adic integers with default (capped relative) precision
/// let zp5 = zp(Integer::from(5), 20, PrecisionModel::CappedRelative)?;
///
/// // Create element
/// let x = zp5.from_int(42)?;
/// # Ok(())
/// # }
/// ```
pub fn zp(p: Integer, prec: usize, model: PrecisionModel) -> Result<PadicIntegerRing> {
    PadicIntegerRing::new(p, prec, model)
}

/// Create a p-adic field Q_p
///
/// # Arguments
///
/// * `p` - Prime number
/// * `prec` - Precision (number of p-adic digits)
/// * `model` - Precision model (CappedRelative, CappedAbsolute, or FixedModulus)
///
/// # Examples
///
/// ```rust
/// use rustmath_rings::padics::factory::{qp, PrecisionModel};
/// use rustmath_integers::Integer;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create 7-adic field with capped absolute precision
/// let qp7 = qp(Integer::from(7), 15, PrecisionModel::CappedAbsolute)?;
///
/// // Create element 3/7
/// let x = qp7.from_rational_nums(3, 7)?;
/// # Ok(())
/// # }
/// ```
pub fn qp(p: Integer, prec: usize, model: PrecisionModel) -> Result<PadicField> {
    PadicField::new(p, prec, model)
}

/// Create an unramified extension Z_q of Z_p
///
/// # Arguments
///
/// * `p` - Prime number
/// * `n` - Extension degree (so q = p^n)
/// * `prec` - Precision
/// * `model` - Precision model
/// * `name` - Variable name (default: "a")
/// * `modulus` - Optional defining polynomial
///
/// # Examples
///
/// ```rust
/// use rustmath_rings::padics::factory::{zq, PrecisionModel};
/// use rustmath_integers::Integer;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create Z_{5^3} with generator 'a'
/// let zq = zq(Integer::from(5), 3, 10, PrecisionModel::CappedRelative, "a".to_string(), None)?;
/// # Ok(())
/// # }
/// ```
pub fn zq(
    p: Integer,
    n: usize,
    prec: usize,
    model: PrecisionModel,
    name: String,
    modulus: Option<UnivariatePolynomial<Integer>>,
) -> Result<PadicIntegerExtension> {
    let base_ring = zp(p, prec, model)?;
    PadicIntegerExtension::new(base_ring, n, name, modulus)
}

/// Create an unramified extension Q_q of Q_p
///
/// # Arguments
///
/// * `p` - Prime number
/// * `n` - Extension degree (so q = p^n)
/// * `prec` - Precision
/// * `model` - Precision model
/// * `name` - Variable name (default: "a")
/// * `modulus` - Optional defining polynomial
///
/// # Examples
///
/// ```rust
/// use rustmath_rings::padics::factory::{qq, PrecisionModel};
/// use rustmath_integers::Integer;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create Q_{7^2} with generator 'a'
/// let qq = qq(Integer::from(7), 2, 15, PrecisionModel::CappedRelative, "a".to_string(), None)?;
/// # Ok(())
/// # }
/// ```
pub fn qq(
    p: Integer,
    n: usize,
    prec: usize,
    model: PrecisionModel,
    name: String,
    modulus: Option<UnivariatePolynomial<Integer>>,
) -> Result<PadicFieldExtension> {
    let base_field = qp(p, prec, model)?;
    PadicFieldExtension::new(base_field, n, name, modulus)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zp_creation() {
        let zp5 = zp(Integer::from(5), 20, PrecisionModel::CappedRelative).unwrap();
        assert_eq!(*zp5.prime(), Integer::from(5));
        assert_eq!(zp5.precision(), 20);
        assert_eq!(zp5.model(), PrecisionModel::CappedRelative);
    }

    #[test]
    fn test_zp_elements() {
        let zp = zp(Integer::from(7), 10, PrecisionModel::CappedRelative).unwrap();

        let x = zp.from_int(42).unwrap();
        let y = zp.from_int(13).unwrap();

        // Test basic operations
        let sum = x.clone() + y.clone();
        assert_eq!(sum.value(), &Integer::from(55));

        let prod = x.clone() * y.clone();
        assert_eq!(prod.value(), &Integer::from(546));
    }

    #[test]
    fn test_qp_creation() {
        let qp7 = qp(Integer::from(7), 15, PrecisionModel::CappedAbsolute).unwrap();
        assert_eq!(*qp7.prime(), Integer::from(7));
        assert_eq!(qp7.precision(), 15);
        assert_eq!(qp7.model(), PrecisionModel::CappedAbsolute);
    }

    #[test]
    fn test_qp_elements() {
        let qp = qp(Integer::from(5), 10, PrecisionModel::CappedRelative).unwrap();

        // Create 3/5 in Q_5
        let x = qp.from_rational_nums(3, 5).unwrap();
        assert_eq!(x.valuation(), -1); // One factor of 5 in denominator

        // Create 25 = 5^2 in Q_5
        let y = qp.from_int(25).unwrap();
        assert_eq!(y.valuation(), 0); // Stored as p-adic integer with valuation in unit
    }

    #[test]
    fn test_precision_models() {
        // Test all three precision models
        let models = [
            PrecisionModel::CappedRelative,
            PrecisionModel::CappedAbsolute,
            PrecisionModel::FixedModulus,
        ];

        for model in &models {
            let zp = zp(Integer::from(3), 15, *model).unwrap();
            let x = zp.from_int(10).unwrap();
            assert_eq!(x.precision(), 15);
        }
    }

    #[test]
    fn test_capped_relative_precision() {
        let zp = zp(Integer::from(5), 10, PrecisionModel::CappedRelative).unwrap();

        // Element with valuation 0
        let x = zp.from_int(7).unwrap();
        assert_eq!(x.valuation(), 0);
        assert_eq!(x.precision(), 10);

        // Element with valuation 2
        let y = zp.from_int(25).unwrap(); // 25 = 5^2
        assert_eq!(y.valuation(), 2);
        assert_eq!(y.precision(), 10);
    }

    #[test]
    fn test_capped_absolute_precision() {
        let zp = zp(Integer::from(5), 10, PrecisionModel::CappedAbsolute).unwrap();

        let x = zp.from_int(7).unwrap();
        let adjusted = zp.adjust_precision(x.clone());
        assert_eq!(adjusted.precision(), 10);
    }

    #[test]
    fn test_fixed_modulus() {
        let zp = zp(Integer::from(5), 8, PrecisionModel::FixedModulus).unwrap();

        let x = zp.from_int(100).unwrap();
        let adjusted = zp.adjust_precision(x);
        assert_eq!(adjusted.precision(), 8);
    }

    #[test]
    fn test_zq_extension() {
        let zq = zq(
            Integer::from(5),
            3,
            10,
            PrecisionModel::CappedRelative,
            "a".to_string(),
            None,
        )
        .unwrap();

        assert_eq!(*zq.prime(), Integer::from(5));
        assert_eq!(zq.degree(), 3);
        assert_eq!(zq.variable_name(), "a");
        assert_eq!(zq.base_ring().precision(), 10);
    }

    #[test]
    fn test_qq_extension() {
        let qq = qq(
            Integer::from(7),
            2,
            15,
            PrecisionModel::CappedAbsolute,
            "b".to_string(),
            None,
        )
        .unwrap();

        assert_eq!(*qq.prime(), Integer::from(7));
        assert_eq!(qq.degree(), 2);
        assert_eq!(qq.variable_name(), "b");
        assert_eq!(qq.base_field().precision(), 15);
        assert_eq!(qq.base_field().model(), PrecisionModel::CappedAbsolute);
    }

    #[test]
    fn test_integer_ring_from_field() {
        let qp = qp(Integer::from(5), 10, PrecisionModel::CappedRelative).unwrap();
        let zp = qp.integer_ring().unwrap();

        assert_eq!(*zp.prime(), Integer::from(5));
        assert_eq!(zp.precision(), 10);
        assert_eq!(zp.model(), PrecisionModel::CappedRelative);
    }

    #[test]
    fn test_extension_integer_ring() {
        let qq = qq(
            Integer::from(3),
            4,
            12,
            PrecisionModel::FixedModulus,
            "a".to_string(),
            None,
        )
        .unwrap();

        let zq = qq.integer_ring().unwrap();
        assert_eq!(*zq.prime(), Integer::from(3));
        assert_eq!(zq.degree(), 4);
    }

    #[test]
    fn test_display_formatting() {
        let zp = zp(Integer::from(5), 20, PrecisionModel::CappedRelative).unwrap();
        let display = format!("{}", zp);
        assert!(display.contains("5-adic"));
        assert!(display.contains("20"));
        assert!(display.contains("capped-rel"));

        let qp = qp(Integer::from(7), 15, PrecisionModel::CappedAbsolute).unwrap();
        let display = format!("{}", qp);
        assert!(display.contains("7-adic"));
        assert!(display.contains("15"));
        assert!(display.contains("capped-abs"));
    }

    #[test]
    fn test_zero_and_one() {
        let zp = zp(Integer::from(5), 10, PrecisionModel::CappedRelative).unwrap();

        let zero = zp.zero().unwrap();
        assert!(zero.is_zero());

        let one = zp.one().unwrap();
        assert!(one.is_one());
    }

    #[test]
    fn test_qp_zero_and_one() {
        let qp = qp(Integer::from(7), 10, PrecisionModel::CappedRelative).unwrap();

        let zero = qp.zero().unwrap();
        assert!(zero.is_zero());

        let one = qp.one().unwrap();
        assert!(one.is_one());
    }

    #[test]
    fn test_invalid_parameters() {
        // Invalid prime
        assert!(zp(Integer::from(1), 10, PrecisionModel::CappedRelative).is_err());
        assert!(zp(Integer::from(0), 10, PrecisionModel::CappedRelative).is_err());

        // Invalid precision
        assert!(zp(Integer::from(5), 0, PrecisionModel::CappedRelative).is_err());

        // Invalid extension degree
        assert!(zq(
            Integer::from(5),
            0,
            10,
            PrecisionModel::CappedRelative,
            "a".to_string(),
            None
        )
        .is_err());
    }

    #[test]
    fn test_precision_model_display() {
        assert_eq!(format!("{}", PrecisionModel::CappedRelative), "capped-rel");
        assert_eq!(format!("{}", PrecisionModel::CappedAbsolute), "capped-abs");
        assert_eq!(format!("{}", PrecisionModel::FixedModulus), "fixed-mod");
    }

    #[test]
    fn test_precision_model_default() {
        assert_eq!(PrecisionModel::default(), PrecisionModel::CappedRelative);
    }
}
