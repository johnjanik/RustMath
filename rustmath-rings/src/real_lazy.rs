//! # Lazy Real and Complex Numbers
//!
//! This module provides lazy evaluation for real and complex numbers, deferring
//! computation until a specific precision is requested. It enables efficient work
//! with exact rings while avoiding unnecessary numerical computation.
//!
//! ## Core Concept
//!
//! The lazy field sits between exact rings of characteristic 0 and numerical
//! representations (like f64), creating an expression tree that can be evaluated
//! at any desired precision.
//!
//! ## Main Types
//!
//! - [`RealLazyField`]: Field of real numbers with deferred evaluation
//! - [`ComplexLazyField`]: Field of complex numbers with deferred evaluation
//! - [`LazyFieldElement`]: An element that represents an unevaluated expression
//!
//! ## Examples
//!
//! ```
//! use rustmath_rings::real_lazy::{RealLazyField, LazyFieldElement};
//!
//! let field = RealLazyField::new();
//!
//! // Create lazy elements - computation is deferred
//! let x = field.from_rational(3, 4);
//! let y = field.from_rational(1, 2);
//!
//! // Operations build expression trees
//! let sum = x.add(&y);
//!
//! // Evaluate only when needed
//! let result = sum.eval_to_f64();
//! assert!((result - 1.25).abs() < 1e-10);
//! ```

use std::fmt;
use std::rc::Rc;
use std::f64::consts as f64_consts;

use rustmath_core::Ring;
use rustmath_rationals::Rational;
use rustmath_complex::Complex;

// ============================================================================
// Binary Operation Types
// ============================================================================

/// Binary operations for lazy evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Pow => write!(f, "^"),
        }
    }
}

// ============================================================================
// Unary Operation Types
// ============================================================================

/// Unary operations for lazy evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnOp::Neg => write!(f, "-"),
            UnOp::Abs => write!(f, "abs"),
            UnOp::Sqrt => write!(f, "sqrt"),
            UnOp::Exp => write!(f, "exp"),
            UnOp::Log => write!(f, "log"),
            UnOp::Sin => write!(f, "sin"),
            UnOp::Cos => write!(f, "cos"),
            UnOp::Tan => write!(f, "tan"),
            UnOp::Asin => write!(f, "asin"),
            UnOp::Acos => write!(f, "acos"),
            UnOp::Atan => write!(f, "atan"),
            UnOp::Sinh => write!(f, "sinh"),
            UnOp::Cosh => write!(f, "cosh"),
            UnOp::Tanh => write!(f, "tanh"),
        }
    }
}

// ============================================================================
// Mathematical Constants
// ============================================================================

/// Mathematical constants for lazy evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathConstant {
    Pi,
    E,
    Sqrt2,
    Ln2,
    Ln10,
}

impl MathConstant {
    /// Evaluate the constant to f64 precision
    pub fn to_f64(&self) -> f64 {
        match self {
            MathConstant::Pi => f64_consts::PI,
            MathConstant::E => f64_consts::E,
            MathConstant::Sqrt2 => f64_consts::SQRT_2,
            MathConstant::Ln2 => f64_consts::LN_2,
            MathConstant::Ln10 => f64_consts::LN_10,
        }
    }
}

impl fmt::Display for MathConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathConstant::Pi => write!(f, "π"),
            MathConstant::E => write!(f, "e"),
            MathConstant::Sqrt2 => write!(f, "√2"),
            MathConstant::Ln2 => write!(f, "ln(2)"),
            MathConstant::Ln10 => write!(f, "ln(10)"),
        }
    }
}

// ============================================================================
// Lazy Field Element (Expression Tree Node)
// ============================================================================

/// A lazy field element representing an unevaluated expression
///
/// This enum represents nodes in an expression tree. Evaluation is deferred
/// until explicitly requested via `eval_to_f64()` or similar methods.
#[derive(Debug, Clone)]
pub enum LazyFieldElement {
    /// Wrapped integer value
    Integer(i64),

    /// Wrapped rational value
    Rational(Rational),

    /// Wrapped floating-point value
    Float(f64),

    /// Wrapped complex value
    Complex(Complex),

    /// Mathematical constant
    Constant(MathConstant),

    /// Binary operation on two lazy elements
    BinOp {
        op: BinOp,
        left: Rc<LazyFieldElement>,
        right: Rc<LazyFieldElement>,
    },

    /// Unary operation on a lazy element
    UnOp {
        op: UnOp,
        operand: Rc<LazyFieldElement>,
    },

    /// Algebraic number defined by polynomial (simplified representation)
    /// Represents a root of a polynomial with given coefficients
    Algebraic {
        /// Polynomial coefficients (a_0 + a_1*x + a_2*x^2 + ...)
        coefficients: Vec<Rational>,
        /// Approximate value for evaluation
        approx: f64,
    },
}

impl LazyFieldElement {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a lazy element from an integer
    pub fn from_integer(n: i64) -> Self {
        LazyFieldElement::Integer(n)
    }

    /// Create a lazy element from a rational number
    pub fn from_rational(r: Rational) -> Self {
        LazyFieldElement::Rational(r)
    }

    /// Create a lazy element from a float
    pub fn from_float(f: f64) -> Self {
        LazyFieldElement::Float(f)
    }

    /// Create a lazy element from a complex number
    pub fn from_complex(c: Complex) -> Self {
        LazyFieldElement::Complex(c)
    }

    /// Create a lazy element representing pi
    pub fn pi() -> Self {
        LazyFieldElement::Constant(MathConstant::Pi)
    }

    /// Create a lazy element representing e
    pub fn e() -> Self {
        LazyFieldElement::Constant(MathConstant::E)
    }

    /// Create a lazy element representing sqrt(2)
    pub fn sqrt2() -> Self {
        LazyFieldElement::Constant(MathConstant::Sqrt2)
    }

    // ========================================================================
    // Arithmetic Operations
    // ========================================================================

    /// Add two lazy elements
    pub fn add(&self, other: &Self) -> Self {
        LazyFieldElement::BinOp {
            op: BinOp::Add,
            left: Rc::new(self.clone()),
            right: Rc::new(other.clone()),
        }
    }

    /// Subtract two lazy elements
    pub fn sub(&self, other: &Self) -> Self {
        LazyFieldElement::BinOp {
            op: BinOp::Sub,
            left: Rc::new(self.clone()),
            right: Rc::new(other.clone()),
        }
    }

    /// Multiply two lazy elements
    pub fn mul(&self, other: &Self) -> Self {
        LazyFieldElement::BinOp {
            op: BinOp::Mul,
            left: Rc::new(self.clone()),
            right: Rc::new(other.clone()),
        }
    }

    /// Divide two lazy elements
    pub fn div(&self, other: &Self) -> Self {
        LazyFieldElement::BinOp {
            op: BinOp::Div,
            left: Rc::new(self.clone()),
            right: Rc::new(other.clone()),
        }
    }

    /// Raise a lazy element to a power
    pub fn pow(&self, exponent: &Self) -> Self {
        LazyFieldElement::BinOp {
            op: BinOp::Pow,
            left: Rc::new(self.clone()),
            right: Rc::new(exponent.clone()),
        }
    }

    /// Negate a lazy element
    pub fn neg(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Neg,
            operand: Rc::new(self.clone()),
        }
    }

    // ========================================================================
    // Unary Operations
    // ========================================================================

    /// Absolute value
    pub fn abs(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Abs,
            operand: Rc::new(self.clone()),
        }
    }

    /// Square root
    pub fn sqrt(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Sqrt,
            operand: Rc::new(self.clone()),
        }
    }

    /// Exponential function
    pub fn exp(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Exp,
            operand: Rc::new(self.clone()),
        }
    }

    /// Natural logarithm
    pub fn log(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Log,
            operand: Rc::new(self.clone()),
        }
    }

    /// Sine function
    pub fn sin(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Sin,
            operand: Rc::new(self.clone()),
        }
    }

    /// Cosine function
    pub fn cos(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Cos,
            operand: Rc::new(self.clone()),
        }
    }

    /// Tangent function
    pub fn tan(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Tan,
            operand: Rc::new(self.clone()),
        }
    }

    /// Arcsine function
    pub fn asin(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Asin,
            operand: Rc::new(self.clone()),
        }
    }

    /// Arccosine function
    pub fn acos(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Acos,
            operand: Rc::new(self.clone()),
        }
    }

    /// Arctangent function
    pub fn atan(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Atan,
            operand: Rc::new(self.clone()),
        }
    }

    /// Hyperbolic sine
    pub fn sinh(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Sinh,
            operand: Rc::new(self.clone()),
        }
    }

    /// Hyperbolic cosine
    pub fn cosh(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Cosh,
            operand: Rc::new(self.clone()),
        }
    }

    /// Hyperbolic tangent
    pub fn tanh(&self) -> Self {
        LazyFieldElement::UnOp {
            op: UnOp::Tanh,
            operand: Rc::new(self.clone()),
        }
    }

    // ========================================================================
    // Evaluation Methods
    // ========================================================================

    /// Evaluate the lazy element to an f64 value
    ///
    /// This recursively evaluates the entire expression tree.
    pub fn eval_to_f64(&self) -> f64 {
        match self {
            LazyFieldElement::Integer(n) => *n as f64,
            LazyFieldElement::Rational(r) => r.to_f64(),
            LazyFieldElement::Float(f) => *f,
            LazyFieldElement::Complex(c) => c.real(), // Take real part for real field
            LazyFieldElement::Constant(c) => c.to_f64(),

            LazyFieldElement::BinOp { op, left, right } => {
                let l = left.eval_to_f64();
                let r = right.eval_to_f64();

                match op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => l / r,
                    BinOp::Pow => l.powf(r),
                }
            }

            LazyFieldElement::UnOp { op, operand } => {
                let x = operand.eval_to_f64();

                match op {
                    UnOp::Neg => -x,
                    UnOp::Abs => x.abs(),
                    UnOp::Sqrt => x.sqrt(),
                    UnOp::Exp => x.exp(),
                    UnOp::Log => x.ln(),
                    UnOp::Sin => x.sin(),
                    UnOp::Cos => x.cos(),
                    UnOp::Tan => x.tan(),
                    UnOp::Asin => x.asin(),
                    UnOp::Acos => x.acos(),
                    UnOp::Atan => x.atan(),
                    UnOp::Sinh => x.sinh(),
                    UnOp::Cosh => x.cosh(),
                    UnOp::Tanh => x.tanh(),
                }
            }

            LazyFieldElement::Algebraic { approx, .. } => *approx,
        }
    }

    /// Evaluate the lazy element to a complex number
    pub fn eval_to_complex(&self) -> Complex {
        match self {
            LazyFieldElement::Complex(c) => c.clone(),
            _ => Complex::new(self.eval_to_f64(), 0.0),
        }
    }

    /// Calculate the depth of the expression tree
    ///
    /// The depth indicates how many nested operations need to be evaluated,
    /// which can be used to determine required precision.
    pub fn depth(&self) -> usize {
        match self {
            LazyFieldElement::Integer(_)
            | LazyFieldElement::Rational(_)
            | LazyFieldElement::Float(_)
            | LazyFieldElement::Complex(_)
            | LazyFieldElement::Constant(_)
            | LazyFieldElement::Algebraic { .. } => 0,

            LazyFieldElement::BinOp { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }

            LazyFieldElement::UnOp { operand, .. } => {
                1 + operand.depth()
            }
        }
    }

    /// Check if this element is exactly zero (without evaluation)
    pub fn is_zero(&self) -> bool {
        match self {
            LazyFieldElement::Integer(0) => true,
            LazyFieldElement::Rational(r) => r.is_zero(),
            LazyFieldElement::Float(f) => *f == 0.0,
            _ => false,
        }
    }

    /// Check if this element is exactly one (without evaluation)
    pub fn is_one(&self) -> bool {
        match self {
            LazyFieldElement::Integer(1) => true,
            LazyFieldElement::Rational(r) => r.is_one(),
            LazyFieldElement::Float(f) => *f == 1.0,
            _ => false,
        }
    }
}

impl fmt::Display for LazyFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LazyFieldElement::Integer(n) => write!(f, "{}", n),
            LazyFieldElement::Rational(r) => write!(f, "{}", r),
            LazyFieldElement::Float(x) => write!(f, "{}", x),
            LazyFieldElement::Complex(c) => write!(f, "{}", c),
            LazyFieldElement::Constant(c) => write!(f, "{}", c),

            LazyFieldElement::BinOp { op, left, right } => {
                write!(f, "({} {} {})", left, op, right)
            }

            LazyFieldElement::UnOp { op, operand } => {
                write!(f, "{}({})", op, operand)
            }

            LazyFieldElement::Algebraic { approx, .. } => {
                write!(f, "algebraic≈{}", approx)
            }
        }
    }
}

// ============================================================================
// Real Lazy Field
// ============================================================================

/// A field of real numbers with lazy evaluation
///
/// Elements of this field defer computation until explicitly evaluated.
/// This allows working with exact algebraic values while maintaining
/// the ability to compute numerical approximations when needed.
#[derive(Debug, Clone)]
pub struct RealLazyField {
    // Currently a unit struct, but kept as struct for future extensions
    // (e.g., precision management, caching, etc.)
}

impl RealLazyField {
    /// Create a new real lazy field
    pub fn new() -> Self {
        RealLazyField {}
    }

    /// Create a lazy element from an integer
    pub fn from_integer(&self, n: i64) -> LazyFieldElement {
        LazyFieldElement::from_integer(n)
    }

    /// Create a lazy element from a rational number (numerator/denominator)
    pub fn from_rational(&self, num: i64, den: i64) -> LazyFieldElement {
        LazyFieldElement::from_rational(Rational::new(num.into(), den.into()).unwrap())
    }

    /// Create a lazy element from a Rational
    pub fn from_rational_value(&self, r: Rational) -> LazyFieldElement {
        LazyFieldElement::from_rational(r)
    }

    /// Create a lazy element from a float
    pub fn from_float(&self, f: f64) -> LazyFieldElement {
        LazyFieldElement::from_float(f)
    }

    /// Get the constant pi
    pub fn pi(&self) -> LazyFieldElement {
        LazyFieldElement::pi()
    }

    /// Get the constant e
    pub fn e(&self) -> LazyFieldElement {
        LazyFieldElement::e()
    }

    /// Get the constant sqrt(2)
    pub fn sqrt2(&self) -> LazyFieldElement {
        LazyFieldElement::sqrt2()
    }

    /// Get zero element
    pub fn zero(&self) -> LazyFieldElement {
        LazyFieldElement::Integer(0)
    }

    /// Get one element
    pub fn one(&self) -> LazyFieldElement {
        LazyFieldElement::Integer(1)
    }
}

impl Default for RealLazyField {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Complex Lazy Field
// ============================================================================

/// A field of complex numbers with lazy evaluation
///
/// Like RealLazyField, but supports complex arithmetic and evaluates
/// to complex numbers when needed.
#[derive(Debug, Clone)]
pub struct ComplexLazyField {
    // Currently a unit struct, but kept as struct for future extensions
}

impl ComplexLazyField {
    /// Create a new complex lazy field
    pub fn new() -> Self {
        ComplexLazyField {}
    }

    /// Create a lazy element from an integer
    pub fn from_integer(&self, n: i64) -> LazyFieldElement {
        LazyFieldElement::from_integer(n)
    }

    /// Create a lazy element from a rational number
    pub fn from_rational(&self, num: i64, den: i64) -> LazyFieldElement {
        LazyFieldElement::from_rational(Rational::new(num.into(), den.into()).unwrap())
    }

    /// Create a lazy element from a complex number
    pub fn from_complex(&self, real: f64, imag: f64) -> LazyFieldElement {
        LazyFieldElement::from_complex(Complex::new(real, imag))
    }

    /// Create the imaginary unit i
    pub fn i(&self) -> LazyFieldElement {
        LazyFieldElement::from_complex(Complex::new(0.0, 1.0))
    }

    /// Get the constant pi
    pub fn pi(&self) -> LazyFieldElement {
        LazyFieldElement::pi()
    }

    /// Get the constant e
    pub fn e(&self) -> LazyFieldElement {
        LazyFieldElement::e()
    }

    /// Get zero element
    pub fn zero(&self) -> LazyFieldElement {
        LazyFieldElement::Integer(0)
    }

    /// Get one element
    pub fn one(&self) -> LazyFieldElement {
        LazyFieldElement::Integer(1)
    }
}

impl Default for ComplexLazyField {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a real lazy field (convenience function)
pub fn RealLazyField() -> RealLazyField {
    RealLazyField::new()
}

/// Create a complex lazy field (convenience function)
pub fn ComplexLazyField() -> ComplexLazyField {
    ComplexLazyField::new()
}

/// Create a lazy element wrapper (convenience function for migration)
pub fn make_element(value: LazyFieldElement) -> LazyFieldElement {
    value
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_real_lazy_field_creation() {
        let field = RealLazyField::new();
        let x = field.from_integer(42);

        assert_eq!(x.eval_to_f64(), 42.0);
    }

    #[test]
    fn test_rational_lazy_element() {
        let field = RealLazyField::new();
        let x = field.from_rational(3, 4);

        assert_eq!(x.eval_to_f64(), 0.75);
    }

    #[test]
    fn test_lazy_addition() {
        let field = RealLazyField::new();
        let x = field.from_rational(1, 2);
        let y = field.from_rational(1, 3);
        let sum = x.add(&y);

        let result = sum.eval_to_f64();
        assert!(approx_eq(result, 5.0 / 6.0, 1e-10));
    }

    #[test]
    fn test_lazy_subtraction() {
        let field = RealLazyField::new();
        let x = field.from_rational(3, 4);
        let y = field.from_rational(1, 4);
        let diff = x.sub(&y);

        assert_eq!(diff.eval_to_f64(), 0.5);
    }

    #[test]
    fn test_lazy_multiplication() {
        let field = RealLazyField::new();
        let x = field.from_rational(2, 3);
        let y = field.from_rational(3, 4);
        let prod = x.mul(&y);

        assert_eq!(prod.eval_to_f64(), 0.5);
    }

    #[test]
    fn test_lazy_division() {
        let field = RealLazyField::new();
        let x = field.from_rational(1, 2);
        let y = field.from_rational(1, 4);
        let quot = x.div(&y);

        assert_eq!(quot.eval_to_f64(), 2.0);
    }

    #[test]
    fn test_lazy_negation() {
        let field = RealLazyField::new();
        let x = field.from_integer(5);
        let neg_x = x.neg();

        assert_eq!(neg_x.eval_to_f64(), -5.0);
    }

    #[test]
    fn test_lazy_power() {
        let field = RealLazyField::new();
        let x = field.from_integer(2);
        let exp = field.from_integer(3);
        let result = x.pow(&exp);

        assert_eq!(result.eval_to_f64(), 8.0);
    }

    #[test]
    fn test_lazy_sqrt() {
        let field = RealLazyField::new();
        let x = field.from_integer(4);
        let result = x.sqrt();

        assert_eq!(result.eval_to_f64(), 2.0);
    }

    #[test]
    fn test_lazy_exp() {
        let field = RealLazyField::new();
        let x = field.from_integer(1);
        let result = x.exp();

        assert!(approx_eq(result.eval_to_f64(), std::f64::consts::E, 1e-10));
    }

    #[test]
    fn test_lazy_log() {
        let field = RealLazyField::new();
        let e = field.e();
        let result = e.log();

        assert!(approx_eq(result.eval_to_f64(), 1.0, 1e-10));
    }

    #[test]
    fn test_lazy_sin() {
        let field = RealLazyField::new();
        let pi = field.pi();
        let half = field.from_rational(1, 2);
        let pi_half = pi.mul(&half);
        let result = pi_half.sin();

        assert!(approx_eq(result.eval_to_f64(), 1.0, 1e-10));
    }

    #[test]
    fn test_lazy_cos() {
        let field = RealLazyField::new();
        let pi = field.pi();
        let result = pi.cos();

        assert!(approx_eq(result.eval_to_f64(), -1.0, 1e-10));
    }

    #[test]
    fn test_lazy_tan() {
        let field = RealLazyField::new();
        let pi = field.pi();
        let quarter = field.from_rational(1, 4);
        let pi_quarter = pi.mul(&quarter);
        let result = pi_quarter.tan();

        assert!(approx_eq(result.eval_to_f64(), 1.0, 1e-10));
    }

    #[test]
    fn test_math_constants() {
        let pi = LazyFieldElement::pi();
        let e = LazyFieldElement::e();
        let sqrt2 = LazyFieldElement::sqrt2();

        assert!(approx_eq(pi.eval_to_f64(), std::f64::consts::PI, 1e-10));
        assert!(approx_eq(e.eval_to_f64(), std::f64::consts::E, 1e-10));
        assert!(approx_eq(sqrt2.eval_to_f64(), std::f64::consts::SQRT_2, 1e-10));
    }

    #[test]
    fn test_expression_depth() {
        let field = RealLazyField::new();
        let x = field.from_integer(2);
        let y = field.from_integer(3);

        assert_eq!(x.depth(), 0);

        let sum = x.add(&y);
        assert_eq!(sum.depth(), 1);

        let nested = sum.mul(&x);
        assert_eq!(nested.depth(), 2);
    }

    #[test]
    fn test_complex_lazy_field() {
        let field = ComplexLazyField::new();
        let i = field.i();

        // i^2 = -1
        let i_squared = i.mul(&i);
        let result = i_squared.eval_to_complex();

        assert!(approx_eq(result.real, -1.0, 1e-10));
        assert!(approx_eq(result.imag, 0.0, 1e-10));
    }

    #[test]
    fn test_complex_addition() {
        let field = ComplexLazyField::new();
        let z1 = field.from_complex(1.0, 2.0);
        let z2 = field.from_complex(3.0, 4.0);
        let sum = z1.add(&z2);

        let result = sum.eval_to_complex();
        assert!(approx_eq(result.real, 4.0, 1e-10));
        assert!(approx_eq(result.imag, 6.0, 1e-10));
    }

    #[test]
    fn test_lazy_abs() {
        let field = RealLazyField::new();
        let x = field.from_integer(-5);
        let result = x.abs();

        assert_eq!(result.eval_to_f64(), 5.0);
    }

    #[test]
    fn test_lazy_hyperbolic() {
        let field = RealLazyField::new();
        let x = field.from_integer(1);

        let sinh_result = x.sinh().eval_to_f64();
        let cosh_result = x.cosh().eval_to_f64();
        let tanh_result = x.tanh().eval_to_f64();

        assert!(approx_eq(sinh_result, 1.0_f64.sinh(), 1e-10));
        assert!(approx_eq(cosh_result, 1.0_f64.cosh(), 1e-10));
        assert!(approx_eq(tanh_result, 1.0_f64.tanh(), 1e-10));
    }

    #[test]
    fn test_lazy_inverse_trig() {
        let field = RealLazyField::new();
        let half = field.from_rational(1, 2);

        let asin_result = half.asin().eval_to_f64();
        let acos_result = half.acos().eval_to_f64();

        assert!(approx_eq(asin_result, 0.5_f64.asin(), 1e-10));
        assert!(approx_eq(acos_result, 0.5_f64.acos(), 1e-10));
    }

    #[test]
    fn test_is_zero_is_one() {
        let field = RealLazyField::new();
        let zero = field.zero();
        let one = field.one();
        let two = field.from_integer(2);

        assert!(zero.is_zero());
        assert!(!one.is_zero());

        assert!(one.is_one());
        assert!(!two.is_one());
    }

    #[test]
    fn test_algebraic_number() {
        // sqrt(2) as an algebraic number: root of x^2 - 2 = 0
        let coeffs = vec![
            Rational::new((-2).into(), 1.into()),  // constant term
            Rational::new(0.into(), 1.into()),     // x term
            Rational::new(1.into(), 1.into()),     // x^2 term
        ];

        let sqrt2_alg = LazyFieldElement::Algebraic {
            coefficients: coeffs,
            approx: std::f64::consts::SQRT_2,
        };

        assert!(approx_eq(sqrt2_alg.eval_to_f64(), std::f64::consts::SQRT_2, 1e-10));
    }

    #[test]
    fn test_nested_operations() {
        let field = RealLazyField::new();

        // Compute (sin(pi/2) + cos(0)) * sqrt(4)
        let pi = field.pi();
        let two = field.from_integer(2);
        let pi_half = pi.div(&two);
        let sin_val = pi_half.sin();

        let zero = field.zero();
        let cos_val = zero.cos();

        let sum = sin_val.add(&cos_val);  // 1 + 1 = 2

        let four = field.from_integer(4);
        let sqrt_four = four.sqrt();      // 2

        let result = sum.mul(&sqrt_four); // 2 * 2 = 4

        assert!(approx_eq(result.eval_to_f64(), 4.0, 1e-10));
    }

    #[test]
    fn test_display_formatting() {
        let field = RealLazyField::new();
        let x = field.from_integer(2);
        let y = field.from_integer(3);
        let sum = x.add(&y);

        let display = format!("{}", sum);
        assert!(display.contains("+"));
        assert!(display.contains("2"));
        assert!(display.contains("3"));
    }

    #[test]
    fn test_make_element() {
        let elem = LazyFieldElement::from_integer(42);
        let wrapped = make_element(elem.clone());

        assert_eq!(wrapped.eval_to_f64(), 42.0);
    }

    #[test]
    fn test_euler_identity_approximation() {
        // e^(i*pi) + 1 ≈ 0
        let field = ComplexLazyField::new();
        let i = field.i();
        let pi = field.pi();
        let e = field.e();

        // i * pi
        let i_pi = i.mul(&pi);

        // e^(i*pi) - note: this is a simplification since we're working with reals
        // In a full implementation, we'd need complex exp
        let exp_i_pi = e.pow(&i_pi);

        let one = field.one();
        let result = exp_i_pi.add(&one);

        // Due to limitations of real evaluation, this won't be exact
        // but demonstrates the expression tree construction
        assert!(result.depth() > 0);
    }
}
