//! Algebraic number descriptors
//!
//! This module defines the internal representation of algebraic numbers.
//! An algebraic number can be described in several ways:
//! - As a rational number
//! - As a root of a polynomial with an isolating interval
//! - As a unary expression (negation, conjugation, etc.)
//! - As a binary expression (sum, product, etc.)

use rustmath_rationals::Rational;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_integers::Integer;
use rustmath_complex::Complex;
use rustmath_core::{Field, Ring};
use std::fmt;

/// Descriptor for an algebraic number
///
/// This enum describes how an algebraic number is represented internally.
/// Different representations have different trade-offs for efficiency.
#[derive(Debug, Clone)]
pub enum AlgebraicDescriptor {
    /// A rational number
    Rational(ANRational),
    /// A root of a polynomial
    Root(ANRoot),
    /// A unary operation
    UnaryExpr(ANUnaryExpr),
    /// A binary operation
    BinaryExpr(ANBinaryExpr),
}

/// A rational number descriptor
#[derive(Debug, Clone, PartialEq)]
pub struct ANRational {
    /// The rational value
    pub value: Rational,
}

impl ANRational {
    pub fn new(value: Rational) -> Self {
        Self { value }
    }
}

/// A polynomial root descriptor
///
/// Represents an algebraic number as a root of a polynomial,
/// identified by an isolating interval or complex region.
#[derive(Debug, Clone)]
pub struct ANRoot {
    /// The minimal polynomial (or a polynomial having this number as a root)
    pub polynomial: UnivariatePolynomial<Integer>,

    /// For real roots: an isolating interval (a, b) where a < root < b
    /// and the polynomial has exactly one root in this interval
    pub isolating_interval: Option<(Rational, Rational)>,

    /// For complex roots: a complex approximation
    pub complex_approximation: Option<Complex>,

    /// The multiplicity of this root
    pub multiplicity: usize,
}

impl ANRoot {
    /// Create a new root descriptor
    pub fn new(
        polynomial: UnivariatePolynomial<Integer>,
        isolating_interval: Option<(Rational, Rational)>,
        complex_approximation: Option<Complex>,
    ) -> Self {
        Self {
            polynomial,
            isolating_interval,
            complex_approximation,
            multiplicity: 1,
        }
    }

    /// Check if this represents a real algebraic number
    pub fn is_real(&self) -> bool {
        self.isolating_interval.is_some()
    }

    /// Refine the isolating interval to higher precision
    pub fn refine_interval(&mut self, target_precision: usize) {
        // TODO: Implement interval refinement using bisection
        // For now, this is a placeholder
        let _ = target_precision;
    }
}

/// Unary operation on an algebraic number
#[derive(Debug, Clone)]
pub struct ANUnaryExpr {
    /// The operation type
    pub op: UnaryOp,

    /// The operand (boxed to avoid infinite size)
    pub operand: Box<AlgebraicDescriptor>,
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Negation: -x
    Neg,
    /// Multiplicative inverse: 1/x
    Inv,
    /// Complex conjugate
    Conj,
    /// Absolute value (for real numbers)
    Abs,
    /// Square root
    Sqrt,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Inv => write!(f, "inv"),
            UnaryOp::Conj => write!(f, "conj"),
            UnaryOp::Abs => write!(f, "abs"),
            UnaryOp::Sqrt => write!(f, "sqrt"),
        }
    }
}

impl ANUnaryExpr {
    pub fn new(op: UnaryOp, operand: AlgebraicDescriptor) -> Self {
        Self {
            op,
            operand: Box::new(operand),
        }
    }
}

/// Binary operation on algebraic numbers
#[derive(Debug, Clone)]
pub struct ANBinaryExpr {
    /// The operation type
    pub op: BinaryOp,

    /// The left operand
    pub left: Box<AlgebraicDescriptor>,

    /// The right operand
    pub right: Box<AlgebraicDescriptor>,
}

/// Binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition: x + y
    Add,
    /// Subtraction: x - y
    Sub,
    /// Multiplication: x * y
    Mul,
    /// Division: x / y
    Div,
    /// Exponentiation: x^n (where n is rational)
    Pow,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Pow => write!(f, "^"),
        }
    }
}

impl ANBinaryExpr {
    pub fn new(op: BinaryOp, left: AlgebraicDescriptor, right: AlgebraicDescriptor) -> Self {
        Self {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

impl AlgebraicDescriptor {
    /// Check if this descriptor represents a rational number
    pub fn is_rational(&self) -> bool {
        matches!(self, AlgebraicDescriptor::Rational(_))
    }

    /// Get the rational value if this is rational
    pub fn as_rational(&self) -> Option<&Rational> {
        match self {
            AlgebraicDescriptor::Rational(r) => Some(&r.value),
            _ => None,
        }
    }

    /// Simplify the descriptor by evaluating expressions when possible
    pub fn simplify(&self) -> AlgebraicDescriptor {
        match self {
            AlgebraicDescriptor::Rational(_) => self.clone(),
            AlgebraicDescriptor::Root(_) => self.clone(),
            AlgebraicDescriptor::UnaryExpr(expr) => {
                let operand = expr.operand.simplify();

                // If operand is rational, we can often evaluate directly
                if let Some(rat) = operand.as_rational() {
                    match expr.op {
                        UnaryOp::Neg => {
                            return AlgebraicDescriptor::Rational(ANRational::new(-rat.clone()));
                        }
                        UnaryOp::Inv => {
                            if let Ok(inv) = rat.inverse() {
                                return AlgebraicDescriptor::Rational(ANRational::new(inv));
                            }
                        }
                        _ => {}
                    }
                }

                AlgebraicDescriptor::UnaryExpr(ANUnaryExpr::new(expr.op, operand))
            }
            AlgebraicDescriptor::BinaryExpr(expr) => {
                let left = expr.left.simplify();
                let right = expr.right.simplify();

                // If both operands are rational, evaluate directly
                if let (Some(l), Some(r)) = (left.as_rational(), right.as_rational()) {
                    match expr.op {
                        BinaryOp::Add => {
                            return AlgebraicDescriptor::Rational(ANRational::new(
                                l.clone() + r.clone(),
                            ));
                        }
                        BinaryOp::Sub => {
                            return AlgebraicDescriptor::Rational(ANRational::new(
                                l.clone() - r.clone(),
                            ));
                        }
                        BinaryOp::Mul => {
                            return AlgebraicDescriptor::Rational(ANRational::new(
                                l.clone() * r.clone(),
                            ));
                        }
                        BinaryOp::Div => {
                            if !r.is_zero() {
                                return AlgebraicDescriptor::Rational(ANRational::new(
                                    l.clone() / r.clone(),
                                ));
                            }
                        }
                        _ => {}
                    }
                }

                AlgebraicDescriptor::BinaryExpr(ANBinaryExpr::new(expr.op, left, right))
            }
        }
    }
}

impl fmt::Display for AlgebraicDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AlgebraicDescriptor::Rational(r) => write!(f, "{}", r.value),
            AlgebraicDescriptor::Root(root) => {
                if let Some((a, b)) = &root.isolating_interval {
                    write!(f, "root of polynomial in ({}, {})", a, b)
                } else if let Some(z) = &root.complex_approximation {
                    write!(f, "root of polynomial near {}", z)
                } else {
                    write!(f, "root of polynomial")
                }
            }
            AlgebraicDescriptor::UnaryExpr(expr) => {
                write!(f, "{}({})", expr.op, expr.operand)
            }
            AlgebraicDescriptor::BinaryExpr(expr) => {
                write!(f, "({} {} {})", expr.left, expr.op, expr.right)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_descriptor() {
        let rat = Rational::new(3, 4).unwrap();
        let desc = AlgebraicDescriptor::Rational(ANRational::new(rat.clone()));

        assert!(desc.is_rational());
        assert_eq!(desc.as_rational().unwrap(), &rat);
    }

    #[test]
    fn test_simplify_rational_operations() {
        let two = AlgebraicDescriptor::Rational(ANRational::new(Rational::new(2, 1).unwrap()));
        let three = AlgebraicDescriptor::Rational(ANRational::new(Rational::new(3, 1).unwrap()));

        let sum = AlgebraicDescriptor::BinaryExpr(ANBinaryExpr::new(
            BinaryOp::Add,
            two.clone(),
            three.clone(),
        ));

        let simplified = sum.simplify();
        assert!(simplified.is_rational());
        assert_eq!(
            simplified.as_rational().unwrap(),
            &Rational::new(5, 1).unwrap()
        );
    }

    #[test]
    fn test_unary_negation() {
        let five = AlgebraicDescriptor::Rational(ANRational::new(Rational::new(5, 1).unwrap()));
        let neg = AlgebraicDescriptor::UnaryExpr(ANUnaryExpr::new(UnaryOp::Neg, five));

        let simplified = neg.simplify();
        assert!(simplified.is_rational());
        assert_eq!(
            simplified.as_rational().unwrap(),
            &Rational::new(-5, 1).unwrap()
        );
    }
}
