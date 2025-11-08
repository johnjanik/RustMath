//! Symbolic expressions

use crate::symbol::Symbol;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

/// Binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    Abs,
    Sign,
    Sinh,
    Cosh,
    Tanh,
    Arcsin,
    Arccos,
    Arctan,
}

/// Symbolic expression
#[derive(Clone, PartialEq)]
pub enum Expr {
    /// Integer constant
    Integer(Integer),
    /// Rational constant
    Rational(Rational),
    /// Symbolic variable
    Symbol(Symbol),
    /// Binary operation
    Binary(BinaryOp, Arc<Expr>, Arc<Expr>),
    /// Unary operation
    Unary(UnaryOp, Arc<Expr>),
}

impl Expr {
    /// Create a symbol expression
    pub fn symbol(name: impl Into<String>) -> Self {
        Expr::Symbol(Symbol::new(name))
    }

    /// Check if the expression is constant (contains no symbols)
    pub fn is_constant(&self) -> bool {
        match self {
            Expr::Integer(_) | Expr::Rational(_) => true,
            Expr::Symbol(_) => false,
            Expr::Binary(_, left, right) => left.is_constant() && right.is_constant(),
            Expr::Unary(_, inner) => inner.is_constant(),
        }
    }

    /// Create a power expression
    pub fn pow(self, exp: Expr) -> Self {
        Expr::Binary(BinaryOp::Pow, Arc::new(self), Arc::new(exp))
    }

    /// Create sin expression
    pub fn sin(self) -> Self {
        Expr::Unary(UnaryOp::Sin, Arc::new(self))
    }

    /// Create cos expression
    pub fn cos(self) -> Self {
        Expr::Unary(UnaryOp::Cos, Arc::new(self))
    }

    /// Create exp expression
    pub fn exp(self) -> Self {
        Expr::Unary(UnaryOp::Exp, Arc::new(self))
    }

    /// Create log expression
    pub fn log(self) -> Self {
        Expr::Unary(UnaryOp::Log, Arc::new(self))
    }

    /// Create sqrt expression
    pub fn sqrt(self) -> Self {
        Expr::Unary(UnaryOp::Sqrt, Arc::new(self))
    }

    /// Create tan expression
    pub fn tan(self) -> Self {
        Expr::Unary(UnaryOp::Tan, Arc::new(self))
    }

    /// Create abs (absolute value) expression
    pub fn abs(self) -> Self {
        Expr::Unary(UnaryOp::Abs, Arc::new(self))
    }

    /// Create sign expression
    /// Returns -1 for negative, 0 for zero, 1 for positive
    pub fn sign(self) -> Self {
        Expr::Unary(UnaryOp::Sign, Arc::new(self))
    }

    /// Create sinh (hyperbolic sine) expression
    pub fn sinh(self) -> Self {
        Expr::Unary(UnaryOp::Sinh, Arc::new(self))
    }

    /// Create cosh (hyperbolic cosine) expression
    pub fn cosh(self) -> Self {
        Expr::Unary(UnaryOp::Cosh, Arc::new(self))
    }

    /// Create tanh (hyperbolic tangent) expression
    pub fn tanh(self) -> Self {
        Expr::Unary(UnaryOp::Tanh, Arc::new(self))
    }

    /// Create arcsin (inverse sine) expression
    pub fn arcsin(self) -> Self {
        Expr::Unary(UnaryOp::Arcsin, Arc::new(self))
    }

    /// Create arccos (inverse cosine) expression
    pub fn arccos(self) -> Self {
        Expr::Unary(UnaryOp::Arccos, Arc::new(self))
    }

    /// Create arctan (inverse tangent) expression
    pub fn arctan(self) -> Self {
        Expr::Unary(UnaryOp::Arctan, Arc::new(self))
    }
}

impl From<i64> for Expr {
    fn from(n: i64) -> Self {
        Expr::Integer(Integer::from(n))
    }
}

impl From<Integer> for Expr {
    fn from(n: Integer) -> Self {
        Expr::Integer(n)
    }
}

impl From<Rational> for Expr {
    fn from(r: Rational) -> Self {
        Expr::Rational(r)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Integer(n) => write!(f, "{}", n),
            Expr::Rational(r) => write!(f, "{}", r),
            Expr::Symbol(s) => write!(f, "{}", s),
            Expr::Binary(op, left, right) => {
                let op_str = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Pow => "^",
                };
                write!(f, "({} {} {})", left, op_str, right)
            }
            Expr::Unary(op, inner) => {
                let op_str = match op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Sin => "sin",
                    UnaryOp::Cos => "cos",
                    UnaryOp::Tan => "tan",
                    UnaryOp::Exp => "exp",
                    UnaryOp::Log => "log",
                    UnaryOp::Sqrt => "sqrt",
                    UnaryOp::Abs => "abs",
                    UnaryOp::Sign => "sign",
                    UnaryOp::Sinh => "sinh",
                    UnaryOp::Cosh => "cosh",
                    UnaryOp::Tanh => "tanh",
                    UnaryOp::Arcsin => "arcsin",
                    UnaryOp::Arccos => "arccos",
                    UnaryOp::Arctan => "arctan",
                };
                match op {
                    UnaryOp::Neg => write!(f, "-{}", inner),
                    _ => write!(f, "{}({})", op_str, inner),
                }
            }
        }
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

// Arithmetic operations
impl Add for Expr {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Expr::Binary(BinaryOp::Add, Arc::new(self), Arc::new(other))
    }
}

impl Sub for Expr {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Expr::Binary(BinaryOp::Sub, Arc::new(self), Arc::new(other))
    }
}

impl Mul for Expr {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Expr::Binary(BinaryOp::Mul, Arc::new(self), Arc::new(other))
    }
}

impl Div for Expr {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Expr::Binary(BinaryOp::Div, Arc::new(self), Arc::new(other))
    }
}

impl Neg for Expr {
    type Output = Self;

    fn neg(self) -> Self {
        Expr::Unary(UnaryOp::Neg, Arc::new(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_building() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");

        // x + y
        let sum = x.clone() + y.clone();
        assert!(!sum.is_constant());

        // 2*x + 3
        let expr = Expr::from(2) * x.clone() + Expr::from(3);
        assert!(!expr.is_constant());

        // Constants
        let constant = Expr::from(5) + Expr::from(3);
        assert!(constant.is_constant());
    }

    #[test]
    fn test_display() {
        let x = Expr::symbol("x");
        let expr = x.clone() + Expr::from(1);
        let display = format!("{}", expr);
        assert!(display.contains("x"));
        assert!(display.contains("1"));
    }

    #[test]
    fn test_trig_functions() {
        let x = Expr::symbol("x");

        let sin_x = x.clone().sin();
        assert_eq!(format!("{}", sin_x), "sin(x)");

        let cos_x = x.clone().cos();
        assert_eq!(format!("{}", cos_x), "cos(x)");

        let tan_x = x.clone().tan();
        assert_eq!(format!("{}", tan_x), "tan(x)");
    }

    #[test]
    fn test_hyperbolic_functions() {
        let x = Expr::symbol("x");

        let sinh_x = x.clone().sinh();
        assert_eq!(format!("{}", sinh_x), "sinh(x)");

        let cosh_x = x.clone().cosh();
        assert_eq!(format!("{}", cosh_x), "cosh(x)");

        let tanh_x = x.clone().tanh();
        assert_eq!(format!("{}", tanh_x), "tanh(x)");
    }

    #[test]
    fn test_inverse_trig_functions() {
        let x = Expr::symbol("x");

        let arcsin_x = x.clone().arcsin();
        assert_eq!(format!("{}", arcsin_x), "arcsin(x)");

        let arccos_x = x.clone().arccos();
        assert_eq!(format!("{}", arccos_x), "arccos(x)");

        let arctan_x = x.clone().arctan();
        assert_eq!(format!("{}", arctan_x), "arctan(x)");
    }

    #[test]
    fn test_abs_and_sign() {
        let x = Expr::symbol("x");

        let abs_x = x.clone().abs();
        assert_eq!(format!("{}", abs_x), "abs(x)");

        let sign_x = x.clone().sign();
        assert_eq!(format!("{}", sign_x), "sign(x)");
    }
}
