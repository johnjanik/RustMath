//! Symbolic expressions

use crate::symbol::Symbol;
use rustmath_core::Ring;
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
    Arcsinh,
    Arccosh,
    Arctanh,
    Gamma,
    Factorial,
    Erf,
    Zeta,
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
    /// Function call with name and arguments
    /// Examples: bessel_j(n, x), custom_func(a, b, c)
    Function(String, Vec<Arc<Expr>>),
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
            Expr::Function(_, args) => args.iter().all(|arg| arg.is_constant()),
        }
    }

    /// Check if the expression contains a specific symbol
    pub fn contains_symbol(&self, var: &Symbol) -> bool {
        match self {
            Expr::Integer(_) | Expr::Rational(_) => false,
            Expr::Symbol(s) => s == var,
            Expr::Binary(_, left, right) => {
                left.contains_symbol(var) || right.contains_symbol(var)
            }
            Expr::Unary(_, inner) => inner.contains_symbol(var),
            Expr::Function(_, args) => args.iter().any(|arg| arg.contains_symbol(var)),
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

    /// Create arcsinh (inverse hyperbolic sine) expression
    pub fn arcsinh(self) -> Self {
        Expr::Unary(UnaryOp::Arcsinh, Arc::new(self))
    }

    /// Create arccosh (inverse hyperbolic cosine) expression
    pub fn arccosh(self) -> Self {
        Expr::Unary(UnaryOp::Arccosh, Arc::new(self))
    }

    /// Create arctanh (inverse hyperbolic tangent) expression
    pub fn arctanh(self) -> Self {
        Expr::Unary(UnaryOp::Arctanh, Arc::new(self))
    }

    /// Create gamma function expression
    ///
    /// The gamma function Γ(x) extends the factorial function to complex numbers.
    /// For positive integers n: Γ(n) = (n-1)!
    pub fn gamma(self) -> Self {
        Expr::Unary(UnaryOp::Gamma, Arc::new(self))
    }

    /// Create factorial expression
    ///
    /// For non-negative integers n: n! = 1·2·3·...·n
    pub fn factorial(self) -> Self {
        Expr::Unary(UnaryOp::Factorial, Arc::new(self))
    }

    /// Create error function expression
    ///
    /// erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
    pub fn erf(self) -> Self {
        Expr::Unary(UnaryOp::Erf, Arc::new(self))
    }

    /// Create Riemann zeta function expression
    ///
    /// ζ(s) = Σ(n=1 to ∞) 1/n^s for Re(s) > 1
    pub fn zeta(self) -> Self {
        Expr::Unary(UnaryOp::Zeta, Arc::new(self))
    }

    /// Create a general function call
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    ///
    /// let x = Expr::symbol("x");
    /// let y = Expr::symbol("y");
    /// // Create custom_func(x, y)
    /// let f = Expr::function("custom_func", vec![x, y]);
    /// ```
    pub fn function(name: impl Into<String>, args: Vec<Expr>) -> Self {
        Expr::Function(
            name.into(),
            args.into_iter().map(Arc::new).collect(),
        )
    }

    /// Create Bessel J function: J_n(x)
    ///
    /// Bessel function of the first kind
    pub fn bessel_j(order: Expr, x: Expr) -> Self {
        Expr::Function("bessel_j".to_string(), vec![Arc::new(order), Arc::new(x)])
    }

    /// Create Bessel Y function: Y_n(x)
    ///
    /// Bessel function of the second kind
    pub fn bessel_y(order: Expr, x: Expr) -> Self {
        Expr::Function("bessel_y".to_string(), vec![Arc::new(order), Arc::new(x)])
    }

    /// Create modified Bessel I function: I_n(x)
    ///
    /// Modified Bessel function of the first kind
    pub fn bessel_i(order: Expr, x: Expr) -> Self {
        Expr::Function("bessel_i".to_string(), vec![Arc::new(order), Arc::new(x)])
    }

    /// Create modified Bessel K function: K_n(x)
    ///
    /// Modified Bessel function of the second kind
    pub fn bessel_k(order: Expr, x: Expr) -> Self {
        Expr::Function("bessel_k".to_string(), vec![Arc::new(order), Arc::new(x)])
    }

    /// Get the underlying symbol if this expression is a symbol
    ///
    /// Returns Some(symbol) if this is Expr::Symbol, None otherwise
    pub fn as_symbol(&self) -> Option<&Symbol> {
        match self {
            Expr::Symbol(s) => Some(s),
            _ => None,
        }
    }

    /// Check if the expression is known to be positive
    ///
    /// Returns:
    /// - Some(true) if definitely positive
    /// - Some(false) if definitely not positive (zero or negative)
    /// - None if unknown
    pub fn is_positive(&self) -> Option<bool> {
        use crate::assumptions::{has_property, Property};

        match self {
            Expr::Integer(n) => Some(n > &Integer::zero()),
            Expr::Rational(r) => Some(r > &Rational::zero()),
            Expr::Symbol(s) => {
                if has_property(s, Property::Positive) {
                    Some(true)
                } else if has_property(s, Property::Negative)
                    || has_property(s, Property::Zero)
                    || has_property(s, Property::NonPositive)
                {
                    Some(false)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if the expression is known to be negative
    ///
    /// Returns:
    /// - Some(true) if definitely negative
    /// - Some(false) if definitely not negative (zero or positive)
    /// - None if unknown
    pub fn is_negative(&self) -> Option<bool> {
        use crate::assumptions::{has_property, Property};

        match self {
            Expr::Integer(n) => Some(n < &Integer::zero()),
            Expr::Rational(r) => Some(r < &Rational::zero()),
            Expr::Symbol(s) => {
                if has_property(s, Property::Negative) {
                    Some(true)
                } else if has_property(s, Property::Positive)
                    || has_property(s, Property::Zero)
                    || has_property(s, Property::NonNegative)
                {
                    Some(false)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if the expression is known to be real
    ///
    /// Returns:
    /// - Some(true) if definitely real
    /// - Some(false) if definitely not real
    /// - None if unknown
    pub fn is_real(&self) -> Option<bool> {
        use crate::assumptions::{has_property, Property};

        match self {
            Expr::Integer(_) | Expr::Rational(_) => Some(true),
            Expr::Symbol(s) => {
                if has_property(s, Property::Real) {
                    Some(true)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if the expression is known to be an integer
    ///
    /// Returns:
    /// - Some(true) if definitely an integer
    /// - Some(false) if definitely not an integer
    /// - None if unknown
    pub fn is_integer(&self) -> Option<bool> {
        use crate::assumptions::{has_property, Property};

        match self {
            Expr::Integer(_) => Some(true),
            Expr::Rational(r) => Some(r.is_integer()),
            Expr::Symbol(s) => {
                if has_property(s, Property::Integer) {
                    Some(true)
                } else {
                    None
                }
            }
            _ => None,
        }
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
                    UnaryOp::Arcsinh => "arcsinh",
                    UnaryOp::Arccosh => "arccosh",
                    UnaryOp::Arctanh => "arctanh",
                    UnaryOp::Gamma => "gamma",
                    UnaryOp::Factorial => "factorial",
                    UnaryOp::Erf => "erf",
                    UnaryOp::Zeta => "zeta",
                };
                match op {
                    UnaryOp::Neg => write!(f, "-{}", inner),
                    _ => write!(f, "{}({})", op_str, inner),
                }
            }
            Expr::Function(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
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
