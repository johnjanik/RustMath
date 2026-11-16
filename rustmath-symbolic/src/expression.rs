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
    Mod,
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
    /// Real constant (floating point)
    Real(f64),
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

thread_local! {
    /// The highest stack address seen on entry to `Drop for Expr` on this thread.
    static EXPR_DROP_STACK_TOP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The leaf that takes the place of a detached child.
    static EXPR_DROP_LEAF: Arc<Expr> = Arc::new(Expr::Real(0.0));
}

/// Whether e has child nodes (a binary or unary node, or a call with arguments).
fn expr_has_children(e: &Expr) -> bool {
    match e {
        Expr::Binary(..) | Expr::Unary(..) => true,
        Expr::Function(_, args) => !args.is_empty(),
        _ => false,
    }
}

/// Moves every child of e that has children to `list`, a shared leaf taking its place.
fn expr_detach(e: &mut Expr, list: &mut Vec<Arc<Expr>>) {
    // the leaf may already be destroyed while the thread's locals are torn down: then a fresh one
    let leaf = || {
        EXPR_DROP_LEAF
            .try_with(Arc::clone)
            .unwrap_or_else(|_| Arc::new(Expr::Real(0.0)))
    };
    let mut take = |a: &mut Arc<Expr>| {
        if expr_has_children(a) {
            list.push(std::mem::replace(a, leaf()));
        }
    };
    match e {
        Expr::Binary(_, l, r) => {
            take(l);
            take(r);
        }
        Expr::Unary(_, a) => take(a),
        Expr::Function(_, args) => args.iter_mut().for_each(take),
        _ => {}
    }
}

impl Drop for Expr {
    /// Teardown with bounded stack use (S1 design §2.14). While the drop runs less than
    /// `GLUE_STACK_BUDGET` bytes below the shallowest drop seen on this thread, the drop glue
    /// recurses as before. Deeper, the subtree is torn down iteratively: every child `Arc` that has
    /// children, shared or not, is moved to a heap list and a shared leaf takes its place, and the
    /// loop takes each with `Arc::into_inner`: the last owner gets the node and moves its children
    /// in turn, any other owner only decrements. So the glue below any node drops leaves only.
    ///
    /// The bound is relative to the shallowest `Expr` drop seen on the thread, not to the top of
    /// the thread's stack: a thread whose first `Expr` drop happens deep in its stack treats that
    /// point as the top, and the glue may use up to `GLUE_STACK_BUDGET` bytes below it before the
    /// iterative teardown starts. A thread with less than 16 KiB of stack left at its first drop
    /// can therefore still overflow there. The address of a local is the proxy for the depth (it
    /// is compared, never dereferenced); on a target whose stack grows upward the difference
    /// saturates to 0 and the drop is the derived glue.
    fn drop(&mut self) {
        const GLUE_STACK_BUDGET: usize = 16 * 1024;
        if !expr_has_children(self) {
            return;
        }
        let marker = 0u8;
        let here = std::ptr::from_ref(std::hint::black_box(&marker)) as usize;
        let used = EXPR_DROP_STACK_TOP
            .try_with(|top| {
                if here > top.get() {
                    top.set(here);
                }
                top.get().saturating_sub(here)
            })
            // the thread's locals are being destroyed: take the iterative teardown
            .unwrap_or(usize::MAX);
        if used < GLUE_STACK_BUDGET {
            return;
        }
        let mut list: Vec<Arc<Expr>> = Vec::new();
        expr_detach(self, &mut list);
        while let Some(a) = list.pop() {
            if let Some(mut e) = Arc::into_inner(a) {
                // the last owner: its children with children move to the list, so e then drops
                // with leaf children only
                expr_detach(&mut e, &mut list);
            }
        }
    }
}

impl Expr {
    /// Create a symbol expression
    pub fn symbol(name: impl Into<String>) -> Self {
        Expr::Symbol(Symbol::new(name))
    }

    /// Check if the expression is constant (contains no symbols)
    pub fn is_constant(&self) -> bool {
        match self {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => true,
            Expr::Symbol(_) => false,
            Expr::Binary(_, left, right) => left.is_constant() && right.is_constant(),
            Expr::Unary(_, inner) => inner.is_constant(),
            Expr::Function(_, args) => args.iter().all(|arg| arg.is_constant()),
        }
    }

    /// Check if the expression contains a specific symbol
    pub fn contains_symbol(&self, var: &Symbol) -> bool {
        match self {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => false,
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
            Expr::Real(x) => Some(*x > 0.0),
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
            Expr::Real(x) => Some(*x < 0.0),
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
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) => Some(true),
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
            Expr::Real(x) => Some(x.fract() == 0.0),
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

impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        // Handle special cases
        if value.is_nan() || value.is_infinite() {
            // Fall back to a symbol or panic - for now, use 0
            return Expr::Integer(Integer::from(0));
        }

        // For integers or simple decimals, convert directly
        if value.fract() == 0.0 && value.abs() < (i64::MAX as f64) {
            return Expr::Integer(Integer::from(value as i64));
        }

        // Convert f64 to rational using decimal approximation
        // Find power of 10 to make this an integer
        let mut scaled = value.abs();
        let mut denominator = 1i64;
        let precision = 10; // decimal places

        for _ in 0..precision {
            if scaled.fract() == 0.0 {
                break;
            }
            scaled *= 10.0;
            denominator *= 10;
        }

        let numerator = scaled.round() as i64;
        let numerator = if value < 0.0 { -numerator } else { numerator };

        // Create rational and simplify
        match Rational::new(numerator, denominator) {
            Ok(r) => Expr::Rational(r),
            Err(_) => Expr::Integer(Integer::from(0)), // Fallback
        }
    }
}

/// The text of a unary operator (`-` for the negation, else the function name).
fn unary_op_name(op: UnaryOp) -> &'static str {
    match op {
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
    }
}

fn binary_op_text(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => " + ",
        BinaryOp::Sub => " - ",
        BinaryOp::Mul => " * ",
        BinaryOp::Div => " / ",
        BinaryOp::Pow => " ^ ",
        BinaryOp::Mod => " % ",
    }
}

/// A negative literal (an `Integer`, `Rational` or `Real` below 0).
fn is_negative_literal(e: &Expr) -> bool {
    match e {
        Expr::Integer(n) => n.signum() < 0,
        Expr::Rational(q) => q.numerator().signum() < 0,
        Expr::Real(f) => *f < 0.0,
        _ => false,
    }
}

/// The number of characters `Display` writes for the float leaf `x`: Rust's `{}` of an f64, which
/// never uses an exponent, then `.0` for a finite integral value. At most 327 (`-5e-324` is a sign,
/// `0.` and 324 digits; `-f64::MAX` is 310 characters and `.0`). S1 fix-up round 3 (review 3, B1):
/// the result caps of the series engine and its bounded messages count a float by this length, not
/// by a constant (24 undercounted 5e-324 by a factor 14). Counted without allocating.
pub(crate) fn real_display_len(x: f64) -> u64 {
    struct Count(u64);
    impl fmt::Write for Count {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            self.0 += s.len() as u64;
            Ok(())
        }
    }
    let mut c = Count(0);
    let _ = fmt::Write::write_fmt(&mut c, format_args!("{}", x));
    if x.is_finite() && x.fract() == 0.0 {
        c.0 += 2;
    }
    c.0
}

/// A `Rational` with denominator above 1 (its text contains a `/`).
fn is_fraction_literal(e: &Expr) -> bool {
    matches!(e, Expr::Rational(q) if !q.denominator().is_one())
}

impl fmt::Display for Expr {
    /// Every binary node is written `(l op r)`. The parser reads `a/b` as a division (S1 design
    /// §2.12), so a fraction literal is put in parentheses when it is an operand of `^` or the
    /// divisor of `/` (`(x ^ (1/2))`, `(x / (1/2))`); the base of `^` when it is negated or a
    /// negative literal (`((-2) ^ 2)`, `((-y) ^ (1/2))`); and the operand of a negation when it is
    /// negated, a negative literal or a fraction (`-(-x)`, `-(-2)`, `-(2/3)`). A finite integral
    /// float keeps a decimal point (`2.0`), so that the parser reads it back as the same f64 and
    /// not as an integer of another value. (S1 design §2.11.)
    ///
    /// Writes with an explicit work list: the stack use does not depend on the depth of the
    /// expression (a result can be twice as deep as its input, and a sum built by a loop is as
    /// deep as it is long). The list holds at most the depth plus the pending arguments of calls.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Item<'a> {
            // the node, and whether it is written in parentheses (the rules above)
            Node(&'a Expr, bool),
            Str(&'static str),
        }
        let mut stack: Vec<Item<'_>> = vec![Item::Node(self, false)];
        while let Some(it) = stack.pop() {
            match it {
                Item::Str(t) => f.write_str(t)?,
                Item::Node(e, paren) => {
                    if paren {
                        f.write_str("(")?;
                        stack.push(Item::Str(")"));
                    }
                    match e {
                        Expr::Integer(n) => write!(f, "{}", n)?,
                        Expr::Rational(r) => write!(f, "{}", r)?,
                        Expr::Real(x) => {
                            write!(f, "{}", x)?;
                            if x.is_finite() && x.fract() == 0.0 {
                                f.write_str(".0")?;
                            }
                        }
                        Expr::Symbol(s) => write!(f, "{}", s)?,
                        Expr::Binary(op, l, r) => {
                            let pl = *op == BinaryOp::Pow
                                && (is_fraction_literal(l)
                                    || is_negative_literal(l)
                                    || matches!(**l, Expr::Unary(UnaryOp::Neg, _)));
                            let pr = matches!(op, BinaryOp::Pow | BinaryOp::Div)
                                && is_fraction_literal(r);
                            f.write_str("(")?;
                            stack.push(Item::Str(")"));
                            stack.push(Item::Node(r, pr));
                            stack.push(Item::Str(binary_op_text(*op)));
                            stack.push(Item::Node(l, pl));
                        }
                        Expr::Unary(UnaryOp::Neg, inner) => {
                            f.write_str("-")?;
                            let p = is_negative_literal(inner)
                                || is_fraction_literal(inner)
                                || matches!(**inner, Expr::Unary(UnaryOp::Neg, _));
                            stack.push(Item::Node(inner, p));
                        }
                        Expr::Unary(op, inner) => {
                            f.write_str(unary_op_name(*op))?;
                            f.write_str("(")?;
                            stack.push(Item::Str(")"));
                            stack.push(Item::Node(inner, false));
                        }
                        Expr::Function(name, args) => {
                            f.write_str(name)?;
                            f.write_str("(")?;
                            stack.push(Item::Str(")"));
                            for (k, a) in args.iter().enumerate().rev() {
                                stack.push(Item::Node(a, false));
                                if k > 0 {
                                    stack.push(Item::Str(", "));
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
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

    /// Runs f on a thread with the bar's debug stack (2 MB).
    fn on_2mb<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(f)
            .expect("spawn")
            .join()
            .expect("the thread panicked or overflowed its stack")
    }

    fn x() -> Expr {
        Expr::symbol("x")
    }

    /// S1 design §4.3 `test_display_work_list` (red team 1, H2): the writer has no recursion, so
    /// deep trees print on a 2 MB thread (HEAD overflowed from about 3000 terms in debug).
    #[test]
    fn test_display_work_list() {
        on_2mb(|| {
            let n = 100_000;
            let mut sum = x();
            for _ in 1..n {
                sum = sum + x();
            }
            let text = format!("{sum}");
            assert_eq!(text.len(), 6 * n - 5);
            assert!(text.starts_with(&"(".repeat(n - 1)));
            assert!(text[n - 1..].starts_with("x + x) + x) + x)"));
            let mut tower = x();
            for _ in 0..n {
                tower = Expr::Binary(BinaryOp::Pow, Arc::new(x()), Arc::new(tower));
            }
            assert!(format!("{tower}").ends_with(&format!("^ x{}", ")".repeat(n))));
            let mut chain = x();
            for _ in 0..n {
                chain = chain.sin();
            }
            assert!(format!("{chain}").starts_with("sin(sin("));
            let mut calls = x();
            for _ in 0..n {
                calls = Expr::Function("f".into(), vec![Arc::new(calls)]);
            }
            assert!(format!("{calls}").starts_with("f(f("));
        });
    }

    /// S1 design §2.11: fractions and negative bases in parentheses, the negated negation, and
    /// integral floats with a decimal point; every other text is HEAD's.
    #[test]
    fn test_display_parenthesis_rules() {
        let half = Expr::Rational(Rational::new(Integer::from(1), Integer::from(2)).unwrap());
        let m_half = Expr::Rational(Rational::new(Integer::from(-1), Integer::from(2)).unwrap());
        let tq = Expr::Rational(Rational::new(Integer::from(3), Integer::from(4)).unwrap());
        let two_thirds = Expr::Rational(Rational::new(Integer::from(2), Integer::from(3)).unwrap());
        let pow = |a: Expr, b: Expr| Expr::Binary(BinaryOp::Pow, Arc::new(a), Arc::new(b));
        assert_eq!(format!("{}", pow(x(), half.clone())), "(x ^ (1/2))");
        assert_eq!(format!("{}", pow(tq, m_half)), "((3/4) ^ (-1/2))");
        assert_eq!(format!("{}", x() / half), "(x / (1/2))");
        assert_eq!(format!("{}", pow(Expr::from(-2), Expr::from(2))), "((-2) ^ 2)");
        assert_eq!(format!("{}", pow(-Expr::symbol("y"), Expr::from(3))), "((-y) ^ 3)");
        assert_eq!(format!("{}", pow(Expr::Real(-1.5), Expr::from(2))), "((-1.5) ^ 2)");
        assert_eq!(format!("{}", -(-x())), "-(-x)");
        assert_eq!(format!("{}", -Expr::from(-2)), "-(-2)");
        assert_eq!(format!("{}", -two_thirds.clone()), "-(2/3)");
        assert_eq!(format!("{}", pow(x(), -two_thirds)), "(x ^ -(2/3))");
        assert_eq!(format!("{}", pow(x(), Expr::from(-2))), "(x ^ -2)");
        assert_eq!(format!("{}", Expr::Real(2.0)), "2.0");
        assert_eq!(format!("{}", Expr::Real(1e23)), "100000000000000000000000.0");
        assert_eq!(format!("{}", Expr::Real(0.5)), "0.5");
        assert_eq!(format!("{}", x() + Expr::from(1)), "(x + 1)");
    }

    /// S1 design §2.14 `drop_left_deep_sum_1e6` and `drop_other_shapes`: deep trees of every
    /// shape drop on a 2 MB debug thread (HEAD's glue overflowed at about 80 000 levels).
    #[test]
    fn drop_deep_shapes() {
        on_2mb(|| {
            let n = 1_000_000;
            let mut sum = x();
            for _ in 1..n {
                sum = sum + x();
            }
            drop(sum);
            let mut tower = x();
            let mut chain = x();
            let mut calls = x();
            let mut zig = x();
            for k in 0..200_000 {
                tower = Expr::Binary(BinaryOp::Pow, Arc::new(x()), Arc::new(tower));
                chain = chain.sin();
                calls = Expr::Function("f".into(), vec![Arc::new(calls), Arc::new(x())]);
                zig = if k % 2 == 0 { zig + x() } else { x() * zig };
            }
            drop((tower, chain, calls, zig));
        });
    }

    /// S1 design §2.14 `drop_shared_children`: a node that holds one child twice, and a child
    /// shared with a sibling's subtree, chained deeply (revision 7rt2 overflowed at 14 500 levels).
    #[test]
    fn drop_shared_children() {
        on_2mb(|| {
            let mut twin = Arc::new(x());
            let mut sib = Arc::new(x());
            for _ in 0..200_000 {
                twin = Arc::new(Expr::Binary(BinaryOp::Mul, twin.clone(), twin));
                let inner = Arc::new(Expr::Unary(UnaryOp::Sin, sib.clone()));
                sib = Arc::new(Expr::Binary(BinaryOp::Add, sib, inner));
            }
            // one extra owner of a deep subtree, dropped after the root
            let keep = match &*twin {
                Expr::Binary(_, l, _) => l.clone(),
                _ => unreachable!(),
            };
            drop(twin);
            drop(sib);
            drop(keep);
        });
    }

    /// S1 design §2.14 `drop_during_tls_teardown` (revision 8): a deep `Expr` held in a
    /// thread-local is dropped by the thread's TLS destructors, possibly after the shared leaf
    /// was destroyed; the drop must not panic.
    #[test]
    fn drop_during_tls_teardown() {
        thread_local! {
            static HELD: std::cell::RefCell<Option<Expr>> = const { std::cell::RefCell::new(None) };
        }
        on_2mb(|| {
            let mut sum = x();
            for _ in 1..300_000 {
                sum = sum + x();
            }
            HELD.with(|h| *h.borrow_mut() = Some(sum));
        });
    }
}
