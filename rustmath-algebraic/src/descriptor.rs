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
    ///
    /// Performs up to `target_precision` bisection steps using exact
    /// `Rational` arithmetic (never floating point), narrowing
    /// `isolating_interval` in place. See `refine_root_interval` for the
    /// underlying (non-mutating) bisection routine shared with the sign /
    /// comparison machinery in `AlgebraicReal`.
    pub fn refine_interval(&mut self, target_precision: usize) {
        if self.isolating_interval.is_some() {
            let refined = refine_root_interval(&*self, target_precision);
            self.isolating_interval = Some(refined);
        }
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

                // Exact reduction using the defining relation of a radical:
                // for alpha = sqrt(n) (minimal polynomial x^2 - n), alpha*alpha
                // is exactly n - no interval refinement needed to see this.
                // This only fires when both operands denote the *same* root
                // (identical polynomial and isolating interval), which is
                // guaranteed for e.g. `sqrt2.clone() * sqrt2.clone()` since
                // `AlgebraicReal::sqrt` is deterministic.
                if expr.op == BinaryOp::Mul {
                    if let (AlgebraicDescriptor::Root(a), AlgebraicDescriptor::Root(b)) =
                        (&left, &right)
                    {
                        if let Some(n) = same_quadratic_surd_square(a, b) {
                            return AlgebraicDescriptor::Rational(ANRational::new(n));
                        }
                    }
                }

                AlgebraicDescriptor::BinaryExpr(ANBinaryExpr::new(expr.op, left, right))
            }
        }
    }
}

/// If `a` and `b` are the same root of a monic quadratic `x^2 - n` (as
/// produced by `AlgebraicReal::sqrt`), return `n`. Such a root satisfies
/// `root * root == n` exactly, by definition of being a root of `x^2 - n`.
fn same_quadratic_surd_square(a: &ANRoot, b: &ANRoot) -> Option<Rational> {
    if a.polynomial != b.polynomial || a.isolating_interval != b.isolating_interval {
        return None;
    }

    let coeffs = a.polynomial.coefficients();
    if coeffs.len() != 3 {
        // Not a quadratic.
        return None;
    }
    if !coeffs[2].is_one() || !coeffs[1].is_zero() {
        // Not monic, or has a nonzero linear term (root*root would then be
        // an affine function of root, not a constant).
        return None;
    }

    Some(Rational::from_integer(-coeffs[0].clone()))
}

// ---------------------------------------------------------------------
// Exact interval arithmetic for sign / comparison determination
// ---------------------------------------------------------------------
//
// `AlgebraicReal::sign` / `Ord::cmp` need to determine the sign of a real
// algebraic expression tree exactly. They do so by refining the isolating
// interval of every `Root` leaf via bisection (using exact `Rational`
// arithmetic - never floating point) and propagating the resulting rational
// bounds through the tree via standard interval arithmetic. `eval_interval`
// below is the entry point; callers retry it with increasing
// `max_iterations` until `interval_sign` stops returning `None`.
//
// This is *sound* whenever it produces `Some(sign)`: the returned bound
// always contains the true value, so a bound strictly on one side of zero
// proves the sign. It is fundamentally unable to *prove* that a value is
// exactly zero this way (no finite amount of bisection can distinguish
// "exactly zero" from "extremely close to zero" via bounds alone) - see the
// refinement-cap fallback in `AlgebraicReal::sign` for how that case is
// handled.

/// Exact sign of a rational number: -1, 0, or 1.
pub(crate) fn sign_of_rational(r: &Rational) -> i32 {
    if r.numerator().is_zero() {
        0
    } else if r.numerator().signum() > 0 {
        1
    } else {
        -1
    }
}

/// Evaluate an integer-coefficient polynomial at a rational point, exactly.
fn eval_poly_at_rational(poly: &UnivariatePolynomial<Integer>, x: &Rational) -> Rational {
    let mut result = Rational::from_i64(0);
    for c in poly.coefficients().iter().rev() {
        result = result * x.clone() + Rational::from_integer(c.clone());
    }
    result
}

/// Refine the isolating interval of a polynomial root via bisection, using
/// exact rational arithmetic throughout. Performs at most `max_iterations`
/// bisection steps, halving the interval width each step.
///
/// Returns a rational interval `(lo, hi)` guaranteed to contain the true
/// root (with `lo == hi` if the root turns out to be exactly rational and
/// bisection happens to land exactly on it). Never mutates `root`: the
/// stored `isolating_interval` is only used as the starting bracket.
pub(crate) fn refine_root_interval(root: &ANRoot, max_iterations: usize) -> (Rational, Rational) {
    let (mut lo, mut hi) = match &root.isolating_interval {
        Some((a, b)) => (a.clone(), b.clone()),
        None => {
            // No real isolating interval is available (e.g. a genuinely
            // complex root). There's nothing sound to say about its
            // real-line sign; return a degenerate bracket so callers can
            // detect there was nothing to refine.
            return (Rational::from_i64(0), Rational::from_i64(0));
        }
    };

    let poly = &root.polynomial;

    let s_lo = sign_of_rational(&eval_poly_at_rational(poly, &lo));
    if s_lo == 0 {
        return (lo.clone(), lo);
    }
    let s_hi = sign_of_rational(&eval_poly_at_rational(poly, &hi));
    if s_hi == 0 {
        return (hi.clone(), hi);
    }
    if s_lo == s_hi {
        // The stored bracket does not actually isolate a sign change, which
        // would violate the documented invariant of `isolating_interval`.
        // Bisecting would be unsound (we wouldn't know which half to keep),
        // so just return the bracket unrefined.
        return (lo, hi);
    }

    for _ in 0..max_iterations {
        let mid = (lo.clone() + hi.clone()) / Rational::from_i64(2);
        let s_mid = sign_of_rational(&eval_poly_at_rational(poly, &mid));
        if s_mid == 0 {
            return (mid.clone(), mid);
        } else if s_mid == s_lo {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    (lo, hi)
}

/// Multiply two rational intervals `[l_lo, l_hi] * [r_lo, r_hi]`.
fn interval_mul(
    l_lo: &Rational,
    l_hi: &Rational,
    r_lo: &Rational,
    r_hi: &Rational,
) -> (Rational, Rational) {
    let candidates = [
        l_lo.clone() * r_lo.clone(),
        l_lo.clone() * r_hi.clone(),
        l_hi.clone() * r_lo.clone(),
        l_hi.clone() * r_hi.clone(),
    ];
    let mut min = candidates[0].clone();
    let mut max = candidates[0].clone();
    for c in &candidates[1..] {
        if *c < min {
            min = c.clone();
        }
        if *c > max {
            max = c.clone();
        }
    }
    (min, max)
}

/// Invert a rational interval, provided it does not straddle (or touch)
/// zero. Returns `None` if it does - the caller should refine further.
fn interval_inverse(lo: &Rational, hi: &Rational) -> Option<(Rational, Rational)> {
    let s_lo = sign_of_rational(lo);
    let s_hi = sign_of_rational(hi);
    if (s_lo > 0 && s_hi > 0) || (s_lo < 0 && s_hi < 0) {
        Some((hi.reciprocal().unwrap(), lo.reciprocal().unwrap()))
    } else {
        None
    }
}

/// Bisect for a rational bound on sqrt(target), given `0 <= target <=
/// bound^2`. Best-effort helper backing `UnaryOp::Sqrt` (not currently
/// constructed anywhere in this crate, but implemented soundly regardless).
fn bisect_sqrt_bound(
    target: &Rational,
    bound: &Rational,
    max_iterations: usize,
    round_up: bool,
) -> Rational {
    let mut lo = Rational::from_i64(0);
    let mut hi = bound.clone();
    for _ in 0..max_iterations {
        let mid = (lo.clone() + hi.clone()) / Rational::from_i64(2);
        let sq = mid.clone() * mid.clone();
        if sq < *target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if round_up {
        hi
    } else {
        lo
    }
}

fn interval_sqrt(
    lo: &Rational,
    hi: &Rational,
    max_iterations: usize,
) -> Option<(Rational, Rational)> {
    if sign_of_rational(hi) < 0 {
        return None;
    }
    let lo_clamped = if sign_of_rational(lo) < 0 {
        Rational::from_i64(0)
    } else {
        lo.clone()
    };
    let bound = if *hi > Rational::from_i64(1) {
        hi.clone()
    } else {
        Rational::from_i64(1)
    };
    let sqrt_lo = bisect_sqrt_bound(&lo_clamped, &bound, max_iterations, false);
    let sqrt_hi = bisect_sqrt_bound(hi, &bound, max_iterations, true);
    Some((sqrt_lo, sqrt_hi))
}

fn interval_pow_nonneg_int(lo: &Rational, hi: &Rational, mut exp: u64) -> (Rational, Rational) {
    let mut result = (Rational::from_i64(1), Rational::from_i64(1));
    let mut base = (lo.clone(), hi.clone());
    while exp > 0 {
        if exp & 1 == 1 {
            result = interval_mul(&result.0, &result.1, &base.0, &base.1);
        }
        base = interval_mul(&base.0, &base.1, &base.0, &base.1);
        exp >>= 1;
    }
    result
}

/// Evaluate a rational interval `(lo, hi)` guaranteed to contain the exact
/// real value of `desc`, refining every `Root` leaf via up to
/// `max_iterations` bisection steps.
///
/// Returns `None` if the value cannot be bounded at this refinement depth
/// (e.g. an intermediate division by an interval that still straddles
/// zero) - the caller should retry with a larger `max_iterations`.
pub(crate) fn eval_interval(
    desc: &AlgebraicDescriptor,
    max_iterations: usize,
) -> Option<(Rational, Rational)> {
    Some(match desc {
        AlgebraicDescriptor::Rational(r) => (r.value.clone(), r.value.clone()),
        AlgebraicDescriptor::Root(root) => refine_root_interval(root, max_iterations),
        AlgebraicDescriptor::UnaryExpr(expr) => {
            let (lo, hi) = eval_interval(&expr.operand, max_iterations)?;
            match expr.op {
                UnaryOp::Neg => (-hi, -lo),
                UnaryOp::Conj => (lo, hi), // real values are self-conjugate
                UnaryOp::Abs => {
                    if sign_of_rational(&lo) >= 0 {
                        (lo, hi)
                    } else if sign_of_rational(&hi) <= 0 {
                        (-hi, -lo)
                    } else {
                        let neg_lo = -lo.clone();
                        let bound = if neg_lo > hi { neg_lo } else { hi.clone() };
                        (Rational::from_i64(0), bound)
                    }
                }
                UnaryOp::Inv => interval_inverse(&lo, &hi)?,
                UnaryOp::Sqrt => interval_sqrt(&lo, &hi, max_iterations)?,
            }
        }
        AlgebraicDescriptor::BinaryExpr(expr) => {
            let (l_lo, l_hi) = eval_interval(&expr.left, max_iterations)?;
            let (r_lo, r_hi) = eval_interval(&expr.right, max_iterations)?;
            match expr.op {
                BinaryOp::Add => (l_lo + r_lo, l_hi + r_hi),
                BinaryOp::Sub => (l_lo - r_hi, l_hi - r_lo),
                BinaryOp::Mul => interval_mul(&l_lo, &l_hi, &r_lo, &r_hi),
                BinaryOp::Div => {
                    let (i_lo, i_hi) = interval_inverse(&r_lo, &r_hi)?;
                    interval_mul(&l_lo, &l_hi, &i_lo, &i_hi)
                }
                BinaryOp::Pow => match expr.right.as_rational() {
                    Some(exp) if exp.denominator().is_one() => {
                        let e = exp.numerator().to_i64();
                        if e >= 0 {
                            interval_pow_nonneg_int(&l_lo, &l_hi, e as u64)
                        } else {
                            let (p_lo, p_hi) = interval_pow_nonneg_int(&l_lo, &l_hi, (-e) as u64);
                            interval_inverse(&p_lo, &p_hi)?
                        }
                    }
                    _ => return None,
                },
            }
        }
    })
}

/// Determine the sign implied by a rational interval known to contain the
/// true value: `Some(1)` / `Some(-1)` if the interval lies strictly to one
/// side of zero, `Some(0)` if it has collapsed to exactly zero, or `None` if
/// it still straddles zero (inconclusive at this refinement depth).
pub(crate) fn interval_sign(lo: &Rational, hi: &Rational) -> Option<i32> {
    let s_lo = sign_of_rational(lo);
    let s_hi = sign_of_rational(hi);
    if s_lo > 0 {
        Some(1)
    } else if s_hi < 0 {
        Some(-1)
    } else if s_lo == 0 && s_hi == 0 {
        Some(0)
    } else {
        None
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
