//! Series expansions: truncated Laurent series over exact constants (Taylor, Maclaurin, Laurent,
//! asymptotic), Big-O notation and Fourier series.
//!
//! # The series engine (S1)
//!
//! `Expr::taylor` and its relatives expand bottom-up in truncated Laurent series
//! `sum c_k h^k + O(h^P)`, h = x - a (or 1/x at infinity), whose coefficients are exact constants
//! (finite Q-linear combinations of monomials in kernels, `series_const.rs`), with a work budget
//! and structural caps. A constant must sometimes be inverted or its sign read; its zero test is
//! three-valued (decided zero, decided nonzero, undecided), with rigorous MPFR enclosures as one
//! of the rules (`series_ball.rs`). Where the expansion itself needs the answer (the leading
//! coefficient of a series it inverts, or takes the log or a power of; the sign of a constant;
//! whether an exponent is an integer), an undecided test is reported
//! (`SeriesError::UndecidedZero`), never guessed. A constant that the input itself divides by is
//! the one exception (see the conventions below).
//!
//! Specification: MAGMA Handbook ch. 49, "Power, Laurent and Puiseux Series", §49.1.3 and §49.1.4
//! (the precision model: relative and absolute precision, exact series) and §49.5 (the
//! transcendental functions of a series); digest
//! `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`. The ch. 49 intrinsics map
//! to rustmath-powerseries; this module is the symbolic engine behind `Expr::taylor`.
//!
//! Algorithms: Knuth, TAOCP vol. 2, 3rd ed., §4.7 (power series manipulation: reciprocal, powers
//! by Miller's recurrence, exp and log through their differential equations); R. P. Brent and
//! H. T. Kung, "Fast algorithms for manipulating formal power series", J. ACM 25(4), 1978,
//! 581-595; S. C. Johnson, "Sparse polynomial arithmetic", ACM SIGSAM Bulletin 8(3), 1974, 63-71,
//! and M. Monagan and R. Pearce, "Sparse polynomial division using a heap", J. Symbolic Comput.
//! 46(7), 2011, 807-822 (the heap merges of sparse products and sums).
//!
//! Conventions, stated once for every function below:
//!
//! - Free symbols other than the variable are generic parameters: a result for `1/(x + a - b)` is
//!   valid for a != b only (the convention of gp, Maxima and the crate's Risch module).
//! - `abs` and `sign` assume a real variable and a real point.
//! - A float literal denotes its exact binary value: `0.1 - 1/10` is a nonzero constant. A literal
//!   that overflows to inf is refused here (`Unsupported`), and one with a nonzero digit that
//!   underflows to 0 (`1.0e-400`) is refused by the parser (S1 fix-up round 3).
//! - A constant that the input divides by (`1/c`, `c^-n`) is taken as nonzero when the zero test
//!   cannot decide it, and a function of a constant at a pole that the constant field cannot
//!   recognise (arctan at a constant that may be i or -i, arctanh at one that may be 1 or -1) is
//!   kept as the kernel f(c). The result then holds where the input is defined: if c is 0 or the
//!   pole, the input has no value. `x/(sin(1)^2 + cos(1)^2 - 1)` at 0 is `Ok`, as is
//!   `x + arctan(sqrt(-1)*x^0)` (the field does not reduce `sqrt(-1)^2` to -1). A decided zero or
//!   pole is `DivisionByZero`. The limit path reads `Finite` only for a rational constant term, so
//!   such a kernel never becomes a finite limit there, but a factor of it can be multiplied by 0
//!   (`x*arctan(sqrt(-1)*x^0)` has the limit 0 at 0 under this convention).
//! - A constant on a branch cut takes the principal value of the logarithmic formula
//!   (arcsin z = -i log(i z + sqrt(1 - z^2)), arccos z = pi/2 - arcsin z, arctanh z =
//!   (log(1 + z) - log(1 - z))/2, arccosh z = log(z + sqrt(z + 1) sqrt(z - 1)), principal log and
//!   square root), as gp's scalar functions do. The series of a power is `exp(alpha log b)` with the
//!   principal log (gp takes the real root for a series power of a negative constant term; the two
//!   differ there).
//!
//! The owner's two rules for the branch functions (decision of 2026-11-15): an argument whose
//! series is an exact constant c is evaluated (f(c), no expansion and no integrand); a
//! non-constant argument at a branch point is `NotLaurent`, never `DivisionByZero`.
//! `DivisionByZero` means that the input divides by an exact zero (or applies a function at an
//! exact pole of a constant, arctanh(1) or arctan(i)).

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::series_const::{
    balanced_arc, balanced_sum, bound_caps, tree_bound, SErr, TextBound, TextMemo, ZS,
};
use crate::series_engine::{self, Point, Render, Ser, Session};
use crate::symbol::Symbol;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;
use std::sync::Arc;

/// The budget of the limit path (`limits.rs`, design §2.13): no library path runs the engine
/// without one (without a budget, `exp(x)^2/x^1000000` exhausts memory).
pub const LIMIT_SERIES_WORK: u64 = 100_000_000;

/// Errors of the series engine (MAGMA ch. 49). `Display` and `Debug` are bounded (a long constant is
/// summarised); `PartialEq` compares the expressions as trees, so its cost is linear in their tree
/// sizes, which can be exponential in the size of their DAG.
#[derive(Clone, PartialEq)]
pub enum SeriesError {
    /// A subexpression has no Laurent expansion at the point: a branch point (sqrt(x) at 0, a
    /// non-constant argument of arcsin at 1), a logarithmic singularity (log(x) at 0, arctanh of a
    /// non-constant argument at 1), an argument with a pole inside exp, sin, arctan, ...
    /// (exp(1/x) at 0), or abs/sign at a zero of odd order. The engine expands each subexpression
    /// on its own, so the function itself may still have one (sqrt(x)^2 and exp(log(x)) at 0).
    NotLaurent(String),
    /// gamma, factorial, zeta, a named function or mod applied to an expression in the variable,
    /// abs/sign of a series with non-rational coefficients, an exponent that overflows i64.
    Unsupported(String),
    /// The expansion must divide by this constant (or read its sign) and cannot decide whether it
    /// is 0. For a power whose exponent alpha must be known to be an integer or not, the constant
    /// is alpha - n with n the nearest integer, or alpha itself when no nearest integer was
    /// computed. The numeric test refuses values outside a safe exponent window, so a constant
    /// whose evaluation would underflow or overflow is reported here, never decided. The variant
    /// holds the whole constant; `Display` prints it only when it is short.
    UndecidedZero(Expr),
    /// The expression divides by an exact zero (1/(x - x), log(0), 0^-1, tan(pi/2) as a constant,
    /// 0^r with Re r decided negative at the point), or applies a function at an exact pole of a
    /// constant (arctanh(1), arctan(i)). exp, sin, cos and tan of an argument whose series is an
    /// exact constant give a truncated series, not the exact value, so a pole reached through
    /// them can be `PrecisionExhausted` instead (`tan(pi/2 + x - x)` at 0).
    DivisionByZero(String),
    /// The point contains the variable, or infinity (oo) appears.
    InvalidPoint(String),
    /// Bounded search: a denominator vanished to this working precision. It may be identically 0.
    PrecisionExhausted { precision: i64 },
    /// The work budget of the call was exceeded. `work` is the count of charged units when the call
    /// stopped. It can include an a-priori charge for a step that was refused before it ran, so it
    /// may exceed both the budget and the work actually done.
    BudgetExceeded { work: u64, budget: u64 },
    /// A structural cap of the engine was reached before the work it bounds: a monomial of more
    /// than 2^15 factors, more than 2^16 kernels, a constant of more than 2^20 terms, a kernel
    /// expression deeper than max(1024, 4 x the input's depth), or a result (or the constant of an
    /// `UndecidedZero`) whose tree has more than 2^21 nodes or holds more than 2^27 characters of
    /// leaf text (the digits of its numbers and the names of its symbols and calls, at every
    /// occurrence: a shared kernel with a long literal is counted at each term that names it). The
    /// text counts the tree that `simplify` makes of the result too (a kernel's source such as
    /// `(2^254+1)^4096` as the literal it folds to), a float by its printed length, and the digits
    /// of an integer above 2^16 digits by their slower printing (`series_const::print_weight`).
    /// S1 fix-up round 4: or a result whose `simplify` may do more than 2^31 units of constant
    /// folding (`StructureLimit::FoldWork`; a kernel's source holds the input's own literal
    /// arithmetic, which `simplify` folds left to right). `value` is the cap.
    LimitExceeded { limit: StructureLimit, value: u64 },
    /// laurent(): the expansion has a nonzero term of exponent `valuation` below `min_order`.
    ValuationBelowMinimum { valuation: i64, min_order: i64 },
    /// series_coefficients(): the expansion has a pole of this order (valuation < 0).
    Pole { valuation: i64 },
}

/// Which structural cap `SeriesError::LimitExceeded` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureLimit {
    MonomialFactors,
    Kernels,
    ConstantTerms,
    KernelDepth,
    TreeSize,
    /// S1 fix-up round 2 (review 2, B2), round 3 (review 3, B1, B2, m1): the characters of leaf
    /// text (digits of the integers and rationals, floats, names of the symbols and calls) at
    /// every occurrence in the tree of the result and in the tree `simplify` makes of it, long
    /// integers weighted by their printing time. S1 fix-up round 5 (review 5, B1): also the
    /// literals that every node of the tree may fold to, at every occurrence, which `simplify`
    /// after `differentiate` may print
    TreeText,
    /// S1 fix-up round 4 (review 4, B1): the work of the constant folds that `simplify` may do on
    /// the result's tree (in the engine's units, about one word operation each), counted from the
    /// sizes of the literals it folds: a kernel's source or a hint holds the input's own literal
    /// arithmetic, which `simplify` folds left to right where the engine folded it in its own order.
    /// S1 fix-up round 5 (review 5, B1): counted at every occurrence in the tree, so that it also
    /// bounds `simplify` after a consumer that rebuilds the tree (`substitute`, `expand`)
    FoldWork,
    /// S1 fix-up round 5 (review 5, B2): the words of the literals that `simplify` may build at
    /// the distinct nodes of the result (`series_const::MAX_RESULT_MEMO_WORDS`, the payload of
    /// `simplify`'s memo), so that `simplify` after `differentiate`, which shares the result's
    /// nodes between their clones, holds them all in its memo
    FoldPayload,
}

/// An estimate of the length of e's Display text, stopped once it passes `limit`: 0.302 bits + 2
/// per integer (both parts of a rational), the printed length per float, the name length per symbol or function
/// and 6 per operator node. A walk with an explicit stack, with no decimal conversion (whose cost
/// is quadratic in the digits), so its cost is O(limit) whatever the size of e.
fn display_estimate(e: &Expr, limit: u64) -> u64 {
    let mut total: u64 = 0;
    let mut stack: Vec<&Expr> = vec![e];
    while let Some(n) = stack.pop() {
        if total > limit {
            break;
        }
        let add = match n {
            Expr::Integer(i) => i.bit_length() * 302 / 1000 + 2,
            Expr::Rational(q) => {
                (q.numerator().bit_length() + q.denominator().bit_length()) * 302 / 1000 + 4
            }
            // S1 fix-up round 3 (review 3, m2): what Display prints (up to 327; 24 undercounted)
            Expr::Real(f) => crate::expression::real_display_len(*f),
            Expr::Symbol(s) => s.name().len() as u64,
            Expr::Binary(_, l, r) => {
                stack.push(r);
                stack.push(l);
                6
            }
            Expr::Unary(_, a) => {
                stack.push(a);
                6
            }
            Expr::Function(name, args) => {
                stack.extend(args.iter().map(|a| &**a));
                name.len() as u64 + 2
            }
        };
        total = total.saturating_add(add);
    }
    total
}

impl fmt::Display for SeriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeriesError::NotLaurent(s) => write!(f, "no Laurent expansion at the point: {s}"),
            SeriesError::Unsupported(s) => write!(f, "not supported by the series engine: {s}"),
            SeriesError::UndecidedZero(c) => {
                let est = display_estimate(c, 200);
                if est <= 200 {
                    write!(f, "cannot decide whether the constant {c} is zero")
                } else {
                    write!(
                        f,
                        "cannot decide whether a constant is zero (its printed form has about \
                         {est} characters or more; the error value holds it)"
                    )
                }
            }
            SeriesError::DivisionByZero(s) => write!(f, "division by an exact zero: {s}"),
            SeriesError::InvalidPoint(s) => write!(f, "invalid expansion point: {s}"),
            SeriesError::PrecisionExhausted { precision } => write!(
                f,
                "a denominator vanished to working precision {precision} (it may be identically 0)"
            ),
            SeriesError::BudgetExceeded { work, budget } => {
                write!(f, "work budget exceeded ({work} units charged, budget {budget})")
            }
            SeriesError::LimitExceeded { limit, value } => {
                let what = match limit {
                    StructureLimit::MonomialFactors => "factors in one monomial",
                    StructureLimit::Kernels => "distinct kernels",
                    StructureLimit::ConstantTerms => "terms in one constant",
                    StructureLimit::KernelDepth => "levels in one kernel expression",
                    StructureLimit::TreeSize => "nodes in the tree of the result",
                    StructureLimit::TreeText => {
                        "characters of leaf text (digits and names) in the tree of the result, of \
                         its simplification, or of the literals its nodes fold to"
                    }
                    StructureLimit::FoldWork => {
                        "units of constant folding in the simplification of the result"
                    }
                    StructureLimit::FoldPayload => {
                        "words of literals built at the distinct nodes of the result by its \
                         simplification"
                    }
                };
                write!(
                    f,
                    "a structural limit of the series engine was reached: more than {value} {what}"
                )
            }
            SeriesError::ValuationBelowMinimum { valuation, min_order } => write!(
                f,
                "the expansion has a nonzero term of exponent {valuation}, below the minimum {min_order}"
            ),
            SeriesError::Pole { valuation } => {
                write!(f, "the expansion has a pole (a nonzero term of exponent {valuation})")
            }
        }
    }
}

/// Written by hand: the variant name, then the Display text. A derived Debug would print the whole
/// `UndecidedZero` constant, whose tree can be exponentially larger than its DAG, so a failing
/// `unwrap()` could print gigabytes.
impl fmt::Debug for SeriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            SeriesError::NotLaurent(_) => "NotLaurent",
            SeriesError::Unsupported(_) => "Unsupported",
            SeriesError::UndecidedZero(_) => "UndecidedZero",
            SeriesError::DivisionByZero(_) => "DivisionByZero",
            SeriesError::InvalidPoint(_) => "InvalidPoint",
            SeriesError::PrecisionExhausted { .. } => "PrecisionExhausted",
            SeriesError::BudgetExceeded { .. } => "BudgetExceeded",
            SeriesError::LimitExceeded { .. } => "LimitExceeded",
            SeriesError::ValuationBelowMinimum { .. } => "ValuationBelowMinimum",
            SeriesError::Pole { .. } => "Pole",
        };
        write!(f, "{name}({self})")
    }
}

impl std::error::Error for SeriesError {}

fn structure_limit(what: &str) -> StructureLimit {
    match what {
        "factors of one monomial" => StructureLimit::MonomialFactors,
        "terms of one constant" => StructureLimit::ConstantTerms,
        "depth of a kernel's expression" => StructureLimit::KernelDepth,
        "nodes of the result's tree" => StructureLimit::TreeSize,
        "characters of leaf text in the result's tree" => StructureLimit::TreeText,
        "characters of the literals at every node of the result's tree" => StructureLimit::TreeText,
        "units of constant folding in the simplification of the result" => StructureLimit::FoldWork,
        "words of the literals that the simplification of the result builds at its distinct nodes" => {
            StructureLimit::FoldPayload
        }
        _ => StructureLimit::Kernels,
    }
}

fn public_error(e: SErr, budget: u64) -> SeriesError {
    match e {
        SErr::NotLaurent(s) => SeriesError::NotLaurent(s),
        SErr::Unsupported(s) => SeriesError::Unsupported(s),
        SErr::Undecided(c) => SeriesError::UndecidedZero(c),
        SErr::DivisionByZero(s) => SeriesError::DivisionByZero(s),
        SErr::BadPoint(s) => SeriesError::InvalidPoint(s),
        // the driver turns a request for precision into PrecisionExhausted; one that escapes a
        // later step is reported the same way
        SErr::NeedPrecision => SeriesError::PrecisionExhausted { precision: 0 },
        SErr::PrecisionExhausted(p) => SeriesError::PrecisionExhausted { precision: p },
        SErr::Budget(work) => SeriesError::BudgetExceeded { work, budget },
        SErr::Limit(what, value) => SeriesError::LimitExceeded {
            limit: structure_limit(what),
            value,
        },
    }
}

/// The point of an expansion: a finite point (an expression free of the variable) or infinity
/// (the expansion in powers of 1/var). `Debug` is written by hand and bounded: a point whose text
/// would be long is summarised by its estimated length (`PartialEq` compares trees).
#[derive(Clone, PartialEq)]
pub enum ExpansionPoint {
    At(Expr),
    Infinity,
}

/// The text of a point, or a summary when it would pass 60 characters (`display_estimate`, so the
/// cost is bounded whatever the size of the point).
fn point_text(p: &ExpansionPoint) -> String {
    match p {
        ExpansionPoint::Infinity => "infinity".to_string(),
        ExpansionPoint::At(a) => {
            let est = display_estimate(a, 60);
            if est <= 60 {
                format!("{a}")
            } else {
                format!("<a point of about {est} characters or more>")
            }
        }
    }
}

impl fmt::Debug for ExpansionPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpansionPoint::Infinity => write!(f, "Infinity"),
            ExpansionPoint::At(_) => write!(f, "At({})", point_text(self)),
        }
    }
}

/// A truncated Laurent expansion sum c_k h^k + O(h^precision), h = var - a (or 1/var at infinity).
/// `Debug` is written by hand: the variable, the precision and the exponents, never a
/// coefficient's tree (a coefficient's tree can be exponentially larger than its DAG).
/// `PartialEq` compares the coefficients as trees (a cost linear in their tree sizes).
#[derive(Clone, PartialEq)]
pub struct SeriesExpansion {
    pub var: Symbol,
    pub point: ExpansionPoint,
    /// (exponent, coefficient) in increasing exponent; canonical constants, none structurally 0;
    /// every exponent below `precision`. Sparse: x^(-10^9) + 1 has two entries. Each coefficient
    /// is the shared node its rendering built (a bare kernel is the kernel's own node), so no node
    /// is copied per term.
    pub terms: Vec<(i64, Arc<Expr>)>,
    /// max_exponent + 1, always (also for an exact result).
    pub precision: i64,
    /// whether the point is the exact zero, recorded by the driver, so `to_expr` renders `x` or
    /// `x - a` without rebuilding the point
    point_is_zero: bool,
    /// h (`x`, or `x - a`), built once by `series_expansion` and charged there
    h: Arc<Expr>,
}

impl fmt::Debug for SeriesExpansion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let point = point_text(&self.point);
        let exps: Vec<i64> = self.terms.iter().map(|(e, _)| *e).take(64).collect();
        write!(
            f,
            "SeriesExpansion {{ var: {}, point: {point}, precision: {}, exponents: {exps:?}{} }}",
            self.var.name(),
            self.precision,
            if self.terms.len() > 64 { " ..." } else { "" }
        )
    }
}

fn is_int(c: &Expr, v: i64) -> bool {
    match c {
        Expr::Integer(n) => n.bit_length() <= 2 && n.to_i64() == Some(v),
        _ => false,
    }
}

fn rational_expr(q: Rational) -> Expr {
    if q.denominator().is_one() {
        Expr::Integer(q.numerator().clone())
    } else {
        Expr::Rational(q)
    }
}

impl SeriesExpansion {
    /// The exponent of the first stored term (None when no term is stored). Every term below it
    /// is exactly zero, so it is a lower bound of the valuation. It equals the valuation when that
    /// coefficient is decided nonzero (for example a rational); an undecided coefficient may be 0:
    /// `(sin(1)^2 + cos(1)^2 - 1)/x + 1` at 0 has first exponent -1 and valuation 0.
    pub fn first_exponent(&self) -> Option<i64> {
        self.terms.first().map(|t| t.0)
    }

    /// None for k >= precision; Some(0) when absent; otherwise the shared coefficient (an `Arc`
    /// increment).
    pub fn coefficient(&self, k: i64) -> Option<Arc<Expr>> {
        if k >= self.precision {
            return None;
        }
        Some(match self.terms.binary_search_by(|(e, _)| e.cmp(&k)) {
            Ok(i) => self.terms[i].1.clone(),
            Err(_) => Arc::new(Expr::from(0)),
        })
    }

    /// The truncated series as an expression (no O-term): per term an `Arc` increment for the
    /// coefficient and at most 4 new nodes, all charged by the expansion; the top node is copied
    /// once only when the result is a single shared coefficient. At infinity the term of exponent
    /// k is c / var^k.
    pub fn to_expr(&self) -> Expr {
        let inf = matches!(self.point, ExpansionPoint::Infinity);
        let h = &self.h;
        let hp = |m: i64| {
            if m == 1 {
                h.clone()
            } else {
                Arc::new(Expr::Binary(
                    BinaryOp::Pow,
                    h.clone(),
                    Arc::new(Expr::from(m)),
                ))
            }
        };
        let mut parts: Vec<Arc<Expr>> = Vec::with_capacity(self.terms.len());
        for (e, c) in &self.terms {
            // the driver refuses an exponent i64::MIN, so the negation cannot overflow
            let e = if inf { -*e } else { *e };
            let t = if e == 0 {
                c.clone()
            } else if e > 0 {
                if is_int(c, 1) {
                    hp(e)
                } else if is_int(c, -1) {
                    Arc::new(Expr::Unary(UnaryOp::Neg, hp(e)))
                } else {
                    Arc::new(Expr::Binary(BinaryOp::Mul, c.clone(), hp(e)))
                }
            } else {
                Arc::new(Expr::Binary(BinaryOp::Div, c.clone(), hp(-e)))
            };
            parts.push(t);
        }
        match balanced_arc(parts, BinaryOp::Add) {
            None => Expr::from(0),
            Some(a) => Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()),
        }
    }

    /// O(h^precision), with h = var - a, or h = 1/var at infinity: O((1/var)^precision). S1 fix-up
    /// round 3 (review 3, semantics m1): at infinity the stored h is the variable itself (`to_expr`
    /// divides by its powers), so this was O(var^precision), true but vacuous as var grows.
    pub fn big_o(&self) -> BigO {
        match self.point {
            ExpansionPoint::Infinity => {
                let inv = Arc::new(Expr::Binary(
                    BinaryOp::Div,
                    Arc::new(Expr::from(1)),
                    self.h.clone(),
                ));
                big_o_of(&inv, self.precision, &self.var)
            }
            ExpansionPoint::At(_) => big_o_of(&self.h, self.precision, &self.var),
        }
    }
}

fn big_o_of(h: &Arc<Expr>, n: i64, var: &Symbol) -> BigO {
    let f = if n == 1 {
        (**h).clone()
    } else {
        Expr::Binary(BinaryOp::Pow, h.clone(), Arc::new(Expr::from(n)))
    };
    BigO::new(f, var.clone())
}

fn h_of(var: &Symbol, point: &ExpansionPoint, point_is_zero: bool) -> Arc<Expr> {
    let x = Arc::new(Expr::Symbol(var.clone()));
    match point {
        ExpansionPoint::Infinity => x,
        ExpansionPoint::At(a) => {
            if point_is_zero {
                x
            } else {
                Arc::new(Expr::Binary(BinaryOp::Sub, x, Arc::new(a.clone())))
            }
        }
    }
}

fn engine_point(point: &ExpansionPoint) -> Point<'_> {
    match point {
        ExpansionPoint::At(a) => Point::At(a),
        ExpansionPoint::Infinity => Point::Infinity,
    }
}

/// A failure inside a session: an engine error (converted with the call's budget) or a public
/// error decided after the expansion (laurent's and series_coefficients' tests).
enum Fail {
    Engine(SErr),
    Public(SeriesError),
}
impl From<SErr> for Fail {
    fn from(e: SErr) -> Fail {
        Fail::Engine(e)
    }
}

/// Runs the driver in one session and hands its result (with whether the point is the exact
/// zero) to `post`, which still sees the session's kernel table (for zero tests and renderings);
/// the session's state is cleared afterwards.
fn with_expansion<T>(
    f: &Expr,
    var: &Symbol,
    point: &ExpansionPoint,
    max_exp: i64,
    budget: Option<u64>,
    render: Render,
    post: impl FnOnce(Ser, bool) -> Result<T, Fail>,
) -> Result<T, SeriesError> {
    let b = budget.unwrap_or(u64::MAX);
    let run = || -> Result<T, Fail> {
        let _session = Session::begin()?;
        let s = series_engine::expansion(f, var, engine_point(point), max_exp, b, render)?;
        let point_is_zero = series_engine::POINT_ZERO.with(|c| c.get());
        post(s, point_is_zero)
    };
    run().map_err(|e| match e {
        Fail::Engine(e) => public_error(e, b),
        Fail::Public(e) => e,
    })
}

/// Whether some stored term of exponent below `below` is nonzero: `Ok(Some(e))` for the first
/// term decided nonzero, `Ok(None)` when all are decided zero, the undecided error otherwise.
fn first_nonzero_below(s: &Ser, below: i64) -> Result<Option<i64>, SErr> {
    for (e, k) in &s.t {
        if *e >= below {
            break;
        }
        let st = k.zero_status();
        crate::series_const::check()?;
        match st {
            ZS::Zero => continue,
            ZS::NonZero => return Ok(Some(*e)),
            ZS::Unknown => return Err(crate::series_const::undecided(k)),
        }
    }
    Ok(None)
}

/// The public expansion of a driver result (every stored term rendered once per distinct
/// coefficient; the charge `render_units` was made by the driver). S1 fix-up round 2 (review 2,
/// B2), round 3 (review 3): refused with `LimitExceeded(TreeText)` when the tree of its `to_expr`,
/// or the tree `simplify` makes of it, weighs more than `MAX_RESULT_TEXT` (`expansion_text`).
fn public_expansion(
    s: Ser,
    var: &Symbol,
    point: &ExpansionPoint,
    max_exp: i64,
    point_is_zero: bool,
) -> Result<SeriesExpansion, Fail> {
    let h = h_of(var, point, point_is_zero);
    // one rendering per distinct coefficient (by the constant's address), so a result that names
    // one shared constant at every term holds one rendering of it
    let mut memo: std::collections::HashMap<usize, Arc<Expr>> = std::collections::HashMap::new();
    let terms: Vec<(i64, Arc<Expr>)> =
        s.t.iter()
            .map(|(e, k)| {
                let key = Arc::as_ptr(&k.t) as usize;
                let r = memo
                    .entry(key)
                    .or_insert_with(|| k.to_expr_arc_raw())
                    .clone();
                (*e, r)
            })
            .collect();
    let e = SeriesExpansion {
        var: var.clone(),
        point: point.clone(),
        terms,
        // the public precision is T = max_exp + 1 (the driver returned trunc_hard(T)); exactness
        // stays internal, so no public value is both an exponent and a sentinel
        precision: max_exp + 1,
        point_is_zero,
        h,
    };
    // S1 fix-up round 4 (review 4, B1): and the work of the folds of its `simplify`; round 5
    // (review 5, B1 and B2): that work per occurrence in the tree, and the literal words of its
    // distinct nodes
    let (text, total, payload, lit_text) = expansion_text(&e);
    if let Some(err) = bound_caps(text, total, payload, lit_text) {
        return Err(Fail::Engine(err));
    }
    Ok(e)
}

/// S1 fix-up round 2 (review 2, B2), round 3 (review 3, B1, B2): the printing weight of the tree
/// of `e.to_expr()` and of the tree `simplify` makes of it, as `tree_text` counts them, from the
/// rendering's parts and in its shape: per term the coefficient, `h` and its exponent (none for a
/// coefficient 1 or -1 times a power of h, which `to_expr` drops; `c / h^k` at infinity), and the
/// balanced sum of the terms. Each coefficient's DAG is walked once (shared memo), h once.
/// Round 4 (review 4, B1): with the fold work of `simplify` on that tree (`TextMemo::work`).
/// Round 5 (review 5, B1, B2): the text, the fold work of the tree with every occurrence counted
/// (`TextBound::total` of the root), the literal words of the distinct nodes (`TextMemo::payload`
/// and the nodes `to_expr` builds above the coefficients), and the printing weight of the
/// literals at every node and occurrence (`TextBound::lit_text` of the root).
fn expansion_text(e: &SeriesExpansion) -> (u64, u64, u64, u64) {
    let mut memo = TextMemo::default();
    let h = tree_bound(&e.h, &mut memo);
    let inf = matches!(e.point, ExpansionPoint::Infinity);
    // S1 fix-up round 5: the literal words of the nodes `to_expr` builds above the coefficients
    // and h (the coefficients' own are in `memo.payload`); their fold work is in the root's
    // `total`, which counts every node of the tree
    let mut lits = 0u64;
    let mut node = |b: TextBound| {
        lits = lits.saturating_add(b.lit());
        b
    };
    let terms: Vec<TextBound> = e
        .terms
        .iter()
        .map(|(k, c)| {
            let k = if inf { -*k } else { *k };
            let hp = |m: i64, node: &mut dyn FnMut(TextBound) -> TextBound| {
                if m == 1 {
                    h.clone()
                } else {
                    node(TextBound::binary(BinaryOp::Pow, &h, &TextBound::int(m)))
                }
            };
            if k == 0 {
                tree_bound(c, &mut memo)
            } else if k > 0 {
                if is_int(c, 1) {
                    hp(k, &mut node)
                } else if is_int(c, -1) {
                    let p = hp(k, &mut node);
                    node(TextBound::unary(UnaryOp::Neg, &p))
                } else {
                    let cb = tree_bound(c, &mut memo);
                    let p = hp(k, &mut node);
                    node(TextBound::binary(BinaryOp::Mul, &cb, &p))
                }
            } else {
                let cb = tree_bound(c, &mut memo);
                let p = hp(-k, &mut node);
                node(TextBound::binary(BinaryOp::Div, &cb, &p))
            }
        })
        .collect();
    // `balanced`'s `lit` is the literal words of the sum's own nodes
    let (text, total, sum_lits, lit_text) = match TextBound::balanced(&terms, BinaryOp::Add) {
        Some(b) => (b.text(), b.total(), b.lit(), b.lit_text()),
        None => (0, 0, 0, 0),
    };
    (
        text,
        total,
        memo.payload.saturating_add(lits).saturating_add(sum_lits),
        lit_text,
    )
}

fn order_to_i64(order: usize) -> Result<i64, SeriesError> {
    match i64::try_from(order) {
        Ok(n) if n < i64::MAX => Ok(n),
        _ => Err(SeriesError::Unsupported("order too large".into())),
    }
}

impl Expr {
    /// The truncated Laurent expansion of self in h = var - a (or h = 1/var at infinity) with every
    /// term of exponent at most `max_exponent`, under a work budget (`None`: unbounded, which only
    /// the structural caps and memory bound). See the module docs for the conventions (generic
    /// parameters, real abs and sign, branch cuts).
    pub fn series_expansion(
        &self,
        var: &Symbol,
        point: &ExpansionPoint,
        max_exponent: i64,
        budget: Option<u64>,
    ) -> Result<SeriesExpansion, SeriesError> {
        if max_exponent == i64::MAX {
            return Err(SeriesError::Unsupported("order too large".into()));
        }
        with_expansion(
            self,
            var,
            point,
            max_exponent,
            budget,
            Render::All,
            |s, pz| public_expansion(s, var, point, max_exponent, pz),
        )
    }

    /// All terms of exponent <= order of the expansion at `point`; poles give negative exponents
    /// (as Sage's and Maxima's taylor). Unbounded work: use `taylor_with_budget` for input from a
    /// user. Free symbols are generic parameters; a float literal denotes its exact binary value.
    pub fn taylor(&self, var: &Symbol, point: &Expr, order: usize) -> Result<Expr, SeriesError> {
        let n = order_to_i64(order)?;
        Ok(self
            .series_expansion(var, &ExpansionPoint::At(point.clone()), n, None)?
            .to_expr())
    }

    /// `taylor` under a work budget (the REPL passes `TAYLOR_WORK`): every big-number operation,
    /// constant, term, kernel and buffer slot is charged before it is made, so the call stops with
    /// `BudgetExceeded` before it uses more than about 3 bytes per unit.
    pub fn taylor_with_budget(
        &self,
        var: &Symbol,
        point: &Expr,
        order: usize,
        budget: u64,
    ) -> Result<Expr, SeriesError> {
        let n = order_to_i64(order)?;
        Ok(self
            .series_expansion(var, &ExpansionPoint::At(point.clone()), n, Some(budget))?
            .to_expr())
    }

    /// The Taylor expansion at 0.
    pub fn maclaurin(&self, var: &Symbol, order: usize) -> Result<Expr, SeriesError> {
        self.taylor(var, &Expr::from(0), order)
    }

    /// The same as `taylor`.
    pub fn series(&self, var: &Symbol, point: &Expr, order: usize) -> Result<Expr, SeriesError> {
        self.taylor(var, point, order)
    }

    /// The terms of exponent in [min_order, max_order]. `Err(ValuationBelowMinimum)` if a term
    /// below min_order is decided nonzero (instead of dropping the principal part),
    /// `Err(UndecidedZero)` if the terms below min_order cannot be decided.
    pub fn laurent(
        &self,
        var: &Symbol,
        point: &Expr,
        min_order: i64,
        max_order: i64,
    ) -> Result<Expr, SeriesError> {
        if max_order == i64::MAX {
            return Err(SeriesError::Unsupported("order too large".into()));
        }
        let pt = ExpansionPoint::At(point.clone());
        with_expansion(self, var, &pt, max_order, None, Render::All, |s, pz| {
            if let Some(valuation) = first_nonzero_below(&s, min_order)? {
                return Err(Fail::Public(SeriesError::ValuationBelowMinimum {
                    valuation,
                    min_order,
                }));
            }
            Ok(public_expansion(s, var, &pt, max_order, pz)?.to_expr())
        })
    }

    /// The expansion at infinity in powers of 1/var: the terms (1/var)^k for k <= order.
    pub fn asymptotic(&self, var: &Symbol, order: usize) -> Result<Expr, SeriesError> {
        let n = order_to_i64(order)?;
        Ok(self
            .series_expansion(var, &ExpansionPoint::Infinity, n, None)?
            .to_expr())
    }

    /// [c_0, ..., c_order] (dense; absent terms are 0); `Err(Pole)` when a term of negative
    /// exponent is decided nonzero, `Err(UndecidedZero)` when it cannot be decided. Like `taylor`
    /// it has no budget; each entry is an owned copy of its coefficient's top node (subtrees
    /// shared). The vector has order + 1 entries whatever the series, so its memory is linear in
    /// the order (one `Expr` each: `x` at order 10^7 takes about 0.6 GB); `series_expansion` is
    /// sparse and budgeted.
    pub fn series_coefficients(
        &self,
        var: &Symbol,
        point: &Expr,
        order: usize,
    ) -> Result<Vec<Expr>, SeriesError> {
        let n = order_to_i64(order)?;
        let pt = ExpansionPoint::At(point.clone());
        let s = with_expansion(self, var, &pt, n, None, Render::All, |s, pz| {
            if let Some(valuation) = first_nonzero_below(&s, 0)? {
                return Err(Fail::Public(SeriesError::Pole { valuation }));
            }
            public_expansion(s, var, &pt, n, pz)
        })?;
        let mut out = Vec::with_capacity(order.saturating_add(1).min(1 << 20));
        for k in 0..=n {
            out.push(match s.coefficient(k) {
                Some(c) => (*c).clone(),
                None => Expr::from(0),
            });
        }
        Ok(out)
    }

    /// `taylor` with its O-term O(h^(order + 1)).
    pub fn series_with_big_o(
        &self,
        var: &Symbol,
        point: &Expr,
        order: usize,
    ) -> Result<(Expr, BigO), SeriesError> {
        let n = order_to_i64(order)?;
        let s = self.series_expansion(var, &ExpansionPoint::At(point.clone()), n, None)?;
        Ok((s.to_expr(), s.big_o()))
    }
}

/// The limit path's entry (design §2.13): the expansion to `order` under `budget`, which charges,
/// bounds and renders the first stored term only, and returns its exponent and, when it is
/// rational, its value; `None` when no term is stored.
pub(crate) fn expansion_first_term(
    f: &Expr,
    var: &Symbol,
    point: &Expr,
    order: usize,
    budget: u64,
) -> Result<Option<(i64, Option<Expr>)>, SeriesError> {
    let n = order_to_i64(order)?;
    let pt = ExpansionPoint::At(point.clone());
    with_expansion(f, var, &pt, n, Some(budget), Render::First, |s, _| {
        Ok(s.t
            .first()
            .map(|(e, k)| (*e, k.as_rational().map(rational_expr))))
    })
}

/// `limit_with_error`'s entry (design §2.13): whether the expansion reaches `order + 1` under
/// `budget`; it renders no term (the caller reads only that it succeeded).
pub(crate) fn expansion_reaches(
    f: &Expr,
    var: &Symbol,
    point: &Expr,
    order: usize,
    budget: u64,
) -> Result<BigO, SeriesError> {
    let n = order_to_i64(order)?;
    let pt = ExpansionPoint::At(point.clone());
    with_expansion(f, var, &pt, n, Some(budget), Render::None, |_, pz| {
        Ok(big_o_of(&h_of(var, &pt, pz), n + 1, var))
    })
}

/// Whether `e` mentions `var`: an explicit stack and a set of visited node addresses, so a DAG
/// (a REPL value built by `e = e + e`) is walked once per distinct node.
pub fn mentions_symbol(e: &Expr, var: &Symbol) -> bool {
    let mut seen: std::collections::HashSet<*const Expr> = std::collections::HashSet::new();
    let mut stack: Vec<&Expr> = vec![e];
    while let Some(n) = stack.pop() {
        if !seen.insert(n as *const Expr) {
            continue;
        }
        match n {
            Expr::Symbol(s) if s == var => return true,
            Expr::Binary(_, l, r) => {
                stack.push(l);
                stack.push(r);
            }
            Expr::Unary(_, a) => stack.push(a),
            Expr::Function(_, args) => stack.extend(args.iter().map(|a| &**a)),
            _ => {}
        }
    }
    false
}
/// Build known Taylor series for common functions
///
/// Each sum is balanced (depth O(log order)), so a high order builds no deep left spine.
pub mod known_series {
    use super::*;

    /// exp(x) = 1 + x + x²/2! + x³/3! + ...
    pub fn exp(var: &Symbol, order: usize) -> Expr {
        let mut terms = Vec::new();
        let mut factorial = Integer::from(1);
        let x = Expr::Symbol(var.clone());

        for n in 0..=order {
            let term = x.clone().pow(Expr::from(n as i64)) / Expr::Integer(factorial.clone());
            terms.push(term);

            if n < order {
                factorial = factorial * Integer::from((n + 1) as i64);
            }
        }

        balanced_sum(terms).simplify()
    }

    /// sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
    pub fn sin(var: &Symbol, order: usize) -> Expr {
        let mut terms = Vec::new();
        let x = Expr::Symbol(var.clone());

        for n in 0..=order {
            let k = 2 * n + 1;
            if k > order {
                break;
            }

            let mut factorial = Integer::from(1);
            for i in 1..=k {
                factorial = factorial * Integer::from(i as i64);
            }

            let sign = if n % 2 == 0 { 1 } else { -1 };
            let term = Expr::from(sign) * x.clone().pow(Expr::from(k as i64))
                / Expr::Integer(factorial);
            terms.push(term);
        }

        balanced_sum(terms)
    }

    /// cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
    pub fn cos(var: &Symbol, order: usize) -> Expr {
        let mut terms = Vec::new();
        let x = Expr::Symbol(var.clone());

        for n in 0..=order {
            let k = 2 * n;
            if k > order {
                break;
            }

            let mut factorial = Integer::from(1);
            for i in 1..=k {
                factorial = factorial * Integer::from(i as i64);
            }

            let sign = if n % 2 == 0 { 1 } else { -1 };
            let term = Expr::from(sign) * x.clone().pow(Expr::from(k as i64))
                / Expr::Integer(factorial);
            terms.push(term);
        }

        balanced_sum(terms)
    }

    /// log(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...
    pub fn log_one_plus(var: &Symbol, order: usize) -> Expr {
        let mut terms = Vec::new();
        let x = Expr::Symbol(var.clone());

        for n in 1..=order {
            let sign = if n % 2 == 1 { 1 } else { -1 };
            let term = Expr::from(sign) * x.clone().pow(Expr::from(n as i64))
                / Expr::from(n as i64);
            terms.push(term);
        }

        balanced_sum(terms)
    }

    /// (1+x)^α = 1 + αx + α(α-1)x²/2! + α(α-1)(α-2)x³/3! + ...
    /// Binomial series (generalized binomial theorem)
    pub fn binomial(var: &Symbol, alpha: &Expr, order: usize) -> Expr {
        let mut terms = vec![Expr::from(1)];
        let x = Expr::Symbol(var.clone());
        let mut coeff = alpha.clone();
        let mut factorial = Integer::from(1);

        for n in 1..=order {
            factorial = factorial * Integer::from(n as i64);

            let term = coeff.clone() * x.clone().pow(Expr::from(n as i64))
                / Expr::Integer(factorial.clone());
            terms.push(term);

            // Update coefficient: multiply by (α - n)
            coeff = coeff * (alpha.clone() - Expr::from(n as i64));
        }

        balanced_sum(terms)
    }
}

// ============================================================================
// Phase 3.4 Enhancement: Big-O Notation Support
// ============================================================================

/// Big-O notation for asymptotic analysis
///
/// Represents O(f(x)) where f(x) is the growth rate function
#[derive(Debug, Clone, PartialEq)]
pub struct BigO {
    /// The function inside O(...)
    pub function: Expr,
    /// The variable (usually x or n)
    pub var: Symbol,
}

impl BigO {
    /// Create a new Big-O expression
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = Symbol::new("x");
    /// let o_x2 = BigO::new(Expr::Symbol(x.clone()).pow(Expr::from(2)), x);
    /// // Represents O(x²)
    /// ```
    pub fn new(function: Expr, var: Symbol) -> Self {
        BigO { function, var }
    }

    /// Create O(1) - constant time
    pub fn constant(var: Symbol) -> Self {
        BigO::new(Expr::from(1), var)
    }

    /// Create O(log n) - logarithmic time
    pub fn logarithmic(var: Symbol) -> Self {
        BigO::new(Expr::Symbol(var.clone()).log(), var)
    }

    /// Create O(n) - linear time
    pub fn linear(var: Symbol) -> Self {
        BigO::new(Expr::Symbol(var.clone()), var)
    }

    /// Create O(n log n) - linearithmic time
    pub fn linearithmic(var: Symbol) -> Self {
        let n = Expr::Symbol(var.clone());
        BigO::new(n.clone() * n.log(), var)
    }

    /// Create O(n²) - quadratic time
    pub fn quadratic(var: Symbol) -> Self {
        BigO::new(Expr::Symbol(var.clone()).pow(Expr::from(2)), var)
    }

    /// Create O(n^k) - polynomial time
    pub fn polynomial(var: Symbol, degree: i64) -> Self {
        BigO::new(Expr::Symbol(var.clone()).pow(Expr::from(degree)), var)
    }

    /// Create O(2^n) - exponential time
    pub fn exponential(var: Symbol) -> Self {
        BigO::new(Expr::from(2).pow(Expr::Symbol(var.clone())), var)
    }

    /// Create O(n!) - factorial time
    pub fn factorial(var: Symbol) -> Self {
        // Factorial not directly representable, use placeholder
        BigO::new(Expr::Symbol(var.clone()).factorial(), var)
    }

    /// Determine the dominant term in an expression for Big-O analysis
    ///
    /// For a sum of terms, returns the fastest-growing term
    ///
    /// # Algorithm
    ///
    /// 1. Separate expression into terms
    /// 2. For each term, determine its growth rate
    /// 3. Return the term with highest growth rate
    pub fn dominant_term(expr: &Expr, var: &Symbol) -> Expr {
        // For sums, find the dominant term
        if let Expr::Binary(BinaryOp::Add, left, right) = expr {
            let left_dom = BigO::dominant_term(left, var);
            let right_dom = BigO::dominant_term(right, var);

            // Compare growth rates
            if BigO::grows_faster(&left_dom, &right_dom, var) {
                left_dom
            } else {
                right_dom
            }
        } else {
            // For non-sums, return the expression itself (simplified)
            expr.clone()
        }
    }

    /// Check if f(x) grows faster than g(x) as x → ∞
    ///
    /// Uses limit comparison: if lim(x→∞) g(x)/f(x) = 0, then f grows faster
    fn grows_faster(f: &Expr, g: &Expr, var: &Symbol) -> bool {
        // Simplified comparison based on degree
        let f_degree = f.degree(var).unwrap_or(0);
        let g_degree = g.degree(var).unwrap_or(0);

        f_degree > g_degree
    }

    /// Derive Big-O complexity from an expression
    ///
    /// Analyzes an expression and determines its asymptotic complexity
    ///
    /// # Examples
    ///
    /// - 3n² + 2n + 1 → O(n²)
    /// - 5n + log(n) → O(n)
    /// - 2^n + n³ → O(2^n)
    pub fn from_expression(expr: &Expr, var: &Symbol) -> Self {
        // Find dominant term
        let dominant = BigO::dominant_term(expr, var);

        // Remove constant coefficients
        let simplified = BigO::remove_constants(&dominant, var);

        BigO::new(simplified, var.clone())
    }

    /// Remove constant coefficients from an expression
    ///
    /// For Big-O analysis, constant factors don't matter
    fn remove_constants(expr: &Expr, var: &Symbol) -> Expr {
        match expr {
            // If expression doesn't contain the variable, treat as constant
            _ if !expr.contains_symbol(var) => Expr::from(1),

            // For products, remove constant factors
            Expr::Binary(BinaryOp::Mul, left, right) => {
                if !left.contains_symbol(var) {
                    BigO::remove_constants(right, var)
                } else if !right.contains_symbol(var) {
                    BigO::remove_constants(left, var)
                } else {
                    // Both contain the variable
                    BigO::remove_constants(left, var) * BigO::remove_constants(right, var)
                }
            }

            // For other expressions, keep as is
            _ => expr.clone(),
        }
    }

    /// Check if this Big-O is equal to or dominates another
    ///
    /// O(f) dominates O(g) if f grows at least as fast as g
    pub fn dominates(&self, other: &BigO) -> bool {
        BigO::grows_faster(&self.function, &other.function, &self.var)
            || self.function == other.function
    }
}

impl std::fmt::Display for BigO {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "O({})", self.function)
    }
}

/// Little-o notation for strict asymptotic bounds
///
/// o(f(x)) represents functions that grow strictly slower than f(x)
#[derive(Debug, Clone, PartialEq)]
pub struct LittleO {
    pub function: Expr,
    pub var: Symbol,
}

impl LittleO {
    pub fn new(function: Expr, var: Symbol) -> Self {
        LittleO { function, var }
    }
}

impl std::fmt::Display for LittleO {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "o({})", self.function)
    }
}

/// Theta notation for tight asymptotic bounds
///
/// Θ(f(x)) represents functions that grow at the same rate as f(x)
#[derive(Debug, Clone, PartialEq)]
pub struct Theta {
    pub function: Expr,
    pub var: Symbol,
}

impl Theta {
    pub fn new(function: Expr, var: Symbol) -> Self {
        Theta { function, var }
    }
}

impl std::fmt::Display for Theta {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Θ({})", self.function)
    }
}

/// Omega notation for lower bounds
///
/// Ω(f(x)) represents functions that grow at least as fast as f(x)
#[derive(Debug, Clone, PartialEq)]
pub struct Omega {
    pub function: Expr,
    pub var: Symbol,
}

impl Omega {
    pub fn new(function: Expr, var: Symbol) -> Self {
        Omega { function, var }
    }
}

impl std::fmt::Display for Omega {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Ω({})", self.function)
    }
}

// ========================================================================
// Phase 2 Enhancement: Fourier Series
// ========================================================================

/// Fourier series representation
///
/// Represents a periodic function as a trigonometric series:
/// f(x) = a₀/2 + Σ(n=1 to ∞) [aₙ cos(nx) + bₙ sin(nx)]
///
/// For period 2L instead of 2π:
/// f(x) = a₀/2 + Σ(n=1 to ∞) [aₙ cos(nπx/L) + bₙ sin(nπx/L)]
#[derive(Debug, Clone, PartialEq)]
pub struct FourierSeries {
    /// Constant term (a₀/2)
    pub a0: Expr,
    /// Cosine coefficients [a₁, a₂, a₃, ...]
    pub a_coeffs: Vec<Expr>,
    /// Sine coefficients [b₁, b₂, b₃, ...]
    pub b_coeffs: Vec<Expr>,
    /// Variable
    pub var: Symbol,
    /// Period (2L, default 2π when L = π)
    pub period: Expr,
}

impl FourierSeries {
    /// Create a new Fourier series
    pub fn new(a0: Expr, a_coeffs: Vec<Expr>, b_coeffs: Vec<Expr>, var: Symbol, period: Expr) -> Self {
        FourierSeries {
            a0,
            a_coeffs,
            b_coeffs,
            var,
            period,
        }
    }

    /// Convert Fourier series to symbolic expression
    ///
    /// Returns the truncated series as a symbolic expression
    pub fn to_expr(&self) -> Expr {
        let mut result = self.a0.clone() / Expr::from(2);
        let x = Expr::Symbol(self.var.clone());

        // Compute angular frequency: ω = 2π/period
        let two_pi = Expr::from(2) * Expr::Symbol(Symbol::new("pi"));
        let omega = two_pi / self.period.clone();

        let n_terms = self.a_coeffs.len().max(self.b_coeffs.len());

        for n in 0..n_terms {
            let n_expr = Expr::from((n + 1) as i64);
            let arg = omega.clone() * n_expr.clone() * x.clone();

            // Add cosine term if available
            if n < self.a_coeffs.len() {
                let cos_term = self.a_coeffs[n].clone() * arg.clone().cos();
                result = result + cos_term;
            }

            // Add sine term if available
            if n < self.b_coeffs.len() {
                let sin_term = self.b_coeffs[n].clone() * arg.sin();
                result = result + sin_term;
            }
        }

        result
    }
}

impl Expr {
    /// Compute Fourier series expansion of a function
    ///
    /// Computes the Fourier series coefficients for a periodic function
    /// over the interval [-L, L].
    ///
    /// # Arguments
    ///
    /// * `var` - Variable to expand with respect to
    /// * `period` - Period of the function (2L)
    /// * `n_terms` - Number of terms to compute
    ///
    /// # Algorithm
    ///
    /// For a function f(x) with period 2L:
    /// - a₀ = (1/L) ∫[-L,L] f(x) dx
    /// - aₙ = (1/L) ∫[-L,L] f(x)cos(nπx/L) dx
    /// - bₙ = (1/L) ∫[-L,L] f(x)sin(nπx/L) dx
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Square wave with period 2π
    /// let x = Symbol::new("x");
    /// let expr = /* piecewise function */;
    /// let fourier = expr.fourier_series(&x, &Expr::from(2*π), 10);
    /// ```
    pub fn fourier_series(&self, var: &Symbol, period: &Expr, n_terms: usize) -> FourierSeries {
        let half_period = period.clone() / Expr::from(2);
        let l = half_period.clone();

        // Compute a₀ = (1/L) ∫[-L,L] f(x) dx
        let a0_integral = self.integrate_definite(
            var,
            &(-l.clone()),
            &l.clone(),
        );
        let a0 = if let Some(integral) = a0_integral {
            integral / l.clone()
        } else {
            // If integration fails, use symbolic representation
            Expr::from(0)
        };

        let mut a_coeffs = Vec::new();
        let mut b_coeffs = Vec::new();

        let pi = Expr::Symbol(Symbol::new("pi"));

        for n in 1..=n_terms {
            let n_expr = Expr::from(n as i64);
            let x = Expr::Symbol(var.clone());

            // Argument: nπx/L
            let arg = n_expr.clone() * pi.clone() * x.clone() / l.clone();

            // Compute aₙ = (1/L) ∫[-L,L] f(x)cos(nπx/L) dx
            let cos_integrand = self.clone() * arg.clone().cos();
            let an_integral = cos_integrand.integrate_definite(
                var,
                &(-l.clone()),
                &l.clone(),
            );
            let an = if let Some(integral) = an_integral {
                integral / l.clone()
            } else {
                Expr::from(0)
            };

            // Compute bₙ = (1/L) ∫[-L,L] f(x)sin(nπx/L) dx
            let sin_integrand = self.clone() * arg.sin();
            let bn_integral = sin_integrand.integrate_definite(
                var,
                &(-l.clone()),
                &l.clone(),
            );
            let bn = if let Some(integral) = bn_integral {
                integral / l.clone()
            } else {
                Expr::from(0)
            };

            a_coeffs.push(an.simplify());
            b_coeffs.push(bn.simplify());
        }

        FourierSeries::new(a0, a_coeffs, b_coeffs, var.clone(), period.clone())
    }

    /// Compute Fourier series for standard 2π period
    ///
    /// This is a convenience method for functions with period 2π
    pub fn fourier_series_2pi(&self, var: &Symbol, n_terms: usize) -> FourierSeries {
        let two_pi = Expr::from(2) * Expr::Symbol(Symbol::new("pi"));
        self.fourier_series(var, &two_pi, n_terms)
    }

    /// Compute discrete Fourier transform (DFT) coefficients
    ///
    /// For a sequence of N samples, computes the DFT coefficients
    /// This is useful for numerical Fourier analysis
    ///
    /// # Note
    ///
    /// This is a symbolic representation. For actual numerical DFT,
    /// use the numerical module with FFT algorithms.
    pub fn discrete_fourier_coefficients(samples: &[Expr], var: &Symbol) -> Vec<Expr> {
        let n = samples.len();
        let mut coefficients = Vec::new();

        for k in 0..n {
            let mut sum = Expr::from(0);
            let two_pi = Expr::from(2) * Expr::Symbol(Symbol::new("pi"));

            for (j, sample) in samples.iter().enumerate() {
                // e^(-2πikj/N) = cos(-2πkj/N) + i*sin(-2πkj/N)
                let angle = -two_pi.clone() * Expr::from(k as i64) * Expr::from(j as i64)
                    / Expr::from(n as i64);

                // For simplicity, just compute the real part (cosine transform)
                let coeff = sample.clone() * angle.cos();
                sum = sum + coeff;
            }

            coefficients.push(sum.simplify());
        }

        coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::known_series;

    /// S1 fix-up round 3 (review 3, resources m2): `display_estimate` counts a float by what
    /// `Display` prints (5e-324 prints 326 characters, not 24).
    #[test]
    fn display_estimate_counts_a_float_by_its_display() {
        for f in [5e-324, -1e-308, 1e300, f64::MAX, 0.5, -2.0] {
            let e = Expr::Real(f);
            let shown = format!("{e}").len() as u64;
            assert!(display_estimate(&e, 10_000) >= shown, "{f:e}");
        }
    }

    /// The coefficients c_0..c_n of `f` at `a` (through `series_coefficients`) as rationals.
    fn rational_coefficients(f: &Expr, x: &Symbol, a: &Expr, n: usize) -> Vec<Rational> {
        f.series_coefficients(x, a, n)
            .unwrap()
            .iter()
            .map(|c| c.eval_rational().expect("a rational coefficient"))
            .collect()
    }

    fn q(n: i64, d: i64) -> Rational {
        Rational::new(Integer::from(n), Integer::from(d)).unwrap()
    }

    #[test]
    fn taylor_at_a_huge_point() {
        // The expansion point 10^30 panicked in the zero test (inherent Integer::to_i64).
        // x^2 is its own Taylor polynomial, so at x = 10^30 + 1 it gives (10^30 + 1)^2.
        let x = Symbol::new("x");
        let a = Integer::from(10).pow(30);
        let f = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let at = Expr::Integer(a.clone() + Integer::from(1));
        let want = Some(Rational::from_integer((a.clone() + Integer::from(1)).pow(2)));
        let t = f.taylor(&x, &Expr::Integer(a.clone()), 2).unwrap();
        assert_eq!(t.substitute(&x, &at).eval_rational(), want);
        let s = f.series(&x, &Expr::Integer(a.clone()), 2).unwrap();
        assert_eq!(s.substitute(&x, &at).eval_rational(), want);
        let (s2, _) = f.series_with_big_o(&x, &Expr::Integer(a), 2).unwrap();
        assert_eq!(s2.substitute(&x, &at).eval_rational(), want);
    }

    /// The S7 point 10^30 through the new entry: x^2 at 10^30 is a^2 + 2a h + h^2 exactly.
    #[test]
    fn big_point_s7() {
        let x = Symbol::new("x");
        let a = Integer::from(10).pow(30);
        let f = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let c = rational_coefficients(&f, &x, &Expr::Integer(a.clone()), 3);
        assert_eq!(
            c,
            vec![
                Rational::from_integer(a.clone() * a.clone()),
                Rational::from_integer(Integer::from(2) * a),
                q(1, 1),
                q(0, 1)
            ]
        );
    }

    #[test]
    fn test_taylor_polynomial() {
        let x = Symbol::new("x");
        // Taylor expansion of x² around x=0 should be x²
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = expr.taylor(&x, &Expr::from(0), 5).unwrap();
        // Should be exactly x² since higher derivatives are 0
        let expected = Expr::Symbol(x.clone()).pow(Expr::from(2));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_maclaurin_exp() {
        // gp: exp(x) + O(x^4) = 1 + x + 1/2*x^2 + 1/6*x^3 + O(x^4)
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).exp();
        let t = expr.maclaurin(&x, 3).unwrap();
        let back = rational_coefficients(&t, &x, &Expr::from(0), 3);
        assert_eq!(back, vec![q(1, 1), q(1, 1), q(1, 2), q(1, 6)]);
        let c = rational_coefficients(&expr, &x, &Expr::from(0), 3);
        assert_eq!(c, vec![q(1, 1), q(1, 1), q(1, 2), q(1, 6)]);
    }

    #[test]
    fn test_known_series_exp() {
        let x = Symbol::new("x");
        let series = known_series::exp(&x, 4);
        // Should be 1 + x + x²/2! + x³/3! + x⁴/4!
        // We can verify the structure (exact comparison is complex due to simplification)
        assert!(!series.is_constant());
    }

    #[test]
    fn test_known_series_sin() {
        let x = Symbol::new("x");
        let series = known_series::sin(&x, 5);
        // Should be x - x³/3! + x⁵/5!
        assert!(!series.is_constant());
    }

    #[test]
    fn test_known_series_cos() {
        let x = Symbol::new("x");
        let series = known_series::cos(&x, 4);
        // Should be 1 - x²/2! + x⁴/4!
        assert!(!series.is_constant());
    }

    #[test]
    fn test_series_coefficients() {
        let x = Symbol::new("x");
        // For f(x) = x², coefficients around 0 are [0, 0, 1, 0]
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let coeffs = expr.series_coefficients(&x, &Expr::from(0), 3).unwrap();

        assert_eq!(coeffs.len(), 4); // Coefficients for x^0, x^1, x^2, x^3
        assert_eq!(coeffs[0], Expr::from(0)); // constant term
        assert_eq!(coeffs[1], Expr::from(0)); // linear term
        assert_eq!(coeffs[2], Expr::from(1));
        assert_eq!(coeffs[3], Expr::from(0));
    }

    #[test]
    fn test_series_coefficients_pole() {
        // 1/x at 0 has a pole of order 1
        let x = Symbol::new("x");
        let f = Expr::from(1) / Expr::Symbol(x.clone());
        match f.series_coefficients(&x, &Expr::from(0), 2) {
            Err(SeriesError::Pole { valuation }) => assert_eq!(valuation, -1),
            other => panic!("expected Pole, got {other:?}"),
        }
    }

    #[test]
    fn test_taylor_around_nonzero() {
        // gp: subst(x^2, x, 1 + h) + O(h^3) = 1 + 2*h + h^2
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let result = expr.taylor(&x, &Expr::from(1), 2).unwrap();
        let h = Expr::Symbol(x.clone()) - Expr::from(1);
        let want = Expr::from(1) + Expr::from(2) * h.clone() + h.pow(Expr::from(2));
        for v in [q(3, 7), q(-5, 2), q(10, 1)] {
            let at = Expr::Rational(v);
            assert_eq!(
                result.substitute(&x, &at).eval_rational(),
                want.substitute(&x, &at).eval_rational()
            );
        }
        let c = rational_coefficients(&expr, &x, &Expr::from(1), 2);
        assert_eq!(c, vec![q(1, 1), q(2, 1), q(1, 1)]);
    }

    #[test]
    fn test_laurent_positive_orders() {
        let x = Symbol::new("x");
        // Laurent series with only positive orders is just Taylor series
        let expr = Expr::Symbol(x.clone()).exp();
        let laurent = expr.laurent(&x, &Expr::from(0), 0, 3).unwrap();
        let taylor = expr.taylor(&x, &Expr::from(0), 3).unwrap();
        assert_eq!(laurent, taylor);
    }

    #[test]
    fn test_laurent_principal_part() {
        // gp: 1/sin(x) + O(x^4) = x^-1 + 1/6*x + 7/360*x^3 + O(x^4)
        let x = Symbol::new("x");
        let f = Expr::from(1) / Expr::Symbol(x.clone()).sin();
        match f.laurent(&x, &Expr::from(0), 0, 3) {
            Err(SeriesError::ValuationBelowMinimum { valuation, min_order }) => {
                assert_eq!((valuation, min_order), (-1, 0))
            }
            other => panic!("expected ValuationBelowMinimum, got {other:?}"),
        }
        let l = f.laurent(&x, &Expr::from(0), -1, 3).unwrap();
        assert_eq!(l, f.taylor(&x, &Expr::from(0), 3).unwrap());
    }

    #[test]
    fn test_binomial_series() {
        let x = Symbol::new("x");
        // (1+x)^2 = 1 + 2x + x²
        let alpha = Expr::from(2);
        let result = known_series::binomial(&x, &alpha, 2);

        // Should contain terms up to x²
        assert!(!result.is_constant());
    }

    #[test]
    fn test_unified_series_function() {
        // gp: exp(x) + O(x^4) = 1 + x + 1/2*x^2 + 1/6*x^3 + O(x^4)
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).exp();
        let series = expr.series(&x, &Expr::from(0), 3).unwrap();
        let c = rational_coefficients(&series, &x, &Expr::from(0), 3);
        assert_eq!(c, vec![q(1, 1), q(1, 1), q(1, 2), q(1, 6)]);
        let k = rational_coefficients(&known_series::exp(&x, 3), &x, &Expr::from(0), 3);
        assert_eq!(c, k);
    }

    #[test]
    fn test_series_at_nonzero_point() {
        // gp: subst(x^2, x, 1 + h) + O(h^3) = 1 + 2*h + h^2
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));
        let series = expr.series(&x, &Expr::from(1), 2).unwrap();
        let c = rational_coefficients(&series, &x, &Expr::from(1), 2);
        assert_eq!(c, vec![q(1, 1), q(2, 1), q(1, 1)]);
    }

    #[test]
    fn test_series_with_big_o() {
        // gp: sin(x) + O(x^4) = x - 1/6*x^3 + O(x^4)
        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).sin();
        let (series, big_o) = expr.series_with_big_o(&x, &Expr::from(0), 3).unwrap();
        let c = rational_coefficients(&series, &x, &Expr::from(0), 3);
        assert_eq!(c, vec![q(0, 1), q(1, 1), q(0, 1), q(-1, 6)]);
        assert_eq!(big_o.var, x);
        assert_eq!(big_o.function, Expr::Symbol(x.clone()).pow(Expr::from(4)));
    }

    #[test]
    fn test_big_o_from_expression() {
        let x = Symbol::new("x");

        // 3n² + 2n + 1 → O(n²)
        let expr = Expr::from(3) * Expr::Symbol(x.clone()).pow(Expr::from(2))
            + Expr::from(2) * Expr::Symbol(x.clone())
            + Expr::from(1);

        let big_o = BigO::from_expression(&expr, &x);

        // Should detect x² as dominant term
        let expected_func = Expr::Symbol(x.clone()).pow(Expr::from(2));
        assert_eq!(big_o.function, expected_func);
    }

    #[test]
    fn test_big_o_dominance() {
        let x = Symbol::new("x");

        // O(n²) dominates O(n)
        let o_n2 = BigO::quadratic(x.clone());
        let o_n = BigO::linear(x.clone());

        assert!(o_n2.dominates(&o_n));
        assert!(!o_n.dominates(&o_n2));
    }

    #[test]
    fn test_big_o_display() {
        let x = Symbol::new("x");
        let big_o = BigO::quadratic(x);

        let display = format!("{}", big_o);
        assert!(display.contains("O("));
    }

    #[test]
    fn test_asymptotic_expansion() {
        let x = Symbol::new("x");

        // Test asymptotic expansion as x → ∞
        // For (x² + x) / (x² + 1), as x → ∞, should approach 1
        let num = Expr::Symbol(x.clone()).pow(Expr::from(2)) + Expr::Symbol(x.clone());
        let den = Expr::Symbol(x.clone()).pow(Expr::from(2)) + Expr::from(1);
        let expr = num / den;

        let asymp = expr.asymptotic(&x, 3).unwrap();

        // gp: subst((x^2+x)/(x^2+1), x, 1/t) + O(t^4) = 1 + t - t^2 - t^3 + O(t^4)
        let t = Symbol::new("t");
        let in_t = asymp.substitute(&x, &(Expr::from(1) / Expr::Symbol(t.clone())));
        for v in [q(1, 3), q(-2, 5), q(7, 11)] {
            let at = Expr::Rational(v.clone());
            let want = q(1, 1) + v.clone() - v.clone() * v.clone() - v.clone() * v.clone() * v;
            assert_eq!(in_t.substitute(&t, &at).eval_rational(), Some(want));
        }
    }

    // ========================================================================
    // Phase 2 Tests: Fourier Series
    // ========================================================================

    #[test]
    fn test_fourier_series_constant() {
        let x = Symbol::new("x");

        // Fourier series of a constant function f(x) = 1
        // Should give a₀ = 2, all other coefficients = 0
        let expr = Expr::from(1);
        let pi = Expr::Symbol(Symbol::new("pi"));
        let period = Expr::from(2) * pi;

        let fourier = expr.fourier_series(&x, &period, 3);

        // Check that we have the expected structure
        assert_eq!(fourier.var, x);
        assert_eq!(fourier.a_coeffs.len(), 3);
        assert_eq!(fourier.b_coeffs.len(), 3);
    }

    #[test]
    fn test_fourier_series_to_expr() {
        let x = Symbol::new("x");

        // Create a simple Fourier series manually
        let a0 = Expr::from(2);
        let a_coeffs = vec![Expr::from(1), Expr::from(0)];
        let b_coeffs = vec![Expr::from(0), Expr::from(1)];
        let pi = Expr::Symbol(Symbol::new("pi"));
        let period = Expr::from(2) * pi;

        let fourier = FourierSeries::new(a0, a_coeffs, b_coeffs, x.clone(), period);

        // Convert to expression
        let expr = fourier.to_expr();

        // Should not be constant
        assert!(!expr.is_constant());
    }

    #[test]
    fn test_fourier_series_2pi() {
        let x = Symbol::new("x");

        // Test the convenience method for 2π period
        let expr = Expr::Symbol(x.clone()).sin();
        let fourier = expr.fourier_series_2pi(&x, 2);

        // Should have computed coefficients
        assert_eq!(fourier.a_coeffs.len(), 2);
        assert_eq!(fourier.b_coeffs.len(), 2);
    }

    #[test]
    fn test_discrete_fourier_coefficients() {
        let x = Symbol::new("x");

        // Test DFT with simple samples
        let samples = vec![
            Expr::from(1),
            Expr::from(2),
            Expr::from(3),
            Expr::from(4),
        ];

        let coeffs = Expr::discrete_fourier_coefficients(&samples, &x);

        // Should have 4 coefficients (same as number of samples)
        assert_eq!(coeffs.len(), 4);
    }

    #[test]
    fn test_fourier_series_even_function() {
        let x = Symbol::new("x");

        // Fourier series of even function f(x) = cos(x)
        // Should have only cosine terms (a coefficients), no sine terms (b = 0)
        let expr = Expr::Symbol(x.clone()).cos();
        let pi = Expr::Symbol(Symbol::new("pi"));
        let period = Expr::from(2) * pi;

        let fourier = expr.fourier_series(&x, &period, 3);

        // For an even function, sine coefficients should be small/zero
        // (In practice, due to numerical integration, they might not be exactly zero)
        assert_eq!(fourier.b_coeffs.len(), 3);
    }
}
